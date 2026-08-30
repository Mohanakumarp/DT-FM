import sys

import torch


def _xpu_available():
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def _load_directml():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            '--device directml requires Python 3.11; current interpreter is {}'.format(
                sys.version.split()[0]
            )
        )
    try:
        import torch_directml
    except ImportError:
        raise RuntimeError(
            '--device directml requires torch-directml in a separate Python 3.11 environment'
        )
    return torch_directml


def resolve_device(args):
    """Resolve the requested compute backend and keep old CLI flags working."""
    requested = getattr(args, 'device', 'auto').lower()
    legacy_use_cuda = getattr(args, 'use_cuda', None)
    if requested == 'auto' and legacy_use_cuda is not None:
        requested = 'cuda' if legacy_use_cuda else 'cpu'

    hip_available = torch.cuda.is_available() and torch.version.hip is not None
    cuda_available = torch.cuda.is_available() and torch.version.hip is None
    xpu_available = _xpu_available()

    if requested == 'auto':
        if hip_available:
            requested = 'rocm'
        elif cuda_available:
            requested = 'cuda'
        elif xpu_available:
            requested = 'xpu'
        else:
            requested = 'cpu'

    if requested == 'cpu':
        device = torch.device('cpu')
        backend = 'cpu'
    elif requested == 'rocm':
        if not hip_available:
            raise RuntimeError(
                '--device rocm requires a ROCm-enabled PyTorch build with an available AMD GPU'
            )
        device = torch.device('cuda', args.cuda_id)
        backend = 'rocm'
    elif requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requires an available CUDA or ROCm device')
        device = torch.device('cuda', args.cuda_id)
        backend = 'rocm' if hip_available else 'cuda'
    elif requested == 'xpu':
        if not xpu_available:
            raise RuntimeError('--device xpu requires an Intel GPU-enabled PyTorch build')
        device = torch.device('xpu', args.cuda_id)
        backend = 'xpu'
    elif requested == 'directml':
        torch_directml = _load_directml()
        directml_id = getattr(args, 'directml_id', 0)
        device_count = torch_directml.device_count()
        if directml_id < 0 or directml_id >= device_count:
            raise RuntimeError(
                '--directml-id {} is invalid; DirectML reports {} device(s)'.format(
                    directml_id, device_count
                )
            )
        device = torch_directml.device(directml_id)
        backend = 'directml'
    else:
        raise ValueError('Unknown --device value: ' + requested)

    args.device_backend = backend
    args.use_cuda = device.type == 'cuda'
    # Keep the first XPU implementation conservative. Intel GPU transfers work
    # without DataLoader pinning, and pinned-memory behavior differs by release.
    args.pin_memory = backend in ('cuda', 'rocm')
    return device


def validate_runtime_args(args, device):
    if args.fp16 and args.device_backend in ('cpu', 'xpu', 'directml'):
        raise ValueError('--fp16 is not supported by the CPU/XPU/DirectML pipeline; use FP32')
    if args.profiling == 'tidy_profiling' and device.type != 'cuda':
        raise ValueError('tidy profiling requires CUDA or ROCm; use --profiling no-profiling')
    if args.device_backend == 'xpu':
        if args.world_size != 1:
            raise ValueError('Intel XPU currently supports only --world-size 1 in DT-FM')
        if args.profiling != 'no-profiling':
            raise ValueError('Intel XPU currently requires --profiling no-profiling')
        if args.data_group_size != 1:
            raise ValueError('Intel XPU data parallelism is not implemented')
    if args.device_backend == 'directml':
        if args.world_size != 1:
            raise ValueError('DirectML currently supports only --world-size 1 in DT-FM')
        if args.pipeline_group_size != 1:
            raise ValueError('DirectML currently supports only --pipeline-group-size 1 in DT-FM')
        if args.profiling != 'no-profiling':
            raise ValueError('DirectML currently requires --profiling no-profiling')
        if args.data_group_size != 1:
            raise ValueError('DirectML data parallelism is not implemented')
    if args.device_backend == 'rocm' and args.world_size != 1:
        raise ValueError('ROCm currently supports only --world-size 1 in DT-FM')
    if device.type == 'cpu' and args.data_group_size != 1:
        raise ValueError('CPU mode currently supports pipeline parallelism only; set --data-group-size 1')
    if args.world_size != args.data_group_size * args.pipeline_group_size:
        raise ValueError('world-size must equal data-group-size times pipeline-group-size')


def describe_device(device, backend):
    if backend == 'cpu':
        return 'CPU'
    if backend == 'xpu':
        index = device.index or 0
        return 'Intel XPU device {} ({})'.format(
            index, torch.xpu.get_device_name(index)
        )
    if backend == 'directml':
        torch_directml = _load_directml()
        index = device.index or 0
        return 'DirectML device {} ({})'.format(
            index, torch_directml.device_name(index)
        )
    name = torch.cuda.get_device_name(device.index or 0)
    if backend == 'rocm':
        return 'AMD ROCm device {} ({})'.format(device.index or 0, name)
    return 'NVIDIA CUDA device {} ({})'.format(device.index or 0, name)
