import torch


def _xpu_available():
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def resolve_device(args):
    """Resolve the requested compute backend and keep old CLI flags working."""
    requested = getattr(args, 'device', 'auto').lower()
    legacy_use_cuda = getattr(args, 'use_cuda', None)
    if requested == 'auto' and legacy_use_cuda is not None:
        requested = 'cuda' if legacy_use_cuda else 'cpu'

    hip_available = torch.cuda.is_available() and torch.version.hip is not None
    cuda_available = torch.cuda.is_available() and torch.version.hip is None

    if requested == 'auto':
        if hip_available:
            requested = 'rocm'
        elif cuda_available:
            requested = 'cuda'
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
        if not _xpu_available():
            raise RuntimeError('--device xpu requires an Intel GPU-enabled PyTorch build')
        device = torch.device('xpu', args.cuda_id)
        backend = 'xpu'
    else:
        raise ValueError('Unknown --device value: ' + requested)

    args.device_backend = backend
    args.use_cuda = device.type == 'cuda'
    args.pin_memory = device.type != 'cpu'
    return device


def validate_runtime_args(args, device):
    if args.fp16 and device.type == 'cpu':
        raise ValueError('--fp16 is not supported by the CPU pipeline; use FP32')
    if args.profiling == 'tidy_profiling' and device.type == 'cpu':
        raise ValueError('tidy profiling requires an accelerator; use --profiling no-profiling on CPU')
    if args.device_backend == 'xpu':
        raise ValueError('Intel XPU execution is not implemented yet; use --device cpu')
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
        return 'Intel XPU device {}'.format(device.index or 0)
    name = torch.cuda.get_device_name(device.index or 0)
    if backend == 'rocm':
        return 'AMD ROCm device {} ({})'.format(device.index or 0, name)
    return 'NVIDIA CUDA device {} ({})'.format(device.index or 0, name)
