"""Read current allocation headroom, without initializing unrelated GPU backends."""
import ctypes
import os
from pathlib import Path
import socket
import time


def host_memory():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in
                ('total_physical', 'available_physical', 'total_commit', 'available_commit',
                 'total_virtual', 'available_virtual', 'extended_virtual')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        api = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
        api.argtypes = [ctypes.POINTER(MemoryStatus)]
        api.restype = ctypes.c_int
        if not api(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(total_bytes=status.total_physical,
                    available_bytes=min(status.available_physical, status.available_commit),
                    physical_available_bytes=status.available_physical,
                    commit_available_bytes=status.available_commit,
                    source='GlobalMemoryStatusEx: min(available physical, available commit)')
    meminfo = Path('/proc/meminfo')
    if not meminfo.is_file():
        raise RuntimeError('automatic host memory discovery requires Windows or Linux')
    fields = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in meminfo.read_text().splitlines() if ':' in line}
    available = fields['MemAvailable']
    # Include every visible ancestor limit. A container's root may already be
    # its cgroup mount; checking root as well covers that namespace arrangement.
    limits = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, group = line.split(':', 2)
        if controllers == '':
            root = Path('/sys/fs/cgroup')
            names = ('memory.max', 'memory.current')
        elif 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory')
            names = ('memory.limit_in_bytes', 'memory.usage_in_bytes')
        else:
            continue
        target = (root / group.lstrip('/')).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError('invalid cgroup path')
        for directory in [target, *target.parents]:
            if directory != root and root not in directory.parents:
                break
            maximum, current = directory / names[0], directory / names[1]
            if maximum.is_file() and current.is_file():
                cap = maximum.read_text().strip()
                if cap != 'max':
                    limits.append(max(0, int(cap) - int(current.read_text())))
    if limits:
        available = min(available, *limits)
    return dict(total_bytes=fields['MemTotal'], available_bytes=available,
                physical_available_bytes=fields['MemAvailable'],
                source='/proc/meminfo MemAvailable, bounded by visible cgroup limits')


def resource_snapshot(args, device):
    import torch
    host = host_memory()
    backend = args.device_backend
    result = dict(host=getattr(args, 'scheduler_host_id', None) or socket.gethostname(), host_memory=host,
                  sampled_at_unix=time.time(), device=backend,
                  cpu_threads=torch.get_num_threads(), logical_cpus=os.cpu_count(),
                  device_memory=None, device_pool=None)
    if backend == 'cpu':
        return result
    if backend == 'directml':
        raise RuntimeError('DirectML does not expose supported free-device-memory telemetry; '
                           'use an explicit static schedule until a memory-budget provider is available')
    api = torch.cuda if backend in ('cuda', 'rocm') else torch.xpu
    getter = getattr(api, 'mem_get_info', None) or getattr(getattr(api, 'memory', None), 'mem_get_info', None)
    if getter is None:
        raise RuntimeError(backend + ' runtime cannot report global free memory')
    free, total = getter(device)
    # UUIDs disambiguate CUDA_VISIBLE_DEVICES remapping. If unavailable, group
    # all same-backend ranks on this host together, conservatively sharing the
    # smallest observed device budget rather than double-counting one adapter.
    properties = api.get_device_properties(device)
    identity = getattr(properties, 'uuid', None)
    result.update(device_memory=dict(available_bytes=int(free), total_bytes=int(total),
                                     source=backend + '.mem_get_info'),
                  device_pool=backend + ':' + (str(identity) if identity else 'shared-unknown-adapter'))
    return result


def memory_model(model, stage, size):
    """Conservative FP32/SGD estimate for this repository's checkpointed GPT.

    Parameters and gradients use exact element counts. Retained activations and
    attention workspace are estimates, with 25% padding; this is not a measured
    peak-memory guarantee. No momentum/Adam/offload state is supported here.
    """
    d, seq, batch, micro, heads, vocab = (model[k] for k in (
        'embedding_dim', 'seq_length', 'batch_size', 'micro_batch_size', 'num_heads', 'vocab_size'))
    layer_parameters = 12 * d * d + 13 * d
    layer_bytes = 8 * layer_parameters + 4 * batch * seq * d * 12
    base = 4 * (batch * seq * d * 6 + micro * heads * seq * seq * 6 + micro * seq * d * 24)
    if stage == 0:
        base += 8 * (vocab * d + seq * d) + 4 * batch * seq * d * 2
    if stage == size - 1:
        base += 8 * (d * d + 3 * d + 2) + 4 * batch * d * 4
    return dict(base_bytes=(base * 5 + 3) // 4,
                per_layer_bytes=(layer_bytes * 5 + 3) // 4,
                semantics='exact FP32 parameter/gradient counts plus estimated activations/workspace, padded 25%')
