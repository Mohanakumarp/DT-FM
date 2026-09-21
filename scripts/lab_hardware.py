"""Read hardware and calibrate CPU threads without changing training settings."""
import gc
import json
import os
import subprocess
from types import SimpleNamespace


def inventory():
    import psutil
    from scheduler.resources import host_memory
    graphics = []
    if os.name == 'nt':
        try:
            text = subprocess.check_output(['powershell', '-NoProfile', '-Command',
                'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress'],
                timeout=20, text=True, encoding='utf-8', errors='replace')
            parsed = json.loads(text)
            graphics = parsed if isinstance(parsed, list) else [parsed]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            graphics = [{'inventory_error': str(exc)}]
    return dict(physical_cores=psutil.cpu_count(logical=False), memory=host_memory(), graphics=graphics)


def model_fields(config):
    defaults = dict(embedding_dim=128, seq_length=64, num_heads=4, batch_size=8, micro_batch_size=2)
    return {key: config.get(key, value) for key, value in defaults.items()}


def memory_preflight(config, rank, computers, vocab_size):
    from scheduler.resources import host_memory, memory_model
    model = dict(model_fields(config), vocab_size=vocab_size)
    available = host_memory()['available_bytes']
    # Compare every requested single-computer baseline before starting any job.
    single = config.get('all_single', False) or rank == 0
    size = 1 if single else computers
    stage = 0 if single else rank
    layers = config['layers'] if single else config['layers'] // computers
    estimate = memory_model(model, stage, size)
    model_bytes = estimate['base_bytes'] + layers * estimate['per_layer_bytes']
    required = model_bytes + 512 * 1048576  # Python/PyTorch/dataset overhead reserve.
    budget = int(available * .75)
    if required > budget:
        raise ValueError('Memory preflight: estimated model + runtime %.2f GiB exceeds safe budget %.2f GiB '
                         'on rank %d. Close unused apps or choose smaller dimensions. '
                         'This is an estimate, not an observed OOM.' % (required / 2**30, budget / 2**30, rank))
    return dict(estimated_model_bytes=model_bytes, runtime_reserve_bytes=512 * 1048576,
                available_bytes=available, safe_budget_bytes=budget,
                semantics='conservative estimate, not a measured memory-capacity result')


def calibrate_threads(config, vocab_size):
    import psutil
    import torch
    from scripts.profile_compute import profile_rank
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count() or physical
    candidates = sorted({1, 2, 4, physical, logical} & set(range(1, logical + 1)))
    samples = []
    for threads in candidates:
        torch.set_num_threads(threads)
        args = SimpleNamespace(**model_fields(config), vocab_size=vocab_size,
            measured_rank=0, max_layers=config['layers'], device='cpu', device_backend='cpu',
            warmup=1, repeats=3)
        print('[CPU calibration] measuring %d threads' % threads, flush=True)
        profile = profile_rank(args, device=torch.device('cpu'))['rank']
        estimated = config['layers'] * profile['layer_ms'] + profile['first_stage_ms'] + profile['last_stage_ms']
        samples.append(dict(threads=threads, estimated_full_model_ms=estimated, component_profile=profile))
        gc.collect()
    best = min(samples, key=lambda item: item['estimated_full_model_ms'])['threads']
    torch.set_num_threads(best)
    print('[CPU calibration] selected %d threads; used for every trial on this computer' % best, flush=True)
    return dict(selected_threads=best, candidates=samples,
                method='median isolated component forward/backward/SGD; estimated full-model time')
