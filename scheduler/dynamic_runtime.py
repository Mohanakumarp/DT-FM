"""Collect live resources and agree on stage sizes before creating model weights."""
import copy
import gc
import json
import math
from pathlib import Path

from scheduler.pipeline_scheduler import apply_stage_assignment, MODEL_FIELDS
from scheduler.resource_scheduler import build_resource_plan
from scheduler.resources import resource_snapshot


def collect(args, action):
    """Exchange local failures as data so peers leave this phase together."""
    import torch.distributed as dist
    try:
        local = dict(value=action(), error=None)
    except Exception as exc:
        local = dict(value=None, error='%s: %s' % (type(exc).__name__, exc))
    values = [local]
    if args.world_size > 1:
        values = [None] * args.world_size
        dist.all_gather_object(values, local)
    failures = ['rank %d: %s' % (r, item['error']) for r, item in enumerate(values) if item['error']]
    if failures:
        raise ValueError('dynamic scheduler failed; ' + '; '.join(failures))
    return [item['value'] for item in values]


def validate_dynamic_args(args):
    if (args.stage_layers is not None or args.data_group_size != 1
            or args.world_size != args.pipeline_group_size or args.pp_mode != 'gpipe'
            or args.tensor_comm != 'gloo' or args.fp16 or args.use_offload
            or args.task != 'SeqClassification' or args.profiling != 'no-profiling'):
        raise ValueError('dynamic scheduling requires FP32, pipeline-only Gloo GPipe, SeqClassification, '
                         'no offload, no profiler, and no --stage-layers')
    if not args.world_size <= args.dynamic_total_layers <= 4096 or args.world_size > 12:
        raise ValueError('dynamic total layers must be between world_size and 4096, with at most 12 ranks')
    if not math.isfinite(args.scheduler_memory_fraction) or not 0 < args.scheduler_memory_fraction <= 1:
        raise ValueError('scheduler memory fraction must be in (0, 1]')
    if args.scheduler_reserve_mb < 0 or args.scheduler_warmup < 0 or args.scheduler_repeats < 1:
        raise ValueError('scheduler reserve/warmup must be nonnegative and repeats must be positive')
    if getattr(args, 'rebalance_every', 0) < 0:
        raise ValueError('rebalance interval must be nonnegative')
    threshold = getattr(args, 'rebalance_min_improvement', .1)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('rebalance minimum improvement must be in [0, 1)')


def measure_live_reports(args, device, model):
    import torch
    from scripts.profile_compute import profile_rank
    snapshots = collect(args, lambda: resource_snapshot(args, device))
    reports = [dict(rank=r, resources=snapshot, timings=dict(layer_ms=1., first_stage_ms=0., last_stage_ms=0.))
               for r, snapshot in enumerate(snapshots)]
    # Check minimum stage footprints before allocating profiling tensors.
    build_resource_plan(model, reports, args.world_size, args.scheduler_memory_fraction,
                        args.scheduler_reserve_mb * 1048576)
    profile_args = copy.copy(args)
    profile_args.measured_rank, profile_args.max_layers = args.rank, 4096
    profile_args.device, profile_args.vocab_size = args.device_backend, model['vocab_size']
    profile_args.warmup, profile_args.repeats = args.scheduler_warmup, args.scheduler_repeats

    def calibrate():
        print('[scheduler] profiling compute on launch rank %d' % args.rank, flush=True)
        result = profile_rank(profile_args, device=device, stage=args.rank, size=args.world_size)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'xpu':
            torch.xpu.empty_cache()
        return {key: result['rank'][key] for key in ('layer_ms', 'first_stage_ms', 'last_stage_ms')}

    # Serial calibration avoids overloading a shared host with simultaneous
    # profiling allocations. External workload still affects the measured rate.
    for active in range(args.world_size):
        timings = collect(args, lambda: calibrate() if args.rank == active else None)
        reports[active]['timings'] = timings[active]
    snapshots = collect(args, lambda: resource_snapshot(args, device))
    for report, snapshot in zip(reports, snapshots):
        report['resources'] = snapshot
    return reports


def negotiate(args, device, local_vocab_size):
    import torch
    collect(args, lambda: validate_dynamic_args(args))
    vocabs = collect(args, lambda: local_vocab_size)
    if vocabs[0] <= 0 or vocabs[-1] != vocabs[0]:
        raise ValueError('dynamic scheduler endpoint vocabularies disagree')
    model = {key: getattr(args, key) for key in MODEL_FIELDS if key != 'vocab_size'}
    model['vocab_size'] = vocabs[0]
    reports = measure_live_reports(args, device, model)
    plan = build_resource_plan(model, reports, args.dynamic_total_layers,
                               args.scheduler_memory_fraction, args.scheduler_reserve_mb * 1048576)
    # Explicit consensus catches accidental code/version differences before any
    # stage allocates model weights. Every rank computed from the same reports.
    encodings = collect(args, lambda: json.dumps(plan, sort_keys=True, allow_nan=False))
    if any(value != encodings[0] for value in encodings):
        raise ValueError('ranks disagree on the live resource plan')
    args.stage_layers = ','.join(map(str, plan['stage_layers']))
    apply_stage_assignment(args)
    args.resource_schedule = plan
    torch.manual_seed(args.seed)

    def persist():
        path = Path(args.metrics_dir) / ('resource_schedule_rank%d.json' % args.rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    collect(args, persist)
    print('[scheduler] live stage layers=%s; rank %d owns [%d, %d)' % (
        args.stage_layers, args.rank, args.layer_start, args.layer_end), flush=True)
    return plan
