"""Periodic live planning and coordinated migration between completed steps."""
import gc
import json
import time

from scheduler.dynamic_runtime import collect, measure_live_reports
from scheduler.migration import migrate, preserve_rng
from scheduler.resource_scheduler import build_resource_plan


def estimated_time(reports, counts):
    return max(count * report['timings']['layer_ms']
               + (report['timings']['first_stage_ms'] if rank == 0 else 0)
               + (report['timings']['last_stage_ms'] if rank == len(counts) - 1 else 0)
               for rank, (report, count) in enumerate(zip(reports, counts)))


def fits_memory(plan, counts):
    estimates = plan['memory_estimates']
    return all(sum(estimates[rank]['base_bytes'] + counts[rank] * estimates[rank]['per_layer_bytes']
                   for rank in pool['ranks']) <= pool['budget_bytes'] for pool in plan['memory_pools'])


def maybe_rebalance(args, pipeline, completed_steps, total_steps, metrics=None):
    interval = getattr(args, 'rebalance_every', 0)
    if not interval or completed_steps % interval or completed_steps >= total_steps:
        return None
    started = time.perf_counter()
    old_counts = [int(n) for n in args.stage_layers.split(',')]
    event = dict(completed_steps=completed_steps, old_stage_layers=old_counts,
                 new_stage_layers=old_counts, status='unchanged', moved_layers=0)
    try:
        with preserve_rng(pipeline.device):
            model = args.resource_schedule['model']
            reports = measure_live_reports(args, pipeline.device, model)
            plans = collect(args, lambda: build_resource_plan(
                model, reports, sum(old_counts), args.scheduler_memory_fraction, args.scheduler_reserve_mb * 1048576))
            plan = plans[0]
            if any(candidate != plan for candidate in plans):
                raise ValueError('ranks disagree on the rebalance plan')
            proposed = plan['stage_layers']
            before, after = estimated_time(reports, old_counts), plan['estimated_bottleneck_ms']
            improvement = (before - after) / before
            memory_pressure = not fits_memory(plan, old_counts)
            event.update(proposed_stage_layers=proposed, estimated_improvement=improvement,
                         memory_pressure=memory_pressure)
            if proposed != old_counts and (memory_pressure or improvement >= args.rebalance_min_improvement):
                event.update(migrate(args, pipeline, proposed))
                args.resource_schedule = plan
                event.update(new_stage_layers=proposed, resource_schedule=plan)
            elif proposed != old_counts:
                event.update(status='skipped', reason='estimated gain below the migration threshold')
    except ValueError as exc:
        # collect() reports a preparation/planning failure on every rank. The
        # original model and SGD state remain active. Transport errors propagate.
        event.update(status='skipped', reason=str(exc))
    gc.collect()
    event['elapsed_s'] = time.perf_counter() - started
    if metrics is not None:
        metrics.record_rebalance(event)
    print('[scheduler] rebalance ' + json.dumps(event, allow_nan=False), flush=True)
    return event
