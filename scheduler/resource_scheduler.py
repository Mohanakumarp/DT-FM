"""Allocate live stage sizes from compute rates and shared memory budgets."""
import copy
import math

from scheduler.pipeline_scheduler import validate_compute_profile
from scheduler.resources import memory_model


def build_resource_plan(model, reports, total_layers, memory_fraction=.7, reserve_bytes=268435456):
    size = len(reports)
    if type(total_layers) is not int or not 1 <= size <= 12 or not size <= total_layers <= 4096:
        raise ValueError('dynamic scheduling requires 1-12 ranks and world_size <= total_layers <= 4096')
    if type(memory_fraction) not in (int, float) or not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError('scheduler memory fraction must be in (0, 1]')
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('scheduler memory reserve must be nonnegative')
    ranks = []
    pools = {}
    footprints = [memory_model(model, rank, size) for rank in range(size)]
    memberships = [[] for _ in reports]
    for rank, report in enumerate(reports):
        resource = report['resources']
        if type(report['rank']) is not int or report['rank'] != rank:
            raise ValueError('live reports must be in launch rank order')
        ranks.append(dict(measured_rank=rank, device=resource['device'], max_layers=4096, **report['timings']))
        if not isinstance(resource['host'], str) or not resource['host']:
            raise ValueError('live report requires a host identity')
        memories = [(('host', resource['host']), resource['host_memory'])]
        if resource['device'] != 'cpu':
            if not isinstance(resource.get('device_pool'), str) or not resource['device_pool']:
                raise ValueError('accelerator report requires a device memory pool')
            memories.append((('device', resource['host'], resource['device_pool']), resource['device_memory']))
        for key, memory in memories:
            free, total = memory['available_bytes'], memory['total_bytes']
            if type(free) is not int or type(total) is not int or not 0 <= free <= total or total <= 0:
                raise ValueError('invalid available/total memory in live report')
            budget = max(0, int(free * memory_fraction) - reserve_bytes)
            pool = pools.setdefault(key, dict(resource=list(key), budget_bytes=budget, ranks=[]))
            pool['budget_bytes'] = min(pool['budget_bytes'], budget)
            pool['ranks'].append(rank)
            memberships[rank].append(key)
    validate_compute_profile(dict(schema_version=1, model=model, ranks=ranks), size)
    counts = [1] * size
    for pool in pools.values():
        pool['estimated_used_bytes'] = sum(footprints[r]['base_bytes'] + footprints[r]['per_layer_bytes']
                                           for r in pool['ranks'])
        if pool['estimated_used_bytes'] > pool['budget_bytes']:
            raise ValueError('insufficient live memory for one layer per rank in %s: need %d, budget %d bytes' % (
                '/'.join(pool['resource']), pool['estimated_used_bytes'], pool['budget_bytes']))

    def compute_ms(rank, count):
        times = reports[rank]['timings']
        return count * times['layer_ms'] + (times['first_stage_ms'] if rank == 0 else 0) + (
            times['last_stage_ms'] if rank == size - 1 else 0)

    # Each extra layer goes to the eligible stage with the lowest resulting
    # compute time. Shared host/device pools constrain aggregate use; they are
    # not divided equally between ranks. Rank order makes ties deterministic.
    for _ in range(total_layers - size):
        candidates = [r for r in range(size) if all(
            pools[key]['estimated_used_bytes'] + footprints[r]['per_layer_bytes'] <= pools[key]['budget_bytes']
            for key in memberships[r])]
        if not candidates:
            raise ValueError('insufficient live memory for all %d layers; only %d fit the estimated budgets' % (
                total_layers, sum(counts)))
        rank = min(candidates, key=lambda r: (compute_ms(r, counts[r] + 1), r))
        counts[rank] += 1
        for key in memberships[rank]:
            pools[key]['estimated_used_bytes'] += footprints[rank]['per_layer_bytes']
    stages, start = [], 0
    for rank, count in enumerate(counts):
        estimate = compute_ms(rank, count)
        if not math.isfinite(estimate):
            raise ValueError('estimated compute time overflows')
        stages.append(dict(rank=rank, num_layers=count, layer_start=start, layer_end=start + count,
                           estimated_compute_ms=estimate,
                           estimated_memory_bytes=footprints[rank]['base_bytes'] + count * footprints[rank]['per_layer_bytes']))
        start += count
    return dict(schema_version=1, method='live_memory_compute_greedy', total_layers=total_layers,
                stage_layers=counts, stages=stages, model=copy.deepcopy(model),
                reports=copy.deepcopy(reports), memory_pools=list(pools.values()),
                memory_fraction=memory_fraction, reserve_bytes=reserve_bytes,
                memory_estimates=footprints,
                estimated_bottleneck_ms=max(stage['estimated_compute_ms'] for stage in stages))


def validate_resource_plan(plan):
    import json
    try:
        expected = build_resource_plan(plan['model'], plan['reports'], plan['total_layers'],
                                       plan['memory_fraction'], plan['reserve_bytes'])
        if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError('resource plan differs from the checked live allocation')
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('invalid resource plan: %s' % exc) from exc
    return plan
