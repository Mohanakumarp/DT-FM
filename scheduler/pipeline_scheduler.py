"""Deterministic, capacity-constrained layer allocation for a network-ordered GPipe."""
import math


MODEL_FIELDS = ('seq_length', 'embedding_dim', 'num_heads', 'batch_size',
                'micro_batch_size', 'vocab_size')


def validate_compute_profile(profile, size):
    if not isinstance(profile, dict) or type(profile.get('schema_version')) is not int or profile['schema_version'] != 1:
        raise ValueError('compute profile requires schema_version 1')
    model = profile.get('model')
    if not isinstance(model, dict) or set(model) != set(MODEL_FIELDS):
        raise ValueError('compute profile model must contain ' + ', '.join(MODEL_FIELDS))
    if any(type(model[k]) is not int or model[k] <= 0 for k in MODEL_FIELDS):
        raise ValueError('compute model dimensions must be positive integers')
    if model['embedding_dim'] % model['num_heads'] or model['batch_size'] % model['micro_batch_size']:
        raise ValueError('embedding and batch sizes must be divisible by heads and microbatch size')
    ranks = profile.get('ranks')
    if not isinstance(ranks, list) or len(ranks) != size:
        raise ValueError('compute profile requires one entry per measured rank')
    for rank, entry in enumerate(ranks):
        if not isinstance(entry, dict) or type(entry.get('measured_rank')) is not int or entry['measured_rank'] != rank:
            raise ValueError('compute entries must be in measured rank order')
        if entry.get('device') not in ('cpu', 'cuda', 'rocm', 'xpu', 'directml'):
            raise ValueError('compute entry requires an explicit device')
        if type(entry.get('max_layers')) is not int or not 1 <= entry['max_layers'] <= 4096:
            raise ValueError('max_layers must be an integer from 1 to 4096')
        for key in ('layer_ms', 'first_stage_ms', 'last_stage_ms'):
            value = entry.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if key == 'layer_ms' else value < 0):
                raise ValueError(key + ' must be finite and ' + ('positive' if key == 'layer_ms' else 'nonnegative'))
    return profile


def allocate_layers(profile, order, total_layers):
    """Minimize max estimated stage compute time for this fixed rank order.

    A layer's time includes forward, backward and SGD for a full batch. Endpoint
    costs are additional embedding/head work. max_layers is an operator-supplied
    capacity including space reserved for endpoints and GPipe activation buffers.
    """
    size = len(order)
    validate_compute_profile(profile, size)
    if (any(type(r) is not int for r in order) or sorted(order) != list(range(size))):
        raise ValueError('order must be a permutation of measured ranks')
    if type(total_layers) is not int or not size <= total_layers <= 4096:
        raise ValueError('total_layers must be an integer between world_size and 4096')
    if sum(r['max_layers'] for r in profile['ranks']) < total_layers:
        raise ValueError('insufficient layer capacity for total_layers')

    # Feasibility is monotonic in the bottleneck time. The optimum is one of
    # the stage costs, so binary-search that finite set, avoiding O(L^2) DP.
    costs = []
    for stage, rank in enumerate(order):
        entry = profile['ranks'][rank]
        overhead = (entry['first_stage_ms'] if stage == 0 else 0) + (
            entry['last_stage_ms'] if stage == size - 1 else 0)
        row = [entry['layer_ms'] * n + overhead
               for n in range(1, min(entry['max_layers'], total_layers - size + 1) + 1)]
        if any(not math.isfinite(value) for value in row):
            raise ValueError('estimated stage time overflows')
        costs.append(row)
    import bisect
    candidates = sorted({value for row in costs for value in row})
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        caps = [bisect.bisect_right(row, candidates[mid]) for row in costs]
        if min(caps) > 0 and sum(caps) >= total_layers:
            hi = mid
        else:
            lo = mid + 1
    bottleneck = candidates[lo]
    caps = [bisect.bisect_right(row, bottleneck) for row in costs]
    # Lexicographically smallest allocation among all minimax solutions.
    remaining = total_layers
    counts = []
    for stage in range(size):
        count = max(1, remaining - sum(caps[stage + 1:]))
        counts.append(count)
        remaining -= count
    stages = []
    start = 0
    for stage, (rank, count) in enumerate(zip(order, counts)):
        stages.append(dict(rank=stage, measured_rank=rank, num_layers=count,
                           layer_start=start, layer_end=start + count,
                           estimated_compute_ms=costs[stage][count - 1]))
        start += count
    return dict(method='network_order_then_minimax_compute', total_layers=total_layers,
                stage_layers=counts, estimated_bottleneck_ms=bottleneck, stages=stages,
                capacity_semantics='operator-supplied layer limits; reserve endpoint and activation memory')


def apply_stage_assignment(args):
    """Resolve a complete stage vector before model construction or tracking."""
    raw = getattr(args, 'stage_layers', None)
    if raw is None:
        return
    try:
        counts = [int(x) for x in raw.split(',')]
    except (ValueError, AttributeError) as exc:
        raise ValueError('--stage-layers requires comma-separated positive integers') from exc
    size = args.pipeline_group_size
    if (len(counts) != size or any(n <= 0 for n in counts) or sum(counts) > 4096
            or args.world_size != size or args.data_group_size != 1
            or not 0 <= args.rank < size or args.pp_mode != 'gpipe'):
        raise ValueError('--stage-layers requires one positive count per rank, at most 4096 total layers, and pipeline-only GPipe')
    args.stage_layers = ','.join(map(str, counts))
    args.num_layers = counts[args.rank]
    args.total_layers = sum(counts)
    args.layer_start = sum(counts[:args.rank])
    args.layer_end = args.layer_start + args.num_layers


def assignment_signature(args):
    """Fields that must agree across ranks, excluding the local layer count."""
    return {key: getattr(args, key, None) for key in (
        'stage_layers', 'seq_length', 'embedding_dim', 'num_heads', 'batch_size',
        'micro_batch_size', 'task', 'synthetic_data', 'synthetic_vocab_size',
        'schedule_vocab_size', 'fp16', 'num_iters', 'num_epochs', 'steps_per_epoch',
        'dynamic_total_layers', 'scheduler_memory_fraction', 'scheduler_reserve_mb',
        'scheduler_warmup', 'scheduler_repeats', 'rebalance_every', 'rebalance_min_improvement')}
