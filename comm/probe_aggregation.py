"""Validate and combine per-rank communication probe measurements."""
import math


def _unavailable_matrix(world_size):
    return [
        [0.0 if src == dst else None for dst in range(world_size)]
        for src in range(world_size)
    ]


def _positive_finite(value, field, src, dst):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('%s for %d->%d must be a number' % (field, src, dst))
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError('%s for %d->%d must be finite and positive' % (
            field, src, dst))
    return value


def aggregate_probe_results(rank_results, world_size):
    """Return complete directed matrices from one contribution per rank.

    Each rank owns the measurements whose source matches that rank. A missing
    direction, or a direction recorded with both values set to ``None``, stays
    unavailable in both matrices. This keeps failed links distinct from the
    zero-valued diagonal.
    """
    if (isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size < 1):
        raise ValueError('world_size must be a positive integer')
    if (not isinstance(rank_results, (list, tuple))
            or len(rank_results) != world_size):
        raise ValueError('expected one probe contribution per rank')

    latency = _unavailable_matrix(world_size)
    bandwidth = _unavailable_matrix(world_size)
    seen_ranks = set()
    seen_directions = set()

    for result in rank_results:
        if not isinstance(result, dict):
            raise ValueError('each probe contribution must be a dictionary')
        rank = result.get('rank')
        measurements = result.get('measurements')
        if (isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 0 <= rank < world_size):
            raise ValueError('probe contribution has an invalid rank')
        if rank in seen_ranks:
            raise ValueError('duplicate probe contribution for rank %d' % rank)
        if not isinstance(measurements, (list, tuple)):
            raise ValueError('measurements for rank %d must be a list' % rank)
        seen_ranks.add(rank)

        for measurement in measurements:
            if not isinstance(measurement, dict):
                raise ValueError('each probe measurement must be a dictionary')
            src = measurement.get('src')
            dst = measurement.get('dst')
            if (isinstance(src, bool) or not isinstance(src, int)
                    or src != rank):
                raise ValueError('rank %d cannot provide source row %r' % (rank, src))
            if (isinstance(dst, bool)
                    or not isinstance(dst, int)
                    or not 0 <= dst < world_size):
                raise ValueError(
                    'measurement from rank %d has an invalid destination' % rank)
            if src == dst:
                raise ValueError('self-links are implicit and must not be measured')
            direction = (src, dst)
            if direction in seen_directions:
                raise ValueError('duplicate probe measurement for %d->%d' % direction)
            seen_directions.add(direction)

            latency_ms = measurement.get('latency_ms')
            bandwidth_mbps = measurement.get('bandwidth_mbps')
            if latency_ms is None and bandwidth_mbps is None:
                continue
            if latency_ms is None or bandwidth_mbps is None:
                raise ValueError(
                    'measurement for %d->%d must provide both values or neither'
                    % direction)
            latency[src][dst] = _positive_finite(
                latency_ms, 'latency_ms', src, dst)
            bandwidth[src][dst] = _positive_finite(
                bandwidth_mbps, 'bandwidth_mbps', src, dst)

    if seen_ranks != set(range(world_size)):
        raise ValueError('probe contributions do not cover every rank')
    return latency, bandwidth
