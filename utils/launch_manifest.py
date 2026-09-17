"""Pure-Python adapter from verify_comm_probe.py JSON to GPipe launches."""
import copy
import math
import re
import shlex
from scheduler.pipeline_scheduler import allocate_layers, validate_compute_profile


def validate_profile(profile):
    """Check the verifier envelope and matrices without filling missing links."""
    if not isinstance(profile, dict) or profile.get('passed') is not True:
        raise ValueError('expected a passed communication-probe verification JSON')
    size = profile.get('world_size')
    if type(size) is not int or not 2 <= size <= 12:
        raise ValueError('world_size must be an integer from 2 to 12')
    for field in ('latency_ms', 'bandwidth_mbps'):
        matrix = profile.get(field)
        if not isinstance(matrix, list) or len(matrix) != size:
            raise ValueError(field + ' must have world_size rows')
        for src, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != size:
                raise ValueError(field + ' must be square')
            for dst, value in enumerate(row):
                if src != dst and value is None:
                    continue
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or (value != 0 if src == dst else value <= 0)):
                    raise ValueError('%s has invalid value at %d->%d' % (field, src, dst))
    for src in range(size):
        for dst in range(size):
            if ((profile['latency_ms'][src][dst] is None)
                    != (profile['bandwidth_mbps'][src][dst] is None)):
                raise ValueError('latency and bandwidth availability must match')
    return size


def _ordering(profile, payload_bytes):
    size = profile['world_size']
    latency, bandwidth = profile['latency_ms'], profile['bandwidth_mbps']

    def edge(src, dst):
        if latency[src][dst] is None or latency[dst][src] is None:
            return None
        # Full measured RTT is deliberately retained as a conservative proxy.
        return sum(latency[a][b] + payload_bytes * 8 / (bandwidth[a][b] * 1000)
                   for a, b in ((src, dst), (dst, src)))

    # Held-Karp path search: O(n^2 * 2^n), deterministic rank-order tie breaks.
    states = {(1 << rank, rank): (0.0, (rank,)) for rank in range(size)}
    for mask in range(1, 1 << size):
        for last in range(size):
            current = states.get((mask, last))
            if current is None:
                continue
            for nxt in range(size):
                if mask & (1 << nxt):
                    continue
                cost = edge(last, nxt)
                if cost is None:
                    continue
                candidate = (current[0] + cost, current[1] + (nxt,))
                if not math.isfinite(candidate[0]):
                    raise ValueError('transfer cost overflows; check profile values')
                key = (mask | (1 << nxt), nxt)
                if key not in states or candidate < states[key]:
                    states[key] = candidate
    complete = [value for (mask, _), value in states.items() if mask == (1 << size) - 1]
    if not complete:
        raise ValueError('no full-rank pipeline with available forward and backward links')
    cost, order = min(complete)
    return list(order), cost


def build_manifest(profile, hosts, devices=None, payload_bytes=1048576, port=9000,
                   log_port=9100, compute_profile=None, total_layers=None, dynamic=False, rebalance_every=0):
    """Hosts/devices are indexed by measured rank, never by selected stage."""
    size = validate_profile(profile)
    if (not isinstance(hosts, list) or len(hosts) != size
            or any(not isinstance(host, str)
                   or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host) for host in hosts)):
        raise ValueError('provide one IPv4 address or DNS hostname per measured rank')
    devices = ['cpu'] * size if devices is None else devices
    if (not isinstance(devices, list) or len(devices) != size
            or any(device not in ('cpu', 'cuda', 'rocm', 'xpu', 'directml') for device in devices)):
        raise ValueError('provide one explicit supported device per measured rank')
    if type(payload_bytes) is not int or not 1 <= payload_bytes <= 2**53:
        raise ValueError('payload_bytes must be an integer from 1 to 2^53')
    if any(type(p) is not int or not 1 <= p <= 65535 for p in (port, log_port)) or port == log_port:
        raise ValueError('port and log_port must be distinct TCP ports from 1 to 65535')
    order, cost = _ordering(profile, payload_bytes)
    schedule = None
    if type(dynamic) is not bool:
        raise ValueError('dynamic must be a boolean')
    if type(rebalance_every) is not int or rebalance_every < 0 or (rebalance_every and not dynamic):
        raise ValueError('rebalance_every must be nonnegative and requires dynamic scheduling')
    if dynamic and (compute_profile is not None or type(total_layers) is not int or not size <= total_layers <= 4096):
        raise ValueError('dynamic scheduling requires world_size <= total_layers <= 4096 and no static compute profile')
    if compute_profile is not None:
        validate_compute_profile(compute_profile, size)
        if [r['device'] for r in compute_profile['ranks']] != devices:
            raise ValueError('compute profile devices must match launch devices in measured rank order')
        schedule = allocate_layers(compute_profile, order, total_layers)
    elif total_layers is not None and not dynamic:
        raise ValueError('total_layers requires a compute profile')
    master = hosts[order[0]]
    ranks = []
    for rank, measured_rank in enumerate(order):
        env = dict(RANK=str(rank), WORLD_SIZE=str(size), PP_SIZE=str(size), DP_SIZE='1',
                   MASTER_IP=master, PORT=str(port), LOG_PORT=str(log_port),
                   DEVICE=devices[measured_rank], TENSOR_COMM='gloo', SKIP_PROBE='true')
        if dynamic:
            env.update(DYNAMIC_TOTAL_LAYERS=str(total_layers))
            if rebalance_every:
                env['REBALANCE_EVERY'] = str(rebalance_every)
        if schedule is not None:
            model = compute_profile['model']
            env.update(STAGE_LAYERS=','.join(map(str, schedule['stage_layers'])),
                       PROFILE='scheduled', SEQ=str(model['seq_length']),
                       EMBED=str(model['embedding_dim']), HEADS=str(model['num_heads']),
                       BATCH=str(model['batch_size']), MICRO=str(model['micro_batch_size']),
                       SYNTHETIC_VOCAB_SIZE=str(model['vocab_size']),
                       SCHEDULE_VOCAB_SIZE=str(model['vocab_size']))
        command = 'env ' + ' '.join(shlex.quote(k + '=' + v) for k, v in env.items())
        command += ' bash scripts/run_rank.sh'
        ranks.append(dict(rank=rank, measured_rank=measured_rank, host=hosts[measured_rank],
                          device=devices[measured_rank], env=env, command=command))
    manifest = dict(schema_version=1, profile=copy.deepcopy(profile),
                hosts=list(hosts), devices=list(devices), world_size=size,
                pipeline_group_size=size, data_group_size=1,
                pipeline_order=order, master_host=master, port=port, log_port=log_port,
                selection=dict(method='minimum_bidirectional_path', payload_bytes=payload_bytes,
                               cost_ms=cost, cost_semantics='sum of directed RTT_ms + bytes*8/(Mbps*1000)'),
                ranks=ranks)
    if schedule is not None:
        manifest.update(schema_version=2, compute_profile=copy.deepcopy(compute_profile),
                        schedule=schedule)
        for entry, stage in zip(ranks, schedule['stages']):
            entry['assignment'] = copy.deepcopy(stage)
    if dynamic:
        manifest.update(schema_version=3, dynamic_schedule=dict(
            total_layers=total_layers, method='live_memory_compute_greedy',
            assignment_time='startup_after_live_resource_discovery'))
        if rebalance_every:
            manifest['dynamic_schedule']['rebalance_every'] = rebalance_every
    return manifest


def validate_manifest(manifest):
    """Recompute all derived fields, including commands, before accepting JSON."""
    try:
        expected = build_manifest(
            manifest['profile'], manifest['hosts'], manifest['devices'],
            manifest['selection']['payload_bytes'], manifest['port'], manifest['log_port'],
            manifest.get('compute_profile'),
            manifest.get('dynamic_schedule', manifest.get('schedule', {})).get('total_layers'),
            'dynamic_schedule' in manifest,
            manifest.get('dynamic_schedule', {}).get('rebalance_every', 0))
        # JSON equality also distinguishes booleans from integer rank identifiers.
        import json
        if json.dumps(manifest, sort_keys=True, allow_nan=False) != json.dumps(
                expected, sort_keys=True, allow_nan=False):
            raise ValueError('manifest differs from the checked profile-to-launch mapping')
    except (KeyError, TypeError, OverflowError, AttributeError) as exc:
        raise ValueError('invalid launch manifest: %s' % exc) from exc
    return manifest
