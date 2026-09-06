"""Run bounded local CPU training and check every persisted probe matrix."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def verify(args):
    root = Path(__file__).resolve().parents[1]
    # A fresh directory prevents stale metrics from satisfying this run.
    parent = Path(args.output_dir).resolve() if args.output_dir else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='dtfm-probe-', dir=parent))
    print('Artifacts: {}'.format(output), flush=True)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]

    command = [
        sys.executable, '-u', str(root / 'dist_runner.py'),
        '--device', 'cpu', '--tensor-comm', 'gloo', '--dist-backend', 'gloo',
        '--dist-url', 'tcp://127.0.0.1:{}'.format(port),
        '--world-size', str(args.world_size),
        '--pipeline-group-size', str(args.world_size), '--data-group-size', '1',
        '--synthetic-data', 'true', '--synthetic-samples', '8',
        '--synthetic-vocab-size', '512', '--seq-length', '32',
        '--embedding-dim', '64', '--num-layers', '1', '--num-heads', '4',
        '--batch-size', '2', '--micro-batch-size', '1', '--num-iters', '2',
        '--metrics-dir', str(output), '--skip-comm-probe', 'false',
        '--use-offload', 'false', '--profiling', 'no-profiling',
    ]
    processes = []
    with ExitStack() as stack:
        if args.occupy_probe_rank is not None:
            blocker = stack.enter_context(socket.socket())
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            blocker.bind(('0.0.0.0', 9200 + args.occupy_probe_rank))
            blocker.listen(1)
        try:
            for rank in range(args.world_size):
                log = stack.enter_context((output / 'rank{}.log'.format(rank)).open('w'))
                processes.append(subprocess.Popen(
                    command + ['--rank', str(rank)], cwd=str(root),
                    stdout=log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + args.timeout
            while True:
                codes = [process.poll() for process in processes]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError('worker failed: {}'.format(codes))
                if all(code == 0 for code in codes):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('workers exceeded {} seconds'.format(args.timeout))
                time.sleep(.1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()

    metrics = [json.loads((output / 'metrics_rank{}.json'.format(rank)).read_text())
               for rank in range(args.world_size)]
    for rank, payload in enumerate(metrics):
        if payload['rank'] != rank or payload['schedule']['completed_steps'] != 2:
            raise ValueError('rank {} did not complete two steps'.format(rank))
        for field in ('latency_ms', 'bandwidth_mbps'):
            matrix = payload[field]
            if matrix != metrics[0][field] or len(matrix) != args.world_size:
                raise ValueError('inconsistent {} on rank {}'.format(field, rank))
            for src, row in enumerate(matrix):
                if len(row) != args.world_size:
                    raise ValueError('invalid matrix width')
                for dst, value in enumerate(row):
                    if src == dst:
                        valid = value == 0.0
                    elif dst == args.occupy_probe_rank:
                        valid = value is None
                    else:
                        valid = (isinstance(value, (int, float))
                                 and math.isfinite(value) and value > 0)
                    if not valid:
                        raise ValueError('invalid {} for {}->{}: {}'.format(
                            field, src, dst, value))
    result = {
        'passed': True, 'world_size': args.world_size, 'steps_per_rank': 2,
        'occupied_probe_rank': args.occupy_probe_rank,
        'scope': 'local CPU processes; physical WAN not tested',
        'latency_ms': metrics[0]['latency_ms'],
        'bandwidth_mbps': metrics[0]['bandwidth_mbps'],
    }
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', type=int, choices=(2, 3), default=3)
    parser.add_argument('--occupy-probe-rank', type=int,
                        help='hold this rank probe port open to test bind failure')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--output-dir', help='parent for a fresh artifact directory')
    args = parser.parse_args()
    if args.occupy_probe_rank is not None and not 0 <= args.occupy_probe_rank < args.world_size:
        parser.error('--occupy-probe-rank must be an existing rank')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    verify(args)


if __name__ == '__main__':
    main()
