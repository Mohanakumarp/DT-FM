"""Run two comparable CPU trials and a two-rank Gloo smoke test."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-uri', default='http://127.0.0.1:5000')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    # Create the experiment before spawning ranks to avoid a first-run race.
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment('DT-FM demo')
    for label, ranks, lr in [('cpu-lr-001', 1, '0.01'),
                              ('cpu-lr-0001', 1, '0.001'),
                              ('two-rank-gloo', 2, '0.01')]:
        group = '%s-%s' % (stamp, label)
        command = [sys.executable, '-u', 'dist_runner.py',
                   '--device', 'cpu', '--spawn-local-ranks', 'true',
                   '--tensor-comm', 'gloo', '--dist-backend', 'gloo',
                   '--dist-url', 'tcp://127.0.0.1:9037',
                   '--world-size', str(ranks), '--pipeline-group-size', str(ranks),
                   '--data-group-size', '1', '--rank', '0',
                   '--synthetic-data', 'true', '--synthetic-vocab-size', '512',
                   '--seq-length', '32', '--embedding-dim', '64',
                   '--num-layers', '1', '--num-heads', '4',
                   '--batch-size', '2', '--micro-batch-size', '1',
                   '--num-iters', str(args.steps), '--lr', lr,
                   '--skip-comm-probe', 'false', '--use-offload', 'false',
                   '--profiling', 'no-profiling',
                   '--metrics-dir', str(root / 'logs' / 'mlflow-demo' / group),
                   '--mlflow-tracking-uri', args.tracking_uri,
                   '--mlflow-experiment', 'DT-FM demo',
                   '--mlflow-run-name', label, '--mlflow-group', group]
        subprocess.run(command, cwd=root, check=True, timeout=300)
    print('Open %s and select DT-FM demo.' % args.tracking_uri)


if __name__ == '__main__':
    main()
