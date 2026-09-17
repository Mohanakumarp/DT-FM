"""Verify real Gloo migration, rollback, and post-migration training continuity."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
from test_migration import run_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', type=int, choices=(2, 3), default=3)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'logs/smoke')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='migration-%drank-' % args.world_size, dir=args.output_dir.resolve()))
    print('Artifacts: ' + str(output), flush=True)
    results = {}
    # Keep the coordinator light while workers run, important on Windows where
    # each PyTorch import consumes additional commit headroom.
    for mode in ('baseline', 'migrate', 'rollback', 'controller'):
        results[mode] = run_workers(output / mode, mode, args.world_size)
        print(mode + ': all ranks completed four steps', flush=True)
    import torch
    def state(mode):
        return {key: value for rank in range(args.world_size) for key, value in torch.load(
            output / mode / ('state_rank%d.pt' % rank), weights_only=True).items()}
    expected = state('baseline')
    for mode in ('migrate', 'rollback', 'controller'):
        actual = state(mode)
        if set(actual) != set(expected):
            raise ValueError('global state keys differ from baseline')
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6, msg=key)
        if results[mode][-1]['losses'] != results['baseline'][-1]['losses']:
            raise ValueError('post-migration losses differ from baseline')
        statuses = ['rolled_back', 'rolled_back'] if mode == 'rollback' else ['committed', 'committed']
        for rank in results[mode]:
            if rank['completed_steps'] != 4 or [event['status'] for event in rank['events']] != statuses:
                raise ValueError('missing migrations or incomplete training')
    report = dict(passed=True, world_size=args.world_size, modes=list(results),
                  scope='local CPU Gloo, controlled assignments/rates, four training steps per mode',
                  parameter_and_momentum_tensors=len(expected),
                  final_stage_losses=results['baseline'][-1]['losses'],
                  checks=['global state checksums across each transition', 'RNG preservation',
                          'two committed migrations', 'unanimous rollback after injected candidate failure',
                          'resource controller integration', 'final weights and momentum match baseline'])
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
