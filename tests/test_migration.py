import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.verify_launch_manifest import stop_workers

ROOT = Path(__file__).resolve().parents[1]


def run_workers(folder, mode, size=2):
    folder.mkdir()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('GLOO_SOCKET_IFNAME', None)
    processes, logs = [], []
    try:
        for rank in range(size):
            log = (folder / ('rank%d.log' % rank)).open('w')
            logs.append(log)
            processes.append(subprocess.Popen([
                sys.executable, str(ROOT / 'tests/migration_worker.py'), '--mode', mode,
                '--world-size', str(size), '--pipeline-group-size', str(size), '--rank', str(rank),
                '--dist-url', 'tcp://127.0.0.1:%d' % port, '--output', str(folder)],
                env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != 'nt'))
        deadline = time.monotonic() + 90
        while any(process.poll() is None for process in processes):
            if time.monotonic() > deadline or any(process.poll() not in (None, 0) for process in processes):
                raise AssertionError('worker failure or timeout: ' + str(folder))
            time.sleep(.1)
        if any(process.returncode != 0 for process in processes):
            raise AssertionError('worker failed: ' + str(folder))
    except Exception as exc:
        for log in logs:
            log.flush()
        details = '\n'.join(path.read_text(errors='replace')[-6000:] for path in folder.glob('rank*.log'))
        raise AssertionError(str(exc) + '\n' + details) from exc
    finally:
        stop_workers(processes, folder)
        for log in logs:
            log.close()
    return [json.loads((folder / ('result_rank%d.json' % r)).read_text()) for r in range(size)]


class MigrationTests(unittest.TestCase):
    def test_state_and_training_continuity_through_moves_and_rollback(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = run_workers(root / 'baseline', 'baseline')
            def tensors(path):
                return {key: value for rank in range(2) for key, value in
                        torch.load(path / ('state_rank%d.pt' % rank), weights_only=True).items()}
            expected = tensors(root / 'baseline')
            self.assertTrue(any('/momentum/' in key for key in expected))
            for mode in ('migrate', 'rollback', 'controller'):
                with self.subTest(mode=mode):
                    results = run_workers(root / mode, mode)
                    actual = tensors(root / mode)
                    self.assertEqual(set(expected), set(actual))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6, msg=key)
                    self.assertEqual(results[-1]['losses'], baseline[-1]['losses'])
                    for rank in results:
                        self.assertEqual(rank['completed_steps'], 4)
                        self.assertEqual([event['status'] for event in rank['events']],
                                         ['rolled_back', 'rolled_back'] if mode == 'rollback' else ['committed', 'committed'])


if __name__ == '__main__':
    unittest.main()
