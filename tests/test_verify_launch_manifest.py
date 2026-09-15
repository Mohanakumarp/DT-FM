import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_launch_manifest import check_metrics
from utils.launch_manifest import build_manifest


class LaunchSmokeVerificationTests(unittest.TestCase):
    def setUp(self):
        profile = json.loads((Path(__file__).parent / 'fixtures/probe_2rank.json').read_text())
        self.manifest = build_manifest(profile, ['localhost', 'localhost'])
        self.metrics = [dict(
            rank=rank, world_size=2, pipeline_group_size=2,
            schedule={'completed_steps': 2},
            training_time_s=dict(forward_incl_p2p=.1, backward_incl_p2p=.2, optimizer=.01),
            latency_ms=None, bandwidth_mbps=None, losses=[.7, .6] if rank == 1 else [])
            for rank in range(2)]

    def check(self, metrics):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for rank, payload in enumerate(metrics):
                (output / ('metrics_rank%d.json' % rank)).write_text(json.dumps(payload))
            return check_metrics(self.manifest, output)

    def test_complete_training_is_accepted(self):
        result = self.check(self.metrics)
        self.assertEqual([r['completed_steps'] for r in result], [2, 2])
        self.assertEqual(result[1]['losses'], [.7, .6])

    def test_incomplete_wrong_rank_and_invalid_losses_rejected(self):
        for field, value in [('schedule', {'completed_steps': 1}), ('rank', 0),
                             ('pipeline_group_size', 1), ('losses', []),
                             ('losses', [.7, float('nan')]), ('latency_ms', [[0, 1], [1, 0]])]:
            metrics = copy.deepcopy(self.metrics)
            metrics[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(metrics)

    def test_missing_rank_artifact_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.check(self.metrics[:1])

    def test_missing_compute_work_rejected(self):
        for field in ('forward_incl_p2p', 'backward_incl_p2p', 'optimizer'):
            for value in (0, -1, float('nan'), float('inf'), True):
                metrics = copy.deepcopy(self.metrics)
                metrics[0]['training_time_s'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check(metrics)


if __name__ == '__main__':
    unittest.main()
