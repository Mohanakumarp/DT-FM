import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from comm.probe_aggregation import aggregate_probe_results
from utils.metrics import RunMetrics


def _result(rank, *measurements):
    return {'rank': rank, 'measurements': list(measurements)}


def _measurement(src, dst, latency_ms, bandwidth_mbps):
    return {
        'src': src,
        'dst': dst,
        'latency_ms': latency_ms,
        'bandwidth_mbps': bandwidth_mbps,
    }


class ProbeAggregationTests(unittest.TestCase):
    def test_reorders_three_rank_contributions_by_owner(self):
        gathered = [
            _result(src, *[_measurement(src, dst, 1 + src * 3 + dst,
                                       100 + src * 3 + dst)
                          for dst in range(3) if dst != src])
            for src in (2, 0, 1)
        ]
        latency, bandwidth = aggregate_probe_results(gathered, 3)
        self.assertEqual(latency, [[0.0, 2.0, 3.0],
                                  [4.0, 0.0, 6.0], [7.0, 8.0, 0.0]])
        self.assertEqual(bandwidth, [[0.0, 101.0, 102.0],
                                    [103.0, 0.0, 105.0], [106.0, 107.0, 0.0]])

    def test_rejects_invalid_ownership_and_duplicate_contributions(self):
        invalid = [
            [_result(0)],
            [_result(0), _result(0)],
            [_result(0), _result(2)],
            [_result(0, _measurement(1, 0, 1, 100)), _result(1)],
            [_result(0, _measurement(False, 1, 1, 100)), _result(1)],
            [_result(0, _measurement(0.0, 1, 1, 100)), _result(1)],
            [_result(0, _measurement(0, 2, 1, 100)), _result(1)],
            [_result(0, _measurement(0, 0, 1, 100)), _result(1)],
            [_result(0, _measurement(0, 1, 1, 100),
                     _measurement(0, 1, 2, 200)), _result(1)],
        ]
        for gathered in invalid:
            with self.subTest(gathered=gathered):
                with self.assertRaises(ValueError):
                    aggregate_probe_results(gathered, 2)

    def test_preserves_asymmetric_directed_measurements(self):
        gathered = [
            _result(0, _measurement(0, 1, 1.25, 125.0)),
            _result(1, _measurement(1, 0, 2.5, 80.0)),
        ]

        latency, bandwidth = aggregate_probe_results(gathered, 2)

        self.assertEqual(latency, [[0.0, 1.25], [2.5, 0.0]])
        self.assertEqual(bandwidth, [[0.0, 125.0], [80.0, 0.0]])

    def test_same_gathered_results_produce_identical_matrices_on_every_rank(self):
        gathered = [
            _result(0, _measurement(0, 1, 1.0, 100.0)),
            _result(1, _measurement(1, 0, 3.0, 75.0)),
        ]

        per_rank_outputs = [
            aggregate_probe_results(gathered, 2) for _rank in range(2)
        ]

        self.assertEqual(per_rank_outputs[0], per_rank_outputs[1])

    def test_missing_and_failed_directions_remain_unavailable(self):
        gathered = [
            _result(
                0,
                _measurement(0, 1, 1.0, 100.0),
                _measurement(0, 2, None, None),
            ),
            _result(1, _measurement(1, 2, 2.0, 50.0)),
            _result(2),
        ]

        latency, bandwidth = aggregate_probe_results(gathered, 3)

        self.assertIsNone(latency[0][2])
        self.assertIsNone(bandwidth[0][2])
        self.assertIsNone(latency[1][0])
        self.assertIsNone(bandwidth[2][1])
        self.assertEqual([latency[i][i] for i in range(3)], [0.0, 0.0, 0.0])

    def test_rejects_invalid_measurement_values(self):
        invalid_values = [0.0, -1.0, math.nan, math.inf, 'fast', True]
        for field in ('latency', 'bandwidth'):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    latency = value if field == 'latency' else 1.0
                    bandwidth = value if field == 'bandwidth' else 100.0
                    gathered = [
                        _result(0, _measurement(0, 1, latency, bandwidth)),
                        _result(1, _measurement(1, 0, 1.0, 100.0)),
                    ]
                    with self.assertRaises(ValueError):
                        aggregate_probe_results(gathered, 2)

        with self.assertRaises(ValueError):
            aggregate_probe_results([
                _result(0, _measurement(0, 1, 1.0, None)),
                _result(1),
            ], 2)

    def test_metrics_serializes_unavailable_links_as_json_null(self):
        args = SimpleNamespace(
            rank=0,
            world_size=2,
            pipeline_group_size=2,
            seq_length=32,
            embedding_dim=64,
            num_layers=1,
            num_heads=4,
            num_epochs=0,
            steps_per_epoch=0,
            num_iters=1,
        )
        metrics = RunMetrics(args)
        metrics.set_comm_matrix(
            [[0.0, None], [2.5, 0.0]],
            [[0.0, None], [80.0, 0.0]],
            ['rank-0', 'rank-1'],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'metrics.json')
            metrics.dump(path)
            with open(path) as handle:
                payload = json.load(handle)

        self.assertIsNone(payload['latency_ms'][0][1])
        self.assertIsNone(payload['bandwidth_mbps'][0][1])
        self.assertEqual(payload['latency_ms'][1][0], 2.5)
        self.assertEqual(payload['bandwidth_mbps'][1][0], 80.0)


if __name__ == '__main__':
    unittest.main()
