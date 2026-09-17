import argparse
import copy
import unittest
from unittest.mock import patch

import torch

from scheduler.rebalance import maybe_rebalance
from scheduler.resource_scheduler import build_resource_plan
from scheduler.resources import memory_model
from utils.launch_manifest import build_manifest, validate_manifest
from test_resource_scheduler import MODEL, reports


class RebalancePolicyTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(rank=0, world_size=1, stage_layers='3,1', rebalance_every=2,
                                       rebalance_min_improvement=.5, scheduler_memory_fraction=1.,
                                       scheduler_reserve_mb=0, resource_schedule={'model': MODEL})
        self.pipeline = argparse.Namespace(device=torch.device('cpu'))

    def test_skips_non_boundary_and_final_step_without_profiling(self):
        with patch('scheduler.rebalance.measure_live_reports') as profile:
            self.assertIsNone(maybe_rebalance(self.args, self.pipeline, 1, 4))
            self.assertIsNone(maybe_rebalance(self.args, self.pipeline, 4, 4))
            profile.assert_not_called()

    def test_small_improvement_keeps_partition(self):
        live = reports()
        live[1]['timings']['layer_ms'] = 1.
        with patch('scheduler.rebalance.measure_live_reports', return_value=live), patch('scheduler.rebalance.migrate') as move:
            result = maybe_rebalance(self.args, self.pipeline, 2, 4)
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['new_stage_layers'], [3, 1])
        move.assert_not_called()

    def test_memory_pressure_can_move_layers_even_when_compute_slows(self):
        live = reports()
        memory = memory_model(MODEL, 0, 2)
        live[0]['resources']['host_memory']['available_bytes'] = memory['base_bytes'] + 2 * memory['per_layer_bytes']
        with patch('scheduler.rebalance.measure_live_reports', return_value=live), patch(
                'scheduler.rebalance.migrate', return_value={'status': 'committed', 'moved_layers': 1}) as move:
            result = maybe_rebalance(self.args, self.pipeline, 2, 4)
        self.assertTrue(result['memory_pressure'])
        self.assertLess(result['estimated_improvement'], 0)
        move.assert_called_once_with(self.args, self.pipeline, [2, 2])
        self.assertEqual(result['new_stage_layers'], [2, 2])

    def test_collective_preparation_failure_keeps_active_plan(self):
        original = copy.deepcopy(self.args.resource_schedule)
        with patch('scheduler.rebalance.measure_live_reports', side_effect=ValueError('rank 1 could not allocate')):
            result = maybe_rebalance(self.args, self.pipeline, 2, 4)
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(self.args.resource_schedule, original)
        self.assertEqual(self.args.stage_layers, '3,1')

    def test_manifest_carries_interval_and_rejects_invalid_modes(self):
        network = dict(passed=True, world_size=2, latency_ms=[[0, 1], [1, 0]],
                       bandwidth_mbps=[[0, 100], [100, 0]])
        manifest = build_manifest(network, ['a', 'b'], total_layers=4, dynamic=True, rebalance_every=20)
        validate_manifest(manifest)
        self.assertEqual(manifest['dynamic_schedule']['rebalance_every'], 20)
        self.assertTrue(all(rank['env']['REBALANCE_EVERY'] == '20' for rank in manifest['ranks']))
        for settings in (dict(rebalance_every=-1), dict(rebalance_every=1),
                         dict(dynamic=True, total_layers=4, rebalance_every=True)):
            with self.assertRaises(ValueError):
                build_manifest(network, ['a', 'b'], **settings)


if __name__ == '__main__':
    unittest.main()
