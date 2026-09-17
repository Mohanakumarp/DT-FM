import copy
import json
from pathlib import Path
import unittest

from scheduler.resource_scheduler import build_resource_plan, validate_resource_plan
from scheduler.resources import host_memory, memory_model, resource_snapshot
from utils.launch_manifest import build_manifest, validate_manifest


MODEL = dict(seq_length=8, embedding_dim=16, num_heads=2, batch_size=2, micro_batch_size=1, vocab_size=64)


def reports(size=2):
    return [dict(rank=r, timings=dict(layer_ms=1. if r == 0 else 10., first_stage_ms=0., last_stage_ms=0.),
                 resources=dict(host='host-%d' % r, device='cpu', device_memory=None, device_pool=None,
                                sampled_at_unix=1., cpu_threads=1, logical_cpus=8,
                                host_memory=dict(available_bytes=1024**3, total_bytes=2 * 1024**3, source='test fixture')))
            for r in range(size)]


class ResourceSchedulerTests(unittest.TestCase):
    def test_compute_rates_determine_weight_not_rank_count(self):
        live = reports()
        first = build_resource_plan(MODEL, live, 8, 1., 0)
        self.assertEqual(first['stage_layers'], [7, 1])
        live[0]['timings']['layer_ms'], live[1]['timings']['layer_ms'] = 10., 1.
        second = build_resource_plan(MODEL, live, 8, 1., 0)
        self.assertEqual(second['stage_layers'], [1, 7])
        self.assertEqual(first['stage_layers'], [7, 1])

    def test_current_available_memory_changes_assignment(self):
        live = reports()
        first = build_resource_plan(MODEL, live, 8, 1., 0)
        footprint = memory_model(MODEL, 0, 2)
        live[0]['resources']['host_memory']['available_bytes'] = footprint['base_bytes'] + 2 * footprint['per_layer_bytes']
        second = build_resource_plan(MODEL, live, 8, 1., 0)
        self.assertEqual(first['stage_layers'], [7, 1])
        self.assertEqual(second['stage_layers'], [2, 6])
        for pool in second['memory_pools']:
            self.assertLessEqual(pool['estimated_used_bytes'], pool['budget_bytes'])
        validate_resource_plan(json.loads(json.dumps(second)))

    def test_shared_host_memory_is_not_counted_once_per_rank(self):
        live = reports()
        live[1]['resources']['host'] = live[0]['resources']['host']
        footprints = [memory_model(MODEL, r, 2) for r in range(2)]
        budget = sum(f['base_bytes'] for f in footprints) + 5 * footprints[0]['per_layer_bytes']
        for report in live:
            report['resources']['host_memory']['available_bytes'] = budget
        plan = build_resource_plan(MODEL, live, 5, 1., 0)
        self.assertEqual(plan['stage_layers'], [4, 1])
        self.assertEqual(len(plan['memory_pools']), 1)
        self.assertEqual(plan['memory_pools'][0]['estimated_used_bytes'], budget)
        with self.assertRaisesRegex(ValueError, 'insufficient live memory'):
            build_resource_plan(MODEL, live, 6, 1., 0)

    def test_shared_gpu_budget_and_host_staging_bound(self):
        live = reports()
        budget = sum(memory_model(MODEL, r, 2)['base_bytes'] for r in range(2)) + 5 * memory_model(MODEL, 0, 2)['per_layer_bytes']
        for report in live:
            report['resources'].update(host='gpu-host', device='cuda', device_pool='cuda:uuid-1',
                                       device_memory=dict(available_bytes=budget, total_bytes=1024**3))
        plan = build_resource_plan(MODEL, live, 5, 1., 0)
        self.assertEqual(len(plan['memory_pools']), 2)
        with self.assertRaisesRegex(ValueError, 'insufficient live memory'):
            build_resource_plan(MODEL, live, 6, 1., 0)
        # A different adapter has a separate GPU budget but still shares RAM.
        live[1]['resources']['device_pool'] = 'cuda:uuid-2'
        build_resource_plan(MODEL, live, 6, 1., 0)
        live[0]['resources']['host_memory']['available_bytes'] = budget
        with self.assertRaisesRegex(ValueError, 'insufficient live memory'):
            build_resource_plan(MODEL, live, 6, 1., 0)

    def test_fraction_and_reserve_apply_once_to_shared_pool(self):
        live = reports()
        live[1]['resources']['host'] = live[0]['resources']['host']
        plan = build_resource_plan(MODEL, live, 5, .5, 1048576)
        self.assertEqual(plan['memory_pools'][0]['budget_bytes'], 1024**3 // 2 - 1048576)

    def test_rejects_invalid_measurements_and_insufficient_memory(self):
        for field, value in [('available_bytes', -1), ('available_bytes', True),
                             ('available_bytes', 3 * 1024**3), ('total_bytes', 0)]:
            live = reports()
            live[0]['resources']['host_memory'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_resource_plan(MODEL, live, 5)
        for fraction in (0, -1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                build_resource_plan(MODEL, reports(), 5, fraction)
        live = reports()
        live[0]['resources']['host_memory']['available_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'one layer per rank'):
            build_resource_plan(MODEL, live, 5)
        for timing in (0, -1, float('inf'), True):
            live = reports()
            live[0]['timings']['layer_ms'] = timing
            with self.assertRaises(ValueError):
                build_resource_plan(MODEL, live, 5)

    def test_ranges_cover_model_and_tampering_is_rejected(self):
        plan = build_resource_plan(MODEL, reports(3), 9)
        indices = [layer for stage in plan['stages'] for layer in range(stage['layer_start'], stage['layer_end'])]
        self.assertEqual(indices, list(range(9)))
        bad = copy.deepcopy(plan)
        bad['stage_layers'] = [3, 3, 3]
        with self.assertRaises(ValueError):
            validate_resource_plan(bad)

    def test_dynamic_manifest_defers_assignment_and_validates(self):
        network = json.loads((Path(__file__).parent / 'fixtures/probe_3rank.json').read_text())
        manifest = build_manifest(network, ['a', 'b', 'c'], total_layers=9, dynamic=True)
        validate_manifest(json.loads(json.dumps(manifest)))
        self.assertEqual(manifest['schema_version'], 3)
        self.assertEqual(manifest['pipeline_order'], [0, 2, 1])
        for entry in manifest['ranks']:
            self.assertEqual(entry['env']['DYNAMIC_TOTAL_LAYERS'], '9')
            self.assertNotIn('STAGE_LAYERS', entry['env'])
            self.assertNotIn('assignment', entry)
        manifest['ranks'][1]['env']['DYNAMIC_TOTAL_LAYERS'] = '6'
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_reads_real_host_headroom(self):
        memory = host_memory()
        self.assertGreater(memory['total_bytes'], 0)
        self.assertGreaterEqual(memory['available_bytes'], 0)
        self.assertLessEqual(memory['available_bytes'], memory['physical_available_bytes'])
        self.assertLessEqual(memory['available_bytes'], memory['total_bytes'])

    def test_layer_parameter_estimate_matches_actual_module(self):
        from modules.gpt_modules import GPTTransformerLayer
        layer = GPTTransformerLayer(16, 2, 64)
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 12 * 16 * 16 + 13 * 16)

    def test_unavailable_gpu_memory_telemetry_fails_explicitly(self):
        import argparse
        from unittest.mock import patch
        with patch('scheduler.resources.host_memory', return_value={'available_bytes': 1024**3}):
            with self.assertRaisesRegex(RuntimeError, 'DirectML'):
                resource_snapshot(argparse.Namespace(device_backend='directml'), None)


if __name__ == '__main__':
    unittest.main()
