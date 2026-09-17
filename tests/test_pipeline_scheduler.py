import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
import unittest

from scheduler.pipeline_scheduler import allocate_layers, apply_stage_assignment, assignment_signature
from utils.launch_manifest import build_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def profile(size=3):
    return dict(schema_version=1,
                model=dict(seq_length=32, embedding_dim=64, num_heads=4,
                           batch_size=2, micro_batch_size=1, vocab_size=512),
                ranks=[dict(measured_rank=r, device='cpu', max_layers=4,
                            layer_ms=float(r + 1), first_stage_ms=2., last_stage_ms=1.)
                       for r in range(size)])


class PipelineSchedulerTests(unittest.TestCase):
    def test_minimax_matches_exhaustive_oracle(self):
        rng = random.Random(42)
        for _ in range(50):
            compute = profile()
            order = rng.sample(range(3), 3)
            for rank in compute['ranks']:
                rank.update(layer_ms=rng.randint(1, 20) / 3,
                            max_layers=rng.randint(1, 5),
                            first_stage_ms=rng.randint(0, 8), last_stage_ms=rng.randint(0, 8))
            total = rng.randint(3, sum(r['max_layers'] for r in compute['ranks']))
            def score(counts):
                return max(compute['ranks'][r]['layer_ms'] * counts[s]
                           + (compute['ranks'][r]['first_stage_ms'] if s == 0 else 0)
                           + (compute['ranks'][r]['last_stage_ms'] if s == 2 else 0)
                           for s, r in enumerate(order))
            candidates = [counts for counts in itertools.product(
                *(range(1, compute['ranks'][r]['max_layers'] + 1) for r in order)) if sum(counts) == total]
            expected = min(candidates, key=lambda counts: (score(counts), counts))
            result = allocate_layers(compute, order, total)
            self.assertEqual(result['stage_layers'], list(expected))
            self.assertAlmostEqual(result['estimated_bottleneck_ms'], score(expected))
            self.assertEqual([i for stage in result['stages'] for i in range(stage['layer_start'], stage['layer_end'])], list(range(total)))

    def test_fast_device_gets_more_layers_and_capacity_is_respected(self):
        compute = profile(2)
        compute['ranks'][0].update(layer_ms=1., first_stage_ms=0.)
        compute['ranks'][1].update(layer_ms=10., last_stage_ms=0.)
        self.assertEqual(allocate_layers(compute, [0, 1], 5)['stage_layers'], [4, 1])
        compute['ranks'][0]['max_layers'] = 2
        self.assertEqual(allocate_layers(compute, [0, 1], 5)['stage_layers'], [2, 3])
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            allocate_layers(compute, [0, 1], 7)

    def test_invalid_inputs(self):
        for key, value in [('layer_ms', 0), ('layer_ms', True), ('layer_ms', float('nan')),
                           ('first_stage_ms', -1), ('last_stage_ms', float('inf')),
                           ('max_layers', 0), ('max_layers', False), ('measured_rank', 1)]:
            compute = profile()
            compute['ranks'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                allocate_layers(compute, [0, 2, 1], 6)
        for total in (True, 2, 4097, None):
            with self.assertRaises(ValueError):
                allocate_layers(profile(), [0, 2, 1], total)
        with self.assertRaises(ValueError):
            allocate_layers(profile(), [0, 0, 1], 6)
        compute = profile()
        compute['model']['num_heads'] = 3
        with self.assertRaises(ValueError):
            allocate_layers(compute, [0, 1, 2], 6)

    def test_manifest_roundtrip_remapping_and_tamper_rejection(self):
        network = json.loads((ROOT / 'tests/fixtures/probe_3rank.json').read_text())
        compute = profile()
        for rank in compute['ranks']:
            rank['max_layers'] = rank['measured_rank'] + 1
        manifest = build_manifest(network, ['a', 'b', 'c'], compute_profile=compute, total_layers=6)
        validate_manifest(json.loads(json.dumps(manifest)))
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(manifest['pipeline_order'], [0, 2, 1])
        self.assertEqual(manifest['schedule']['stage_layers'], [1, 3, 2])
        self.assertEqual([r['assignment']['measured_rank'] for r in manifest['ranks']], [0, 2, 1])
        self.assertTrue(all(r['env']['STAGE_LAYERS'] == '1,3,2' for r in manifest['ranks']))
        for field in ('schedule', 'ranks', 'compute_profile'):
            bad = copy.deepcopy(manifest)
            if field == 'schedule':
                bad[field]['stage_layers'] = [2, 2, 2]
            elif field == 'ranks':
                bad[field][0]['env']['STAGE_LAYERS'] = '2,2,2'
            else:
                bad[field]['ranks'][0]['device'] = 'xpu'
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_manifest(bad)

    def test_runtime_resolves_before_model_creation(self):
        from modules.dist_gpt_pp_module import GPTStageFirst, GPTStageMiddle, GPTStageLast
        from modules.gpt_modules import GPTTransformerLayer
        signatures = []
        for rank, cls in enumerate((GPTStageFirst, GPTStageMiddle, GPTStageLast)):
            args = argparse.Namespace(stage_layers='1,3,2', world_size=3, pipeline_group_size=3,
                                      data_group_size=1, pp_mode='gpipe', rank=rank, num_layers=99,
                                      task='SeqClassification', seq_length=8, embedding_dim=16, num_heads=2)
            apply_stage_assignment(args)
            model = cls(args, 64, 2, 'cpu')
            self.assertEqual(sum(isinstance(m, GPTTransformerLayer) for m in model.modules()), [1, 3, 2][rank])
            self.assertEqual((args.layer_start, args.layer_end), [(0, 1), (1, 4), (4, 6)][rank])
            self.assertEqual(args.total_layers, 6)
            signatures.append(assignment_signature(args))
        self.assertTrue(all(sig == signatures[0] for sig in signatures))

    def test_runtime_rejects_invalid_assignments_and_dp(self):
        args = argparse.Namespace(world_size=3, pipeline_group_size=3, data_group_size=1,
                                  pp_mode='gpipe', rank=0)
        for vector in ('1,2', '1,0,2', '1,a,2', '1,2,4096'):
            args.stage_layers = vector
            with self.assertRaises(ValueError):
                apply_stage_assignment(args)
        args.stage_layers = '1,3,2'
        args.data_group_size = 2
        with self.assertRaises(ValueError):
            apply_stage_assignment(args)

    def test_cli_merges_rank_profiles_and_rejects_mismatched_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'launch.json'
            compute = profile(2)
            paths = [Path(directory) / ('rank%d.json' % r) for r in range(2)]
            records = [dict(schema_version=1, model=copy.deepcopy(compute['model']), rank=r) for r in compute['ranks']]
            for p, record in zip(paths, records):
                p.write_text(json.dumps(record))
            cmd = [sys.executable, str(ROOT / 'scripts/profile_to_launch_manifest.py'),
                   str(ROOT / 'tests/fixtures/probe_2rank.json'), '--hosts', 'a', 'b',
                   '--compute-profiles', *map(str, reversed(paths)), '--total-layers', '5', '--output', str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            validate_manifest(json.loads(output.read_text()))
            output.unlink()
            records[1]['model']['seq_length'] = 64
            paths[1].write_text(json.dumps(records[1]))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_live_ranks_reject_disagreement_and_wrong_vocabulary(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('GLOO_SOCKET_IFNAME', None)
        for scenario in ('assignment', 'vocabulary', 'resource', 'dynamic_mismatch'):
            with self.subTest(scenario=scenario), socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
                listener.close()
                base = [sys.executable, '-u', str(ROOT / 'dist_runner.py'),
                        '--device', 'cpu', '--dist-backend', 'gloo', '--tensor-comm', 'gloo',
                        '--dist-url', 'tcp://127.0.0.1:%d' % port, '--world-size', '2',
                        '--pipeline-group-size', '2', '--data-group-size', '1',
                        '--synthetic-data', 'true', '--synthetic-vocab-size', '64',
                        '--seq-length', '8', '--embedding-dim', '16', '--num-heads', '2',
                        '--batch-size', '2', '--micro-batch-size', '1', '--num-iters', '1', '--use-offload', 'false']
                processes = []
                try:
                    for rank in range(2):
                        vector = '2,1' if scenario == 'assignment' and rank == 1 else '1,2'
                        extra = ['--schedule-vocab-size', '128'] if scenario == 'vocabulary' else []
                        schedule_args = ['--stage-layers', vector]
                        if scenario in ('resource', 'dynamic_mismatch'):
                            schedule_args = ['--dynamic-total-layers', '6' if scenario == 'dynamic_mismatch' and rank else '5']
                            if scenario == 'resource':
                                extra = ['--scheduler-reserve-mb', '999999999']
                        processes.append(subprocess.Popen(base + ['--rank', str(rank)] + schedule_args + extra,
                                                          cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
                    for process in processes:
                        output, _ = process.communicate(timeout=60)
                        self.assertNotEqual(process.returncode, 0, output)
                        expected = {'assignment': 'ranks disagree', 'vocabulary': 'dataset vocabulary differs',
                                    'resource': 'insufficient live memory', 'dynamic_mismatch': 'ranks disagree'}[scenario]
                        self.assertIn(expected, output)
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.communicate()


if __name__ == '__main__':
    unittest.main()
