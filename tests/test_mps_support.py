"""MPS selection, shared-memory budgets and native hardware smoke coverage."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import torch

from comm.comm_utils import _synchronize_tensor
from scheduler.resources import host_memory, resource_snapshot
from scheduler.resource_scheduler import build_resource_plan
from scripts.run_macos_smoke import run_smoke
from utils.device_utils import resolve_device, resolve_task_device, validate_runtime_args, synchronize_device, describe_device
from utils.dist_args_utils import add_device_arguments
from utils.launch_manifest import build_manifest, validate_manifest


def runtime(**fields):
    defaults = dict(device='mps', cuda_id=0, world_size=2, pipeline_group_size=2,
                    data_group_size=1, tensor_comm='gloo', fp16=False, profiling='no-profiling')
    defaults.update(fields)
    return SimpleNamespace(**defaults)


@unittest.skipUnless(hasattr(torch.backends, 'mps'), 'requires PyTorch with MPS APIs')
class MPSCompatibilityTests(unittest.TestCase):
    def test_explicit_mps_does_not_probe_unrelated_accelerators(self):
        args = runtime()
        with patch('torch.backends.mps.is_built', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=True), \
                patch('torch.cuda.is_available', side_effect=AssertionError('CUDA was probed')), \
                patch('utils.device_utils._xpu_available', side_effect=AssertionError('XPU was probed')):
            device = resolve_device(args)
        self.assertEqual(device.type, 'mps')
        self.assertEqual(args.device_backend, 'mps')
        self.assertFalse(args.use_cuda)
        self.assertFalse(args.pin_memory)
        self.assertIn('MPS', describe_device(device, args.device_backend))

    def test_auto_selects_mps_and_retains_cpu_fallback_and_legacy_flag(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('utils.device_utils._xpu_available', return_value=False), \
                patch('torch.backends.mps.is_built', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(resolve_device(runtime(device='auto')).type, 'mps')
            self.assertEqual(resolve_device(runtime(device='auto', use_cuda=False)).type, 'cpu')
            with patch('torch.backends.mps.is_available', return_value=False):
                self.assertEqual(resolve_device(runtime(device='auto')).type, 'cpu')

    def test_cuda_auto_priority_is_preserved(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.version.hip', None), \
                patch('utils.device_utils._xpu_available', return_value=False), \
                patch('torch.backends.mps.is_available', side_effect=AssertionError('MPS was probed')):
            self.assertEqual(resolve_device(runtime(device='auto')).type, 'cuda')

    def test_explicit_unavailable_mps_has_actionable_errors(self):
        with patch('torch.backends.mps.is_built', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'arm64'):
                resolve_task_device('mps')
        with patch('torch.backends.mps.is_built', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'macOS version'):
                resolve_task_device('mps')

    def test_runtime_accepts_pipeline_gloo_and_rejects_unsupported_modes(self):
        args = runtime(tensor_comm='auto', device_backend='mps')
        validate_runtime_args(args, torch.device('mps'))
        self.assertEqual(args.tensor_comm, 'gloo')
        for changes in (dict(fp16=True), dict(tensor_comm='nccl'), dict(profiling='tidy_profiling'),
                        dict(world_size=4, data_group_size=2)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_runtime_args(runtime(device_backend='mps', **changes), torch.device('mps'))

    def test_cli_and_manifest_preserve_mps_after_rank_remapping(self):
        parser = argparse.ArgumentParser()
        add_device_arguments(parser)
        self.assertEqual(parser.parse_args(['--device', 'mps']).device, 'mps')
        profile = json.loads((Path(__file__).parent / 'fixtures/probe_3rank.json').read_text())
        manifest = build_manifest(profile, ['mac', 'linux', 'win'], devices=['mps', 'cuda', 'cpu'])
        validate_manifest(manifest)
        for rank in manifest['ranks']:
            self.assertEqual(rank['env']['DEVICE'], ['mps', 'cuda', 'cpu'][rank['measured_rank']])
            self.assertEqual(rank['env']['TENSOR_COMM'], 'gloo')

    def test_compute_and_tensor_synchronization_use_mps_api(self):
        tensor = SimpleNamespace(device=torch.device('mps'))
        with patch('torch.mps.synchronize') as synchronize:
            synchronize_device(torch.device('mps'))
            _synchronize_tensor(tensor)
        self.assertEqual(synchronize.call_count, 2)

    def test_macos_host_memory_uses_available_unified_memory(self):
        psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=16 * 1024**3, available=6 * 1024**3))
        with patch('scheduler.resources.platform.system', return_value='Darwin'), \
                patch.dict(sys.modules, psutil=psutil):
            memory = host_memory()
        self.assertEqual(memory['available_bytes'], 6 * 1024**3)
        self.assertEqual(memory['total_bytes'], 16 * 1024**3)
        self.assertIn('macOS', memory['source'])

    def test_mps_working_set_is_bounded_by_host_headroom(self):
        host = dict(available_bytes=4 * 1024**3, total_bytes=16 * 1024**3)
        with patch('scheduler.resources.host_memory', return_value=host), \
                patch('torch.mps.recommended_max_memory', return_value=12 * 1024**3), \
                patch('torch.mps.driver_allocated_memory', return_value=2 * 1024**3):
            report = resource_snapshot(runtime(device_backend='mps'), torch.device('mps'))
        self.assertEqual(report['device_memory']['available_bytes'], host['available_bytes'])
        with patch('scheduler.resources.host_memory', return_value=host), \
                patch('torch.mps.recommended_max_memory', return_value=3 * 1024**3), \
                patch('torch.mps.driver_allocated_memory', return_value=4 * 1024**3):
            self.assertEqual(resource_snapshot(runtime(device_backend='mps'), None)['device_memory']['available_bytes'], 0)

    def test_cpu_and_mps_share_one_host_budget(self):
        from test_resource_scheduler import MODEL, reports
        from scheduler.resources import memory_model
        live = reports()
        footprint = [memory_model(MODEL, rank, 2) for rank in range(2)]
        budget = sum(item['base_bytes'] for item in footprint) + 5 * footprint[0]['per_layer_bytes']
        for item in live:
            item['resources']['host'] = 'one-mac'
            item['resources']['host_memory']['available_bytes'] = budget
        live[1]['resources'].update(device='mps', device_pool='mps:shared-metal-gpu',
                                    device_memory=dict(available_bytes=budget, total_bytes=1024**3))
        plan = build_resource_plan(MODEL, live, 5, 1., 0)
        host_pools = [pool for pool in plan['memory_pools'] if pool['resource'][0] == 'host']
        self.assertEqual(len(host_pools), 1)
        self.assertEqual(host_pools[0]['ranks'], [0, 1])
        with self.assertRaisesRegex(ValueError, 'insufficient live memory'):
            build_resource_plan(MODEL, live, 6, 1., 0)

    def test_qa_checkpoint_restores_mps_random_state(self):
        from utils.qa_checkpoints import save_checkpoint, load_common_checkpoint
        module = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=.01)
        stage = SimpleNamespace(boundaries=[0, 1], pretrained_state_dict=module.state_dict,
                                load_pretrained_state_dict=module.load_state_dict)
        args = runtime(world_size=1, pipeline_size=1, rank=0, model='tiny-bert', smoke=True,
                       gradient_compression='none')
        state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        def gather(objects, value):
            objects[:] = [value]
        with tempfile.TemporaryDirectory() as directory, \
                patch('utils.qa_checkpoints.collect_errors', side_effect=lambda error: self.assertIsNone(error)), \
                patch('torch.distributed.all_gather_object', side_effect=gather), \
                patch('torch.mps.get_rng_state', return_value=state), \
                patch('torch.mps.set_rng_state') as restore:
            save_checkpoint(directory, args, stage, optimizer, None, 2, 'test-run', {'model': 'tiny'})
            checkpoint = load_common_checkpoint(directory, args, {'model': 'tiny'}, stage, optimizer, None)
            self.assertEqual(checkpoint['step'], 2)
            torch.testing.assert_close(restore.call_args.args[0], state)

    def test_rebalance_restores_mps_random_state_even_after_failure(self):
        from scheduler.migration import preserve_rng
        state = torch.tensor([3, 2, 1], dtype=torch.uint8)
        with patch('torch.mps.get_rng_state', return_value=state), \
                patch('torch.mps.set_rng_state') as restore:
            with self.assertRaisesRegex(RuntimeError, 'profiling failed'):
                with preserve_rng(torch.device('mps')):
                    raise RuntimeError('profiling failed')
            torch.testing.assert_close(restore.call_args.args[0], state)


@unittest.skipUnless(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(), 'requires a native macOS Metal GPU')
class NativeMPSTrainingTests(unittest.TestCase):
    def smoke(self, workflow, mixed):
        with tempfile.TemporaryDirectory() as directory:
            run_smoke('mps', mixed, output=Path(directory) / 'run', workflow=workflow)

    def test_single_rank_forward_backward_optimizer(self):
        self.smoke('core', False)

    def test_cpu_mps_pipeline_activation_and_gradient_transfer(self):
        self.smoke('core', True)

    def test_cpu_mps_qa_pipeline_checkpoint(self):
        self.smoke('qa', True)

    def test_cpu_mps_mt5_pipeline_validation_and_checkpoint(self):
        self.smoke('summarization', True)


if __name__ == '__main__':
    unittest.main()
