"""Numerical sparse SUM/error-feedback checks on real Gloo groups."""
import copy
from datetime import timedelta
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.gradient_compression import SparseGradientSynchronizer


def compression_worker(rank, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=3, timeout=timedelta(seconds=30))
    try:
        # Rank 1 represents another pipeline stage. It must not contribute to
        # the stage's gradients or be needed in sparse collectives.
        group = dist.new_group([0, 2])
        if rank == 1:
            return
        peer = rank // 2
        parameters = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(5))]
        compressor = SparseGradientSynchronizer(
            [("a", parameters[0]), ("b", parameters[1])], mode=mode,
            keep_ratio=0.25, warmup_ratio=0.25, warmup_steps=0,
            momentum=0.5, bucket_size=4,
        )
        residuals = [torch.zeros(8), torch.zeros(8)]
        momentums = [torch.zeros(8), torch.zeros(8)]
        total_inputs = torch.zeros(8)
        total_sent = torch.zeros(8)
        for step in range(4):
            inputs = [torch.tensor([10., 1., 2., 3., 4., 5., 6., 20.]),
                      torch.tensor([12., 2., 1., 4., 5., 6., 30., 3.])]
            if step == 1:
                inputs = [torch.tensor([0., 2., 1., 10., 0., 7., 2., 3.]),
                          torch.tensor([1., 0., 9., 2., 3., 0., 4., 8.])]
            elif step > 1:
                inputs = [torch.zeros(8), torch.zeros(8)]
            expected = torch.zeros(8)
            total_inputs += sum(inputs)
            for source in range(2):
                if mode == "dgc":
                    momentums[source].mul_(0.5).add_(inputs[source])
                    residuals[source].add_(momentums[source])
                else:
                    residuals[source].add_(inputs[source])
                for start in (0, 4):
                    segment = residuals[source][start:start + 4]
                    index = segment.abs().argmax().item() + start
                    expected[index] += residuals[source][index]
                    residuals[source][index] = 0
                    if mode == "dgc":
                        momentums[source][index] = 0
            for parameter, gradient in zip(parameters, inputs[peer].split([3, 5])):
                parameter.grad = gradient.clone() if step < 2 else None
            # Sparse mode must never all-reduce a dense bucket.
            with patch.object(dist, "all_reduce", side_effect=AssertionError("dense exchange in sparse path")):
                record = compressor.synchronize(group)
            actual = torch.cat([p.grad for p in parameters])
            torch.testing.assert_close(actual, expected)
            total_sent += actual
            state = compressor.state_dict()
            torch.testing.assert_close(torch.cat(state["residuals"]), residuals[peer])
            if mode == "dgc":
                torch.testing.assert_close(torch.cat(state["momentums"]), momentums[peer])
            else:
                torch.testing.assert_close(total_sent + sum(residuals), total_inputs)
            assert record["gradient_payload_bytes"] == 16
            assert record["gradient_dense_bytes"] == 32
            assert record["gradient_payload_compression"] == 2
            if step == 1:
                # Preserve rank-local residuals, momentum, and schedule when a
                # new synchronizer is constructed, as on checkpoint resume.
                restored = SparseGradientSynchronizer(
                    [("a", parameters[0]), ("b", parameters[1])], mode=mode,
                    keep_ratio=0.25, warmup_ratio=0.25, warmup_steps=0,
                    momentum=0.5, bucket_size=4,
                )
                restored.load_state_dict(copy.deepcopy(state))
                compressor = restored
    finally:
        dist.destroy_process_group()


def dense_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        parameter = torch.nn.Parameter(torch.zeros(8))
        compressor = SparseGradientSynchronizer([("p", parameter)], keep_ratio=1,
                                               warmup_ratio=1, warmup_steps=0)
        for step in range(2):
            parameter.grad = torch.arange(8, dtype=torch.float32) * (rank + 1 + step)
            record = compressor.synchronize(dist.group.WORLD)
            torch.testing.assert_close(parameter.grad, torch.arange(8, dtype=torch.float32) * (3 + 2 * step))
            assert record["gradient_payload_bytes"] == 32
            assert not compressor.buckets[0]["residual"].count_nonzero()
    finally:
        dist.destroy_process_group()


class GradientCompressionTests(unittest.TestCase):
    def test_topk_sum_and_residual_conservation_in_stage_subgroup(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(compression_worker, args=(root + "/rendezvous", "topk"), nprocs=3, join=True)

    def test_dgc_momentum_correction_and_sent_coordinate_masking(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(compression_worker, args=(root + "/rendezvous", "dgc"), nprocs=3, join=True)

    def test_full_density_falls_back_to_dense_sum(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(dense_worker, args=(root + "/rendezvous",), nprocs=2, join=True)

    def test_warmup_checkpoint_round_trip_and_configuration_mismatch(self):
        parameter = torch.nn.Parameter(torch.zeros(20))
        compressor = SparseGradientSynchronizer([("p", parameter)], warmup_steps=4)
        self.assertEqual(compressor.current_keep_ratio(), 0.25)
        compressor.completed_steps = 2
        self.assertAlmostEqual(compressor.current_keep_ratio(), (0.25 * 0.001)**0.5)
        compressor.buckets[0]["residual"][4] = 7
        with tempfile.TemporaryFile() as checkpoint:
            torch.save(compressor.state_dict(), checkpoint)
            checkpoint.seek(0)
            state = torch.load(checkpoint, weights_only=True)
        restored = SparseGradientSynchronizer([("p", parameter)], warmup_steps=4)
        restored.load_state_dict(state)
        self.assertEqual(restored.completed_steps, 2)
        self.assertEqual(restored.current_keep_ratio(), compressor.current_keep_ratio())
        self.assertEqual(restored.buckets[0]["residual"][4], 7)
        restored.completed_steps = 4
        self.assertEqual(restored.current_keep_ratio(), 0.001)
        incompatible = SparseGradientSynchronizer([("p", parameter)], keep_ratio=0.01)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            incompatible.load_state_dict(state)

    def test_invalid_compression_settings(self):
        for settings in ({"keep_ratio": 0}, {"keep_ratio": float("nan")},
                         {"momentum": 1}, {"warmup_steps": -1}, {"bucket_size": 2**31}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                SparseGradientSynchronizer([], **settings)


if __name__ == "__main__":
    unittest.main()
