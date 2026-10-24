"""Numerical split-model parity; these tests require requirements-mt5.txt."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
import torch
try:
    from transformers import MT5Config, MT5ForConditionalGeneration
except ImportError:
    MT5Config = MT5ForConditionalGeneration = None

from modules.mt5_partition import MT5Partition, pipeline_layout


@unittest.skipIf(MT5Config is None, "requires requirements-mt5.txt")
class MT5PipelineTests(unittest.TestCase):
    def test_two_process_training_resume_and_checkpoint_assembly(self):
        self._check_training_resume_and_checkpoint_assembly(2)

    def test_three_process_training_resume_and_checkpoint_assembly(self):
        self._check_training_resume_and_checkpoint_assembly(3)

    def test_four_stage_pipeline(self):
        self._check_training_resume_and_checkpoint_assembly(4, resume=False)

    def test_pipeline_with_three_data_replicas(self):
        self._check_training_resume_and_checkpoint_assembly(6, pipeline_size=2)

    def test_single_rank(self):
        self._check_training_resume_and_checkpoint_assembly(1, resume=False)

    def test_topk_pipeline_replicas_resume_matches_uninterrupted_training(self):
        self._check_training_resume_and_checkpoint_assembly(
            4, pipeline_size=2, extra_args=["--gradient-compression", "topk",
                                          "--compression-warmup-steps", "4",
                                          "--compression-bucket-size", "1024"],
        )

    def test_dgc_pipeline_replicas_resume_matches_uninterrupted_training(self):
        self._check_training_resume_and_checkpoint_assembly(
            4, pipeline_size=2, extra_args=["--gradient-compression", "dgc", "--optimizer", "sgd",
                                          "--compression-warmup-steps", "4",
                                          "--compression-bucket-size", "1024"],
        )

    def test_compression_rejects_unsupported_optimizer_and_single_replica(self):
        repository = Path(__file__).resolve().parents[1]
        for flags, message in (
            (["--gradient-compression", "dgc"], "requires --optimizer sgd"),
            (["--gradient-compression", "topk"], "requires at least two data replicas"),
        ):
            with self.subTest(flags=flags):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                command = [sys.executable, "scripts/train_mt5_pipeline.py", "--smoke", "--rank", "0",
                           "--world-size", "1", "--dist-url", "tcp://127.0.0.1:%d" % port] + flags
                result = subprocess.run(command, cwd=repository, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_layout_accepts_varying_world_sizes_without_idle_stages(self):
        config = MT5Config(num_layers=8, num_decoder_layers=8)
        for world_size in range(1, 101):
            stages, replicas, boundaries = pipeline_layout(config, world_size)
            self.assertEqual(stages * replicas, world_size)
            self.assertEqual(boundaries[0], 0)
            self.assertEqual(boundaries[-1], 18)
            self.assertTrue(all(start < end for start, end in zip(boundaries, boundaries[1:])))
        with self.assertRaises(ValueError):
            pipeline_layout(config, 7, 3)
        with self.assertRaises(ValueError):
            pipeline_layout(config, 30, 30)

    def _check_training_resume_and_checkpoint_assembly(self, world_size, pipeline_size=None, resume=True, extra_args=None):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def launch(steps, resume=False, output=None):
                output = output or root
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                processes = []
                logs = []
                try:
                    for rank in range(world_size):
                        log_path = root / ("process-%d-%d.log" % (steps, rank))
                        log = log_path.open("w")
                        logs.append(log)
                        command = [
                            sys.executable, "scripts/train_mt5_pipeline.py", "--smoke",
                            "--rank", str(rank), "--steps", str(steps),
                            "--world-size", str(world_size),
                            "--dist-url", "tcp://127.0.0.1:%d" % port,
                            "--timeout-seconds", "30", "--validation-batches", "1",
                            "--output-dir", str(output),
                        ]
                        if resume:
                            command += ["--resume-from", str(root / ("rank%d" % rank))]
                        if pipeline_size is not None:
                            command += ["--pipeline-size", str(pipeline_size)]
                        command += extra_args or []
                        processes.append(subprocess.Popen(
                            command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        ))
                    for rank, process in enumerate(processes):
                        result = process.wait(timeout=60)
                        logs[rank].flush()
                        self.assertEqual(result, 0, (root / ("process-%d-%d.log" % (steps, rank))).read_text())
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                    for log in logs:
                        log.close()

            launch(1)
            initial = torch.load(root / "rank0/checkpoint.pt", weights_only=True)
            if world_size == 2 and not extra_args and resume:
                # Dense checkpoints written before compression support had no
                # optimizer name or compressor metadata.
                for rank in range(world_size):
                    path = root / ("rank%d/checkpoint.pt" % rank)
                    legacy = torch.load(path, weights_only=True)
                    for key in ("optimizer_name", "momentum", "gradient_compression",
                                "compression_max_grad_norm", "compressor"):
                        legacy.pop(key)
                    torch.save(legacy, path)
            if resume:
                launch(2, resume=True)
            final = torch.load(root / "rank0/checkpoint.pt", weights_only=True)
            self.assertEqual(final["step"], 2 if resume else 1)
            if resume:
                self.assertFalse(torch.equal(initial["pretrained"]["shared.weight"], final["pretrained"]["shared.weight"]))
            if extra_args:
                self.assertEqual(final["compressor"]["completed_steps"], 2)
                self.assertTrue(any(tensor.count_nonzero() for tensor in final["compressor"]["residuals"]))
                records = [json.loads(line) for line in (root / "rank0/metrics.jsonl").read_text().splitlines()
                           if "train_loss" in json.loads(line)]
                self.assertGreater(records[-1]["gradient_payload_compression"], records[0]["gradient_payload_compression"])
                self.assertGreater(records[-1]["gradient_sync_seconds"], 0)
                launch(2, output=root / "continuous")
                for rank in range(world_size):
                    resumed = torch.load(root / ("rank%d/checkpoint.pt" % rank), weights_only=True)
                    continuous = torch.load(root / ("continuous/rank%d/checkpoint.pt" % rank), weights_only=True)
                    for key in resumed["pretrained"]:
                        torch.testing.assert_close(resumed["pretrained"][key], continuous["pretrained"][key], rtol=0, atol=0)
                    for key in ("residuals", "momentums"):
                        for actual, expected in zip(resumed["compressor"][key], continuous["compressor"][key]):
                            if actual is not None:
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if resumed["gradient_compression"] == "dgc":
                        self.assertEqual(resumed["optimizer"]["param_groups"][0]["momentum"], 0)
            command = [
                sys.executable, "scripts/assemble_mt5_checkpoint.py",
                "--output-dir", str(root / "merged"),
                "--rank-dirs",
            ] + [str(root / ("rank%d" % rank)) for rank in range(final["pipeline_size"])]
            result = subprocess.run(command, cwd=repository, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            merged = MT5ForConditionalGeneration.from_pretrained(root / "merged")
            torch.testing.assert_close(merged.shared.weight, final["pretrained"]["shared.weight"])
            generated = merged.generate(torch.tensor([[5, 6, 1]]), max_new_tokens=3)
            self.assertGreater(generated.numel(), 1)

            if final["pipeline_size"] > 1:
                incomplete_command = command[:-1]
                incomplete = subprocess.run(incomplete_command, cwd=repository, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(incomplete.returncode, 0)
                self.assertIn("--rank-dirs", incomplete.stderr)

            if final["replicas"] > 1:
                for stage_rank in range(final["pipeline_size"]):
                    expected = torch.load(root / ("rank%d/checkpoint.pt" % stage_rank), weights_only=True)["pretrained"]
                    for replica in range(1, final["replicas"]):
                        rank = replica * final["pipeline_size"] + stage_rank
                        actual = torch.load(root / ("rank%d/checkpoint.pt" % rank), weights_only=True)["pretrained"]
                        for key in expected:
                            torch.testing.assert_close(expected[key], actual[key])

    def test_split_loss_and_all_parameter_gradients_match_unsplit_model(self):
        for world_size in (1, 2, 3, 4, 6):
            with self.subTest(world_size=world_size):
                self._check_loss_and_gradients(world_size)

    def _check_loss_and_gradients(self, world_size):
        torch.manual_seed(7)
        config = MT5Config(
            vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_heads=2,
            num_layers=2, num_decoder_layers=2, dropout_rate=0.0,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        )
        reference = MT5ForConditionalGeneration(config)
        partitioned = copy.deepcopy(reference)
        _, _, boundaries = pipeline_layout(config, world_size, world_size)
        stages = [MT5Partition(partitioned, rank, boundaries) for rank in range(world_size)]
        inputs = torch.tensor([[5, 6, 1, 0], [8, 9, 10, 1]])
        mask = (inputs != 0).long()
        labels = torch.tensor([[11, 1, -100], [12, 13, 1]])
        expected = reference(input_ids=inputs, attention_mask=mask, labels=labels).loss
        expected.backward()
        # Unequal target lengths and micro-batches must preserve token weighting.
        actual = 0.0
        tokens = (labels != -100).sum()
        for i in range(2):
            state, edges = {}, []
            for rank, stage in enumerate(stages):
                output = stage(state, mask[i:i + 1], labels[i:i + 1], input_ids=inputs[i:i + 1] if rank == 0 else None)
                if rank < world_size - 1:
                    state = {key: tensor.detach().requires_grad_() for key, tensor in output.items()}
                    edges.append((output, state))
            loss = output / tokens
            actual += loss.detach()
            loss.backward()
            for local, remote in reversed(edges):
                gradients = [remote[key].grad if remote[key].grad is not None else torch.zeros_like(remote[key]) for key in local]
                torch.autograd.backward(tuple(local.values()), gradients)
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
        parameters = {}
        state = {}
        for stage in stages:
            for name, parameter in stage.named_parameters():
                name = name.replace("encoder_blocks.", "encoder.block.").replace("decoder_blocks.", "decoder.block.")
                name = name.replace("encoder_norm.", "encoder.final_layer_norm.").replace("decoder_norm.", "decoder.final_layer_norm.")
                self.assertNotIn(name, parameters)
                parameters[name] = parameter
            state.update(stage.pretrained_state_dict())
        for name, parameter in reference.named_parameters():
            self.assertIn(name, parameters)
            torch.testing.assert_close(parameters[name].grad, parameter.grad, rtol=2e-4, atol=2e-6)
        reference.load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
