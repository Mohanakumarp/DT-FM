"""BERT span parity, SQuAD preprocessing/scoring, and real 6-rank Gloo training."""
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
    from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast
    from scripts.train_qa_pipeline import load_model, synthetic_batch
except ImportError:
    BertConfig = None

from modules.bert_qa_partition import BertQAPartition, qa_layout, span_loss
from task_datasets.question_answering import answer_scores, score_predictions, tokenize_qa


@unittest.skipIf(BertConfig is None, "requires requirements-mt5.txt")
class QAPipelineTests(unittest.TestCase):
    def test_partition_logits_loss_and_gradients_match_full_bert(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = BertConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=4, num_attention_heads=2,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        config._attn_implementation = "eager"
        original = BertForQuestionAnswering(config)
        inputs = torch.tensor([[5, 6, 7, 8, 0], [9, 10, 11, 12, 13]])
        mask = (inputs != 0).long()
        types = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 1]])
        starts, ends = torch.tensor([2, 3]), torch.tensor([3, 4])
        reference = original(input_ids=inputs, attention_mask=mask, token_type_ids=types,
                             start_positions=starts, end_positions=ends)
        reference.loss.backward()
        for count in (1, 2, 3, 6):
            with self.subTest(stages=count):
                model = copy.deepcopy(original)
                model.zero_grad(set_to_none=True)
                boundaries = qa_layout(config, count, count)
                stages = [BertQAPartition(model, i, boundaries) for i in range(count)]
                hidden, edges = None, []
                for i, stage in enumerate(stages):
                    output = stage(hidden, mask, types, input_ids=inputs if i == 0 else None)
                    if i < count - 1:
                        hidden = output.detach().requires_grad_()
                        edges.append((output, hidden))
                torch.testing.assert_close(output[:, :, 0], reference.start_logits)
                torch.testing.assert_close(output[:, :, 1], reference.end_logits)
                actual = span_loss(output, starts, ends) / len(inputs)
                torch.testing.assert_close(actual, reference.loss)
                actual.backward()
                for local, remote in reversed(edges):
                    local.backward(remote.grad)
                for (name, expected), (_, parameter) in zip(original.named_parameters(), model.named_parameters()):
                    torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-4, atol=1e-6, msg=name)
                state = {}
                for stage in stages:
                    local = stage.pretrained_state_dict()
                    self.assertFalse(state.keys() & local.keys())
                    state.update(local)
                model.load_state_dict(state, strict=True)

    def test_overflow_windows_label_spans_and_preserve_context_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocab.txt"
            vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                        "what", "alpha", "beta", "gamma", "delta", "epsilon"]))
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            context = "alpha beta gamma delta epsilon alpha beta gamma delta epsilon"
            answer = context.rindex("delta")
            examples = {"id": ["q"], "question": ["what"], "context": [context],
                        "answers": [{"answer_start": [answer], "text": ["delta"]}]}
            features = tokenize_qa(examples, tokenizer, max_length=10, stride=2)
            self.assertGreater(len(features["input_ids"]), 1)
            found = False
            for index, start in enumerate(features["start_positions"]):
                if start == features["input_ids"][index].index(tokenizer.cls_token_id):
                    self.assertEqual(features["end_positions"][index], start)
                else:
                    left = features["offset_mapping"][index][start][0]
                    right = features["offset_mapping"][index][features["end_positions"][index]][1]
                    self.assertEqual(context[left:right], "delta")
                    found = True
                self.assertIsNone(features["offset_mapping"][index][0])
            self.assertTrue(found)

    def test_answer_scoring_combines_windows_and_rejects_missing_features(self):
        examples = [{"id": "q", "context": "alpha beta gamma", "answers": {"text": ["beta"], "answer_start": [6]}}]
        features = [{"example_id": "q", "offset_mapping": [None, [0, 5], [6, 10], [11, 16]]}] * 2
        predictions = [(0, [0, 2, 0, 0], [0, 2, 0, 0]), (1, [0, 0, 5, 0], [0, 0, 5, 0])]
        scores, answers = score_predictions(examples, features, predictions)
        self.assertEqual(scores["exact_match"], 100)
        self.assertEqual(scores["f1"], 100)
        self.assertEqual(scores["validation_examples"], 1)
        self.assertEqual(answers["q"], "beta")
        self.assertEqual(answer_scores("The beta!", "beta"), (1, 1))
        self.assertEqual(answer_scores("beta gamma", "beta"), (0, 2 / 3))
        with self.assertRaisesRegex(ValueError, "Missing"):
            score_predictions(examples, features, predictions[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score_predictions(examples, features, predictions + predictions[:1])

    def test_six_ranks_three_stages_two_replicas_and_export(self):
        self._check_distributed_training(6, 3)

    def test_topk_four_ranks_two_stages_two_replicas_and_export(self):
        self._check_distributed_training(4, 2, ["--gradient-compression", "topk",
                                              "--compression-keep-ratio", "0.01",
                                              "--compression-warmup-steps", "1",
                                              "--compression-bucket-size", "1024"])

    def test_resume_restores_dense_and_compressed_updates_and_falls_back_to_common_step(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def launch(output, steps, flags=(), resume=False):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                processes, handles = [], []
                try:
                    for rank in range(4):
                        handle = (root / ("launch-%d-%d.log" % (steps, rank))).open("w")
                        handles.append(handle)
                        command = [sys.executable, "scripts/train_qa_pipeline.py", "--smoke", "--rank", str(rank),
                                   "--world-size", "4", "--pipeline-size", "2", "--batch-size", "4",
                                   "--micro-batch-size", "3", "--epochs", "3", "--max-steps", str(steps),
                                   "--warmup-steps", "0", "--timeout-seconds", "60", "--output-dir", str(output),
                                   "--dist-url", "tcp://127.0.0.1:%d" % port, *flags]
                        if resume:
                            command.append("--resume")
                        processes.append(subprocess.Popen(command, cwd=repository, stdout=handle, stderr=subprocess.STDOUT,
                                                          env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))
                    for rank, process in enumerate(processes):
                        self.assertEqual(process.wait(timeout=100), 0, (root / ("launch-%d-%d.log" % (steps, rank))).read_text())
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                    for handle in handles:
                        handle.close()

            for mode in ("none", "topk"):
                with self.subTest(compression=mode):
                    flags = ("--gradient-compression", mode)
                    if mode == "topk":
                        flags += ("--compression-bucket-size", "1024", "--compression-warmup-steps", "1")
                    continuous, recovered = root / (mode + "-continuous"), root / (mode + "-recovered")
                    launch(continuous, 3, flags)
                    launch(recovered, 2, flags)
                    if mode == "none":
                        # A failed save can leave the newest version absent on one
                        # host. Its checkpoint.pt may still point at that update;
                        # resume must select step 1 from the immutable versions.
                        for path in (recovered / "rank3").glob("checkpoint-step-00000002-*.pt"):
                            path.unlink()
                    launch(recovered, 3, flags, resume=True)
                    for rank in range(4):
                        expected = torch.load(continuous / ("rank%d/checkpoint.pt" % rank), weights_only=True)
                        actual = torch.load(recovered / ("rank%d/checkpoint.pt" % rank), weights_only=True)
                        for key in expected["pretrained"]:
                            torch.testing.assert_close(actual["pretrained"][key], expected["pretrained"][key], rtol=0, atol=0)
                        for index, state in expected["optimizer"]["state"].items():
                            for key, value in state.items():
                                torch.testing.assert_close(actual["optimizer"]["state"][index][key], value, rtol=0, atol=0)
                        torch.testing.assert_close(actual["rng_cpu"], expected["rng_cpu"], rtol=0, atol=0)
                        if mode == "topk":
                            self.assertEqual(actual["compressor"]["completed_steps"], 3)
                            for actual_buffer, expected_buffer in zip(actual["compressor"]["residuals"], expected["compressor"]["residuals"]):
                                torch.testing.assert_close(actual_buffer, expected_buffer, rtol=0, atol=0)
                        self.assertEqual(len(list((recovered / ("rank%d" % rank)).glob("checkpoint-step-*.pt"))), 2)
                        summary = json.loads((recovered / ("rank%d/summary.json" % rank)).read_text())
                        self.assertEqual(summary["resumed_from_step"], 1 if mode == "none" else 2)
                    records = [json.loads(line) for line in (recovered / "rank0/metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([record["step"] for record in records if "train_loss" in record], [1, 2, 3])

    def _check_distributed_training(self, world_size, pipeline_size, extra_args=None):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            processes, logs = [], []
            try:
                for rank in range(world_size):
                    log = (root / ("process%d.log" % rank)).open("w")
                    logs.append(log)
                    command = [sys.executable, "scripts/train_qa_pipeline.py", "--smoke", "--rank", str(rank),
                               "--world-size", str(world_size), "--pipeline-size", str(pipeline_size), "--batch-size", "4",
                               "--micro-batch-size", "3", "--max-steps", "2", "--warmup-steps", "0",
                               "--dist-url", "tcp://127.0.0.1:%d" % port, "--timeout-seconds", "60",
                               "--output-dir", str(root)] + (extra_args or [])
                    processes.append(subprocess.Popen(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                                                      env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))
                for rank, process in enumerate(processes):
                    self.assertEqual(process.wait(timeout=100), 0, (root / ("process%d.log" % rank)).read_text())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                for log in logs:
                    log.close()
            for stage in range(pipeline_size):
                expected = torch.load(root / ("rank%d/checkpoint.pt" % stage), weights_only=True)
                actual = torch.load(root / ("rank%d/checkpoint.pt" % (stage + pipeline_size)), weights_only=True)
                for key in expected["pretrained"]:
                    torch.testing.assert_close(actual["pretrained"][key], expected["pretrained"][key], rtol=0, atol=0)
                if extra_args:
                    self.assertEqual(expected["gradient_compression"], "topk")
                    self.assertEqual(expected["compressor"]["completed_steps"], 2)
                    self.assertTrue(any(residual.count_nonzero() for residual in expected["compressor"]["residuals"]))
            records = [json.loads(line) for line in (root / "rank0/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["global_features"], 8)
            self.assertEqual(records[-1]["completed_steps"], 2)
            self.assertGreater(records[-1]["features_per_second"], 0)
            if extra_args:
                self.assertEqual(records[0]["gradient_keep_ratio"], 0.05)
                self.assertAlmostEqual(records[1]["gradient_keep_ratio"], 0.01)
                self.assertGreater(records[1]["gradient_payload_compression"], records[0]["gradient_payload_compression"])
                self.assertGreater(records[1]["gradient_payload_compression"], 1)
                self.assertLess(records[1]["gradient_payload_bytes"], records[1]["gradient_dense_bytes"])
            progress = [json.loads(line) for line in (root / "rank0/progress.jsonl").read_text().splitlines()]
            phases = [entry["phase"] for entry in progress]
            self.assertIn("waiting_for_all_ranks", phases)
            self.assertIn("forward", phases)
            self.assertIn("receiving_backward", phases)
            self.assertIn("synchronizing_data_replica_gradients", phases)
            self.assertEqual(phases[-1], "complete")
            # The first hybrid update must use the global, not per-replica, loss.
            from argparse import Namespace
            args = Namespace(smoke=True, seed=42, batch_size=4, replica_rank=0)
            reference = load_model(args)
            first = synthetic_batch(args, reference.config)
            args.replica_rank = 1
            second = synthetic_batch(args, reference.config)
            batch = {key: torch.cat([first[key], second[key]]) for key in first}
            with torch.no_grad():
                expected_loss = reference(**batch).loss.item()
            self.assertAlmostEqual(records[0]["train_loss"], expected_loss, places=5)
            command = [sys.executable, "scripts/assemble_mt5_checkpoint.py", "--rank-dirs",
                       *[str(root / ("rank%d" % i)) for i in range(pipeline_size)], "--output-dir", str(root / "merged")]
            result = subprocess.run(command, cwd=repository, capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            merged = BertForQuestionAnswering.from_pretrained(root / "merged")
            self.assertEqual(merged.config.num_hidden_layers, 4)
            self.assertFalse(torch.equal(merged.qa_outputs.weight, reference.qa_outputs.weight))


if __name__ == "__main__":
    unittest.main()
