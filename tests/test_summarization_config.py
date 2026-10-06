"""Runtime dataset selection and multiple model-family training paths."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
try:
    from transformers import (AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5Config,
                              BartConfig, LlamaConfig, Qwen2Config, PreTrainedTokenizerFast,
                              Adafactor, DataCollatorForSeq2Seq)
except ImportError:
    T5Config = None

from modules.mt5_partition import FullModelStage
from task_datasets.summarization import load_summarization_splits, tokenize_summaries


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                                    "article": 3, "target": 4, "summary": 5, "other": 6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")


def tiny_configs():
    return [
        T5Config(vocab_size=16, d_model=16, d_ff=32, d_kv=8, num_heads=2,
                 num_layers=1, num_decoder_layers=1, dropout_rate=0, decoder_start_token_id=0),
        BartConfig(vocab_size=16, d_model=16, encoder_layers=1, decoder_layers=1,
                   encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                   decoder_ffn_dim=32, dropout=0, attention_dropout=0, activation_dropout=0,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1),
        LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=1),
        Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=1),
    ]


@unittest.skipIf(T5Config is None, "requires requirements-mt5.txt")
class SummarizationConfigurationTests(unittest.TestCase):
    def test_runtime_columns_and_causal_prompt_masking(self):
        args = SimpleNamespace(source_column="article", target_column="answer", source_prefix="",
                               source_length=8, target_length=4, summary_separator="\nSummary:\n",
                               is_encoder_decoder=False)
        result = tokenize_summaries({"article": ["article other"], "answer": ["target summary"]}, tiny_tokenizer(), args)
        labels = result["labels"][0]
        self.assertEqual(labels[-3:], [4, 5, 1])
        self.assertTrue(all(label == -100 for label in labels[:-3]))
        self.assertEqual(len(labels), len(result["input_ids"][0]))
        args.source_length = 3
        args.summary_separator = " summary"
        truncated = tokenize_summaries({"article": ["article other article other"], "answer": ["target"]}, tiny_tokenizer(), args)
        self.assertEqual(truncated["input_ids"][0][:3], [3, 6, 5])
        self.assertEqual(truncated["labels"][0][:3], [-100, -100, -100])
        args.source_length = 8
        args.is_encoder_decoder = True
        result = tokenize_summaries({"article": ["article other"], "answer": ["target summary"]}, tiny_tokenizer(), args)
        self.assertEqual(result["labels"], [[4, 5]])

    def test_local_json_splits_and_missing_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "validation"):
                (root / (split + ".jsonl")).write_text(json.dumps({"article": "article", "answer": "summary"}) + "\n")
            args = SimpleNamespace(dataset="json", dataset_config=None, train_file=str(root / "train.jsonl"),
                                   validation_file=str(root / "validation.jsonl"), train_split="train", validation_split="validation",
                                   validation_batches=1, source_column="article", target_column="answer")
            train, validation = load_summarization_splits(args)
            self.assertEqual((len(train), len(validation)), (1, 1))
            args.source_column = "missing"
            with self.assertRaisesRegex(ValueError, "available columns"):
                load_summarization_splits(args)

    def test_full_model_loss_and_gradients_for_t5_bart_llama_qwen(self):
        ids = torch.tensor([[3, 6, 4, 5, 1], [3, 4, 5, 1, 0]])
        mask = (ids != 0).long()
        for config in tiny_configs():
            with self.subTest(model_type=config.model_type):
                torch.manual_seed(9)
                factory = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
                reference = factory.from_config(config).eval()
                wrapped = FullModelStage(copy.deepcopy(reference)).eval()
                if config.is_encoder_decoder:
                    labels = torch.tensor([[4, 5, 1], [4, 1, -100]])
                    tokens = (labels != -100).sum()
                else:
                    labels = ids.clone()
                    labels[:, :2] = -100
                    labels[mask == 0] = -100
                    tokens = (labels[:, 1:] != -100).sum()
                expected = reference(input_ids=ids, attention_mask=mask, labels=labels).loss
                expected.backward()
                actual = wrapped({}, mask, labels, input_ids=ids) / tokens
                actual.backward()
                torch.testing.assert_close(actual, expected)
                for (_, parameter), (_, other) in zip(reference.named_parameters(), wrapped.model.named_parameters()):
                    torch.testing.assert_close(parameter.grad, other.grad)

    def test_runtime_model_and_dataset_in_two_processes(self):
        self._check_runtime_model_and_dataset_in_two_processes()

    def test_full_density_topk_preserves_global_token_weighting(self):
        self._check_runtime_model_and_dataset_in_two_processes(
            ["--gradient-compression", "topk", "--compression-keep-ratio", "1",
             "--compression-warmup-ratio", "1", "--compression-warmup-steps", "0"],
        )

    def _check_runtime_model_and_dataset_in_two_processes(self, extra_args=None):
        repository = Path(__file__).resolve().parents[1]
        for config in (tiny_configs()[0], tiny_configs()[-1]):
            with self.subTest(model_type=config.model_type), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                factory = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
                factory.from_config(config).save_pretrained(root / "model")
                tiny_tokenizer().save_pretrained(root / "model")
                for split in ("train", "validation"):
                    rows = [{"article": "article other", "answer": "target summary"},
                            {"article": "other article", "answer": "summary"},
                            {"article": "article", "answer": "target"}]
                    (root / (split + ".jsonl")).write_text("\n".join(map(json.dumps, rows)) + "\n")
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                processes, logs = [], []
                try:
                    for rank in range(2):
                        log = (root / ("rank%d.log" % rank)).open("w")
                        logs.append(log)
                        command = [sys.executable, "scripts/train_mt5_pipeline.py", "--model", str(root / "model"),
                                   "--dataset", "json", "--train-file", str(root / "train.jsonl"),
                                   "--validation-file", str(root / "validation.jsonl"), "--source-column", "article",
                                   "--target-column", "answer", "--world-size", "2", "--rank", str(rank),
                                   "--dist-url", "tcp://127.0.0.1:%d" % port, "--batch-size", "1", "--steps", "1",
                                   "--validation-batches", "1", "--source-length", "8", "--target-length", "4",
                                   "--timeout-seconds", "30", "--output-dir", str(root / "output")]
                        command += extra_args or []
                        processes.append(subprocess.Popen(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                                                          env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))
                    for rank, process in enumerate(processes):
                        result = process.wait(timeout=60)
                        logs[rank].flush()
                        self.assertEqual(result, 0, (root / ("rank%d.log" % rank)).read_text())
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                    for log in logs:
                        log.close()
                first = torch.load(root / "output/rank0/checkpoint.pt", weights_only=True)
                second = torch.load(root / "output/rank1/checkpoint.pt", weights_only=True)
                # Compare the distributed update against training an unsplit model
                # on the union of both replicas' first samples. Summary lengths
                # differ, so averaging replica means would give the wrong result.
                selected = [rows[next(iter(torch.utils.data.DistributedSampler(rows, num_replicas=2, rank=rank, seed=42)))]
                            for rank in range(2)]
                args = SimpleNamespace(source_column="article", target_column="answer", source_prefix="",
                                       source_length=8, target_length=4, summary_separator="\nSummary:\n",
                                       is_encoder_decoder=config.is_encoder_decoder)
                encoded = tokenize_summaries({"article": [row["article"] for row in selected],
                                              "answer": [row["answer"] for row in selected]}, tiny_tokenizer(), args)
                features = [{key: values[index] for key, values in encoded.items()} for index in range(2)]
                batch = DataCollatorForSeq2Seq(tiny_tokenizer(), label_pad_token_id=-100)(features)
                reference = factory.from_pretrained(root / "model").train()
                optimizer = Adafactor(reference.parameters(), lr=3e-4, relative_step=False, scale_parameter=False, warmup_init=False)
                expected_loss = reference(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                          labels=batch["labels"], use_cache=False).loss
                expected_loss.backward()
                norm = sum(parameter.grad.double().square().sum() for parameter in reference.parameters() if parameter.grad is not None).sqrt()
                factor = min(1.0, 1.0 / (norm.item() + 1e-6))
                for parameter in reference.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)
                optimizer.step()
                record = json.loads((root / "output/rank0/metrics.jsonl").read_text().splitlines()[0])
                self.assertAlmostEqual(record["train_loss"], expected_loss.item(), places=5)
                for key in first["pretrained"]:
                    torch.testing.assert_close(first["pretrained"][key], reference.state_dict()[key], rtol=2e-5, atol=2e-6)
                self.assertEqual(first["pipeline_size"], 1)
                self.assertEqual(first["replicas"], 2)
                for key in first["pretrained"]:
                    torch.testing.assert_close(first["pretrained"][key], second["pretrained"][key])
                result = subprocess.run([sys.executable, "scripts/assemble_mt5_checkpoint.py", "--rank-dirs",
                                         str(root / "output/rank0"), "--output-dir", str(root / "merged")],
                                        cwd=repository, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                restored = factory.from_pretrained(root / "merged")
                for key in first["pretrained"]:
                    torch.testing.assert_close(restored.state_dict()[key], first["pretrained"][key])


if __name__ == "__main__":
    unittest.main()
