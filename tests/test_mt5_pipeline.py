"""Numerical split-model parity; these tests require requirements-mt5.txt."""
import copy
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

from modules.mt5_stages import MT5EncoderStage, MT5DecoderStage


@unittest.skipIf(MT5Config is None, "requires requirements-mt5.txt")
class MT5PipelineTests(unittest.TestCase):
    def test_two_process_training_resume_and_checkpoint_assembly(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def launch(steps, resume=False):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                processes = []
                logs = []
                try:
                    for rank in (0, 1):
                        log_path = root / ("process-%d-%d.log" % (steps, rank))
                        log = log_path.open("w")
                        logs.append(log)
                        command = [
                            sys.executable, "scripts/train_mt5_pipeline.py", "--smoke",
                            "--rank", str(rank), "--steps", str(steps),
                            "--dist-url", "tcp://127.0.0.1:%d" % port,
                            "--timeout-seconds", "30", "--validation-batches", "1",
                            "--output-dir", str(root),
                        ]
                        if resume:
                            command += ["--resume-from", str(root / ("rank%d" % rank))]
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
            launch(2, resume=True)
            final = torch.load(root / "rank0/checkpoint.pt", weights_only=True)
            self.assertEqual(final["step"], 2)
            self.assertFalse(torch.equal(initial["pretrained"]["shared.weight"], final["pretrained"]["shared.weight"]))
            result = subprocess.run([
                sys.executable, "scripts/assemble_mt5_checkpoint.py",
                "--encoder-dir", str(root / "rank0"), "--decoder-dir", str(root / "rank1"),
                "--output-dir", str(root / "merged"),
            ], cwd=repository, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            merged = MT5ForConditionalGeneration.from_pretrained(root / "merged")
            torch.testing.assert_close(merged.shared.weight, final["pretrained"]["shared.weight"])
            generated = merged.generate(torch.tensor([[5, 6, 1]]), max_new_tokens=3)
            self.assertGreater(generated.numel(), 1)

    def test_split_loss_and_all_parameter_gradients_match_unsplit_model(self):
        torch.manual_seed(7)
        config = MT5Config(
            vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_heads=2,
            num_layers=2, num_decoder_layers=2, dropout_rate=0.0,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        )
        reference = MT5ForConditionalGeneration(config)
        partitioned = copy.deepcopy(reference)
        encoder = MT5EncoderStage(partitioned)
        decoder = MT5DecoderStage(partitioned)
        inputs = torch.tensor([[5, 6, 1, 0], [8, 9, 10, 1]])
        mask = (inputs != 0).long()
        labels = torch.tensor([[11, 1, -100], [12, 13, 1]])
        expected = reference(input_ids=inputs, attention_mask=mask, labels=labels).loss
        expected.backward()
        # Unequal target lengths and micro-batches must preserve token weighting.
        actual = 0.0
        tokens = (labels != -100).sum()
        for i in range(2):
            encoded, embeds = encoder(inputs[i:i + 1], mask[i:i + 1], labels[i:i + 1])
            remote_encoded = encoded.detach().requires_grad_()
            remote_embeds = embeds.detach().requires_grad_()
            loss = decoder(remote_encoded, remote_embeds, mask[i:i + 1], labels[i:i + 1]) / tokens
            actual += loss.detach()
            loss.backward()
            torch.autograd.backward((encoded, embeds), (remote_encoded.grad, remote_embeds.grad))
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
        parameters = dict(encoder.named_parameters())
        parameters.update(dict(decoder.named_parameters()))
        for name, parameter in reference.named_parameters():
            split_name = "encoder.embed_tokens.weight" if name == "shared.weight" else name
            self.assertIn(split_name, parameters)
            torch.testing.assert_close(parameters[split_name].grad, parameter.grad, rtol=2e-4, atol=2e-6)
        state = encoder.pretrained_state_dict()
        state.update(decoder.pretrained_state_dict())
        reference.load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
