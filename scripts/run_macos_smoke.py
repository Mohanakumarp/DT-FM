#!/usr/bin/env python3
"""Bounded native CPU/MPS training smoke; no model or dataset downloads."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_smoke(device="mps", mixed_cpu=False, iterations=2, timeout_seconds=120,
              output=None, workflow="core"):
    from utils.device_utils import resolve_task_device
    import torch
    resolve_task_device(device)
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        raise RuntimeError("This PyTorch build does not include Gloo")
    output = Path(output or ROOT / "logs/smoke" / ("macos-%s-%d" % (workflow, time.time_ns()))).resolve()
    # Reusing old metrics could make an incomplete run look successful.
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    devices = ["cpu", device] if mixed_cpu else [device]
    size = len(devices)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    if sys.platform == "darwin":
        environment["GLOO_SOCKET_IFNAME"] = "lo0"
    processes, handles = [], []
    try:
        for rank, compute in enumerate(devices):
            if workflow == "core":
                arguments = ["dist_runner.py", "--device", compute, "--tensor-comm", "gloo",
                    "--dist-backend", "gloo", "--dist-url", "tcp://127.0.0.1:%d" % port,
                    "--world-size", str(size), "--pipeline-group-size", str(size), "--data-group-size", "1",
                    "--rank", str(rank), "--synthetic-data", "true", "--synthetic-samples", "8",
                    "--synthetic-vocab-size", "64", "--seq-length", "8", "--embedding-dim", "16",
                    "--num-layers", "1", "--num-heads", "2", "--batch-size", "2",
                    "--micro-batch-size", "1", "--num-iters", str(iterations), "--metrics-dir", str(output),
                    "--skip-comm-probe", "true", "--use-offload", "false", "--profiling", "no-profiling"]
            else:
                script = "scripts/train_qa_pipeline.py" if workflow == "qa" else "scripts/train_mt5_pipeline.py"
                arguments = [script, "--device", compute, "--rank", str(rank), "--world-size", str(size),
                    "--pipeline-size", str(size), "--dist-url", "tcp://127.0.0.1:%d" % port,
                    "--smoke", "--batch-size", "2", "--micro-batch-size", "1", "--output-dir", str(output)]
                if workflow == "qa":
                    arguments += ["--epochs", str(iterations), "--max-steps", str(iterations), "--max-length", "8",
                                  "--doc-stride", "2", "--warmup-steps", "0"]
                else:
                    arguments += ["--steps", str(iterations), "--source-length", "8", "--target-length", "4",
                                  "--validation-batches", "1"]
            handle = (output / ("rank%d.log" % rank)).open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                env=environment, stdout=handle, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + timeout_seconds
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("A training rank failed; inspect logs in " + str(output))
            if time.monotonic() >= deadline:
                raise TimeoutError("Training exceeded the smoke timeout; inspect " + str(output))
            time.sleep(.1)
        if any(process.returncode for process in processes):
            raise RuntimeError("Training failed; inspect logs in " + str(output))
        for rank, compute in enumerate(devices):
            if workflow == "core":
                metrics = json.loads((output / ("metrics_rank%d.json" % rank)).read_text())
                if metrics["schedule"]["completed_steps"] != iterations or metrics["device"]["backend"] != compute:
                    raise RuntimeError("Core training did not complete on the requested device")
                if not all(math.isfinite(metrics["training_time_s"][key]) and metrics["training_time_s"][key] > 0 for key in
                           ("forward_incl_p2p", "backward_incl_p2p", "optimizer")):
                    raise RuntimeError("Missing forward/backward/optimizer evidence")
                losses = metrics["losses"]
            else:
                records = [json.loads(line) for line in (output / ("rank%d" % rank) / "metrics.jsonl").read_text().splitlines()]
                losses = [item["train_loss"] for item in records if "train_loss" in item]
                if len(losses) != iterations or not (output / ("rank%d" % rank) / "checkpoint.pt").is_file():
                    raise RuntimeError("Task training lacks completed steps or a checkpoint")
            # Only the core last stage logs losses. All task stages log them.
            if (workflow != "core" or rank == size - 1) and (
                    len(losses) != iterations or not all(math.isfinite(loss) and loss > 0 for loss in losses)):
                raise RuntimeError("Training lacks finite positive losses")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for handle in handles:
            handle.close()
    print("%s smoke passed on %s; %d optimizer steps per rank; logs: %s" % (
        workflow, ", ".join(devices), iterations, output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--mixed-cpu", action="store_true", help="CPU rank 0 and selected device rank 1")
    parser.add_argument("--workflow", choices=["core", "qa", "summarization"], default="core")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.timeout_seconds < 1:
        parser.error("iterations and timeout-seconds must be positive")
    output = args.output or ROOT / "logs/smoke" / ("macos-%s-%d" % (args.workflow, time.time_ns()))
    run_smoke(args.device, args.mixed_cpu, args.iterations, args.timeout_seconds, output, args.workflow)


if __name__ == "__main__":
    main()
