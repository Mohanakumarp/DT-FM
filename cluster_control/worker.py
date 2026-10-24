"""Pair a computer, then execute approved DT-FM jobs without shell commands."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import uuid

from cluster_control.spec import METRIC_KEYS, validate_config


REPOSITORY = Path(__file__).resolve().parents[1]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Worker API redirects are disabled; use the final server URL")


class Client:
    def __init__(self, server, token=None):
        parsed = urlsplit(server)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Use an http(s) server origin, without credentials or a path")
        self.server, self.token = server.rstrip("/"), token
        self.opener = build_opener(NoRedirect)

    def call(self, path, body):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(self.server + path, data=json.dumps(body, allow_nan=False).encode(), headers=headers)
        with self.opener.open(request, timeout=3) as response:
            return json.load(response)


def inventory(cpu_threads=1):
    identity = platform.node() + ":" + str(uuid.getnode())
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if path.is_file():
            identity = path.read_text().strip()
            break
    result = {"host_id": hashlib.sha256(identity.encode()).hexdigest(), "platform": platform.platform(),
              "cpu_threads": cpu_threads, "runtime_ready": False,
              "devices": [{"id": "cpu", "name": platform.processor() or "CPU", "backend": "cpu", "index": 0}]}
    try:
        import torch
        import transformers
        import datasets
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            raise RuntimeError("This PyTorch build does not include Gloo")
        if torch.cuda.is_available():
            backend = "rocm" if torch.version.hip else "cuda"
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpu_uuid = str(getattr(props, "uuid", "") or "")
                device_id = re.sub(r"[^A-Za-z0-9_-]", "", gpu_uuid) or "%s-%d" % (backend, index)
                result["devices"].append({"id": device_id, "name": props.name, "backend": backend,
                                          "index": index, "memory_bytes": props.total_memory})
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for index in range(torch.xpu.device_count()):
                result["devices"].append({"id": "xpu-%d" % index, "name": torch.xpu.get_device_name(index),
                                          "backend": "xpu", "index": index})
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["devices"].append({"id": "mps-0", "name": "Apple Metal GPU", "backend": "mps", "index": 0})
        result["runtime_ready"] = True
        result["runtime_error"] = ""
    except (ImportError, RuntimeError, OSError) as error:
        result["runtime_error"] = str(error)
    return result


def select_devices(caps, selection):
    if selection == "all":
        return [item["id"] for item in caps["devices"]]
    choices = selection.split(",") if selection else []
    selected = []
    for choice in choices:
        device = next((item for item in caps["devices"] if item["id"] == choice or
                       "%s:%d" % (item["backend"], item["index"]) == choice), None)
        if not device or device["id"] in selected:
            raise ValueError("Unknown or duplicate device: " + choice)
        selected.append(device["id"])
    return selected


def save_identity(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL protects an existing identity; the initial token file is never world-readable on POSIX.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(state, stream, indent=2)


def command_for(spec, rank, output, prepare=False):
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported job schema")
    config = validate_config(spec["config"])
    size = config["pipeline_size"] * config["replicas"]
    if spec.get("world_size") != size or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("Invalid rank topology")
    parsed = urlsplit(spec["dist_url"])
    if parsed.scheme != "tcp" or not parsed.hostname or parsed.port != config["port"] or parsed.path or parsed.username:
        raise ValueError("Invalid training coordinator address")
    assignment = spec["assignments"][rank]
    if assignment["rank"] != rank or assignment["device"]["backend"] not in ("cpu", "cuda"):
        raise ValueError("Unsupported assignment")
    arguments = [sys.executable, "-u", str(REPOSITORY / "scripts/train_qa_pipeline.py"),
                 "--rank", str(rank), "--world-size", str(size), "--pipeline-size", str(config["pipeline_size"]),
                 "--dist-url", spec["dist_url"], "--device", assignment["device"]["backend"],
                 "--model", config["model"], "--dataset", config["dataset"],
                 "--model-revision", config["model_revision"], "--dataset-revision", config["dataset_revision"],
                 "--output-dir", str(output), "--lr", str(config["learning_rate"]), "--warmup-steps", "0",
                 "--log-interval", "2"]
    for setting in ("batch_size", "micro_batch_size", "epochs", "max_steps", "max_length", "doc_stride",
                    "train_examples", "validation_examples", "checkpoint_every", "timeout_seconds"):
        arguments.extend(["--" + setting.replace("_", "-"), str(config[setting])])
    if config["workflow"] == "smoke":
        arguments.append("--smoke")
    if prepare:
        arguments.append("--prepare-only")
    return arguments


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)


class Execution:
    def __init__(self, command, root, interface=None):
        self.job_id, self.rank, self.spec = command["job_id"], command["rank"], command["spec"]
        if not re.fullmatch(r"[a-f0-9]{32}", self.job_id):
            raise ValueError("Invalid job identity")
        self.assignment = self.spec["assignments"][self.rank]
        self.output = Path(root) / self.job_id
        self.interface = interface
        self.state, self.reported, self.process = "preparing", None, None
        self.lines = queue.Queue(maxsize=1000)
        self.metrics = {}
        self.reader = None
        self.spawn(prepare=True)

    def spawn(self, prepare):
        if prepare:
            for item in self.spec["assignments"]:
                socket.getaddrinfo(item["address"], self.spec["config"]["port"], type=socket.SOCK_STREAM)
            if self.rank == 0:
                with socket.socket() as probe:
                    probe.bind(("0.0.0.0", self.spec["config"]["port"]))
        output = self.output / "preflight" if prepare else self.output
        output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                           OMP_NUM_THREADS=str(self.assignment["cpu_threads"]),
                           MKL_NUM_THREADS=str(self.assignment["cpu_threads"]))
        if self.assignment["device"]["backend"] == "cuda":
            # The runner uses device 0; narrow visibility to the contributor-selected adapter.
            inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
            index = self.assignment["device"]["index"]
            environment["CUDA_VISIBLE_DEVICES"] = inherited.split(",")[index] if inherited else str(index)
        if self.interface:
            environment["GLOO_SOCKET_IFNAME"] = self.interface
        self.process = subprocess.Popen(command_for(self.spec, self.rank, output, prepare), cwd=REPOSITORY,
                                        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", shell=False,
                                        start_new_session=os.name != "nt")
        self.reader = threading.Thread(target=self.read_output, args=(self.process, output), daemon=True)
        self.reader.start()

    def read_output(self, process, output):
        try:
            with (output / ("worker-rank%d.log" % self.rank)).open("a", encoding="utf-8") as logfile:
                for line in iter(lambda: process.stdout.readline(8192), ""):
                    logfile.write(line)
                    logfile.flush()
                    try:
                        self.lines.put_nowait(line.strip()[-4000:])
                    except queue.Full:
                        pass
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            self.metrics.update({key: value for key, value in record.items() if key in METRIC_KEYS})
                    except (ValueError, TypeError):
                        pass
        finally:
            process.stdout.close()

    def advance(self, action):
        if action == "stop":
            self.stop()
            return
        code = self.process.poll()
        if self.state == "preparing" and code is not None:
            self.reader.join(timeout=2)
            self.state = "ready" if code == 0 else "failed"
        if action == "run" and self.state == "ready" and self.reported == "ready":
            self.state = "running"
            self.metrics = {}
            self.spawn(prepare=False)
        elif self.state == "running" and code is not None:
            self.reader.join(timeout=2)
            summary = self.output / ("rank%d" % self.rank) / "summary.json"
            self.state = "failed"
            if code == 0 and summary.is_file():
                try:
                    record = json.loads(summary.read_text())
                    if record.get("rank") == self.rank and record.get("completed_steps", 0) > 0:
                        self.metrics.update({key: value for key, value in record.items() if key in METRIC_KEYS})
                        self.state = "succeeded"
                except (ValueError, OSError):
                    pass

    def report(self, client):
        lines = []
        while len(lines) < 40:
            try:
                lines.append(self.lines.get_nowait())
            except queue.Empty:
                break
        changed = self.state != self.reported and self.state != "preparing"
        if not changed and not lines:
            return
        metrics = dict(self.metrics)
        # Python's JSON encoder disallows non-finite values for the transport.
        import math
        metrics = {key: value for key, value in metrics.items() if value is None or isinstance(value, (str, int, float, bool))}
        metrics = {key: value for key, value in metrics.items() if not isinstance(value, float) or math.isfinite(value)}
        if self.state == "failed":
            lines.append("Worker rank failed; inspect its local log for the full traceback.")
        if changed and "train_loss" in metrics:
            lines.append(json.dumps(metrics, allow_nan=False))
        client.call("/api/worker/report", {"job_id": self.job_id, "rank": self.rank,
                    "state": self.state if changed else "progress", "message": "\n".join(lines)[-12000:],
                    "metrics": metrics, "exit_code": self.process.poll() if self.state in ("succeeded", "failed", "stopped") else None})
        if changed:
            self.reported = self.state

    def stop(self):
        stop_process(self.process)
        if self.reader:
            self.reader.join(timeout=2)
        self.state = "stopped"


class Worker:
    def __init__(self, state, root, interface=None, poll_seconds=0.5):
        self.state = state
        self.client = Client(state["server"], state["token"])
        self.root, self.interface, self.poll_seconds = Path(root), interface, poll_seconds
        self.stopped = threading.Event()
        self.executions = {}
        self.lost = set()

    def expire_local_lease(self):
        for key, task in list(self.executions.items()):
            task.stop()
            self.lost.add(key)
        self.executions.clear()

    def run(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        last_contact = time.monotonic()
        lease = 10
        try:
            while not self.stopped.is_set():
                if time.monotonic() - last_contact >= lease:
                    self.expire_local_lease()
                try:
                    response = self.client.call("/api/worker/poll", {})
                    last_contact = time.monotonic()
                    lease = min(10, response.get("lease_seconds", 10))
                    for command in response["commands"]:
                        key = (command["job_id"], command["rank"])
                        task = self.executions.get(key)
                        if command["action"] == "stop":
                            if task:
                                task.stop()
                                task.report(self.client)
                            else:
                                self.client.call("/api/worker/report", {"job_id": key[0], "rank": key[1], "state": "stopped"})
                            self.executions.pop(key, None)
                            self.lost.discard(key)
                            continue
                        if key in self.lost or (task is None and command["action"] != "prepare"):
                            self.client.call("/api/worker/report", {"job_id": key[0], "rank": key[1], "state": "failed",
                                             "message": "Worker restarted or its control connection lease expired; create a new job."})
                            continue
                        if task is None:
                            try:
                                assignment = command["spec"]["assignments"][key[1]]
                                if (assignment["agent_id"] != self.state["agent_id"] or
                                        assignment["device_id"] not in self.state["shared_devices"] or
                                        assignment["device_id"] not in response["shared_devices"]):
                                    raise ValueError("Assignment exceeds this worker's locally approved device sharing")
                                task = Execution(command, self.root, self.interface)
                                self.executions[key] = task
                            except Exception as error:
                                self.client.call("/api/worker/report", {"job_id": key[0], "rank": key[1], "state": "failed", "message": str(error)[:2000]})
                                continue
                        task.advance(command["action"])
                        task.report(self.client)
                        if task.reported in ("succeeded", "failed", "stopped"):
                            self.executions.pop(key, None)
                except HTTPError as error:
                    code = error.code
                    error.close()
                    if code in (401, 403):
                        print("Worker access was revoked; stopping all local training.", file=sys.stderr, flush=True)
                        break
                    print("Control request failed (%d); retrying." % code, file=sys.stderr, flush=True)
                except (OSError, RuntimeError, ValueError) as error:
                    print("Control connection unavailable: %s" % error, file=sys.stderr, flush=True)
                if time.monotonic() - last_contact >= lease:
                    self.expire_local_lease()
                self.stopped.wait(self.poll_seconds)
        finally:
            for task in self.executions.values():
                task.stop()
            self.executions.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("pair", help="register a computer using an approved member's one-time code")
    pair.add_argument("--server", required=True)
    pair.add_argument("--code", help="omit to paste privately at the prompt")
    pair.add_argument("--name", default=platform.node() or "My computer")
    pair.add_argument("--address", required=True, help="IPv4/DNS address reachable by the other training workers")
    pair.add_argument("--devices", default="cpu", help="comma-separated IDs or aliases: cpu,cuda:0; or all")
    pair.add_argument("--cpu-threads", type=int, default=1)
    pair.add_argument("--state", default=".cluster-data/worker.json")
    run = commands.add_parser("run", help="contribute the paired devices; Ctrl+C stops local jobs")
    run.add_argument("--state", default=".cluster-data/worker.json")
    run.add_argument("--output", default=".cluster-data/worker-artifacts")
    run.add_argument("--interface", help="optional local Gloo network interface, e.g. tailscale0")
    commands.add_parser("inspect", help="show hardware and training-runtime availability")
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inventory(), indent=2))
    elif args.command == "pair":
        if Path(args.state).exists():
            parser.error("Worker identity already exists; use a different --state path to pair another agent")
        caps = inventory(args.cpu_threads)
        selected = select_devices(caps, args.devices)
        client = Client(args.server)
        result = client.call("/api/worker/pair", {"code": args.code or getpass.getpass("Pairing code: "),
            "name": args.name, "address": args.address, "capabilities": caps, "shared_devices": selected})
        save_identity(args.state, {**result, "server": client.server, "shared_devices": selected, "capabilities": caps})
        print("Paired %s. Run: %s -m cluster_control.worker run --state %s" % (args.name, sys.executable, args.state))
    else:
        state = json.loads(Path(args.state).read_text())
        worker = Worker(state, args.output, args.interface)
        actual = inventory(state["capabilities"]["cpu_threads"])
        worker.client.call("/api/worker/heartbeat", {"capabilities": actual})
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: worker.stopped.set())
        print("Contributing paired devices. Ctrl+C stops local training.", flush=True)
        worker.run()


if __name__ == "__main__":
    main()
