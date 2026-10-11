"""Actual worker pairing/launch over HTTP, real training, artifacts and cancellation."""
from http.cookiejar import CookieJar
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import build_opener, HTTPCookieProcessor, Request

from cluster_control.server import ControlServer
from cluster_control.service import Service
from cluster_control.store import Store


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers") and
                     importlib.util.find_spec("datasets"), "requires CPU PyTorch and requirements-mt5.txt")
class WorkerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.directory = Path(self.root.name)
        self.store = Store(self.directory / "control.sqlite3")
        self.service = Service(self.store)
        self.server = ControlServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = "http://127.0.0.1:%d" % self.server.server_port
        self.client = build_opener(HTTPCookieProcessor(CookieJar()))
        auth = self.call("/api/auth/register", {"name": "Owner", "email": "owner@example.com", "password": "long test password"})
        self.csrf = auth["csrf"]
        self.cluster = self.call("/api/clusters", {"name": "Integration lab"})["id"]
        self.processes, self.handles = [], []
        self.selections = []
        for index in range(2):
            state = self.directory / ("worker-%d.json" % index)
            code = self.call("/api/clusters/%s/pairing" % self.cluster, {})["code"]
            paired = subprocess.run([sys.executable, "-m", "cluster_control.worker", "pair", "--server", self.origin,
                "--code", code, "--name", "CPU worker %d" % index, "--address", "127.0.0.1", "--devices", "cpu", "--state", str(state)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(paired.returncode, 0, paired.stderr)
            identity = json.loads(state.read_text())
            self.selections.append(identity["agent_id"] + ":cpu")
            if os.name != "nt":
                self.assertEqual(state.stat().st_mode & 0o777, 0o600)
            handle = (self.directory / ("worker-%d.log" % index)).open("w")
            self.handles.append(handle)
            process = subprocess.Popen([sys.executable, "-m", "cluster_control.worker", "run", "--state", str(state),
                "--output", str(self.directory / ("artifacts-%d" % index))], stdout=handle, stderr=subprocess.STDOUT)
            self.processes.append(process)

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for handle in self.handles:
            handle.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.store.close()
        self.root.cleanup()

    def call(self, path, body=None):
        request = Request(self.origin + path, data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "X-CSRF-Token": getattr(self, "csrf", "")})
        with self.client.open(request, timeout=5) as response:
            return json.load(response)

    def create_job(self, steps=2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        job = self.call("/api/clusters/%s/jobs" % self.cluster, {"config": {"pipeline_size": 2,
            "epochs": steps, "max_steps": steps, "max_length": 16, "doc_stride": 4, "port": port}, "devices": self.selections})
        self.call("/api/jobs/%s/start" % job["id"], {})
        return job

    def wait_job(self, job, statuses, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.call("/api/jobs/" + job["id"])
            if result["status"] in statuses:
                return result
            for process in self.processes:
                if process.poll() is not None:
                    self.fail("Worker exited unexpectedly: " + "\n".join(p.read_text() for p in self.directory.glob("worker-*.log")))
            time.sleep(0.25)
        self.fail("Job timed out: " + json.dumps(result))

    def test_http_to_two_rank_training_and_checkpoint_artifacts(self):
        job = self.create_job()
        result = self.wait_job(job, {"completed", "failed"})
        self.assertEqual(result["status"], "completed", json.dumps(result))
        self.assertEqual([a["state"] for a in result["assignments"]], ["succeeded", "succeeded"])
        for rank in range(2):
            output = self.directory / ("artifacts-%d" % rank) / job["id"] / ("rank%d" % rank)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed_steps"], 2)
            self.assertTrue(list(output.glob("checkpoint*.pt")))
            metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            losses = [m["train_loss"] for m in metrics if "train_loss" in m]
            self.assertEqual(len(losses), 2)
            self.assertTrue(all(0 < loss < 100 for loss in losses))
        self.assertTrue(any("complete" in event["message"] for event in result["events"]))

    def test_cancel_stops_running_processes_and_releases_devices(self):
        job = self.create_job(steps=10000)
        self.wait_job(job, {"running", "failed"})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            active = self.call("/api/jobs/" + job["id"])
            if all(a["state"] == "running" for a in active["assignments"]) and any(
                    a["metrics"].get("step", 0) >= 1 for a in active["assignments"]):
                break
            time.sleep(0.25)
        else:
            self.fail("Ranks did not begin optimizer steps before cancellation")
        self.call("/api/jobs/%s/cancel" % job["id"], {})
        result = self.wait_job(job, {"cancelled", "failed"}, timeout=20)
        self.assertEqual(result["status"], "cancelled", json.dumps(result))
        self.assertTrue(all(a["state"] == "stopped" for a in result["assignments"]))
        another = self.create_job()
        result = self.wait_job(another, {"completed", "failed"})
        self.assertEqual(result["status"], "completed", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
