"""HTTP boundaries and local fail-closed behavior with an actual child process."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from cluster_control.server import ControlServer
from cluster_control.service import Service
from cluster_control.store import Store
from cluster_control.worker import Worker, stop_process


class SafetyTests(unittest.TestCase):
    def test_static_headers_head_and_request_validation(self):
        store = Store(":memory:")
        server = ControlServer(("127.0.0.1", 0), Service(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = "http://127.0.0.1:%d" % server.server_port
        try:
            with urlopen(Request(origin + "/", method="HEAD")) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"")
                self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            for path in ("/../../.git/config", "/.cluster-data/control.sqlite3"):
                with self.assertRaises(HTTPError) as error:
                    urlopen(origin + path)
                self.assertEqual(error.exception.code, 404)
                error.exception.close()
            for payload in (b'{"name":NaN}', b'[]', b'{bad'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(origin + "/api/auth/register", data=payload, headers={"Content-Type": "application/json"}))
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            store.close()

    def test_loss_of_control_connection_stops_a_real_local_process(self):
        from unittest.mock import patch
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=sys.platform != "win32")
        class Task:
            def stop(self):
                stop_process(process)
        class Offline:
            def call(self, *args):
                raise OSError("offline")
        with tempfile.TemporaryDirectory() as directory:
            worker = Worker({"server": "http://127.0.0.1:1", "token": "test"}, directory, poll_seconds=0.01)
            worker.client = Offline()
            worker.executions[("test-job", 0)] = Task()
            # Advance only the lease clock; use a real process and real process-group termination.
            actual = time.monotonic
            started = actual()
            with patch("cluster_control.worker.time.monotonic", side_effect=lambda: (actual() - started) * 100 + started):
                thread = threading.Thread(target=worker.run)
                try:
                    thread.start()
                    process.wait(timeout=3)
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn(("test-job", 0), worker.lost)
                finally:
                    worker.stopped.set()
                    thread.join(timeout=5)
                    stop_process(process)


if __name__ == "__main__":
    unittest.main()
