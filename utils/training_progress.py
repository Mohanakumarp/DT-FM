"""Immediate phase logs and periodic heartbeats while a training rank waits."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time


class TrainingProgress:
    def __init__(self, rank, output_dir, interval=15, append=False):
        self.rank = rank
        self.started = self.phase_started = time.monotonic()
        self.current_phase = "starting"
        self.details = {}
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        root = Path(output_dir) / ("rank%d" % rank)
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / "progress.jsonl").open("a" if append else "w")
        self.phase("starting")
        self.thread = threading.Thread(target=self._heartbeat, args=(interval,), daemon=True)
        self.thread.start()

    def _emit(self, event):
        now = time.monotonic()
        record = {"event": event, "rank": self.rank,
                  "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "phase": self.current_phase, "elapsed_seconds": round(now - self.started, 1),
                  "phase_seconds": round(now - self.phase_started, 1), **self.details}
        line = json.dumps(record)
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()

    def phase(self, name, **details):
        with self.lock:
            self.current_phase = name
            self.phase_started = time.monotonic()
            self.details = details
            self._emit("progress")

    def _heartbeat(self, interval):
        while not self.stopped.wait(interval):
            with self.lock:
                self._emit("heartbeat")

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        with self.lock:
            self.file.close()
