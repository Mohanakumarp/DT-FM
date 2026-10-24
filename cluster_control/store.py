"""SQLite persistence. Each service operation is one serialized transaction."""
from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import threading


class Store:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:" and os.name != "nt":
            os.chmod(path, 0o600)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                password TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                csrf TEXT NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                owner_id TEXT NOT NULL REFERENCES users(id), created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memberships (
                cluster_id TEXT NOT NULL REFERENCES clusters(id), user_id TEXT NOT NULL REFERENCES users(id),
                role TEXT NOT NULL CHECK(role IN ('owner', 'contributor')),
                status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected', 'revoked')),
                created REAL NOT NULL, PRIMARY KEY(cluster_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS pairing_codes (
                code_hash TEXT PRIMARY KEY, cluster_id TEXT NOT NULL REFERENCES clusters(id),
                user_id TEXT NOT NULL REFERENCES users(id), expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY, cluster_id TEXT NOT NULL REFERENCES clusters(id),
                user_id TEXT NOT NULL REFERENCES users(id), token_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL, address TEXT NOT NULL, capabilities TEXT NOT NULL,
                shared_devices TEXT NOT NULL, enabled INTEGER NOT NULL, last_seen REAL NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, cluster_id TEXT NOT NULL REFERENCES clusters(id),
                owner_id TEXT NOT NULL REFERENCES users(id), spec TEXT NOT NULL,
                status TEXT NOT NULL, error TEXT, created REAL NOT NULL, updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assignments (
                job_id TEXT NOT NULL REFERENCES jobs(id), rank INTEGER NOT NULL,
                agent_id TEXT NOT NULL REFERENCES agents(id), device_id TEXT NOT NULL,
                state TEXT NOT NULL, exit_code INTEGER, metrics TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(job_id, rank)
            );
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
                rank INTEGER, timestamp REAL NOT NULL, message TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS assignment_agent ON assignments(agent_id);
            CREATE INDEX IF NOT EXISTS event_job ON job_events(job_id, id);
        """)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def close(self):
        with self.lock:
            self.connection.close()
