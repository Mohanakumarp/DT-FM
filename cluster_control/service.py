"""Authentication and authorization shared by the HTTP API and tests."""
import hashlib
import json
import re
import secrets
import sqlite3
import time
import uuid


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def identifier():
    return uuid.uuid4().hex


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def text(value, label, minimum=1, maximum=160):
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise APIError(400, "%s must contain %d–%d characters" % (label, minimum, maximum))
    return value.strip()


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    result = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
    return salt + ":" + result.hex()


def public_user(row):
    return {key: row[key] for key in ("id", "email", "name")}


class Service:
    SESSION_SECONDS = 24 * 3600

    def __init__(self, store, clock=time.time):
        self.store = store
        self.clock = clock

    def _session(self, db, user):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        db.execute("DELETE FROM sessions WHERE expires <= ?", (self.clock(),))
        db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                   (digest(token), user["id"], csrf, self.clock() + self.SESSION_SECONDS))
        return {"user": public_user(user), "csrf": csrf}, token

    def register(self, body):
        name = text(body.get("name"), "Name", maximum=80)
        email = text(body.get("email"), "Email", maximum=254).casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            raise APIError(400, "Enter a valid email address")
        password = body.get("password")
        if not isinstance(password, str) or not 12 <= len(password) <= 256:
            raise APIError(400, "Password must contain 12–256 characters")
        hashed = password_hash(password)
        with self.store.transaction() as db:
            try:
                db.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                           (identifier(), email, name, hashed, self.clock()))
            except sqlite3.IntegrityError:
                raise APIError(409, "An account with this email already exists") from None
            return self._session(db, db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone())

    def login(self, body):
        email = text(body.get("email"), "Email", maximum=254).casefold()
        password = body.get("password")
        if not isinstance(password, str) or len(password) > 256:
            raise APIError(401, "Invalid email or password")
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            # Do the same expensive hash for an unknown account.
            stored = row["password"] if row else "00" * 16 + ":" + "00" * 64
            if not secrets.compare_digest(password_hash(password, stored.split(":")[0]), stored) or row is None:
                raise APIError(401, "Invalid email or password")
            return self._session(db, row)

    def authenticate(self, token, csrf=None, mutation=False):
        if not token:
            raise APIError(401, "Sign in to continue")
        with self.store.transaction() as db:
            row = db.execute("SELECT u.*, s.csrf FROM sessions s JOIN users u ON u.id=s.user_id "
                             "WHERE s.token_hash=? AND s.expires>?", (digest(token), self.clock())).fetchone()
            if row is None:
                raise APIError(401, "Session expired; sign in again")
            if mutation and (not csrf or not secrets.compare_digest(csrf, row["csrf"])):
                raise APIError(403, "Invalid request token; refresh the page")
            return {"user": public_user(row), "csrf": row["csrf"]}

    def logout(self, token):
        with self.store.transaction() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (digest(token),))

    def _member(self, db, cluster_id, user_id, owner=False):
        cluster = db.execute("SELECT * FROM clusters WHERE id=?", (cluster_id,)).fetchone()
        if cluster is None:
            raise APIError(404, "Cluster not found")
        member = db.execute("SELECT * FROM memberships WHERE cluster_id=? AND user_id=?",
                            (cluster_id, user_id)).fetchone()
        if not member or member["status"] != "approved" or (owner and member["role"] != "owner"):
            raise APIError(403, "Cluster owner access required" if owner else "Approved cluster membership required")
        return cluster

    def clusters(self, user_id):
        with self.store.transaction() as db:
            return [dict(row) for row in db.execute("""
                SELECT c.*, u.name AS owner_name, m.status AS membership, m.role,
                (SELECT COUNT(*) FROM memberships WHERE cluster_id=c.id AND status='approved') AS member_count
                FROM clusters c JOIN users u ON u.id=c.owner_id
                LEFT JOIN memberships m ON m.cluster_id=c.id AND m.user_id=? ORDER BY c.created DESC
            """, (user_id,))]

    def create_cluster(self, user_id, body):
        name = text(body.get("name"), "Cluster name", maximum=80)
        description = text(body.get("description", ""), "Description", minimum=0, maximum=500)
        cluster_id = identifier()
        with self.store.transaction() as db:
            db.execute("INSERT INTO clusters VALUES (?, ?, ?, ?, ?)",
                       (cluster_id, name, description, user_id, self.clock()))
            db.execute("INSERT INTO memberships VALUES (?, ?, 'owner', 'approved', ?)",
                       (cluster_id, user_id, self.clock()))
        return {"id": cluster_id}

    def request_join(self, user_id, cluster_id):
        with self.store.transaction() as db:
            if db.execute("SELECT id FROM clusters WHERE id=?", (cluster_id,)).fetchone() is None:
                raise APIError(404, "Cluster not found")
            current = db.execute("SELECT status FROM memberships WHERE cluster_id=? AND user_id=?",
                                 (cluster_id, user_id)).fetchone()
            if current and current["status"] in ("approved", "pending"):
                raise APIError(409, "Already a member or awaiting approval")
            db.execute("INSERT INTO memberships VALUES (?, ?, 'contributor', 'pending', ?) "
                       "ON CONFLICT(cluster_id, user_id) DO UPDATE SET status='pending', created=excluded.created",
                       (cluster_id, user_id, self.clock()))
        return {"status": "pending"}

    def decide_membership(self, user_id, cluster_id, member_id, decision):
        statuses = {"approve": "approved", "reject": "rejected", "revoke": "revoked"}
        if decision not in statuses:
            raise APIError(400, "Choose approve, reject, or revoke")
        with self.store.transaction() as db:
            self._member(db, cluster_id, user_id, owner=True)
            member = db.execute("SELECT * FROM memberships WHERE cluster_id=? AND user_id=?",
                                (cluster_id, member_id)).fetchone()
            if not member:
                raise APIError(404, "Membership not found")
            if member["role"] == "owner":
                raise APIError(400, "The owner's membership cannot be changed")
            db.execute("UPDATE memberships SET status=? WHERE cluster_id=? AND user_id=?",
                       (statuses[decision], cluster_id, member_id))
            if decision != "approve":
                for agent in db.execute("SELECT id FROM agents WHERE cluster_id=? AND user_id=?", (cluster_id, member_id)):
                    self._withdraw(db, agent["id"], "Contributor membership was revoked")
                    db.execute("UPDATE agents SET token_hash=?, enabled=0, shared_devices='[]' WHERE id=?",
                               (digest(secrets.token_urlsafe(32)), agent["id"]))
                db.execute("DELETE FROM pairing_codes WHERE cluster_id=? AND user_id=?", (cluster_id, member_id))
        return {"status": statuses[decision]}

    def cluster(self, user_id, cluster_id):
        with self.store.transaction() as db:
            cluster = dict(self._member(db, cluster_id, user_id))
            is_owner = cluster["owner_id"] == user_id
            cluster["members"] = [dict(row) for row in db.execute(
                "SELECT u.id, u.name, m.role, m.status FROM memberships m JOIN users u ON u.id=m.user_id "
                "WHERE cluster_id=?" + ("" if is_owner else " AND m.status='approved'") + " ORDER BY m.created",
                (cluster_id,))]
            return cluster

    def create_pairing(self, user_id, cluster_id):
        code = secrets.token_urlsafe(18)
        with self.store.transaction() as db:
            self._member(db, cluster_id, user_id)
            db.execute("DELETE FROM pairing_codes WHERE expires<=?", (self.clock(),))
            db.execute("INSERT INTO pairing_codes VALUES (?, ?, ?, ?)",
                       (digest(code), cluster_id, user_id, self.clock() + 600))
        return {"code": code, "expires_in": 600}

    def pair_agent(self, body):
        from cluster_control.spec import validate_capabilities
        code = text(body.get("code"), "Pairing code", maximum=80)
        name = text(body.get("name"), "Computer name", maximum=80)
        address = text(body.get("address"), "Private network address", maximum=253)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", address):
            raise APIError(400, "Use an IPv4 address or DNS hostname reachable by the other workers")
        capabilities = validate_capabilities(body.get("capabilities"))
        shared = self._shared(body.get("shared_devices", []), capabilities)
        token, agent_id = secrets.token_urlsafe(32), identifier()
        with self.store.transaction() as db:
            pair = db.execute("SELECT * FROM pairing_codes WHERE code_hash=? AND expires>?",
                              (digest(code), self.clock())).fetchone()
            if not pair:
                raise APIError(401, "Pairing code expired or already used")
            self._member(db, pair["cluster_id"], pair["user_id"])
            db.execute("DELETE FROM pairing_codes WHERE code_hash=?", (digest(code),))
            db.execute("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                       (agent_id, pair["cluster_id"], pair["user_id"], digest(token), name, address,
                        json.dumps(capabilities), json.dumps(shared), self.clock(), self.clock()))
            return {"agent_id": agent_id, "token": token, "cluster_id": pair["cluster_id"]}

    def _shared(self, selected, capabilities):
        if (not isinstance(selected, list) or any(not isinstance(item, str) for item in selected)
                or len(set(selected)) != len(selected)
                or set(selected) - {item["id"] for item in capabilities["devices"]}):
            raise APIError(400, "Choose devices reported by this computer")
        return selected

    def _agent(self, db, token):
        if not token:
            raise APIError(401, "Worker token required")
        row = db.execute("SELECT * FROM agents WHERE token_hash=?", (digest(token),)).fetchone()
        if not row:
            raise APIError(401, "Invalid worker token")
        self._member(db, row["cluster_id"], row["user_id"])
        return row

    def agents(self, user_id, cluster_id):
        with self.store.transaction() as db:
            self._member(db, cluster_id, user_id)
            values = [self._public_agent(row) for row in db.execute(
                "SELECT a.*, u.name AS contributor FROM agents a JOIN users u ON u.id=a.user_id "
                "JOIN memberships m ON m.cluster_id=a.cluster_id AND m.user_id=a.user_id "
                "WHERE a.cluster_id=? AND m.status='approved' ORDER BY a.created", (cluster_id,))]
            for value in values:
                value["reservations"] = [dict(row) for row in db.execute(
                    "SELECT a.device_id, a.job_id, a.rank, a.state FROM assignments a WHERE a.agent_id=? "
                    "AND a.state IN ('preparing', 'ready', 'running', 'stopping')", (value["id"],))]
            return values

    def _public_agent(self, row):
        keys = ("id", "cluster_id", "user_id", "name", "address", "last_seen", "enabled")
        value = {key: row[key] for key in keys}
        value.update(capabilities=json.loads(row["capabilities"]), shared_devices=json.loads(row["shared_devices"]),
                     online=self.clock() - row["last_seen"] < 30)
        if "contributor" in row.keys():
            value["contributor"] = row["contributor"]
        return value

    def set_sharing(self, user_id, cluster_id, agent_id, body):
        with self.store.transaction() as db:
            self._member(db, cluster_id, user_id)
            agent = db.execute("SELECT * FROM agents WHERE id=? AND cluster_id=?", (agent_id, cluster_id)).fetchone()
            if not agent:
                raise APIError(404, "Computer not found")
            if agent["user_id"] != user_id:
                raise APIError(403, "Only the contributor can change their computer's sharing")
            shared = self._shared(body.get("shared_devices", []), json.loads(agent["capabilities"]))
            if type(body.get("enabled")) is not bool:
                raise APIError(400, "Enabled must be true or false")
            db.execute("UPDATE agents SET shared_devices=?, enabled=? WHERE id=?",
                       (json.dumps(shared), int(body["enabled"]), agent_id))
            removed = set(json.loads(agent["shared_devices"])) - set(shared)
            if not body["enabled"] or removed:
                self._withdraw(db, agent_id, "Contributor paused or withdrew device sharing", removed if body["enabled"] else None)
        return {"ok": True}

    def heartbeat(self, token, capabilities=None):
        with self.store.transaction() as db:
            agent = self._agent(db, token)
            if capabilities is not None:
                from cluster_control.spec import validate_capabilities
                capabilities = validate_capabilities(capabilities)
                if capabilities != json.loads(agent["capabilities"]):
                    self._withdraw(db, agent["id"], "Worker capabilities changed; recreate the job")
                shared = [item for item in json.loads(agent["shared_devices"])
                          if item in {d["id"] for d in capabilities["devices"]}]
                db.execute("UPDATE agents SET capabilities=?, shared_devices=? WHERE id=?",
                           (json.dumps(capabilities), json.dumps(shared), agent["id"]))
            db.execute("UPDATE agents SET last_seen=? WHERE id=?", (self.clock(), agent["id"]))
            return {"agent_id": agent["id"], "enabled": bool(agent["enabled"]),
                    "shared_devices": json.loads(agent["shared_devices"])}

    def _preview(self, db, user_id, cluster_id, body):
        from cluster_control.spec import validate_config
        self._member(db, cluster_id, user_id, owner=True)
        config = validate_config(body.get("config", {}))
        selected = body.get("devices")
        world_size = config["pipeline_size"] * config["replicas"]
        if (not isinstance(selected, list) or len(selected) != world_size or
                any(not isinstance(item, str) for item in selected) or len(set(selected)) != world_size):
            raise APIError(400, "Select exactly %d distinct devices for %d stages × %d replicas" % (
                world_size, config["pipeline_size"], config["replicas"]))
        assignments, physical_gpus = [], set()
        for rank, key in enumerate(selected):
            pieces = key.split(":")
            if len(pieces) != 2:
                raise APIError(400, "Invalid device selection")
            agent_id, device_id = pieces
            agent = db.execute("SELECT * FROM agents WHERE id=? AND cluster_id=?", (agent_id, cluster_id)).fetchone()
            if not agent:
                raise APIError(400, "Selected computer is not in this cluster")
            self._member(db, cluster_id, agent["user_id"])
            if not agent["enabled"] or self.clock() - agent["last_seen"] >= 30:
                raise APIError(409, "Selected computer is offline or sharing is paused")
            capabilities = json.loads(agent["capabilities"])
            device = next((item for item in capabilities["devices"] if item["id"] == device_id), None)
            if not device or device_id not in json.loads(agent["shared_devices"]):
                raise APIError(403, "This device has not been shared by its contributor")
            if device["backend"] not in ("cpu", "cuda") or not capabilities["runtime_ready"]:
                raise APIError(400, "QA jobs currently require a ready CPU or NVIDIA CUDA runtime")
            physical = (capabilities["host_id"], device_id)
            if device["backend"] == "cuda" and physical in physical_gpus:
                raise APIError(409, "The same physical GPU was selected through multiple agents")
            physical_gpus.add(physical)
            assignments.append({"rank": rank, "agent_id": agent_id, "device_id": device_id,
                                "device": device, "computer": agent["name"], "address": agent["address"],
                                "host_id": capabilities["host_id"], "cpu_threads": capabilities["cpu_threads"],
                                "pipeline_stage": rank % config["pipeline_size"],
                                "replica": rank // config["pipeline_size"]})
        if len({item["host_id"] for item in assignments}) > 1 and any(
                item["address"] == "localhost" or item["address"].startswith("127.") for item in assignments):
            raise APIError(400, "Multi-computer jobs need reachable private addresses, not loopback addresses")
        return {"schema_version": 1, "config": config, "world_size": world_size, "assignments": assignments,
                "dist_url": "tcp://%s:%d" % (assignments[0]["address"], config["port"]),
                "warnings": ["Memory fit is checked by the runtime; it is not guaranteed by this preview.",
                             "Membership and rank assignments stay fixed during a job."]}

    def preview_job(self, user_id, cluster_id, body):
        with self.store.transaction() as db:
            return self._preview(db, user_id, cluster_id, body)

    def create_job(self, user_id, cluster_id, body):
        job_id = identifier()
        with self.store.transaction() as db:
            spec = self._preview(db, user_id, cluster_id, body)
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, 'draft', NULL, ?, ?)",
                       (job_id, cluster_id, user_id, json.dumps(spec), self.clock(), self.clock()))
            for assignment in spec["assignments"]:
                db.execute("INSERT INTO assignments(job_id, rank, agent_id, device_id, state) VALUES (?, ?, ?, ?, 'pending')",
                           (job_id, assignment["rank"], assignment["agent_id"], assignment["device_id"]))
        return {"id": job_id, "status": "draft", "spec": spec}

    def jobs(self, user_id, cluster_id):
        with self.store.transaction() as db:
            self._member(db, cluster_id, user_id)
            return [self._public_job(row) for row in db.execute(
                "SELECT * FROM jobs WHERE cluster_id=? ORDER BY created DESC LIMIT 100", (cluster_id,))]

    def _public_job(self, row):
        value = dict(row)
        value["spec"] = json.loads(value["spec"])
        return value

    def job(self, user_id, job_id, after=0):
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise APIError(404, "Job not found")
            self._member(db, row["cluster_id"], user_id)
            value = self._public_job(row)
            value["assignments"] = [dict(item) for item in db.execute(
                "SELECT * FROM assignments WHERE job_id=? ORDER BY rank", (job_id,))]
            for assignment in value["assignments"]:
                assignment["metrics"] = json.loads(assignment["metrics"])
            value["events"] = [dict(item) for item in db.execute(
                "SELECT * FROM job_events WHERE job_id=? AND id>? ORDER BY id LIMIT 200", (job_id, after))]
            return value

    def _event(self, db, job_id, message, rank=None):
        db.execute("INSERT INTO job_events(job_id, rank, timestamp, message) VALUES (?, ?, ?, ?)",
                   (job_id, rank, self.clock(), str(message)[-12000:]))
        db.execute("DELETE FROM job_events WHERE job_id=? AND id NOT IN "
                   "(SELECT id FROM job_events WHERE job_id=? ORDER BY id DESC LIMIT 2000)", (job_id, job_id))

    def _stop(self, db, job_id, status, reason):
        db.execute("UPDATE jobs SET status=?, error=?, updated=? WHERE id=?", (status, reason, self.clock(), job_id))
        db.execute("UPDATE assignments SET state='stopping' WHERE job_id=? "
                   "AND state NOT IN ('succeeded', 'failed', 'stopped')", (job_id,))
        self._event(db, job_id, reason)

    def _withdraw(self, db, agent_id, reason, removed=None):
        rows = list(db.execute("SELECT a.job_id, a.device_id FROM assignments a JOIN jobs j ON j.id=a.job_id "
                               "WHERE a.agent_id=? AND j.status IN ('preparing', 'running')", (agent_id,)))
        for row in rows:
            if removed is None or row["device_id"] in removed:
                self._stop(db, row["job_id"], "failed", reason)

    def start_job(self, user_id, job_id):
        with self.store.transaction() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job:
                raise APIError(404, "Job not found")
            self._member(db, job["cluster_id"], user_id, owner=True)
            if job["status"] != "draft":
                raise APIError(409, "Only a draft job can be started; create a new job to retry")
            spec = json.loads(job["spec"])
            fresh = self._preview(db, user_id, job["cluster_id"], {"config": spec["config"],
                "devices": [a["agent_id"] + ":" + a["device_id"] for a in spec["assignments"]]})
            if spec != fresh:
                raise APIError(409, "Worker capabilities changed; create a new job")
            for assignment in spec["assignments"]:
                if db.execute("SELECT 1 FROM assignments WHERE agent_id=? AND device_id=? "
                              "AND state IN ('preparing', 'ready', 'running', 'stopping')",
                              (assignment["agent_id"], assignment["device_id"])).fetchone():
                    raise APIError(409, "A selected device is already reserved by another job")
            # Separate agent identities cannot bypass a physical GPU reservation.
            for other in db.execute("SELECT spec FROM jobs WHERE id IN (SELECT job_id FROM assignments "
                                    "WHERE state IN ('preparing', 'ready', 'running', 'stopping'))"):
                other_spec = json.loads(other["spec"])
                if spec["dist_url"] == other_spec["dist_url"]:
                    raise APIError(409, "The coordinator address and port are already in use by another job")
                for a in spec["assignments"]:
                    for b in other_spec["assignments"]:
                        if a["device"]["backend"] == "cuda" and (a["host_id"], a["device_id"]) == (b["host_id"], b["device_id"]):
                            raise APIError(409, "A selected physical GPU is already reserved")
            db.execute("UPDATE jobs SET status='preparing', updated=? WHERE id=?", (self.clock(), job_id))
            db.execute("UPDATE assignments SET state='preparing' WHERE job_id=?", (job_id,))
            self._event(db, job_id, "Resources reserved. Waiting for all workers to prepare the runtime and data.")
        return {"status": "preparing"}

    def cancel_job(self, user_id, job_id):
        with self.store.transaction() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job:
                raise APIError(404, "Job not found")
            self._member(db, job["cluster_id"], user_id, owner=True)
            if job["status"] == "draft":
                db.execute("UPDATE jobs SET status='cancelled', updated=? WHERE id=?", (self.clock(), job_id))
            elif job["status"] in ("preparing", "running"):
                self._stop(db, job_id, "cancelling", "Cancelled by the cluster owner")
            return {"status": db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0]}

    def _reconcile(self, db):
        now = self.clock()
        for job in list(db.execute("SELECT * FROM jobs WHERE status IN ('preparing', 'running', 'cancelling', 'failed')")):
            assignments = list(db.execute("SELECT a.*, g.last_seen FROM assignments a JOIN agents g ON g.id=a.agent_id "
                                          "WHERE a.job_id=?", (job["id"],)))
            if job["status"] in ("preparing", "running"):
                stale = [a for a in assignments if a["state"] not in ("succeeded", "failed", "stopped") and now - a["last_seen"] >= 30]
                if stale:
                    self._stop(db, job["id"], "failed", "Worker heartbeat lost; remaining ranks are being stopped")
                elif job["status"] == "preparing" and now - job["updated"] > json.loads(job["spec"])["config"]["timeout_seconds"]:
                    self._stop(db, job["id"], "failed", "Worker preparation timed out")
            # Agents enforce a 10-second local lease. After 30 seconds we can release a lost agent's reservation.
            db.execute("UPDATE assignments SET state='stopped' WHERE job_id=? AND state='stopping' AND agent_id IN "
                       "(SELECT id FROM agents WHERE last_seen<=?)", (job["id"], now - 30))
            if job["status"] == "cancelling" and not db.execute(
                    "SELECT 1 FROM assignments WHERE job_id=? AND state='stopping'", (job["id"],)).fetchone():
                db.execute("UPDATE jobs SET status='cancelled', updated=? WHERE id=?", (now, job["id"]))

    def reconcile(self):
        with self.store.transaction() as db:
            self._reconcile(db)

    def poll_worker(self, token):
        with self.store.transaction() as db:
            self._reconcile(db)
            agent = self._agent(db, token)
            db.execute("UPDATE agents SET last_seen=? WHERE id=?", (self.clock(), agent["id"]))
            commands = []
            for row in db.execute("SELECT a.*, j.spec, j.status AS job_status FROM assignments a JOIN jobs j ON j.id=a.job_id "
                                  "WHERE a.agent_id=? AND a.state IN ('preparing', 'ready', 'running', 'stopping')", (agent["id"],)):
                if row["state"] == "stopping":
                    action = "stop"
                elif row["job_status"] == "running":
                    action = "run"
                else:
                    action = "prepare" if row["state"] == "preparing" else "wait"
                commands.append({"action": action, "job_id": row["job_id"], "rank": row["rank"], "spec": json.loads(row["spec"])})
            return {"commands": commands, "lease_seconds": 10, "enabled": bool(agent["enabled"]),
                    "shared_devices": json.loads(agent["shared_devices"])}

    def report_worker(self, token, body):
        job_id, rank, state = body.get("job_id"), body.get("rank"), body.get("state", "progress")
        if not isinstance(job_id, str) or type(rank) is not int or state not in ("ready", "running", "succeeded", "failed", "stopped", "progress"):
            raise APIError(400, "Invalid worker report")
        message, metrics = body.get("message", ""), body.get("metrics", {})
        if not isinstance(message, str) or len(message) > 12000 or not isinstance(metrics, dict) or len(json.dumps(metrics)) > 8000:
            raise APIError(400, "Worker report is too large or invalid")
        import math
        from cluster_control.spec import METRIC_KEYS
        if set(metrics) - METRIC_KEYS or any(
                value is not None and (type(value) not in (str, int, float, bool) or
                (isinstance(value, str) and len(value) > 200) or
                (isinstance(value, float) and not math.isfinite(value))) for value in metrics.values()):
            raise APIError(400, "Invalid worker metrics")
        with self.store.transaction() as db:
            agent = self._agent(db, token)
            row = db.execute("SELECT a.*, j.status AS job_status FROM assignments a JOIN jobs j ON j.id=a.job_id "
                             "WHERE a.job_id=? AND a.rank=? AND a.agent_id=?", (job_id, rank, agent["id"])).fetchone()
            if not row:
                raise APIError(403, "Rank is not assigned to this worker")
            db.execute("UPDATE agents SET last_seen=? WHERE id=?", (self.clock(), agent["id"]))
            if message:
                self._event(db, job_id, message, rank)
            if metrics:
                current = json.loads(row["metrics"])
                current.update(metrics)
                db.execute("UPDATE assignments SET metrics=? WHERE job_id=? AND rank=?", (json.dumps(current), job_id, rank))
            if row["state"] in ("succeeded", "failed", "stopped"):
                return {"ok": True}
            if state == "progress":
                return {"ok": True}
            if row["state"] == "stopping":
                if state in ("stopped", "failed", "succeeded"):
                    db.execute("UPDATE assignments SET state='stopped' WHERE job_id=? AND rank=?", (job_id, rank))
                self._reconcile(db)
                return {"ok": True}
            if state == "ready":
                if row["state"] == "ready":
                    return {"ok": True}
                if row["state"] not in ("preparing", "ready") or row["job_status"] != "preparing":
                    raise APIError(409, "Rank cannot become ready in this job state")
            elif state == "running":
                if row["state"] not in ("ready", "running") or row["job_status"] != "running":
                    raise APIError(409, "All workers must be ready before training starts")
            elif state == "succeeded":
                if row["state"] != "running" or body.get("exit_code") != 0 or type(metrics.get("completed_steps")) is not int or metrics["completed_steps"] < 1:
                    raise APIError(409, "Success requires a running rank, a zero exit code and completed training steps")
            elif state == "stopped":
                state = "failed"
            exit_code = body.get("exit_code")
            if exit_code is not None and type(exit_code) is not int:
                raise APIError(400, "Exit code must be an integer")
            db.execute("UPDATE assignments SET state=?, exit_code=? WHERE job_id=? AND rank=?", (state, exit_code, job_id, rank))
            if state == "failed":
                self._stop(db, job_id, "failed", message or "A training rank failed")
            elif state == "ready" and not db.execute(
                    "SELECT 1 FROM assignments WHERE job_id=? AND state!='ready'", (job_id,)).fetchone():
                db.execute("UPDATE jobs SET status='running', updated=? WHERE id=?", (self.clock(), job_id))
                self._event(db, job_id, "All workers prepared. Starting the fixed rank assignments.")
            elif state == "succeeded" and not db.execute(
                    "SELECT 1 FROM assignments WHERE job_id=? AND state!='succeeded'", (job_id,)).fetchone():
                db.execute("UPDATE jobs SET status='completed', updated=? WHERE id=?", (self.clock(), job_id))
                self._event(db, job_id, "All training ranks completed successfully. Checkpoints are stored on the workers.")
        return {"ok": True}
