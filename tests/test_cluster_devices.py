"""Device consent, expiring pairing, immutable job configuration and topology."""
import unittest

from cluster_control.service import APIError, Service
from cluster_control.store import Store


def capabilities(backend="cpu", host="test-host"):
    return {"host_id": host, "platform": "Linux", "runtime_ready": True, "cpu_threads": 1,
            "devices": [{"id": "cpu" if backend == "cpu" else "gpu-0", "name": "Test device", "backend": backend}]}


class DeviceFixture:
    def setUp(self):
        self.store = Store(":memory:")
        self.now = 1000
        self.service = Service(self.store, clock=lambda: self.now)
        self.owner = self.service.register({"name": "Owner", "email": "o@example.com", "password": "long test password"})[0]["user"]["id"]
        self.peer = self.service.register({"name": "Peer", "email": "p@example.com", "password": "long test password"})[0]["user"]["id"]
        self.cluster = self.service.create_cluster(self.owner, {"name": "Lab"})["id"]
        self.service.request_join(self.peer, self.cluster)
        self.service.decide_membership(self.owner, self.cluster, self.peer, "approve")
        self.agent = self.pair(self.peer)
        self.selection = [self.agent["agent_id"] + ":cpu"]

    def tearDown(self):
        self.store.close()

    def pair(self, user, caps=None):
        caps = caps or capabilities()
        code = self.service.create_pairing(user, self.cluster)["code"]
        return self.service.pair_agent({"code": code, "name": "Laptop", "address": "127.0.0.1", "capabilities": caps,
                                        "shared_devices": [caps["devices"][0]["id"]]})


class DeviceTests(DeviceFixture, unittest.TestCase):

    def test_single_use_expiring_and_membership_bound_pairing(self):
        code = self.service.create_pairing(self.peer, self.cluster)["code"]
        body = {"code": code, "name": "Laptop", "address": "localhost", "capabilities": capabilities()}
        self.service.pair_agent(body)
        with self.assertRaisesRegex(APIError, "already used"):
            self.service.pair_agent(body)
        body["code"] = self.service.create_pairing(self.peer, self.cluster)["code"]
        self.now += 601
        with self.assertRaisesRegex(APIError, "expired"):
            self.service.pair_agent(body)
        self.service.decide_membership(self.owner, self.cluster, self.peer, "revoke")
        with self.assertRaises(APIError):
            self.service.heartbeat(self.agent["token"])

    def test_only_contributor_can_change_sharing(self):
        body = {"enabled": False, "shared_devices": []}
        with self.assertRaisesRegex(APIError, "Only the contributor"):
            self.service.set_sharing(self.owner, self.cluster, self.agent["agent_id"], body)
        self.service.set_sharing(self.peer, self.cluster, self.agent["agent_id"], body)
        self.assertFalse(self.service.heartbeat(self.agent["token"])["enabled"])
        with self.assertRaisesRegex(APIError, "paused"):
            self.service.preview_job(self.owner, self.cluster, {"devices": self.selection})
        self.assertNotIn("token_hash", self.service.agents(self.owner, self.cluster)[0])

    def test_owner_only_versioned_snapshot_and_rank_topology(self):
        second = self.pair(self.owner)
        body = {"config": {"pipeline_size": 2, "replicas": 1},
                "devices": self.selection + [second["agent_id"] + ":cpu"]}
        with self.assertRaisesRegex(APIError, "owner access"):
            self.service.create_job(self.peer, self.cluster, body)
        job = self.service.create_job(self.owner, self.cluster, body)
        self.assertEqual(job["spec"]["schema_version"], 1)
        self.assertEqual(job["spec"]["world_size"], 2)
        self.assertEqual([a["rank"] for a in job["spec"]["assignments"]], [0, 1])
        body["config"]["pipeline_size"] = 1
        self.assertEqual(self.service.job(self.peer, job["id"])["spec"]["config"]["pipeline_size"], 2)

    def test_rejects_invalid_settings_devices_and_unready_backends(self):
        for config in ({"pipeline_size": True}, {"replicas": 2}, {"learning_rate": float("nan")},
                       {"shell": "rm -rf /"}, {"model": "../../model"}, {"doc_stride": 128}):
            with self.subTest(config=config), self.assertRaises(APIError):
                self.service.preview_job(self.owner, self.cluster, {"config": config, "devices": self.selection})
        agent = self.pair(self.owner, capabilities("xpu"))
        with self.assertRaisesRegex(APIError, "CPU or NVIDIA"):
            self.service.preview_job(self.owner, self.cluster, {"devices": [agent["agent_id"] + ":gpu-0"]})
        self.now += 31
        with self.assertRaisesRegex(APIError, "offline"):
            self.service.preview_job(self.owner, self.cluster, {"devices": self.selection})

    def test_prevents_duplicate_physical_gpu_assignment(self):
        one = self.pair(self.owner, capabilities("cuda"))
        two = self.pair(self.peer, capabilities("cuda"))
        with self.assertRaisesRegex(APIError, "same physical GPU"):
            self.service.preview_job(self.owner, self.cluster, {"config": {"pipeline_size": 2},
                "devices": [one["agent_id"] + ":gpu-0", two["agent_id"] + ":gpu-0"]})

    def test_multi_host_jobs_reject_loopback_addresses(self):
        second = self.pair(self.owner, capabilities(host="other-host"))
        with self.assertRaisesRegex(APIError, "private addresses"):
            self.service.preview_job(self.owner, self.cluster, {"config": {"pipeline_size": 2},
                "devices": self.selection + [second["agent_id"] + ":cpu"]})

    def test_reapproval_does_not_restore_revoked_worker_tokens_or_pairing_codes(self):
        code = self.service.create_pairing(self.peer, self.cluster)["code"]
        self.service.decide_membership(self.owner, self.cluster, self.peer, "revoke")
        self.service.request_join(self.peer, self.cluster)
        self.service.decide_membership(self.owner, self.cluster, self.peer, "approve")
        with self.assertRaisesRegex(APIError, "Invalid worker token"):
            self.service.heartbeat(self.agent["token"])
        with self.assertRaisesRegex(APIError, "expired or already used"):
            self.service.pair_agent({"code": code, "name": "Old identity", "address": "localhost", "capabilities": capabilities()})
        fresh = self.pair(self.peer)
        self.service.heartbeat(fresh["token"])


if __name__ == "__main__":
    unittest.main()
