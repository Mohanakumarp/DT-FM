"""Coordinator readiness barrier, reservations, failure propagation and leases."""
import unittest
from test_cluster_devices import DeviceFixture
from cluster_control.service import APIError


class LifecycleTests(DeviceFixture, unittest.TestCase):
    def make_job(self, count=2):
        agents = [self.agent] + [self.pair(self.owner) for _ in range(count - 1)]
        body = {"config": {"pipeline_size": count}, "devices": [a["agent_id"] + ":cpu" for a in agents]}
        job = self.service.create_job(self.owner, self.cluster, body)
        self.service.start_job(self.owner, job["id"])
        return job, agents

    def report(self, agent, job, rank, state, **extra):
        return self.service.report_worker(agent["token"], {"job_id": job["id"], "rank": rank, "state": state, **extra})

    def test_all_workers_ready_before_training_and_success(self):
        job, agents = self.make_job()
        self.assertEqual(self.service.poll_worker(agents[0]["token"])["commands"][0]["action"], "prepare")
        with self.assertRaisesRegex(APIError, "All workers"):
            self.report(agents[0], job, 0, "running")
        self.report(agents[0], job, 0, "ready")
        self.assertEqual(self.service.poll_worker(agents[0]["token"])["commands"][0]["action"], "wait")
        self.report(agents[1], job, 1, "ready")
        self.assertEqual(self.service.job(self.owner, job["id"])["status"], "running")
        for rank, agent in enumerate(agents):
            self.assertEqual(self.service.poll_worker(agent["token"])["commands"][0]["action"], "run")
            self.report(agent, job, rank, "running")
            self.report(agent, job, rank, "succeeded", exit_code=0, metrics={"completed_steps": 2})
        self.assertEqual(self.service.job(self.owner, job["id"])["status"], "completed")

    def test_rank_identity_and_device_reservation(self):
        job, agents = self.make_job()
        with self.assertRaisesRegex(APIError, "not assigned"):
            self.report(agents[0], job, 1, "ready")
        draft = self.service.create_job(self.owner, self.cluster, {"devices": self.selection})
        with self.assertRaisesRegex(APIError, "reserved"):
            self.service.start_job(self.owner, draft["id"])
        with self.assertRaisesRegex(APIError, "owner access"):
            self.service.cancel_job(self.peer, job["id"])

    def test_cancel_waits_for_worker_stop_acknowledgements(self):
        job, agents = self.make_job()
        self.service.cancel_job(self.owner, job["id"])
        self.assertEqual(self.service.job(self.owner, job["id"])["status"], "cancelling")
        for rank, agent in enumerate(agents):
            self.assertEqual(self.service.poll_worker(agent["token"])["commands"][0]["action"], "stop")
            self.report(agent, job, rank, "stopped")
        self.assertEqual(self.service.job(self.owner, job["id"])["status"], "cancelled")

    def test_failure_stops_peers_and_holds_resources_until_stopped(self):
        job, agents = self.make_job()
        self.report(agents[0], job, 0, "failed", exit_code=1, message="preflight failed")
        self.assertEqual(self.service.job(self.owner, job["id"])["status"], "failed")
        self.assertEqual(self.service.poll_worker(agents[1]["token"])["commands"][0]["action"], "stop")
        draft = self.service.create_job(self.owner, self.cluster, {"devices": [agents[1]["agent_id"] + ":cpu"]})
        with self.assertRaisesRegex(APIError, "reserved"):
            self.service.start_job(self.owner, draft["id"])
        self.report(agents[1], job, 1, "stopped")
        self.service.start_job(self.owner, draft["id"])

    def test_heartbeat_loss_and_contributor_withdrawal_stop_job(self):
        job, agents = self.make_job()
        self.now += 31
        self.service.reconcile()
        self.assertEqual(self.service.job(self.owner, job["id"])["status"], "failed")
        self.assertEqual(self.service.poll_worker(agents[0]["token"])["commands"], [])
        for agent in agents:
            self.service.heartbeat(agent["token"])
        other, _ = self.make_job()
        self.service.set_sharing(self.peer, self.cluster, self.agent["agent_id"], {"enabled": False, "shared_devices": []})
        self.assertEqual(self.service.job(self.owner, other["id"])["status"], "failed")
