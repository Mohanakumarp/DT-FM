from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch

from scripts.lab_demo import Coordinator, Client, ThreadingHTTPServer, handler_for, make_jobs, validate_result, execute_job, local_ipv4_for, command_for
from scripts.lab_report import load_trials


def result(rank, size, layers):
    return dict(process_wall_seconds=12, peak_process_rss_bytes=1000, threads=2,
        metrics=dict(rank=rank, world_size=size, pipeline_group_size=size,
            resource_schedule=dict(stage_layers=layers, total_layers=6),
            schedule=dict(completed_steps=3),
            model=dict(total_layers=6, embedding_dim=128, seq_length=64, num_heads=4,
                       layers_per_stage=layers[rank], stage_layers=','.join(map(str, layers))),
            iterations=[dict(iter_s=1, forward_s=.3, backward_s=.5, optim_s=.2, train_elapsed_s=t)
                        for t in (1, 3, 5)], losses=[.8, .7, .6] if rank == size - 1 else []))


class LabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.config = dict(steps=2, warmup=1, repeats=1, layers=6)
        self.coordinator = Coordinator(self.config, self.output, 'test-secret')
        self.member = dict(code_sha256='code', python='3.12.10', torch='2.13.0', numpy='2.2.6',
                           data=dict(rows=100, dataset='QQP'), hostname='test')

    def register(self):
        for rank in range(3):
            self.coordinator.dispatch('/register', dict(rank=rank, member=self.member))

    def test_rejects_mixed_code_before_training(self):
        self.coordinator.dispatch('/register', dict(rank=0, member=self.member))
        other = dict(self.member, code_sha256='different')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.coordinator.dispatch('/register', dict(rank=1, member=other))
        self.assertEqual(self.coordinator.state()['status'], 'failed')

    def test_does_not_advance_until_all_ranks_finish(self):
        self.register()
        for rank in (0, 1):
            self.coordinator.dispatch('/result', dict(rank=rank, job_id=0, result=result(rank, 3, [2, 2, 2])))
            self.assertEqual(self.coordinator.current, 0)
        self.coordinator.dispatch('/result', dict(rank=2, job_id=0, result=result(2, 3, [2, 2, 2])))
        self.assertEqual(self.coordinator.current, 1)

    def test_two_computers_complete_without_waiting_for_rank_two(self):
        config = dict(self.config, computers=2)
        coordinator = Coordinator(config, self.output, 'test-secret')
        for rank in range(2):
            coordinator.dispatch('/register', dict(rank=rank, member=self.member))
        self.assertEqual(coordinator.state()['status'], 'running')
        for job in make_jobs(1, 2):
            layers = [6] if job['mode'] == 'single' else [3, 3]
            for rank in job['ranks']:
                coordinator.dispatch('/result', dict(rank=rank, job_id=job['id'], result=result(rank, len(layers), layers)))
        self.assertEqual(coordinator.state()['status'], 'complete')
        _, machines, trials = load_trials(self.output)
        self.assertEqual(len(machines), 2)
        self.assertEqual([t['layers'] for t in trials], ['3,3', '3,3', '6'])

    def test_each_host_single_baseline_uses_local_rank_zero(self):
        config = dict(self.config, computers=2, all_single=True)
        coordinator = Coordinator(config, self.output, 'test-secret')
        for rank in range(2):
            coordinator.dispatch('/register', dict(rank=rank, member=self.member))
        for job in make_jobs(1, 2, True):
            size = len(job['ranks'])
            for rank in job['ranks']:
                payload = result(job['ranks'].index(rank), size, [6] if size == 1 else [3,3])
                payload['host_rank'] = rank
                coordinator.dispatch('/result', dict(rank=rank, job_id=job['id'], result=payload))
        self.assertEqual(coordinator.state()['status'], 'complete')
        _, _, trials = load_trials(self.output)
        self.assertEqual(len(trials), 4)
        self.assertEqual(trials[-1]['results'][0]['host_rank'], 1)
        self.assertEqual(trials[-1]['results'][0]['metrics']['rank'], 0)

    def test_preflight_rejects_oversized_model_without_allocating_it(self):
        from scripts.lab_hardware import memory_preflight
        with patch('scheduler.resources.host_memory', return_value={'available_bytes': 1024 * 1024}):
            with self.assertRaisesRegex(ValueError, 'estimate, not an observed OOM'):
                memory_preflight(dict(self.config, all_single=True), 0, 2, 28998)

    def test_incomplete_and_nonfinite_results_rejected(self):
        job = make_jobs(1)[0]
        broken = result(2, 3, [2, 2, 2])
        broken['metrics']['schedule']['completed_steps'] = 2
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_result(broken, self.config, job, 2)
        broken = result(2, 3, [2, 2, 2])
        broken['metrics']['losses'][-1] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite loss'):
            validate_result(broken, self.config, job, 2)

    def test_rank_disagreement_rejected(self):
        self.register()
        self.coordinator.dispatch('/result', dict(rank=0, job_id=0, result=result(0, 3, [2, 2, 2])))
        with self.assertRaisesRegex(ValueError, 'allocation'):
            self.coordinator.dispatch('/result', dict(rank=1, job_id=0, result=result(1, 3, [1, 3, 2])))

    def test_report_excludes_warmup_counts_examples_once(self):
        self.register()
        for job in make_jobs(1):
            layers = [6] if job['mode'] == 'single' else [2, 2, 2]
            for rank in job['ranks']:
                self.coordinator.dispatch('/result', dict(rank=rank, job_id=job['id'], result=result(rank, len(layers), layers)))
        _, _, trials = load_trials(self.output)
        self.assertEqual([t['examples_per_second'] for t in trials], [4, 4, 4])
        self.assertEqual(self.coordinator.state()['status'], 'complete')
        self.coordinator.fail('forced failure')
        with self.assertRaisesRegex(ValueError, 'failed'):
            load_trials(self.output)

    def test_authenticated_http_and_duplicate_rank(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.coordinator))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = Client('127.0.0.1', server.server_port, 'test-secret')
            self.assertEqual(client.call('/config'), self.config)
            with self.assertRaises(urllib.error.HTTPError) as forbidden:
                client.opener.open(client.url + '/source-update.zip')
            self.assertEqual(forbidden.exception.code, 403)
            request = urllib.request.Request(client.url + '/other-file',
                headers={'Authorization': 'Bearer test-secret'})
            with self.assertRaises(urllib.error.HTTPError) as missing:
                client.opener.open(request)
            self.assertEqual(missing.exception.code, 404)
            with self.assertRaisesRegex(RuntimeError, 'rejected'):
                Client('127.0.0.1', server.server_port, 'wrong').call('/config')
            client.call('/register', dict(rank=0, member=self.member))
            with self.assertRaisesRegex(RuntimeError, 'already registered'):
                client.call('/register', dict(rank=0, member=self.member))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_launcher_selects_explicit_ipv4_and_short_collective_timeout(self):
        self.assertEqual(local_ipv4_for('127.0.0.1', 29500), '127.0.0.1')
        config = dict(self.config, master='127.0.0.1', train_port=29500, synthetic=True)
        job = make_jobs(1, 2)[0]
        for rank in range(2):
            command = command_for(config, job, rank, self.output)
            self.assertEqual(command[command.index('--gloo-bind-address') + 1], '127.0.0.1')
            self.assertEqual(command[command.index('--dist-timeout-seconds') + 1], '90')

    def test_peer_failure_terminates_running_training_child(self):
        import psutil
        marker = self.output / 'child.pid'
        command = [sys.executable, '-c',
                   "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(marker)]
        class FailedClient:
            def call(self, _):
                return dict(status='failed', error='another rank failed')
        with patch('scripts.lab_demo.command_for', return_value=command):
            with self.assertRaisesRegex(RuntimeError, 'another rank failed'):
                execute_job(FailedClient(), dict(timeout=30), make_jobs(1)[0], 0, self.output / 'worker', 2)
        self.assertFalse(psutil.pid_exists(int(marker.read_text())))


if __name__ == '__main__':
    unittest.main()
