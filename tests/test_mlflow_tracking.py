import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from utils.metrics import RunMetrics
from utils.mlflow_tracking import tracked_run


class TrackingTests(unittest.TestCase):
    def test_disabled_requires_no_tracking_configuration(self):
        with tracked_run(argparse.Namespace()) as tracker:
            self.assertIsNone(tracker)

    def test_step_alignment_and_rank_throughput(self):
        tracker = Mock()
        metrics = RunMetrics(argparse.Namespace(batch_size=8), tracker)
        metrics.record_iter({'iter_s': 2.0})
        metrics.record_loss(0.4)
        tracker.log_metrics.assert_called_once_with(
            {'iter_s': 2.0, 'samples_per_second': 4.0}, step=1, synchronous=True)
        tracker.log_metric.assert_called_once_with(
            'last_microbatch_loss', 0.4, step=1, synchronous=True)

    def test_real_store_failure_status_and_history(self):
        try:
            import mlflow
        except ImportError:
            self.skipTest('optional MLflow dependency not installed')
        with tempfile.TemporaryDirectory() as directory:
            uri = 'sqlite:///' + str(Path(directory) / 'tracking.db').replace('\\', '/')
            old_uri = mlflow.get_tracking_uri()
            args = argparse.Namespace(mlflow_tracking_uri=uri,
                                      mlflow_experiment='integration-test',
                                      mlflow_run_name='failure-test', mlflow_group='group',
                                      rank=1, synthetic_data=True, batch_size=2)
            try:
                mlflow.set_tracking_uri(uri)
                mlflow.create_experiment('integration-test',
                                         artifact_location=Path(directory).as_uri())
                with self.assertRaisesRegex(ValueError, 'training failed'):
                    with tracked_run(args) as tracker:
                        run_id = tracker.active_run().info.run_id
                        metrics = RunMetrics(args, tracker)
                        metrics.record_iter({'iter_s': 0.5})
                        metrics.record_loss(0.7)
                        raise ValueError('training failed')
                client = mlflow.MlflowClient(tracking_uri=uri)
                run = client.get_run(run_id)
                self.assertEqual(run.info.status, 'FAILED')
                self.assertEqual(run.data.params['rank'], '1')
                history = client.get_metric_history(run_id, 'last_microbatch_loss')
                self.assertEqual([(point.step, point.value) for point in history], [(1, 0.7)])

                # Exercise the successful finalization and artifact path against
                # the same real store, including unavailable probe links.
                args.world_size = args.pipeline_group_size = 2
                args.seq_length, args.embedding_dim = 32, 64
                args.num_layers, args.num_heads, args.num_iters = 1, 4, 2
                args.stage_layers, args.total_layers = '2,1', 3
                args.layer_start, args.layer_end = 2, 3
                with tracked_run(args) as tracker:
                    success_id = tracker.active_run().info.run_id
                    metrics = RunMetrics(args, tracker)
                    metrics.mark('train_start')
                    for loss in (0.8, 0.6):
                        metrics.record_iter({'iter_s': 0.5})
                        metrics.record_loss(loss)
                    metrics.mark('train_end')
                    metrics.set_comm_matrix([[0, None], [2.5, 0]],
                                            [[0, None], [80, 0]], ['rank-0', 'rank-1'])
                    payload = metrics.dump(str(Path(directory) / 'metrics_rank1.json'))
                success = client.get_run(success_id)
                self.assertEqual(success.info.status, 'FINISHED')
                self.assertEqual(success.data.tags['launch_group'], 'group')
                self.assertEqual(success.data.tags['pipeline_stage'], '1/2')
                self.assertEqual(success.data.tags['pipeline_role'], 'last_stage')
                self.assertEqual(success.data.metrics['completed_steps'], 2)
                self.assertEqual(success.data.metrics['samples_per_second'], 4)
                self.assertEqual(success.data.params['stage_layers'], '2,1')
                self.assertEqual(success.data.params['total_layers'], '3')
                self.assertEqual(payload['model']['layer_start'], 2)
                self.assertEqual(payload['model']['layer_end'], 3)
                history = client.get_metric_history(success_id, 'last_microbatch_loss')
                self.assertEqual([(point.step, point.value) for point in history],
                                 [(1, 0.8), (2, 0.6)])
                downloaded = client.download_artifacts(success_id,
                                                       'measurements/metrics_rank1.json')
                self.assertEqual(json.loads(Path(downloaded).read_text()), payload)
                # Dynamic assignment is unknown when the tracking run opens.
                # Logging it later must not conflict with placeholder params.
                args.dynamic_total_layers = 5
                args.num_layers, args.stage_layers = 99, None
                with tracked_run(args) as tracker:
                    dynamic_id = tracker.active_run().info.run_id
                    tracker.log_params(dict(num_layers=3, stage_layers='2,3', total_layers=5,
                                            layer_start=2, layer_end=5))
                dynamic = client.get_run(dynamic_id)
                self.assertEqual(dynamic.data.params['num_layers'], '3')
                self.assertEqual(dynamic.data.params['stage_layers'], '2,3')
            finally:
                mlflow.set_tracking_uri(old_uri)
                # Release SQLite handles before Windows removes the temp directory.
                from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
                engine_map = getattr(SqlAlchemyStore, '_engine_map', None)
                if engine_map is not None:
                    engine = engine_map.pop(uri, None)
                    if engine is not None:
                        engine.dispose()


if __name__ == '__main__':
    unittest.main()
