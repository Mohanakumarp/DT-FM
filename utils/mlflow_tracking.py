"""Optional, per-rank tracking for the existing training loop."""
from contextlib import contextmanager
import socket
import sys


def add_tracking_arguments(parser):
    parser.add_argument('--mlflow-tracking-uri', default=None,
                        help='Enable MLflow, e.g. http://127.0.0.1:5000')
    parser.add_argument('--mlflow-experiment', default='DT-FM training')
    parser.add_argument('--mlflow-run-name', default='training')
    parser.add_argument('--mlflow-group', default=None,
                        help='Shared identifier for all ranks in a distributed launch')


@contextmanager
def tracked_run(args):
    if not getattr(args, 'mlflow_tracking_uri', None):
        yield None
        return
    # MLflow prints Unicode run links even when Windows stdout is redirected
    # to a legacy-encoded file. Console decoration must not fail training.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError('Install requirements-mlflow.txt to enable tracking') from exc
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name='%s / rank %d' % (args.mlflow_run_name, args.rank)):
        mlflow.set_tags({
            'host': socket.gethostname(),
            'launch_group': args.mlflow_group or args.mlflow_run_name,
            'dataset': 'synthetic smoke test' if args.synthetic_data else 'QQP',
            'mlflow.note.content': (
                'Real forward/backward/optimizer execution. Synthetic data is not a quality benchmark. '
                'Loss is the final microbatch loss, not the batch average. '
                'Stage timings include communication and synchronization; they are not pure compute. '
                'Each run represents one rank; samples/s is not summed across pipeline ranks.'
            ),
        })
        # Explicit allowlist avoids sending tracking credentials or arbitrary paths.
        keys = ('rank', 'world_size', 'pipeline_group_size', 'data_group_size',
                'device', 'seq_length', 'embedding_dim', 'num_layers', 'num_heads',
                'batch_size', 'micro_batch_size', 'gradient_accumulate_step',
                'lr', 'seed', 'num_iters', 'num_epochs', 'steps_per_epoch',
                'synthetic_data', 'synthetic_vocab_size', 'task', 'pp_mode',
                'tensor_comm', 'fp16')
        mlflow.log_params({key: getattr(args, key) for key in keys if hasattr(args, key)})
        yield mlflow
