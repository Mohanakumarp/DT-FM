"""Optional, per-rank tracking for the existing training loop."""
from contextlib import contextmanager
import os
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
        rank = getattr(args, 'rank', 0)
        pp_size = getattr(args, 'pipeline_group_size', 1)
        if pp_size == 1:
            role = 'single_stage'
        elif rank == 0:
            role = 'first_stage'
        elif rank == pp_size - 1:
            role = 'last_stage'
        else:
            role = 'middle_stage'

        is_bert = ((getattr(args, 'seq_length', None) == 128 and getattr(args, 'embedding_dim', None) in (768, 1024, 1152, 1280, 1344, 1408)) or
                   'bert' in getattr(args, 'mlflow_run_name', '').lower() or
                   'bert' in getattr(args, 'mlflow_experiment', '').lower())
        if is_bert:
            if getattr(args, 'embedding_dim', None) == 1280:
                model_arch = 'BERT-500M'
            elif getattr(args, 'embedding_dim', None) == 1024:
                model_arch = 'BERT-Large'
            else:
                model_arch = 'BERT'
        else:
            model_arch = 'GPT'

        tags = {
            'host': socket.gethostname(),
            'launch_group': args.mlflow_group or args.mlflow_run_name,
            'dataset': 'synthetic smoke test' if getattr(args, 'synthetic_data', False) else 'QQP',
            'pipeline_stage': '%d/%d' % (rank, pp_size),
            'pipeline_role': role,
            'model_architecture': model_arch,
            'network_interface': os.environ.get('GLOO_SOCKET_IFNAME', 'auto'),
            'mlflow.note.content': (
                'Real forward/backward/optimizer execution. Synthetic data is not a quality benchmark. '
                'Loss is the final microbatch loss, not the batch average. '
                'Stage timings include communication and synchronization; they are not pure compute. '
                'Each run represents one rank; samples/s is not summed across pipeline ranks.'
            ),
        }
        if hasattr(args, 'device'):
            tags['device_requested'] = str(args.device)
        if hasattr(args, 'device_backend'):
            tags['device_backend'] = str(args.device_backend)
        if hasattr(args, 'tensor_comm'):
            tags['tensor_comm'] = str(args.tensor_comm)
        if hasattr(args, 'dist_backend'):
            tags['dist_backend'] = str(args.dist_backend)
        mlflow.set_tags(tags)

        # Explicit allowlist avoids sending tracking credentials or arbitrary paths.
        keys = ('rank', 'world_size', 'pipeline_group_size', 'data_group_size',
                'device', 'device_backend', 'seq_length', 'embedding_dim', 'num_layers', 'num_heads',
                'batch_size', 'micro_batch_size', 'gradient_accumulate_step',
                'lr', 'seed', 'num_iters', 'num_epochs', 'steps_per_epoch',
                'synthetic_data', 'synthetic_vocab_size', 'task', 'pp_mode',
                'tensor_comm', 'dist_backend', 'dist_url', 'fp16', 'stage_layers',
                'total_layers', 'layer_start', 'layer_end',
                'rebalance_every', 'rebalance_min_improvement', 'tokenizer_type')
        if getattr(args, 'dynamic_total_layers', None) is not None:
            # These immutable MLflow params are known only after negotiation.
            keys = tuple(key for key in keys if key not in
                         ('num_layers', 'stage_layers', 'total_layers', 'layer_start', 'layer_end'))
            keys += ('dynamic_total_layers', 'scheduler_memory_fraction', 'scheduler_reserve_mb')
        mlflow.log_params({key: getattr(args, key) for key in keys if hasattr(args, key)})
        yield mlflow
