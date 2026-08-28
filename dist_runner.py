import argparse
import os
import sys
import subprocess
import torch
import torch.autograd.profiler as profiler
import torch.distributed as dist
from task_datasets.qqp import get_glue_qqp_train_data_loader
from task_datasets.synthetic import get_synthetic_train_data_loader
from task_datasets.tokenizer import build_tokenizer
from pipeline_parallel.dist_pp_utils import get_pp_module
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from comm.comm_utils import *
from comm.comm_probe import measure_comm_matrix
from utils.metrics import RunMetrics
from utils.device_utils import describe_device, resolve_device, validate_runtime_args


def _local_rank_argv(rank):
    argv = [sys.executable, '-u', os.path.abspath(sys.argv[0])]
    args = sys.argv[1:]
    i = 0
    saw_rank = False
    while i < len(args):
        tok = args[i]
        if tok == '--rank':
            argv.extend(['--rank', str(rank)])
            i += 2
            saw_rank = True
            continue
        if tok.startswith('--rank='):
            argv.append('--rank=' + str(rank))
            i += 1
            saw_rank = True
            continue
        if tok == '--spawn-local-ranks':
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                i += 2
            else:
                i += 1
            continue
        if tok.startswith('--spawn-local-ranks='):
            i += 1
            continue
        argv.append(tok)
        i += 1
    if not saw_rank:
        argv.extend(['--rank', str(rank)])
    return argv


def spawn_local_ranks(args):
    if not args.spawn_local_ranks:
        return []
    if args.rank != 0:
        return []
    if args.world_size <= 1:
        return []

    children = []
    for rank in range(1, args.world_size):
        cmd = _local_rank_argv(rank)
        print("Spawning local rank", rank, ":", " ".join(cmd))
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        children.append(subprocess.Popen(cmd, env=env))
    return children


def wait_local_ranks(children):
    rc = 0
    for proc in children:
        child_rc = proc.wait()
        print("Local child pid", proc.pid, "exited with", child_rc)
        if child_rc != 0:
            rc = child_rc
    return rc


def main():
    parser = argparse.ArgumentParser(description='Decentralized-GPT3-XL')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_qqp_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='no-profiling', metavar='S',
                        help='profiling mode: no-profiling, tidy_profiling, or pytorch_profiling')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    args = parser.parse_args()
    print("==== Process rank", args.rank, "pid", os.getpid(),
          "world_size", args.world_size,
          "pipeline_group_size", args.pipeline_group_size)
    children = spawn_local_ranks(args)
    try:
        run_training(args)
    except Exception:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        raise
    else:
        child_rc = wait_local_ranks(children)
        if child_rc != 0:
            raise SystemExit(child_rc)


def run_training(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args)
    validate_runtime_args(args, device)
    print('==== Compute device:', describe_device(device, args.device_backend))

    init_communicators(args)

    if get_pipeline_parallel_rank() == 0 or get_pipeline_parallel_rank() == args.pipeline_group_size-1:
        if args.synthetic_data:
            vocab_size = args.synthetic_vocab_size
            train_data_loader = get_synthetic_train_data_loader(args)
            print('synthetic vocab size:', vocab_size)
        else:
            tokenizer = build_tokenizer(args)
            print("token vocab size:", tokenizer.vocab_size)
            train_data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
            vocab_size = tokenizer.vocab_size
        num_classes = 2
    else:
        train_data_loader = None
        num_classes = 2
        vocab_size = -1

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")

    pipe = get_pp_module(args, vocab_size, num_classes, device, use_dp)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    print("Rank", args.rank, "stage params:", n_params)

    metrics = RunMetrics(args)
    # First/last ranks load all of QQP (~minutes). The middle rank skips the
    # dataset and used to enter the blocking comm probe immediately, so Gloo
    # send/recv hung until the 30min timeout. Wait here so every rank has
    # finished init before any send.
    if args.world_size > 1:
        print("Rank", args.rank, "waiting at post-init barrier before comm probe / train")
        dist.barrier()
        print("Rank", args.rank, "post-init barrier done")

    skip_probe = getattr(args, 'skip_comm_probe', True)
    metrics.mark('probe_start')
    if args.world_size == 1:
        print('[probe] skipped for a single-process run.')
    elif skip_probe:
        print('[probe] skipped (--skip-comm-probe). All ranks must skip or all must probe.')
    else:
        try:
            lat, bw = measure_comm_matrix(
                get_pipeline_parallel_comm(),
                args.rank,
                args.world_size,
                device='cpu',
            )
            peers = ['rank-%d' % i for i in range(args.world_size)]
            metrics.set_comm_matrix(lat, bw, peers)
        except Exception as exc:
            print('[probe] comm matrix failed:', repr(exc))
            print('[probe] do not start training until every rank leaves the probe')
            raise
    metrics.mark('probe_end')
    if args.world_size > 1:
        dist.barrier()

    if args.profiling == 'no-profiling':
        distributed_train_foo_iter(args, pipe, device, train_data_loader, metrics=metrics)
        metrics.dump('%s/metrics_rank%d.json' % (args.metrics_dir, args.rank))
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            distributed_train_foo_iter(args, pipe, device, train_data_loader, metrics=metrics)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                distributed_train_foo_iter(args, pipe, device, train_data_loader, metrics=metrics)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
        metrics.dump('%s/metrics_rank%d.json' % (args.metrics_dir, args.rank))


if __name__ == '__main__':
    main()
