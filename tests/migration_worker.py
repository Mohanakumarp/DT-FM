"""Real Gloo worker used only by the state-continuity integration tests."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from comm.comm_utils import init_communicators
from pipeline_parallel.dist_gpipe_pipeline_async import GpipeAsync
from scheduler.dynamic_runtime import collect
from scheduler.migration import migrate, stage_units, state_fingerprints, unit_tensors
from scheduler.pipeline_scheduler import apply_stage_assignment
from scheduler.rebalance import maybe_rebalance
from scheduler.resources import resource_snapshot
from utils.device_utils import resolve_device
from utils.dist_args_utils import (add_device_arguments, add_torch_distributed_arguments, add_model_arguments,
                                   add_qqp_task_arguments, add_training_hyper_parameter_arguments,
                                   add_mixed_precision_arguments, add_parallel_schema_arguments)


def main():
    parser = argparse.ArgumentParser()
    for add in (add_device_arguments, add_torch_distributed_arguments, add_model_arguments,
                add_qqp_task_arguments, add_training_hyper_parameter_arguments,
                add_mixed_precision_arguments, add_parallel_schema_arguments):
        add(parser)
    parser.add_argument('--mode', choices=('baseline', 'migrate', 'rollback', 'controller'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.set_defaults(device='cpu', dist_backend='gloo', tensor_comm='gloo', data_group_size=1,
                        task='SeqClassification', seq_length=8, embedding_dim=16, num_heads=2,
                        batch_size=2, micro_batch_size=1, use_offload=False)
    args = parser.parse_args()
    args.profiling, args.seed = 'no-profiling', 123
    args.stage_layers = '1,3' if args.world_size == 2 else '1,2,3'
    original = list(map(int, args.stage_layers.split(',')))
    alternate = list(reversed(original))
    apply_stage_assignment(args)
    device = resolve_device(args)
    init_communicators(args)
    torch.manual_seed(123 + args.rank)
    pipeline = GpipeAsync(args, 64, 2, device)
    pipeline.optimizer = torch.optim.SGD(pipeline.model.parameters(), lr=.01, momentum=.9)
    args.resource_schedule = dict(model=dict(seq_length=8, embedding_dim=16, num_heads=2,
                                            batch_size=2, micro_batch_size=1, vocab_size=64))
    args.dynamic_total_layers, args.rebalance_every, args.rebalance_min_improvement = sum(original), 1, 0.
    events, losses = [], []
    try:
        for step in range(4):
            generator = torch.Generator().manual_seed(700 + step)
            tokens = torch.randint(64, (2, 8), generator=generator)
            labels = torch.randint(2, (2,), generator=generator)
            pipeline.sgd_iter(tokens if args.rank == 0 else None,
                              labels if args.rank == args.world_size - 1 else None)
            if pipeline.last_loss is not None:
                losses.append(pipeline.last_loss)
            if args.mode != 'baseline' and step < 2:
                before = collect(args, lambda: state_fingerprints(args, pipeline))
                rng = torch.get_rng_state().clone()
                old_model, old_optimizer = pipeline.model, pipeline.optimizer
                target = alternate if step == 0 else original
                if args.mode == 'rollback':
                    try:
                        with patch('scheduler.migration.nn.Sequential', side_effect=RuntimeError('injected candidate failure')) if args.rank == 1 else nullcontext():
                            migrate(args, pipeline, alternate)
                    except ValueError as exc:
                        if 'injected candidate failure' not in str(exc):
                            raise
                    else:
                        raise AssertionError('injected preparation failure was not propagated')
                    assert pipeline.model is old_model and pipeline.optimizer is old_optimizer
                    event = dict(status='rolled_back')
                elif args.mode == 'controller':
                    snapshots = collect(args, lambda: resource_snapshot(args, device))
                    reports = [dict(rank=r, resources=snapshot, timings=dict(
                        layer_ms=1. if r == (0 if step == 0 else args.world_size - 1) else 100.,
                        first_stage_ms=0., last_stage_ms=0.)) for r, snapshot in enumerate(snapshots)]
                    with patch('scheduler.rebalance.measure_live_reports', return_value=reports):
                        event = maybe_rebalance(args, pipeline, step + 1, 4)
                    assert event['status'] == 'committed', event
                else:
                    event = migrate(args, pipeline, target)
                after = collect(args, lambda: state_fingerprints(args, pipeline))
                assert {k: v for part in before for k, v in part.items()} == {k: v for part in after for k, v in part.items()}
                assert torch.equal(rng, torch.get_rng_state())
                events.append(event)
        state = {unit + '/' + name: tensor.detach().cpu().clone()
                 for unit, module in stage_units(args, pipeline.model).items()
                 for name, tensor in unit_tensors(module, pipeline.optimizer).items()}
        torch.save(state, args.output / ('state_rank%d.pt' % args.rank))
        (args.output / ('result_rank%d.json' % args.rank)).write_text(json.dumps(
            dict(losses=losses, events=events, layers=args.stage_layers, completed_steps=4)))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
