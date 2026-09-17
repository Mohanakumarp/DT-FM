"""Measure local FP32 SeqClassification component times for the pipeline scheduler."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheduler.pipeline_scheduler import MODEL_FIELDS, validate_compute_profile


def benchmark(module, inputs, device, warmup, repeats, classification=False):
    import torch
    from pipeline_parallel.dist_gpipe_pipeline_async import _synchronize_device
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    samples = []
    for iteration in range(warmup + repeats):
        for item in inputs:
            item.grad = None
        optimizer.zero_grad(set_to_none=True)
        _synchronize_device(device)
        start = time.perf_counter()
        outputs = [module(item) for item in inputs]
        for output in outputs:
            if classification:
                torch.nn.functional.cross_entropy(
                    output, torch.zeros(output.shape[0], dtype=torch.long, device=device)).backward()
            else:
                output.backward(torch.ones_like(output))
        optimizer.step()
        _synchronize_device(device)
        elapsed = (time.perf_counter() - start) * 1000
        if iteration >= warmup:
            samples.append(elapsed)
    return statistics.median(samples)


def profile_rank(args, device=None, stage=None, size=None):
    import torch
    from modules.gpt_modules import GPTEmbedding, GPTTransformerLayer
    from modules.task_modules import SeqClassification
    from utils.device_utils import resolve_device, describe_device
    model = {key: getattr(args, key) for key in MODEL_FIELDS}
    rank = dict(measured_rank=args.measured_rank, device=args.device,
                max_layers=args.max_layers, layer_ms=1., first_stage_ms=0., last_stage_ms=0.)
    # Validate before allocating tensors. Reindex only for this single-rank check.
    validate_compute_profile(dict(schema_version=1, model=model, ranks=[dict(rank, measured_rank=0)]), 1)
    if args.measured_rank < 0 or args.warmup < 0 or args.repeats < 1:
        raise ValueError('rank/warmup must be nonnegative and repeats must be positive')
    torch.manual_seed(1)
    if device is None:
        device = resolve_device(args)
    rank['device'] = args.device_backend
    microbatches = args.batch_size // args.micro_batch_size
    shape = (args.micro_batch_size, args.seq_length, args.embedding_dim)
    inputs = [torch.randn(shape, device=device, requires_grad=True) for _ in range(microbatches)]
    layer = GPTTransformerLayer(args.embedding_dim, args.num_heads, args.embedding_dim * 4).to(device)
    rank['layer_ms'] = benchmark(layer, inputs, device, args.warmup, args.repeats)
    del layer
    if stage is None or stage == size - 1:
        head = SeqClassification(args.embedding_dim, 2).to(device)
        rank['last_stage_ms'] = benchmark(head, inputs, device, args.warmup, args.repeats, classification=True)
        del head
    del inputs
    if stage is None or stage == 0:
        embedding = GPTEmbedding(args.vocab_size, args.embedding_dim, args.seq_length).to(device)
        tokens = [torch.randint(args.vocab_size, (args.micro_batch_size, args.seq_length), device=device)
                  for _ in range(microbatches)]
        rank['first_stage_ms'] = benchmark(embedding, tokens, device, args.warmup, args.repeats)
    validate_compute_profile(dict(schema_version=1, model=model, ranks=[dict(rank, measured_rank=0)]), 1)
    return dict(schema_version=1, model=model, rank=rank,
                measurement=dict(device=describe_device(device, args.device_backend, getattr(args, 'directml_name', None)),
                                 warmup=args.warmup, repeats=args.repeats, statistic='median',
                                 scope='isolated FP32 components, full-batch forward/backward/SGD; no network',
                                 capacity='max_layers supplied by operator, not measured'))


def main():
    from utils.dist_args_utils import add_device_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_arguments(parser)
    parser.set_defaults(device='cpu')
    parser.add_argument('--measured-rank', type=int, required=True)
    parser.add_argument('--max-layers', type=int, required=True,
                        help='safe layer limit with room reserved for endpoints and GPipe activations')
    for key, default in zip(MODEL_FIELDS, (32, 64, 4, 2, 1, 512)):
        parser.add_argument('--' + key.replace('_', '-'), type=int, default=default)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = profile_rank(args)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
