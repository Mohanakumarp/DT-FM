#!/usr/bin/env python3
"""Configurable summarisation training: mT5 pipeline stages and data replicas."""
import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import (
    Adafactor, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, MT5Config,
    MT5ForConditionalGeneration,
)

from modules.mt5_partition import MT5Partition, FullModelStage, activation_shapes, pipeline_layout
from task_datasets.summarization import load_summarization_splits, tokenize_summaries
from utils.gradient_compression import SparseGradientSynchronizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--pipeline-size", type=int, help="stages per replica; must divide world-size; default automatic")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--dataset", default="csebuetnlp/xlsum")
    parser.add_argument("--dataset-config", help="dataset configuration; XL-Sum defaults to tamil")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--source-column", default="text")
    parser.add_argument("--target-column", default="summary")
    parser.add_argument("--source-prefix", default="", help="e.g. 'summarize: ' for T5")
    parser.add_argument("--summary-separator", default="\nSummary:\n", help="article/summary separator for causal models")
    parser.add_argument("--train-file", help="local training file for --dataset json/csv/parquet")
    parser.add_argument("--validation-file", help="local validation file")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:29500")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--source-length", type=int, default=512)
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adafactor", "sgd"], default="adafactor")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum; accumulated locally for DGC")
    parser.add_argument("--gradient-compression", choices=["none", "topk", "dgc"], default="none",
                        help="data-replica gradient exchange; dgc requires --optimizer sgd")
    parser.add_argument("--compression-keep-ratio", type=float, default=0.001, help="fraction kept, e.g. 0.001 = top 0.1%%")
    parser.add_argument("--compression-warmup-steps", type=int, default=100,
                        help="steps to decay the kept fraction exponentially; 0 disables warm-up")
    parser.add_argument("--compression-warmup-ratio", type=float, default=0.25, help="initial kept fraction")
    parser.add_argument("--compression-bucket-size", type=int, default=4 * 1024 * 1024,
                        help="maximum elements per CPU Top-K bucket")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", default="logs/mt5-tamil")
    parser.add_argument("--resume-from", help="local rank checkpoint directory on each host")
    parser.add_argument("--smoke", action="store_true", help="tiny random mT5 and synthetic tokens; no downloads")
    args = parser.parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error("world-size must be positive and rank must be in [0, world-size)")
    if args.pipeline_size is not None and (args.pipeline_size < 1 or args.world_size % args.pipeline_size):
        parser.error("pipeline-size must be positive and divide world-size")
    for name in ("batch_size", "micro_batch_size", "source_length", "target_length", "steps", "timeout_seconds"):
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.validation_batches < 0 or args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error("validation-batches must be nonnegative; lr and max-grad-norm must be positive")
    if not all(math.isfinite(value) for value in (args.lr, args.max_grad_norm, args.momentum)):
        parser.error("lr, max-grad-norm, and momentum must be finite")
    if not 0 <= args.momentum < 1:
        parser.error("momentum must be in [0, 1)")
    if not 0 < args.compression_keep_ratio <= args.compression_warmup_ratio <= 1:
        parser.error("require 0 < compression-keep-ratio <= compression-warmup-ratio <= 1")
    if args.compression_warmup_steps < 0 or not 1 <= args.compression_bucket_size < 2**31:
        parser.error("compression-warmup-steps must be nonnegative and bucket-size must be in [1, 2^31)")
    if args.gradient_compression == "dgc" and args.optimizer != "sgd":
        parser.error("DGC momentum correction requires --optimizer sgd; use topk with Adafactor")
    return args


def send(tensor, dst):
    dist.send(tensor.detach().to("cpu").contiguous(), dst=dst)


def receive(shape, dtype, device, src, requires_grad=False):
    tensor = torch.empty(shape, dtype=dtype)
    dist.recv(tensor, src=src)
    return tensor.to(device).requires_grad_(requires_grad)


def create_groups(args):
    args.pipeline_rank = args.rank % args.pipeline_size
    args.replica_rank = args.rank // args.pipeline_size
    args.pipeline_leader = args.replica_rank * args.pipeline_size
    args.pipeline_group = args.data_group = None
    # Every global rank creates groups in the same order.
    if args.pipeline_size > 1:
        for replica in range(args.replicas):
            ranks = list(range(replica * args.pipeline_size, (replica + 1) * args.pipeline_size))
            group = dist.new_group(ranks)
            if args.rank in ranks:
                args.pipeline_group = group
    if args.replicas > 1:
        for stage_rank in range(args.pipeline_size):
            ranks = [stage_rank + replica * args.pipeline_size for replica in range(args.replicas)]
            group = dist.new_group(ranks)
            if args.rank in ranks:
                args.data_group = group


def synchronize_gradients(stage, args, compressor=None):
    if args.replicas == 1:
        return {}
    if compressor is not None:
        return compressor.synchronize(args.data_group)
    dense_bytes = 0
    for parameter in stage.parameters():
        gradient = parameter.grad.detach().cpu().contiguous() if parameter.grad is not None else torch.zeros_like(parameter, device="cpu")
        dist.all_reduce(gradient, group=args.data_group)
        if parameter.grad is None:
            parameter.grad = gradient.to(parameter.device)
        else:
            parameter.grad.copy_(gradient)
        dense_bytes += gradient.numel() * gradient.element_size()
    return {"gradient_dense_bytes": dense_bytes, "gradient_payload_bytes": dense_bytes,
            "gradient_payload_compression": 1.0}


def clip_gradients(stage, max_norm, args):
    # Norm covers every stage, exactly once per unique parameter.
    norm_squared = torch.zeros((), dtype=torch.float64)
    for p in stage.parameters():
        if p.grad is not None:
            norm_squared += p.grad.detach().double().square().sum().cpu()
    if args.pipeline_size > 1:
        dist.all_reduce(norm_squared, group=args.pipeline_group)
    coefficient = min(1.0, max_norm / (norm_squared.sqrt().item() + 1e-6))
    for p in stage.parameters():
        if p.grad is not None:
            p.grad.mul_(coefficient)


def pipeline_batch(stage, batch, micro_batch_size, device, training, args):
    rank = args.pipeline_rank
    leader = args.pipeline_leader
    header = torch.zeros(4, dtype=torch.long)
    if rank == 0:
        header[:] = torch.tensor([
            len(batch["input_ids"]), batch["input_ids"].shape[1],
            batch["labels"].shape[1], (batch["labels"] != -100).sum().item(),
        ])
    if args.pipeline_size > 1:
        dist.broadcast(header, src=leader, group=args.pipeline_group)
    size, source_length, target_length, tokens = header.tolist()
    if tokens == 0:
        raise ValueError("batch contains no target tokens")
    token_count = torch.tensor(tokens, dtype=torch.long)
    if args.replicas > 1:
        dist.all_reduce(token_count, group=args.data_group)
    global_tokens = token_count.item()
    mask = batch["attention_mask"].cpu().contiguous() if rank == 0 else torch.empty((size, source_length), dtype=torch.long)
    labels = batch["labels"].cpu().contiguous() if rank == 0 else torch.empty((size, target_length), dtype=torch.long)
    if args.pipeline_size > 1:
        dist.broadcast(mask, src=leader, group=args.pipeline_group)
        dist.broadcast(labels, src=leader, group=args.pipeline_group)
    loss_sum = torch.zeros((), dtype=torch.float64)
    with torch.set_grad_enabled(training):
        for start in range(0, size, micro_batch_size):
            count = min(micro_batch_size, size - start)
            inputs = {}
            if rank > 0:
                shapes = activation_shapes(stage.config, stage.start - 1, count, source_length, target_length)
                inputs = {key: receive(shape, torch.float32, device, args.rank - 1, training)
                          for key, shape in shapes.items()}
            output = stage(inputs, mask[start:start + count].to(device), labels[start:start + count].to(device),
                           input_ids=batch["input_ids"][start:start + count].to(device) if rank == 0 else None)
            if rank == args.pipeline_size - 1:
                loss_sum += output.detach().double().cpu()
                if training:
                    (output / global_tokens).backward()
            else:
                for tensor in output.values():
                    send(tensor, args.rank + 1)
                if training:
                    gradients = [receive(t.shape, t.dtype, device, args.rank + 1) for t in output.values()]
                    torch.autograd.backward(tuple(output.values()), gradients)
            if training and rank > 0:
                for tensor in inputs.values():
                    send(tensor.grad if tensor.grad is not None else torch.zeros_like(tensor), args.rank - 1)
    if args.pipeline_size > 1:
        dist.broadcast(loss_sum, src=leader + args.pipeline_size - 1, group=args.pipeline_group)
    if args.replicas > 1:
        dist.all_reduce(loss_sum, group=args.data_group)
    return loss_sum.item(), global_tokens


def make_loaders(args, model):
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed + args.replica_rank)
        batch = {
            "input_ids": torch.randint(2, model.config.vocab_size, (args.batch_size, 8), generator=generator),
            "attention_mask": torch.ones(args.batch_size, 8, dtype=torch.long),
            "labels": torch.randint(2, model.config.vocab_size, (args.batch_size, 5), generator=generator),
        }
        batch["labels"][:, -1] = model.config.eos_token_id
        return [batch], [batch], None
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.model_type != "mt5")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw_train, raw_validation = load_summarization_splits(args)
    processed = {}
    for split, raw in (("train", raw_train), ("validation", raw_validation)):
        if raw is None:
            continue
        processed[split] = raw.map(
            lambda batch: tokenize_summaries(batch, tokenizer, args),
            batched=True, remove_columns=raw.column_names,
        )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
    train = torch.utils.data.DataLoader(
        processed["train"], batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(processed["train"], num_replicas=args.replicas,
                                                     rank=args.replica_rank, seed=args.seed), collate_fn=collator,
    )
    validation = torch.utils.data.DataLoader(
        processed["validation"], batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(processed["validation"], num_replicas=args.replicas,
                                                     rank=args.replica_rank, shuffle=False), collate_fn=collator,
    ) if "validation" in processed else []
    return train, validation, tokenizer


def save_checkpoint(args, stage, optimizer, step, config, tokenizer, compressor=None):
    root = Path(args.output_dir) / ("rank%d" % args.rank)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "checkpoint.tmp"
    torch.save({
        "rank": args.rank, "step": step, "model": args.model, "smoke": args.smoke,
        "run_id": args.run_id,
        "world_size": args.world_size,
        "pipeline_size": args.pipeline_size, "replicas": args.replicas,
        "boundaries": stage.boundaries, "partition_version": 1,
        "stage": stage.state_dict(), "optimizer": optimizer.state_dict(),
        "optimizer_name": args.optimizer, "momentum": args.momentum,
        "gradient_compression": args.gradient_compression,
        "compression_max_grad_norm": args.max_grad_norm,
        "compressor": compressor.state_dict() if compressor is not None else None,
        "pretrained": stage.pretrained_state_dict(),
    }, temporary)
    temporary.replace(root / "checkpoint.pt")
    config.save_pretrained(root)
    if tokenizer is not None:
        tokenizer.save_pretrained(root)


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.smoke:
        model = MT5ForConditionalGeneration(MT5Config(
            vocab_size=64, d_model=16, d_ff=32, d_kv=8,
            num_heads=2, num_layers=8, num_decoder_layers=8,
            dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        ))
    else:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        factory = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
        # Each host loads on CPU first, then keeps its own stage.
        model = factory.from_pretrained(args.model, trust_remote_code=False, torch_dtype=torch.float32)
    config = model.config
    args.is_encoder_decoder = config.is_encoder_decoder
    args.model_type = config.model_type
    partitioned = config.model_type == "mt5" and not config.tie_word_embeddings
    if partitioned:
        args.pipeline_size, args.replicas, boundaries = pipeline_layout(config, args.world_size, args.pipeline_size)
    else:
        if args.pipeline_size not in (None, 1):
            raise ValueError("This architecture uses full-model data parallelism; set --pipeline-size 1")
        args.pipeline_size, args.replicas, boundaries = 1, args.world_size, [0, 1]
    if args.gradient_compression != "none" and args.replicas < 2:
        raise ValueError("gradient compression requires at least two data replicas; choose a smaller --pipeline-size dividing --world-size")
    layouts = [None] * args.world_size
    dist.all_gather_object(layouts, {"config": config.to_dict(), "pipeline_size": args.pipeline_size,
                                    "replicas": args.replicas, "boundaries": boundaries})
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("ranks disagree on model architecture or pipeline topology")
    create_groups(args)
    train = validation = tokenizer = None
    if args.pipeline_rank == 0:
        train, validation, tokenizer = make_loaders(args, model)
    stage = MT5Partition(model, args.pipeline_rank, boundaries) if partitioned else FullModelStage(model)
    stage = stage.to(device)
    del model
    compressor = None
    if args.gradient_compression != "none":
        compressor = SparseGradientSynchronizer(
            stage.named_parameters(), mode=args.gradient_compression, keep_ratio=args.compression_keep_ratio,
            warmup_steps=args.compression_warmup_steps, warmup_ratio=args.compression_warmup_ratio,
            momentum=args.momentum, bucket_size=args.compression_bucket_size,
        )
    if args.optimizer == "sgd":
        # DGC communicates momentum-corrected velocities; do not apply momentum twice.
        optimizer = torch.optim.SGD(stage.parameters(), lr=args.lr,
                                    momentum=0.0 if args.gradient_compression == "dgc" else args.momentum)
    else:
        optimizer = Adafactor(stage.parameters(), lr=args.lr, relative_step=False, scale_parameter=False, warmup_init=False)
    start_step = 0
    resume_run_id = None
    if args.resume_from:
        checkpoint = torch.load(Path(args.resume_from) / "checkpoint.pt", map_location="cpu", weights_only=True)
        if (checkpoint["rank"], checkpoint["model"], checkpoint["smoke"]) != (args.rank, args.model, args.smoke):
            raise ValueError("checkpoint rank/model/smoke mode differs from this run")
        if checkpoint.get("world_size", 2) != args.world_size:
            raise ValueError("checkpoint world-size differs from this run")
        if checkpoint.get("partition_version") != 1 or checkpoint.get("pipeline_size") != args.pipeline_size or checkpoint.get("boundaries") != boundaries:
            raise ValueError("checkpoint partition differs; assemble it and use the merged model for a new topology")
        if checkpoint.get("optimizer_name", "adafactor") != args.optimizer:
            raise ValueError("checkpoint optimizer differs from this run")
        if checkpoint.get("gradient_compression", "none") != args.gradient_compression:
            raise ValueError("checkpoint gradient compression mode differs from this run")
        if args.optimizer == "sgd" and checkpoint.get("momentum") != args.momentum:
            raise ValueError("checkpoint SGD momentum differs from this run")
        if compressor is not None:
            if checkpoint.get("compression_max_grad_norm") != args.max_grad_norm:
                raise ValueError("checkpoint compression clipping threshold differs from this run")
            compressor.load_state_dict(checkpoint["compressor"])
            if compressor.completed_steps != checkpoint["step"]:
                raise ValueError("checkpoint compression step differs from training step")
        stage.load_state_dict(checkpoint["stage"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        resume_run_id = checkpoint["run_id"]
    signature = {
        "model": args.model, "config": config.to_dict(), "smoke": args.smoke,
        "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
        "steps": args.steps, "start_step": start_step,
        "validation_batches": args.validation_batches, "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "optimizer": args.optimizer, "momentum": args.momentum,
        "gradient_compression": args.gradient_compression,
        "compression_keep_ratio": args.compression_keep_ratio,
        "compression_warmup_steps": args.compression_warmup_steps,
        "compression_warmup_ratio": args.compression_warmup_ratio,
        "compression_bucket_size": args.compression_bucket_size,
        "resume_run_id": resume_run_id,
        "world_size": args.world_size,
        "pipeline_size": args.pipeline_size, "boundaries": boundaries,
        "source_length": args.source_length, "target_length": args.target_length,
        "seed": args.seed,
        "dataset": args.dataset, "dataset_config": args.dataset_config,
        "train_split": args.train_split, "validation_split": args.validation_split,
        "source_column": args.source_column, "target_column": args.target_column,
        "source_prefix": args.source_prefix, "summary_separator": args.summary_separator,
        "train_file": args.train_file, "validation_file": args.validation_file,
    }
    signatures = [None] * args.world_size
    dist.all_gather_object(signatures, signature)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("ranks disagree on model, batch settings, steps, or checkpoint step")
    if start_step >= args.steps:
        raise ValueError("--steps must exceed the checkpoint step")
    run_ids = [str(uuid.uuid4()) if args.rank == 0 else None]
    dist.broadcast_object_list(run_ids, src=0)
    args.run_id = run_ids[0]
    print(json.dumps({"rank": args.rank, "pipeline_rank": args.pipeline_rank, "pipeline_size": args.pipeline_size,
                      "replica": args.replica_rank, "replicas": args.replicas, "units": [stage.start, stage.end],
                      "optimizer": args.optimizer, "gradient_compression": args.gradient_compression,
                      "device": str(device), "parameters": sum(p.numel() for p in stage.parameters())}), flush=True)
    root = Path(args.output_dir) / ("rank%d" % args.rank)
    root.mkdir(parents=True, exist_ok=True)
    iterator = iter(train) if args.pipeline_rank == 0 else None
    epoch = 0
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        for step in range(start_step, args.steps):
            stage.train()
            optimizer.zero_grad(set_to_none=True)
            batch = None
            if args.pipeline_rank == 0:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if hasattr(train, "sampler") and hasattr(train.sampler, "set_epoch"):
                        train.sampler.set_epoch(epoch)
                    iterator = iter(train)
                    batch = next(iterator)
            started = time.monotonic()
            loss_sum, tokens = pipeline_batch(stage, batch, args.micro_batch_size, device, True, args)
            sync_started = time.monotonic()
            if args.gradient_compression == "dgc":
                # Gradients already include the 1/global_tokens scaling. The DGC
                # local threshold is global_threshold / sqrt(data_replicas).
                clip_gradients(stage, args.max_grad_norm / math.sqrt(args.replicas), args)
            communication = synchronize_gradients(stage, args, compressor)
            sync_seconds = time.monotonic() - sync_started
            if args.gradient_compression != "dgc":
                clip_gradients(stage, args.max_grad_norm, args)
            optimizer.step()
            record = {"step": step + 1, "train_loss": loss_sum / tokens, "seconds": time.monotonic() - started,
                      "gradient_sync_seconds": sync_seconds, **communication}
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
        stage.eval()
        count = torch.tensor(min(args.validation_batches, len(validation)) if args.pipeline_rank == 0 else 0)
        if args.pipeline_size > 1:
            dist.broadcast(count, src=args.pipeline_leader, group=args.pipeline_group)
        iterator = iter(validation) if args.pipeline_rank == 0 else None
        total_loss = total_tokens = 0
        for _ in range(count.item()):
            batch = next(iterator) if args.pipeline_rank == 0 else None
            loss_sum, tokens = pipeline_batch(stage, batch, args.micro_batch_size, device, False, args)
            total_loss += loss_sum
            total_tokens += tokens
        if total_tokens:
            record = {"step": args.steps, "validation_loss": total_loss / total_tokens, "validation_batches": count.item()}
            metrics.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    save_checkpoint(args, stage, optimizer, args.steps, config, tokenizer, compressor)
    dist.barrier()


def main():
    args = parse_args()
    dist.init_process_group("gloo", init_method=args.dist_url, rank=args.rank, world_size=args.world_size, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
