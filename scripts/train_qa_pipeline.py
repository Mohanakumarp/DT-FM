#!/usr/bin/env python3
"""BERT extractive QA with GPipe microbatches and stage-wise data replicas."""
import argparse
from datetime import timedelta
import json
import hashlib
import math
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering

from modules.bert_qa_partition import BertQAPartition, qa_layout, span_loss
from task_datasets.question_answering import tokenize_qa, score_predictions
from scripts.train_mt5_pipeline import send, receive, clip_gradients
from utils.training_progress import TrainingProgress
from utils.gradient_compression import SparseGradientSynchronizer
from utils.qa_checkpoints import checkpoint_candidates, collect_errors, load_common_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="prajjwal1/bert-mini")
    parser.add_argument("--dataset", default="rajpurkar/squad")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--pipeline-size", type=int, default=1)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:29500")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=24, help="features per data replica, NOT per pipeline stage")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=8000)
    parser.add_argument("--validation-examples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--doc-stride", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="optional optimizer-step cap; 0 runs all epochs")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-compression", choices=["none", "topk"], default="none",
                        help="Top-K with error feedback for stage-wise data-replica gradients; retains AdamW")
    parser.add_argument("--compression-keep-ratio", type=float, default=0.01)
    parser.add_argument("--compression-warmup-steps", type=int, default=20)
    parser.add_argument("--compression-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--compression-bucket-size", type=int, default=4_194_304)
    parser.add_argument("--warmup-steps", type=int, default=10, help="steps excluded from throughput timing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--log-interval", type=float, default=15, help="seconds between progress heartbeats")
    parser.add_argument("--output-dir", default="logs/qa-squad")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="save every N optimizer steps; 0 saves only before validation")
    parser.add_argument("--resume", action="store_true", help="resume the newest checkpoint step available on every rank in output-dir")
    parser.add_argument("--smoke", action="store_true", help="tiny random BERT and synthetic spans; no downloads")
    parser.add_argument("--prepare-only", action="store_true", help="cache model, tokenizer and tokenized SQuAD before joining ranks")
    args = parser.parse_args()
    for name in ("world_size", "pipeline_size", "batch_size", "micro_batch_size", "train_examples",
                 "validation_examples", "max_length", "epochs", "timeout_seconds"):
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if not 0 <= args.rank < args.world_size or args.world_size % args.pipeline_size:
        parser.error("rank must be in [0, world-size); pipeline-size must divide world-size")
    if not 0 <= args.doc_stride < args.max_length or args.max_steps < 0 or args.warmup_steps < 0:
        parser.error("invalid doc-stride, max-steps, or warmup-steps")
    if any(not math.isfinite(value) or value <= 0 for value in (args.lr, args.max_grad_norm)):
        parser.error("lr and max-grad-norm must be finite and positive")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        parser.error("log-interval must be finite and positive")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every must be nonnegative")
    args.replicas = args.world_size // args.pipeline_size
    if not 0 < args.compression_keep_ratio <= args.compression_warmup_ratio <= 1:
        parser.error("require 0 < compression-keep-ratio <= compression-warmup-ratio <= 1")
    if args.compression_warmup_steps < 0 or not 1 <= args.compression_bucket_size < 2**31:
        parser.error("compression-warmup-steps must be nonnegative and bucket-size must be in [1, 2^31)")
    if args.gradient_compression != "none" and args.replicas < 2:
        parser.error("gradient compression requires at least two data replicas")
    args.pipeline_rank = args.rank % args.pipeline_size
    args.replica_rank = args.rank // args.pipeline_size
    args.pipeline_leader = args.replica_rank * args.pipeline_size
    return args


def make_groups(args):
    args.pipeline_group = args.data_group = None
    timeout = timedelta(seconds=args.timeout_seconds)
    # Identical collective creation order on every rank.
    for replica in range(args.replicas):
        ranks = list(range(replica * args.pipeline_size, (replica + 1) * args.pipeline_size))
        group = dist.new_group(ranks, timeout=timeout)
        if args.rank in ranks:
            args.pipeline_group = group
    for stage in range(args.pipeline_size):
        ranks = list(range(stage, args.world_size, args.pipeline_size))
        group = dist.new_group(ranks, timeout=timeout)
        if args.rank in ranks:
            args.data_group = group


def load_model(args):
    torch.manual_seed(args.seed)
    if args.smoke:
        config = BertConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=4, num_attention_heads=2,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        config._attn_implementation = "eager"
        return BertForQuestionAnswering(config)
    return BertForQuestionAnswering.from_pretrained(
        args.model, revision=getattr(args, "model_revision", "main"), attn_implementation="eager", torch_dtype=torch.float32,
    )


def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=getattr(args, "model_revision", "main"), use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("QA requires a fast tokenizer with offset mappings")
    dataset = load_dataset(args.dataset, revision=getattr(args, "dataset_revision", "main"))
    examples, features = {}, {}
    for split, limit in (("train", args.train_examples), ("validation", args.validation_examples)):
        raw = dataset[split].shuffle(seed=args.seed).select(range(min(limit, len(dataset[split]))))
        examples[split] = raw
        features[split] = raw.map(
            tokenize_qa, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length, "stride": args.doc_stride},
            batched=True, remove_columns=raw.column_names,
        )
    return tokenizer, examples, features


def synthetic_batch(args, config):
    generator = torch.Generator().manual_seed(args.seed + args.replica_rank)
    return {
        "input_ids": torch.randint(2, config.vocab_size, (args.batch_size, 8), generator=generator),
        "attention_mask": torch.ones(args.batch_size, 8, dtype=torch.long),
        "token_type_ids": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]] * args.batch_size),
        "start_positions": torch.full((args.batch_size,), 4, dtype=torch.long),
        "end_positions": torch.full((args.batch_size,), 5, dtype=torch.long),
    }


def pipeline_batch(stage, batch, args, device, training=True, progress=None, step=None):
    """GPipe: fill forward microbatches, then drain backward in reverse order."""
    header = torch.zeros(2, dtype=torch.long)
    if progress:
        progress.phase("batch_metadata", step=step, training=training)
    if args.pipeline_rank == 0:
        header[:] = torch.tensor(batch["input_ids"].shape)
    dist.broadcast(header, src=args.pipeline_leader, group=args.pipeline_group)
    size, length = header.tolist()
    values = {}
    for key, shape in (("attention_mask", (size, length)), ("token_type_ids", (size, length)),
                       ("start_positions", (size,)), ("end_positions", (size,))):
        tensor = batch[key].cpu().contiguous() if args.pipeline_rank == 0 else torch.empty(shape, dtype=torch.long)
        dist.broadcast(tensor, src=args.pipeline_leader, group=args.pipeline_group)
        values[key] = tensor.to(device)
    global_size = torch.tensor(size, dtype=torch.long)
    if training and args.replicas > 1:
        dist.all_reduce(global_size, group=args.data_group)
    last = args.pipeline_rank == args.pipeline_size - 1
    pending, predictions = [], []
    loss_sum = torch.zeros((), dtype=torch.float64)
    with torch.set_grad_enabled(training):
        for start in range(0, size, args.micro_batch_size):
            stop = min(start + args.micro_batch_size, size)
            micro = start // args.micro_batch_size + 1
            if progress:
                progress.phase("receiving_forward" if args.pipeline_rank else "forward",
                               step=step, micro_batch=micro, training=training)
            hidden = None if args.pipeline_rank == 0 else receive(
                (stop - start, length, stage.config.hidden_size), torch.float32,
                device, args.rank - 1, training,
            )
            if progress and args.pipeline_rank:
                progress.phase("forward", step=step, micro_batch=micro, training=training)
            output = stage(hidden, values["attention_mask"][start:stop], values["token_type_ids"][start:stop],
                           input_ids=batch["input_ids"][start:stop].to(device) if args.pipeline_rank == 0 else None)
            if last:
                loss = span_loss(output, values["start_positions"][start:stop], values["end_positions"][start:stop])
                loss_sum += loss.detach().double().cpu()
                if not training:
                    predictions.append(output.detach().cpu())
                if training:
                    pending.append((hidden, loss / global_size.item()))
            else:
                if progress:
                    progress.phase("sending_forward", step=step, micro_batch=micro, peer=args.rank + 1)
                send(output, args.rank + 1)
                if training:
                    pending.append((hidden, output))
        if training:
            for index in range(len(pending) - 1, -1, -1):
                hidden, output = pending[index]
                if progress:
                    progress.phase("backward" if last else "receiving_backward", step=step, micro_batch=index + 1)
                if last:
                    output.backward()
                else:
                    gradient = receive(output.shape, output.dtype, device, args.rank + 1)
                    if progress:
                        progress.phase("backward", step=step, micro_batch=index + 1)
                    output.backward(gradient)
                if hidden is not None:
                    if progress:
                        progress.phase("sending_backward", step=step, micro_batch=index + 1, peer=args.rank - 1)
                    send(hidden.grad, args.rank - 1)
    if progress:
        progress.phase("reporting_loss" if training else "collecting_validation_logits", step=step)
    if last and training and args.replicas > 1:
        dist.all_reduce(loss_sum, group=args.data_group)
    dist.broadcast(loss_sum, src=args.pipeline_leader + args.pipeline_size - 1, group=args.pipeline_group)
    logits = None
    if not training:
        logits = torch.cat(predictions) if last else torch.empty(size, length, 2)
        dist.broadcast(logits, src=args.pipeline_leader + args.pipeline_size - 1, group=args.pipeline_group)
    return loss_sum.item() / global_size.item(), global_size.item(), logits


def synchronize_gradients(stage, args, compressor=None):
    if args.replicas == 1:
        return {}
    if compressor is not None:
        return compressor.synchronize(args.data_group)
    parameters = list(stage.parameters())
    packed = torch.cat([p.grad.detach().reshape(-1).cpu() if p.grad is not None else
                        torch.zeros(p.numel()) for p in parameters])
    dist.all_reduce(packed, group=args.data_group)  # losses already use global feature count
    offset = 0
    for parameter in parameters:
        gradient = packed[offset:offset + parameter.numel()].view_as(parameter).to(parameter.device)
        if parameter.grad is None:
            parameter.grad = gradient.clone()
        else:
            parameter.grad.copy_(gradient)
        offset += parameter.numel()
    dense_bytes = packed.numel() * packed.element_size()
    return {"gradient_dense_bytes": dense_bytes, "gradient_payload_bytes": dense_bytes,
            "gradient_payload_compression": 1.0}


def evaluate(stage, args, device, features, examples, progress=None):
    stage.eval()
    predictions = []
    indices = list(range(args.replica_rank, len(features["validation"]), args.replicas)) if args.pipeline_rank == 0 else []
    batches = torch.tensor(math.ceil(len(indices) / args.batch_size), dtype=torch.long)
    dist.broadcast(batches, src=args.pipeline_leader, group=args.pipeline_group)
    for index in range(batches.item()):
        selected = indices[index * args.batch_size:(index + 1) * args.batch_size]
        batch = None
        if args.pipeline_rank == 0:
            rows = features["validation"].select(selected)
            batch = {key: torch.tensor(rows[key], dtype=torch.long) for key in
                     ("input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions")}
        _, _, logits = pipeline_batch(stage, batch, args, device, training=False, progress=progress)
        if args.pipeline_rank == 0:
            predictions.extend((feature, logits[i, :, 0].tolist(), logits[i, :, 1].tolist())
                               for i, feature in enumerate(selected))
    collected = [None] * args.world_size if args.rank == 0 else None
    if progress:
        progress.phase("gathering_validation_predictions")
    dist.gather_object(predictions, collected, dst=0)
    result = [{}]
    if args.rank == 0:
        try:
            metrics, answers = score_predictions(examples["validation"], features["validation"],
                                                 [item for peer in collected for item in peer])
            result[0] = metrics
            root = Path(args.output_dir) / "rank0"
            (root / "predictions.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2))
        except Exception as error:
            result[0] = {"error": str(error)}
    dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise ValueError("Validation scoring failed: " + result[0]["error"])
    return result[0]


def run(args, progress=None):
    def phase(name, **details):
        if progress:
            progress.phase(name, **details)

    phase("loading_model")
    print(json.dumps({"event": "loading_model", "rank": args.rank, "model": args.model}), flush=True)
    model = load_model(args)  # CPU load before the rendezvous, including on CUDA hosts.
    if args.prepare_only:
        if not args.smoke:
            phase("preparing_dataset")
            _, examples, features = load_data(args)
            print(json.dumps({"event": "prepared", "train_examples": len(examples["train"]),
                              "train_features": len(features["train"]),
                              "validation_examples": len(examples["validation"])}), flush=True)
        phase("prepared")
        return
    boundaries = qa_layout(model.config, args.world_size, args.pipeline_size)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length exceeds the model's position embeddings")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    phase("waiting_for_all_ranks", world_size=args.world_size, coordinator=args.dist_url)
    dist.init_process_group("gloo", init_method=args.dist_url, rank=args.rank, world_size=args.world_size,
                            timeout=timedelta(seconds=args.timeout_seconds))
    try:
        phase("creating_pipeline_and_data_groups")
        make_groups(args)
        signature = {key: value for key, value in vars(args).items() if key not in
                     ("rank", "device", "output_dir", "pipeline_rank", "replica_rank", "pipeline_leader",
                      "pipeline_group", "data_group")}
        signature["config"] = model.config.to_dict()
        signatures = [None] * args.world_size
        dist.all_gather_object(signatures, signature)
        if any(value != signatures[0] for value in signatures):
            raise ValueError("Ranks disagree on model, topology or training settings")
        phase("placing_model_partition", device=str(device))
        stage = BertQAPartition(model, args.pipeline_rank, boundaries).to(device)
        config = model.config
        del model
        # Avoid CUDA foreach temporaries proportional to the entire partition
        # on the 6 GB laptop GPUs used by the larger-model experiment.
        optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr, foreach=False)
        compressor = None
        if args.gradient_compression == "topk":
            phase("allocating_gradient_compression_buffers")
            compressor = SparseGradientSynchronizer(
                stage.named_parameters(), mode="topk", keep_ratio=args.compression_keep_ratio,
                warmup_steps=args.compression_warmup_steps, warmup_ratio=args.compression_warmup_ratio,
                bucket_size=args.compression_bucket_size,
            )
        tokenizer = examples = features = train = sampler = None
        data_error = None
        if args.pipeline_rank == 0:
            try:
                phase("preparing_dataset", replica=args.replica_rank)
                if args.smoke:
                    train = [synthetic_batch(args, config)]
                else:
                    tokenizer, examples, features = load_data(args)
                    columns = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
                    training = features["train"].with_format("torch", columns=columns)
                    sampler = torch.utils.data.DistributedSampler(training, num_replicas=args.replicas,
                                                                 rank=args.replica_rank, seed=args.seed)
                    train = torch.utils.data.DataLoader(
                        training, batch_size=args.batch_size, sampler=sampler,
                        generator=torch.Generator().manual_seed(args.seed),
                    )
            except Exception as error:
                data_error = str(error)
        phase("waiting_for_replica_dataset_preparation")
        errors = [None] * args.world_size
        dist.all_gather_object(errors, data_error)
        if any(errors):
            raise ValueError("Dataset preparation failed: " + str(errors))
        counts = torch.tensor([len(train), len(features["train"]) if features else args.batch_size * args.replicas]
                              if args.pipeline_rank == 0 else [0, 0], dtype=torch.long)
        dist.broadcast(counts, src=args.pipeline_leader, group=args.pipeline_group)
        lengths = [None] * args.world_size
        dist.all_gather_object(lengths, counts.tolist())
        if any(value != lengths[0] for value in lengths):
            raise ValueError("Replicas have different training feature counts")
        run_id = [str(uuid.uuid4()) if args.rank == 0 else None]
        dist.broadcast_object_list(run_id, src=0)
        root = Path(args.output_dir) / ("rank%d" % args.rank)
        root.mkdir(parents=True, exist_ok=True)
        identity = {key: value for key, value in signature.items() if key not in
                    ("resume", "epochs", "max_steps", "checkpoint_every", "warmup_steps",
                     "dist_url", "timeout_seconds", "log_interval", "prepare_only")}
        # Default revisions retain compatibility with checkpoints from the older CLI.
        # Explicit pins still participate in identity validation on resume.
        for revision in ("model_revision", "dataset_revision"):
            if identity.get(revision) == "main":
                identity.pop(revision)
        identity["training_batches"], identity["training_features"] = counts.tolist()
        data_hash = [hashlib.sha256(json.dumps(examples["train"].to_dict(), sort_keys=True).encode()).hexdigest()
                     if examples is not None else None]
        dist.broadcast_object_list(data_hash, src=args.pipeline_leader, group=args.pipeline_group)
        identity["training_data_sha256"] = data_hash[0]
        identities = [None] * args.world_size
        dist.all_gather_object(identities, identity)
        if any(value != identities[0] for value in identities):
            raise ValueError("Replicas have different training data")
        error = None
        try:
            if not args.resume and (checkpoint_candidates(root) or (root / "checkpoint.pt").exists()):
                raise ValueError("Existing checkpoints in output-dir; use --resume or a new output directory")
            config.save_pretrained(root)
            if tokenizer is not None:
                tokenizer.save_pretrained(root)
        except Exception as exception:
            error = str(exception)
        collect_errors(error)
        print(json.dumps({"event": "ready", "rank": args.rank, "replica": args.replica_rank,
                          "pipeline_stage": args.pipeline_rank, "units": [stage.start, stage.end],
                          "parameters": sum(p.numel() for p in stage.parameters()), "device": str(device),
                          "pipeline_size": args.pipeline_size, "data_replicas": args.replicas,
                          "gradient_compression": args.gradient_compression,
                          "train_features": counts[1].item(), "global_batch_size": args.batch_size * args.replicas}), flush=True)
        torch.manual_seed(args.seed + args.rank)  # replica-specific dropout, identical initial weights
        step = 0
        if args.resume:
            phase("loading_checkpoint")
            checkpoint = load_common_checkpoint(root, args, identity, stage, optimizer, compressor)
            step = checkpoint["step"]
            run_id[0] = checkpoint["run_id"]
            print(json.dumps({"event": "resumed", "rank": args.rank, "step": step}), flush=True)
        resumed_from = step
        measured_examples = measured_steps = 0
        measured_seconds = checkpoint_seconds = 0.0
        start_time = time.monotonic()
        if args.resume and (root / "metrics.jsonl").exists():
            retained = []
            for line in (root / "metrics.jsonl").read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "train_loss" in record and record["step"] <= step:
                    retained.append(line)
            (root / "metrics.jsonl").write_text("".join(line + "\n" for line in retained))
        saved_step = step if args.resume else None
        with (root / "metrics.jsonl").open("a" if args.resume else "w") as metrics:
            for epoch in range(step // counts[0].item(), args.epochs):
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(train) if args.pipeline_rank == 0 else None
                skip = step % counts[0].item() if epoch == resumed_from // counts[0].item() else 0
                for _ in range(skip):
                    if iterator is not None:
                        next(iterator)
                for _ in range(skip, counts[0].item()):
                    if args.max_steps and step >= args.max_steps:
                        break
                    stage.train()
                    phase("loading_batch", step=step + 1, epoch=epoch + 1)
                    started = time.monotonic()
                    batch = next(iterator) if iterator is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    loss, count, _ = pipeline_batch(stage, batch, args, device, progress=progress, step=step + 1)
                    phase("synchronizing_data_replica_gradients", step=step + 1,
                          gradient_compression=args.gradient_compression,
                          gradient_keep_ratio=compressor.current_keep_ratio() if compressor else 1.0)
                    sync_started = time.monotonic()
                    communication = synchronize_gradients(stage, args, compressor)
                    sync_seconds = time.monotonic() - sync_started
                    phase("clipping_gradients", step=step + 1)
                    clip_gradients(stage, args.max_grad_norm, args)
                    phase("optimizer", step=step + 1)
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    seconds = time.monotonic() - started
                    step += 1
                    if step > args.warmup_steps:
                        measured_seconds += seconds
                        measured_examples += count
                        measured_steps += 1
                    record = {"step": step, "epoch": epoch + 1, "train_loss": loss, "seconds": seconds,
                              "gradient_sync_seconds": sync_seconds, "global_features": count, **communication}
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print(json.dumps({"rank": args.rank, **record}), flush=True)
                    if args.checkpoint_every and step % args.checkpoint_every == 0:
                        phase("saving_checkpoint", step=step)
                        checkpoint_started = time.monotonic()
                        save_checkpoint(root, args, stage, optimizer, compressor, step, run_id[0], identity)
                        checkpoint_seconds += time.monotonic() - checkpoint_started
                        saved_step = step
                        print(json.dumps({"event": "checkpoint_saved", "rank": args.rank, "step": step}), flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            if saved_step != step:
                phase("saving_checkpoint", step=step)
                checkpoint_started = time.monotonic()
                save_checkpoint(root, args, stage, optimizer, compressor, step, run_id[0], identity)
                checkpoint_seconds += time.monotonic() - checkpoint_started
            training_wall = torch.tensor(time.monotonic() - start_time, dtype=torch.float64)
            window = torch.tensor(measured_seconds, dtype=torch.float64)
            dist.all_reduce(training_wall, op=dist.ReduceOp.MAX)
            dist.all_reduce(window, op=dist.ReduceOp.MAX)
            phase("validation")
            evaluation = {} if args.smoke else evaluate(stage, args, device, features, examples, progress)
            summary = {"run_id": run_id[0], "rank": args.rank, "world_size": args.world_size,
                       "pipeline_size": args.pipeline_size, "data_replicas": args.replicas,
                       "gradient_compression": args.gradient_compression,
                       "compression_keep_ratio": args.compression_keep_ratio if compressor else None,
                       "completed_steps": step, "measured_steps": measured_steps,
                       "resumed_from_step": resumed_from, "session_steps": step - resumed_from,
                       "checkpoint_seconds": checkpoint_seconds,
                       "measured_features": measured_examples, "measured_seconds": window.item(),
                       "features_per_second": measured_examples / window.item() if window.item() else None,
                       "training_wall_seconds": training_wall.item(),
                       "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                       "smoke": args.smoke, **evaluation}
            (root / "summary.json").write_text(json.dumps(summary, indent=2))
            metrics.write(json.dumps(summary) + "\n")
            print(json.dumps({"event": "complete", **summary}), flush=True)
            phase("complete", completed_steps=step)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    arguments = parse_args()
    logger = TrainingProgress(arguments.rank, arguments.output_dir, arguments.log_interval, append=arguments.resume)
    try:
        run(arguments, logger)
    except Exception as error:
        logger.phase("failed", error=str(error))
        raise
    finally:
        logger.close()
