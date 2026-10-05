#!/usr/bin/env python3
"""Synchronous two-host mT5 pipeline, with CPU-staged Gloo transport."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import (
    Adafactor, AutoTokenizer, DataCollatorForSeq2Seq, MT5Config,
    MT5ForConditionalGeneration,
)

from modules.mt5_stages import MT5EncoderStage, MT5DecoderStage
from task_datasets.xlsum import load_tamil_xlsum, tokenize_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--rank", type=int, choices=[0, 1], required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", default="logs/mt5-tamil")
    parser.add_argument("--resume-from", help="local rank checkpoint directory on each host")
    parser.add_argument("--smoke", action="store_true", help="tiny random mT5 and synthetic tokens; no downloads")
    args = parser.parse_args()
    for name in ("batch_size", "micro_batch_size", "source_length", "target_length", "steps", "timeout_seconds"):
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.validation_batches < 0 or args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error("validation-batches must be nonnegative; lr and max-grad-norm must be positive")
    return args


def send(tensor, dst):
    dist.send(tensor.detach().to("cpu").contiguous(), dst=dst)


def receive(shape, dtype, device, src, requires_grad=False):
    tensor = torch.empty(shape, dtype=dtype)
    dist.recv(tensor, src=src)
    return tensor.to(device).requires_grad_(requires_grad)


def clip_gradients(stage, max_norm):
    # Norm covers both stages, exactly once per unique parameter.
    norm_squared = torch.zeros((), dtype=torch.float64)
    for p in stage.parameters():
        if p.grad is not None:
            norm_squared += p.grad.detach().double().square().sum().cpu()
    dist.all_reduce(norm_squared)
    coefficient = min(1.0, max_norm / (norm_squared.sqrt().item() + 1e-6))
    for p in stage.parameters():
        if p.grad is not None:
            p.grad.mul_(coefficient)


def pipeline_batch(stage, batch, micro_batch_size, device, training, rank):
    header = torch.zeros(4, dtype=torch.long)
    if rank == 0:
        header[:] = torch.tensor([
            len(batch["input_ids"]), batch["input_ids"].shape[1],
            batch["labels"].shape[1], (batch["labels"] != -100).sum().item(),
        ])
    dist.broadcast(header, src=0)
    size, source_length, target_length, tokens = header.tolist()
    if tokens == 0:
        raise ValueError("batch contains no target tokens")
    loss_sum = torch.zeros((), dtype=torch.float64)
    with torch.set_grad_enabled(training):
        for start in range(0, size, micro_batch_size):
            count = min(micro_batch_size, size - start)
            if rank == 0:
                part = {k: v[start:start + count].to(device) for k, v in batch.items()}
                encoded, decoder_embeds = stage(**part)
                for tensor in (part["attention_mask"], part["labels"], encoded, decoder_embeds):
                    send(tensor, 1)
                if training:
                    gradients = [receive(t.shape, t.dtype, device, 1) for t in (encoded, decoder_embeds)]
                    torch.autograd.backward((encoded, decoder_embeds), gradients)
            else:
                mask = receive((count, source_length), torch.long, device, 0)
                labels = receive((count, target_length), torch.long, device, 0)
                encoded = receive((count, source_length, stage.config.d_model), torch.float32, device, 0, training)
                decoder_embeds = receive((count, target_length, stage.config.d_model), torch.float32, device, 0, training)
                loss = stage(encoded, decoder_embeds, mask, labels)
                loss_sum += loss.detach().double().cpu()
                if training:
                    (loss / tokens).backward()
                    send(encoded.grad, 0)
                    send(decoder_embeds.grad, 0)
    dist.broadcast(loss_sum, src=1)
    return loss_sum.item(), tokens


def make_loaders(args, model):
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed)
        batch = {
            "input_ids": torch.randint(2, model.config.vocab_size, (args.batch_size, 8), generator=generator),
            "attention_mask": torch.ones(args.batch_size, 8, dtype=torch.long),
            "labels": torch.randint(2, model.config.vocab_size, (args.batch_size, 5), generator=generator),
        }
        batch["labels"][:, -1] = model.config.eos_token_id
        return [batch], [batch], None
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    raw = load_tamil_xlsum()
    processed = {}
    for split in ("train", "validation"):
        processed[split] = raw[split].map(
            lambda batch: tokenize_batch(batch, tokenizer, args.source_length, args.target_length),
            batched=True, remove_columns=raw[split].column_names,
        )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
    train = torch.utils.data.DataLoader(
        processed["train"], batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), collate_fn=collator,
    )
    validation = torch.utils.data.DataLoader(
        processed["validation"], batch_size=args.batch_size, collate_fn=collator,
    )
    return train, validation, tokenizer


def save_checkpoint(args, stage, optimizer, step, config, tokenizer):
    root = Path(args.output_dir) / ("rank%d" % args.rank)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "checkpoint.tmp"
    torch.save({
        "rank": args.rank, "step": step, "model": args.model, "smoke": args.smoke,
        "run_id": args.run_id,
        "stage": stage.state_dict(), "optimizer": optimizer.state_dict(),
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
            num_heads=2, num_layers=1, num_decoder_layers=1,
            dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        ))
    else:
        # Each host loads on CPU first, then discards the other stage.
        model = MT5ForConditionalGeneration.from_pretrained(args.model)
    config = model.config
    if config.tie_word_embeddings:
        raise ValueError("Use mT5 with untied output weights")
    train = validation = tokenizer = None
    if args.rank == 0:
        train, validation, tokenizer = make_loaders(args, model)
    stage = (MT5EncoderStage(model) if args.rank == 0 else MT5DecoderStage(model)).to(device)
    del model
    optimizer = Adafactor(stage.parameters(), lr=args.lr, relative_step=False, scale_parameter=False, warmup_init=False)
    start_step = 0
    resume_run_id = None
    if args.resume_from:
        checkpoint = torch.load(Path(args.resume_from) / "checkpoint.pt", map_location="cpu", weights_only=True)
        if (checkpoint["rank"], checkpoint["model"], checkpoint["smoke"]) != (args.rank, args.model, args.smoke):
            raise ValueError("checkpoint rank/model/smoke mode differs from this run")
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
        "resume_run_id": resume_run_id,
    }
    signatures = [None, None]
    dist.all_gather_object(signatures, signature)
    if signatures[0] != signatures[1]:
        raise ValueError("ranks disagree on model, batch settings, steps, or checkpoint step")
    if start_step >= args.steps:
        raise ValueError("--steps must exceed the checkpoint step")
    run_ids = [str(uuid.uuid4()) if args.rank == 0 else None]
    dist.broadcast_object_list(run_ids, src=0)
    args.run_id = run_ids[0]
    print(json.dumps({"rank": args.rank, "device": str(device), "parameters": sum(p.numel() for p in stage.parameters())}), flush=True)
    root = Path(args.output_dir) / ("rank%d" % args.rank)
    root.mkdir(parents=True, exist_ok=True)
    iterator = iter(train) if args.rank == 0 else None
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        for step in range(start_step, args.steps):
            stage.train()
            optimizer.zero_grad(set_to_none=True)
            batch = None
            if args.rank == 0:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train)
                    batch = next(iterator)
            started = time.monotonic()
            loss_sum, tokens = pipeline_batch(stage, batch, args.micro_batch_size, device, True, args.rank)
            clip_gradients(stage, args.max_grad_norm)
            optimizer.step()
            record = {"step": step + 1, "train_loss": loss_sum / tokens, "seconds": time.monotonic() - started}
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
        stage.eval()
        count = torch.tensor(min(args.validation_batches, len(validation)) if args.rank == 0 else 0)
        dist.broadcast(count, src=0)
        iterator = iter(validation) if args.rank == 0 else None
        total_loss = total_tokens = 0
        for _ in range(count.item()):
            batch = next(iterator) if args.rank == 0 else None
            loss_sum, tokens = pipeline_batch(stage, batch, args.micro_batch_size, device, False, args.rank)
            total_loss += loss_sum
            total_tokens += tokens
        if total_tokens:
            record = {"step": args.steps, "validation_loss": total_loss / total_tokens, "validation_batches": count.item()}
            metrics.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    save_checkpoint(args, stage, optimizer, args.steps, config, tokenizer)
    dist.barrier()


def main():
    args = parse_args()
    dist.init_process_group("gloo", init_method=args.dist_url, rank=args.rank, world_size=2, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
