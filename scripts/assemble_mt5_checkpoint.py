#!/usr/bin/env python3
"""Assemble matching stage checkpoints into a Hugging Face model directory."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def assemble(encoder_dir=None, decoder_dir=None, output_dir=None, head_dir=None, rank_dirs=None):
    directories = [Path(root) for root in rank_dirs] if rank_dirs else [Path(root) for root in
                   (encoder_dir, decoder_dir, head_dir) if root is not None]
    if not directories:
        raise ValueError("Provide --rank-dirs, one directory per pipeline stage from one replica")
    checkpoints = [torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True) for root in directories]
    encoder = checkpoints[0]
    expected_size = encoder.get("pipeline_size", encoder.get("world_size", 2))
    replica = encoder["rank"] // expected_size
    for rank, checkpoint in enumerate(checkpoints):
        if checkpoint["rank"] != replica * expected_size + rank:
            raise ValueError("Expected checkpoint directories in pipeline order from the same replica")
        if checkpoint.get("pipeline_size", checkpoint.get("world_size", 2)) != len(directories):
            raise ValueError("Provide all pipeline stage directories using --rank-dirs (or --head-dir for legacy three-rank runs)")
        for key in ("step", "model", "smoke", "run_id", "world_size", "pipeline_size", "boundaries", "partition_version"):
            if encoder.get(key) != checkpoint.get(key):
                raise ValueError("Checkpoints differ in " + key)
    configs = [AutoConfig.from_pretrained(root) for root in directories]
    # Loading from local directories changes _name_or_path, which is not architectural.
    signatures = [{k: v for k, v in config.to_dict().items() if k != "_name_or_path"} for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Checkpoints have different model configurations")
    state = dict(encoder["pretrained"])
    for checkpoint in checkpoints[1:]:
        if state.keys() & checkpoint["pretrained"].keys():
            raise ValueError("Checkpoint stage weights overlap")
        state.update(checkpoint["pretrained"])
    factory = AutoModelForSeq2SeqLM if configs[0].is_encoder_decoder else AutoModelForCausalLM
    model = factory.from_config(configs[0])
    model.load_state_dict(state, strict=True)
    model.save_pretrained(output_dir)
    if not encoder["smoke"]:
        AutoTokenizer.from_pretrained(directories[0], use_fast=configs[0].model_type != "mt5").save_pretrained(output_dir)
    print("Saved merged model to", output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dirs", nargs="+", help="one directory per pipeline stage, in order, from one replica")
    parser.add_argument("--encoder-dir", help="legacy encoder checkpoint directory")
    parser.add_argument("--decoder-dir", help="legacy decoder checkpoint directory")
    parser.add_argument("--head-dir", help="rank2 checkpoint directory for three-rank runs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    assemble(args.encoder_dir, args.decoder_dir, args.output_dir, args.head_dir, args.rank_dirs)


if __name__ == "__main__":
    main()
