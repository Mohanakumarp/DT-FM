#!/usr/bin/env python3
"""Assemble matching pipeline checkpoints into a Hugging Face mT5 directory."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, MT5Config, MT5ForConditionalGeneration


def assemble(encoder_dir, decoder_dir, output_dir):
    directories = [Path(encoder_dir), Path(decoder_dir)]
    checkpoints = [torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True) for root in directories]
    encoder, decoder = checkpoints
    if encoder["rank"] != 0 or decoder["rank"] != 1:
        raise ValueError("Expected rank0 encoder and rank1 decoder checkpoints")
    for key in ("step", "model", "smoke", "run_id"):
        if encoder[key] != decoder[key]:
            raise ValueError("Checkpoints differ in " + key)
    configs = [MT5Config.from_pretrained(root) for root in directories]
    # Loading from local directories changes _name_or_path, which is not architectural.
    signatures = [{k: v for k, v in config.to_dict().items() if k != "_name_or_path"} for config in configs]
    if signatures[0] != signatures[1]:
        raise ValueError("Checkpoints have different model configurations")
    state = dict(encoder["pretrained"])
    if state.keys() & decoder["pretrained"].keys():
        raise ValueError("Checkpoint stage weights overlap")
    state.update(decoder["pretrained"])
    model = MT5ForConditionalGeneration(configs[0])
    model.load_state_dict(state, strict=True)
    model.save_pretrained(output_dir)
    if not encoder["smoke"]:
        AutoTokenizer.from_pretrained(encoder_dir, use_fast=False).save_pretrained(output_dir)
    print("Saved merged model to", output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", required=True)
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    assemble(args.encoder_dir, args.decoder_dir, args.output_dir)


if __name__ == "__main__":
    main()
