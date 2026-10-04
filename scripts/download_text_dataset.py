#!/usr/bin/env python3
"""
Download and prepare WikiText-103 for DT-FM GPT-2 Foundation Model training.
- Compressed download: ~181 MB (bandwidth-friendly).
- Uncompressed text: ~540 MB of clean, high-quality Wikipedia text.
- Formats text into TSV files compatible with DT-FM data loaders.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path


WIKITEXT_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "task_datasets" / "data"
OUTPUT_DIR = DATA_DIR / "wikitext-103"


def download_with_progress(url, dest_path):
    print(f"Downloading WikiText-103 from {url}...")
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\nDownload complete!")


def convert_tokens_to_tsv(input_file, output_file, max_samples=None):
    """
    Groups clean text lines into paragraph samples and formats as TSV:
    id \t qid1 \t qid2 \t text_a \t text_b \t 0
    """
    print(f"Converting {input_file.name} -> {output_file.name}...")
    uid = 0
    buffer = []
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        # Write header expected by DT-FM dataset loader
        fout.write("id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n")
        
        for line in fin:
            line = line.strip()
            # Skip empty lines and Wikipedia article section headings like "= = = Section = = ="
            if not line or line.startswith("="):
                if len(buffer) >= 2:
                    # Form a training sample from accumulated text
                    mid = len(buffer) // 2
                    text_a = " ".join(buffer[:mid]).replace("\t", " ").strip()
                    text_b = " ".join(buffer[mid:]).replace("\t", " ").strip()
                    if text_a and text_b:
                        fout.write(f"{uid}\t{uid*2}\t{uid*2+1}\t{text_a}\t{text_b}\t0\n")
                        uid += 1
                        if max_samples and uid >= max_samples:
                            break
                    buffer = []
                continue
            
            buffer.append(line)
            if len(buffer) >= 6:
                mid = len(buffer) // 2
                text_a = " ".join(buffer[:mid]).replace("\t", " ").strip()
                text_b = " ".join(buffer[mid:]).replace("\t", " ").strip()
                if text_a and text_b:
                    fout.write(f"{uid}\t{uid*2}\t{uid*2+1}\t{text_a}\t{text_b}\t0\n")
                    uid += 1
                    if max_samples and uid >= max_samples:
                        break
                buffer = []

    print(f"  Created {uid} formatted training samples in {output_file}.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_dest = DATA_DIR / "wikitext-103-v1.zip"
    raw_dir = DATA_DIR / "wikitext-103"

    train_tsv = OUTPUT_DIR / "train.tsv"
    test_tsv = OUTPUT_DIR / "test.tsv"

    if train_tsv.exists() and test_tsv.exists():
        print(f"Dataset already exists at: {OUTPUT_DIR}")
        print(f"  - {train_tsv} ({train_tsv.stat().st_size / (1024*1024):.1f} MB)")
        print(f"  - {test_tsv} ({test_tsv.stat().st_size / (1024*1024):.1f} MB)")
        return

    if not zip_dest.exists():
        download_with_progress(WIKITEXT_URL, zip_dest)
    else:
        print(f"Found existing archive at: {zip_dest}")

    print("Extracting archive...")
    with zipfile.ZipFile(zip_dest, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    train_raw = raw_dir / "wiki.train.tokens"
    valid_raw = raw_dir / "wiki.valid.tokens"

    convert_tokens_to_tsv(train_raw, train_tsv)
    convert_tokens_to_tsv(valid_raw, test_tsv)

    print("\nDataset preparation completed successfully!")
    print(f"Train dataset: {train_tsv}")
    print(f"Validation dataset: {test_tsv}")


if __name__ == "__main__":
    main()
