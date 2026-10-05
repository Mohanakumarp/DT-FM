#!/usr/bin/env bash
''''exec python3 "$0" "$@" # '''
"""
Download and prepare WikiText-103 for DT-FM GPT-2 Foundation Model training.
- Uses FastAI's reliable high-speed AWS mirror.
- Shows live download progress bar with speed.
- Automatically handles extraction (.tgz / .tar.gz / .zip) and format conversion to TSV.
"""

import os
import sys
import time
import shutil
import tarfile
import zipfile
import subprocess
import urllib.request
from pathlib import Path


# FastAI S3 mirror (reliable 200 OK endpoint, ~180 MB)
WIKITEXT_URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz"
FALLBACK_URL = "https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "task_datasets" / "data"
OUTPUT_DIR = DATA_DIR / "wikitext-103"


def download_with_progress(url, dest_path):
    print(f"\n[1/3] Downloading WikiText-103 (~180 MB) from:\n      {url}\n")
    
    # Try using system curl if available for maximum speed and progress bar
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--progress-bar", "--fail", "-o", str(dest_path), url]
        res = subprocess.run(cmd)
        if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
            print("\n  --> Download finished successfully via curl!")
            return
        print("\n  curl encountered an error; falling back to Python downloader...")

    # Python urllib fallback with redirect handling and live progress bar
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    
    start_time = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as fout:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 64

        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            fout.write(chunk)
            downloaded += len(chunk)
            
            elapsed = time.time() - start_time
            speed_mb = (downloaded / elapsed) / (1024 * 1024) if elapsed > 0 else 0
            
            if total_size > 0:
                percent = min(100, int(downloaded * 100 / total_size))
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                bar_len = 30
                filled = int(bar_len * percent / 100)
                bar = "=" * filled + ">" + " " * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
                sys.stdout.write(f"\r  [{bar}] {percent:3d}% | {mb_down:6.1f} / {mb_total:6.1f} MB | {speed_mb:4.1f} MB/s")
                sys.stdout.flush()

    print("\n  --> Download finished successfully!")


def extract_archive(archive_path, extract_to):
    print(f"\n[2/3] Extracting archive {archive_path.name}...")
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(extract_to)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")
    print("  --> Extraction complete.")


def find_data_file(search_dirs, pattern_keywords):
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and any(k in p.name.lower() for k in pattern_keywords):
                if p.suffix.lower() in [".tokens", ".txt", ".csv", ".tsv"]:
                    return p
    return None


def convert_to_tsv(input_file, output_file, max_samples=None):
    print(f"\n[3/3] Formatting {input_file.name} -> {output_file.name}...")
    uid = 0
    buffer = []
    total_lines = 0
    
    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        fout.write("id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n")
        
        for line in fin:
            total_lines += 1
            line = line.strip().strip('"')
            
            if total_lines % 250000 == 0:
                sys.stdout.write(f"\r  Processed {total_lines:,} lines -> {uid:,} samples formatted...")
                sys.stdout.flush()

            # Skip empty lines and Wikipedia article section headings like "= = = Section = = ="
            if not line or line.startswith("="):
                if len(buffer) >= 2:
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

    print(f"\r  --> Completed: {uid:,} samples in {output_file.name} ({output_file.stat().st_size / (1024*1024):.1f} MB)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_dest = DATA_DIR / "wikitext-103.tgz"
    train_tsv = OUTPUT_DIR / "train.tsv"
    test_tsv = OUTPUT_DIR / "test.tsv"

    # Check if already generated
    if train_tsv.exists() and test_tsv.exists() and train_tsv.stat().st_size > 1024 * 1024:
        print(f"\nDataset already exists and is ready at: {OUTPUT_DIR}")
        print(f"  - train.tsv: {train_tsv.stat().st_size / (1024*1024):.1f} MB")
        print(f"  - test.tsv:  {test_tsv.stat().st_size / (1024*1024):.1f} MB")
        return

    # Download archive if not present
    if not archive_dest.exists() or archive_dest.stat().st_size < 1024 * 1024:
        try:
            download_with_progress(WIKITEXT_URL, archive_dest)
        except Exception:
            print("\nRetrying with fallback mirror...")
            download_with_progress(FALLBACK_URL, archive_dest)
    else:
        print(f"\nFound existing archive: {archive_dest}")

    # Extract
    extract_archive(archive_dest, DATA_DIR)

    # Locate extracted training and validation files
    search_paths = [DATA_DIR / "wikitext-103", DATA_DIR]
    train_raw = find_data_file(search_paths, ["train"])
    valid_raw = find_data_file(search_paths, ["valid", "test"])

    if not train_raw:
        raise FileNotFoundError(f"Could not locate extracted training file in {search_paths}")
    if not valid_raw:
        valid_raw = train_raw  # Fallback to train if test split not separate

    convert_to_tsv(train_raw, train_tsv)
    convert_to_tsv(valid_raw, test_tsv)

    print("\n=======================================================")
    print("Dataset ready for DT-FM GPT-2 training!")
    print(f"  Train: {train_tsv}")
    print(f"  Test:  {test_tsv}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
