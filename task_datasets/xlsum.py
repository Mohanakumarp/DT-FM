"""Load the original XL-Sum JSONL files without executing its legacy Hub script."""


def load_tamil_xlsum(cache_dir=None):
    from datasets import DownloadConfig, DownloadManager, load_dataset

    url = (
        "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/"
        "data/tamil_XLSum_v2.0.tar.bz2"
    )
    from pathlib import Path
    root = Path(DownloadManager(download_config=DownloadConfig(cache_dir=cache_dir)).download_and_extract(url))
    files = {
        "train": str(root / "tamil_train.jsonl"),
        "validation": str(root / "tamil_val.jsonl"),
        "test": str(root / "tamil_test.jsonl"),
    }
    return load_dataset("json", data_files=files, cache_dir=cache_dir)


def tokenize_batch(batch, tokenizer, source_length, target_length):
    inputs = tokenizer(
        batch["text"], max_length=source_length, truncation=True
    )
    targets = tokenizer(
        text_target=batch["summary"], max_length=target_length, truncation=True
    )
    inputs["labels"] = targets["input_ids"]
    return inputs
