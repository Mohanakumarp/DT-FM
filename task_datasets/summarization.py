"""Runtime-configured article/summary datasets and local JSON/CSV/Parquet."""
import re
from pathlib import Path


def load_summarization_splits(args):
    from datasets import DownloadManager, load_dataset

    splits = [args.train_split]
    if args.validation_batches:
        splits.append(args.validation_split)
    data_files = None
    if args.dataset == "csebuetnlp/xlsum":
        language = args.dataset_config or "tamil"
        if not re.fullmatch(r"[a-z_]+", language):
            raise ValueError("XL-Sum configuration must be a language name")
        archive = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/%s_XLSum_v2.0.tar.bz2" % language
        root = Path(DownloadManager().download_and_extract(archive))
        data_files = {"train": str(root / (language + "_train.jsonl")),
                      "validation": str(root / (language + "_val.jsonl")),
                      "test": str(root / (language + "_test.jsonl"))}
        loaded = load_dataset("json", data_files=data_files, split=splits)
    else:
        if args.train_file:
            data_files = {"train": args.train_file}
            if args.validation_file:
                data_files["validation"] = args.validation_file
        loaded = load_dataset(args.dataset, name=args.dataset_config, data_files=data_files,
                              split=splits, trust_remote_code=False)
    for dataset in loaded:
        missing = {args.source_column, args.target_column} - set(dataset.column_names)
        if missing:
            raise ValueError("dataset is missing columns %s; available columns: %s" %
                             (sorted(missing), dataset.column_names))
        if len(dataset) == 0:
            raise ValueError("dataset split is empty")
    return loaded[0], loaded[1] if len(loaded) > 1 else None


def tokenize_summaries(batch, tokenizer, args):
    sources = batch[args.source_column]
    targets = batch[args.target_column]
    if any(not isinstance(text, str) or not text.strip() for text in sources + targets):
        raise ValueError("source and target columns must contain nonempty strings")
    if not args.is_encoder_decoder:
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        separator = tokenizer(args.summary_separator, add_special_tokens=False)["input_ids"]
        source_budget = args.source_length - len(separator)
        if source_budget < 1:
            raise ValueError("source-length must leave room for article tokens and the summary separator")
        for source, target in zip(sources, targets):
            # Keep the summary cue even when a long article is truncated.
            prompt = tokenizer(args.source_prefix + source,
                               max_length=source_budget, truncation=True)["input_ids"] + separator
            summary = tokenizer(target, add_special_tokens=False,
                                max_length=args.target_length, truncation=True)["input_ids"]
            if tokenizer.eos_token_id is not None and (not summary or summary[-1] != tokenizer.eos_token_id):
                summary = summary[:max(0, args.target_length - 1)] + [tokenizer.eos_token_id]
            if not prompt or not summary:
                raise ValueError("tokenizer produced an empty prompt or summary")
            result["input_ids"].append(prompt + summary)
            result["attention_mask"].append([1] * (len(prompt) + len(summary)))
            # Article/prompt tokens and padding do not contribute to causal loss.
            result["labels"].append([-100] * len(prompt) + summary)
        return result
    inputs = tokenizer([args.source_prefix + text for text in sources],
                       max_length=args.source_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=args.target_length, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs
