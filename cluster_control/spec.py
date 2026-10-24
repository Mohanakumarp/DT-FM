"""A versioned, bounded job format; owners select settings, never shell commands."""
import math
import re

from cluster_control.service import APIError


MODELS = {
    "tiny-random-bert": {"label": "Tiny random BERT (offline smoke)", "units": 6},
    "prajjwal1/bert-mini": {"label": "BERT Mini", "units": 6},
    "google-bert/bert-base-uncased": {"label": "BERT Base", "units": 14},
    "google-bert/bert-large-uncased": {"label": "BERT Large", "units": 26},
}
METRIC_KEYS = {"event", "phase", "step", "epoch", "train_loss", "seconds", "completed_steps",
               "features_per_second", "training_wall_seconds", "peak_gpu_allocated_bytes", "parameters",
               "checkpoint_seconds", "exact_match", "f1"}
DEFAULTS = {
    "workflow": "smoke", "model": "tiny-random-bert", "dataset": "synthetic",
    "model_revision": "main", "dataset_revision": "main", "pipeline_size": 1, "replicas": 1,
    "batch_size": 2, "micro_batch_size": 1, "epochs": 2, "max_steps": 2,
    "learning_rate": 0.00005, "max_length": 128, "doc_stride": 32,
    "train_examples": 100, "validation_examples": 20, "checkpoint_every": 1,
    "port": 29500, "timeout_seconds": 120,
}


def validate_config(body):
    if not isinstance(body, dict) or set(body) - set(DEFAULTS):
        raise APIError(400, "Unknown training settings")
    config = {**DEFAULTS, **body}
    if config["workflow"] not in ("smoke", "qa"):
        raise APIError(400, "Choose smoke or qa workflow")
    if not isinstance(config["model"], str) or config["model"] not in MODELS:
        raise APIError(400, "Choose a supported BERT model")
    if config["workflow"] == "smoke":
        if config["model"] != "tiny-random-bert" or config["dataset"] != "synthetic":
            raise APIError(400, "Smoke requires tiny-random-bert and synthetic data")
    elif config["model"] == "tiny-random-bert" or not isinstance(config["dataset"], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", config["dataset"]):
        raise APIError(400, "QA requires a pretrained BERT and a Hugging Face dataset ID with SQuAD-compatible splits")
    for key in ("model_revision", "dataset_revision"):
        if not isinstance(config[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", config[key]):
            raise APIError(400, "Invalid " + key)
    bounds = {"pipeline_size": (1, MODELS[config["model"]]["units"]), "replicas": (1, 16),
              "batch_size": (1, 1024), "micro_batch_size": (1, 1024), "epochs": (1, 10000),
              "max_steps": (0, 1000000), "max_length": (8, 512), "doc_stride": (0, 511),
              "train_examples": (1, 1000000), "validation_examples": (1, 100000),
              "checkpoint_every": (0, 100000), "port": (1024, 65535), "timeout_seconds": (10, 1800)}
    for key, (low, high) in bounds.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise APIError(400, "%s must be an integer from %d to %d" % (key, low, high))
    if config["pipeline_size"] * config["replicas"] > 16:
        raise APIError(400, "This version supports at most 16 ranks per job")
    if config["micro_batch_size"] > config["batch_size"] or config["doc_stride"] >= config["max_length"]:
        raise APIError(400, "Microbatch must fit the batch and doc stride must be smaller than sequence length")
    lr = config["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 1:
        raise APIError(400, "Learning rate must be finite and between 0 and 1")
    return config


def validate_capabilities(body):
    if not isinstance(body, dict):
        raise APIError(400, "Invalid capability report")
    devices = body.get("devices")
    if not isinstance(devices, list) or not 1 <= len(devices) <= 16:
        raise APIError(400, "Report 1–16 devices")
    result = []
    for item in devices:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,96}", item["id"]):
            raise APIError(400, "Invalid device identity")
        if item.get("backend") not in ("cpu", "cuda", "rocm", "xpu", "directml", "mps"):
            raise APIError(400, "Unknown device backend")
        if not isinstance(item.get("name"), str) or not 1 <= len(item["name"]) <= 160:
            raise APIError(400, "Invalid device name")
        if type(item.get("index", 0)) is not int or not 0 <= item.get("index", 0) <= 15:
            raise APIError(400, "Invalid device index")
        memory = item.get("memory_bytes")
        if memory is not None and (type(memory) is not int or not 0 <= memory <= 2**60):
            raise APIError(400, "Invalid memory report")
        result.append({"id": item["id"], "name": item["name"], "backend": item["backend"],
                       "index": item.get("index", 0), "memory_bytes": memory})
    if len({item["id"] for item in result}) != len(result):
        raise APIError(400, "Duplicate device identities")
    host_id = body.get("host_id")
    if not isinstance(host_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", host_id):
        raise APIError(400, "Invalid host identity")
    threads = body.get("cpu_threads", 1)
    if type(threads) is not int or not 1 <= threads <= 64:
        raise APIError(400, "CPU thread limit must be 1–64")
    return {"devices": result, "host_id": host_id, "platform": str(body.get("platform", "unknown"))[:80],
            "runtime_ready": body.get("runtime_ready") is True,
            "runtime_error": str(body.get("runtime_error", ""))[:500], "cpu_threads": threads}
