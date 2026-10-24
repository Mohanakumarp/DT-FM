"""Atomic per-rank checkpoints and resume from a step available on every rank."""
import os
from pathlib import Path
import shutil

import torch
import torch.distributed as dist


def collect_errors(error):
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError("Checkpoint operation failed: " + str(errors))


def checkpoint_candidates(root):
    return {path.name for path in Path(root).glob("checkpoint-step-*.pt")}


def load_common_checkpoint(root, args, identity, stage, optimizer, compressor):
    candidates = [None] * args.world_size
    dist.all_gather_object(candidates, checkpoint_candidates(root))
    common = set.intersection(*candidates)
    if not common:
        raise ValueError("No resumable checkpoint step exists on every rank; the older runner saved only at completion")
    name = max(common, key=lambda value: int(value.split("-")[2]))
    state = None
    error = None
    try:
        state = torch.load(Path(root) / name, map_location="cpu", weights_only=True)
        if state.get("resume_version") != 1 or state.get("identity") != identity:
            raise ValueError("Checkpoint model, data, topology or optimizer settings differ")
        if state["rank"] != args.rank or state["device"] != args.device:
            raise ValueError("Checkpoint rank/device differs")
        stage.load_pretrained_state_dict(state["pretrained"])
        optimizer.load_state_dict(state["optimizer"])
        if compressor:
            compressor.load_state_dict(state["compressor"])
            if compressor.completed_steps != state["step"]:
                raise ValueError("Compressor and checkpoint step differ")
        torch.set_rng_state(state["rng_cpu"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(state["rng_cuda"])
    except Exception as exception:
        error = str(exception)
    collect_errors(error)
    metadata = [None] * args.world_size
    dist.all_gather_object(metadata, (state["run_id"], state["step"]))
    if any(value != metadata[0] for value in metadata):
        raise ValueError("Checkpoints belong to different runs or steps")
    return state


def save_checkpoint(root, args, stage, optimizer, compressor, step, run_id, identity):
    """Retain the latest two versions so a failed save can fall back one step."""
    root = Path(root)
    name = "checkpoint-step-%08d-%s.pt" % (step, run_id)
    path = root / name
    temporary = root / (name + ".tmp")
    error = None
    try:
        state = {"resume_version": 1, "identity": identity, "device": args.device,
                 "task": "qa", "rank": args.rank, "step": step, "model": args.model,
                 "smoke": args.smoke, "run_id": run_id, "world_size": args.world_size,
                 "pipeline_size": args.pipeline_size, "boundaries": stage.boundaries,
                 "partition_version": 1, "gradient_compression": args.gradient_compression,
                 "compressor": compressor.state_dict() if compressor else None,
                 "pretrained": stage.pretrained_state_dict(), "optimizer": optimizer.state_dict(),
                 "rng_cpu": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state() if args.device == "cuda" else None}
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception as exception:
        error = str(exception)
    collect_errors(error)  # All ranks now have this version before updating pointers.
    error = None
    try:
        latest = root / "checkpoint.tmp"
        latest.unlink(missing_ok=True)
        try:
            os.link(path, latest)
        except OSError:
            shutil.copyfile(path, latest)
        latest.replace(root / "checkpoint.pt")
        versions = sorted(checkpoint_candidates(root), key=lambda value: int(value.split("-")[2]), reverse=True)
        for old in versions[2:]:
            (root / old).unlink()
    except Exception as exception:
        error = str(exception)
    collect_errors(error)
