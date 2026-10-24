"""CPU/Gloo sparse SUM exchange for globally token-normalized gradients.

Top-K keeps local error feedback. DGC additionally accumulates local SGD
momentum and masks it at transmitted coordinates (Eq. 7 of Lin et al., 2018).
Local clipping belongs to the caller, before accumulation. This is deliberately
not a DDP hook: the pipeline runner has already divided by global target tokens.
"""
import math

import torch
import torch.distributed as dist


class SparseGradientSynchronizer:
    def __init__(self, named_parameters, *, mode="topk", keep_ratio=0.001,
                 warmup_steps=100, warmup_ratio=0.25, momentum=0.9,
                 bucket_size=4 * 1024 * 1024):
        if mode not in ("topk", "dgc"):
            raise ValueError("compression mode must be topk or dgc")
        if not 0 < keep_ratio <= warmup_ratio <= 1:
            raise ValueError("require 0 < keep_ratio <= warmup_ratio <= 1")
        if warmup_steps < 0 or not 0 <= momentum < 1 or not 1 <= bucket_size < 2**31:
            raise ValueError("invalid warmup steps, momentum, or bucket size")
        parameters = [(name, p) for name, p in named_parameters if p.requires_grad]
        self.config = {
            "mode": mode, "keep_ratio": keep_ratio, "warmup_steps": warmup_steps,
            "warmup_ratio": warmup_ratio, "momentum": momentum,
            "bucket_size": bucket_size,
            "parameters": [(name, list(p.shape)) for name, p in parameters],
        }
        self.completed_steps = 0
        self.buckets = []
        pieces, size = [], 0
        for name, parameter in parameters:
            offset = 0
            while offset < parameter.numel():
                count = min(parameter.numel() - offset, bucket_size - size)
                pieces.append((parameter, offset, count))
                size += count
                offset += count
                if size == bucket_size:
                    self._add_bucket(pieces, size)
                    pieces, size = [], 0
        if size:
            self._add_bucket(pieces, size)

    def _add_bucket(self, pieces, size):
        self.buckets.append({
            "pieces": pieces, "residual": torch.zeros(size, dtype=torch.float32),
            "momentum": torch.zeros(size, dtype=torch.float32) if self.config["mode"] == "dgc" else None,
        })

    def current_keep_ratio(self):
        target = self.config["keep_ratio"]
        steps = self.config["warmup_steps"]
        if not steps or self.completed_steps >= steps:
            return target
        start = self.config["warmup_ratio"]
        return start * (target / start) ** (self.completed_steps / steps)

    @torch.no_grad()
    def synchronize(self, group):
        replicas = dist.get_world_size(group)
        if replicas < 2:
            raise ValueError("gradient compression requires at least two data replicas")
        ratio = self.current_keep_ratio()
        dense_bytes = payload_bytes = selected = 0
        for bucket in self.buckets:
            for parameter, _, _ in bucket["pieces"]:
                if parameter.grad is not None and not parameter.grad.is_contiguous():
                    parameter.grad = parameter.grad.contiguous()
        for bucket in self.buckets:
            residual, momentum = bucket["residual"], bucket["momentum"]
            local = torch.zeros_like(residual)
            cursor = 0
            for parameter, offset, count in bucket["pieces"]:
                if parameter.grad is not None:
                    local[cursor:cursor + count].copy_(parameter.grad.detach().reshape(-1)[offset:offset + count])
                cursor += count
            if momentum is not None:
                momentum.mul_(self.config["momentum"]).add_(local)
                residual.add_(momentum)
            else:
                residual.add_(local)
            size = residual.numel()
            k = max(1, min(size, math.ceil(size * ratio)))
            dense_bytes += size * 4
            # FP32 values + int32 bucket-local indices. Avoid expanding dense
            # traffic when the requested selection costs more than all-reduce.
            if k * 8 >= size * 4:
                local.copy_(residual)
                dist.all_reduce(local, group=group)
                residual.zero_()
                if momentum is not None:
                    momentum.zero_()
                payload_bytes += size * 4
                selected += size
            else:
                indices = torch.topk(residual.abs(), k, sorted=False).indices
                values = residual[indices].contiguous()
                wire_indices = indices.to(torch.int32)
                peer_indices = [torch.empty_like(wire_indices) for _ in range(replicas)]
                peer_values = [torch.empty_like(values) for _ in range(replicas)]
                dist.all_gather(peer_indices, wire_indices, group=group)
                dist.all_gather(peer_values, values, group=group)
                local.zero_()
                for positions, updates in zip(peer_indices, peer_values):
                    local.index_add_(0, positions.long(), updates)
                # Only the local selection is removed, never another rank's.
                residual[indices] = 0
                if momentum is not None:
                    momentum[indices] = 0
                payload_bytes += k * 8
                selected += k
            cursor = 0
            for parameter, offset, count in bucket["pieces"]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.view(-1)[offset:offset + count].copy_(local[cursor:cursor + count])
                cursor += count
        self.completed_steps += 1
        return {
            "gradient_keep_ratio": ratio, "gradient_selected_elements": selected,
            "gradient_dense_bytes": dense_bytes, "gradient_payload_bytes": payload_bytes,
            "gradient_payload_compression": dense_bytes / payload_bytes if payload_bytes else 1.0,
        }

    def state_dict(self):
        return {
            "version": 1, "config": self.config, "completed_steps": self.completed_steps,
            "residuals": [bucket["residual"] for bucket in self.buckets],
            "momentums": [bucket["momentum"] for bucket in self.buckets],
        }

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("config") != self.config:
            raise ValueError("checkpoint gradient compression configuration differs from this run")
        if not isinstance(state.get("completed_steps"), int) or state["completed_steps"] < 0:
            raise ValueError("invalid gradient compression step in checkpoint")
        if len(state["residuals"]) != len(self.buckets) or len(state["momentums"]) != len(self.buckets):
            raise ValueError("checkpoint gradient compression bucket count differs")
        for bucket, residual, momentum in zip(self.buckets, state["residuals"], state["momentums"]):
            for key, saved in (("residual", residual), ("momentum", momentum)):
                destination = bucket[key]
                if destination is None:
                    if saved is not None:
                        raise ValueError("unexpected compression momentum in checkpoint")
                elif saved is None or saved.shape != destination.shape or saved.dtype != destination.dtype:
                    raise ValueError("checkpoint gradient compression tensor differs")
                else:
                    destination.copy_(saved)
        self.completed_steps = state["completed_steps"]
