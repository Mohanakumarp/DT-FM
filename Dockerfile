# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.22 AS uv

# NVIDIA runtime: build explicitly with --target cuda. CPU is the default target.
FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS cuda
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH \
    DTFM_DEVICE=cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates curl unzip git openssh-client \
        iproute2 libgomp1 cuda-cudart-dev-12-6 cuda-nvrtc-dev-12-6 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
RUN UV_HTTP_TIMEOUT=600 uv pip install --python /opt/venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt requirements-mlflow.txt requirements-mt5.txt ./
COPY docker/requirements.txt ./docker/requirements.txt
RUN UV_HTTP_TIMEOUT=600 uv pip install --python /opt/venv/bin/python -r docker/requirements.txt cupy-cuda12x==13.6.0 \
    && uv pip check --python /opt/venv/bin/python \
    && python -c "from pathlib import Path; libs=sorted(Path('/opt/venv/lib/python3.10/site-packages/nvidia').glob('*/lib')); Path('/etc/ld.so.conf.d/dtfm-nvidia.conf').write_text(''.join(str(p) + '\\n' for p in libs))" \
    && ldconfig \
    && python -c "import torch, cupy; from cupy.cuda import nccl; print(torch.__version__, cupy.__version__, nccl.get_version())"
COPY . .
RUN find scripts docker -type f -name '*.sh' -exec sed -i 's/\r$//' {} + \
    && mkdir -p logs trace_json task_datasets/data
ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["smoke"]

# Shared CPU runtime for linux/amd64 and linux/arm64.
FROM python:3.10-slim-bookworm AS cpu-runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    DTFM_DEVICE=cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip git openssh-client iproute2 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
RUN uv pip install --python /usr/local/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-mlflow.txt requirements-mt5.txt ./
COPY docker/requirements.txt ./docker/requirements.txt
RUN uv pip install --python /usr/local/bin/python -r docker/requirements.txt \
    && uv pip check --python /usr/local/bin/python
COPY . .
RUN find scripts docker -type f -name '*.sh' -exec sed -i 's/\r$//' {} + \
    && mkdir -p logs trace_json task_datasets/data
ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["smoke"]

# Apple silicon Docker Desktop runs Linux ARM64 with CPU compute.
# MPS training requires the native macOS setup, outside this container.
FROM cpu-runtime AS mac
ARG TARGETARCH
RUN test "${TARGETARCH}" = arm64 || \
    (echo 'Build the mac target with --platform linux/arm64. Intel Macs use the cpu target.' >&2; exit 1)
LABEL org.opencontainers.image.description="DT-FM ARM64 CPU training for Apple silicon Docker Desktop. Apple MPS GPU training requires native macOS."

# Keep CPU as the default target for existing docker build/Compose commands.
FROM cpu-runtime AS cpu
