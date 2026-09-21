# MLflow Tracking Guide for Distributed BERT Training

This guide explains how to set up, run, and monitor distributed BERT model training over a decentralized computer network using **Tailscale** and **Gloo**, with support for **heterogeneous GPU/CPU drivers** (NVIDIA CUDA, AMD ROCm, Intel XPU, DirectML, and CPU) and full experiment tracking via **MLflow**.

---

## 1. Architecture Overview

```
                        +------------------------------------+
                        |       Tailscale Mesh Network       |
                        |      (tailscale0 / 100.x.y.z)      |
                        +-----------------+------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
+------------------+             +------------------+             +------------------+
|      Rank 0      |             |      Rank 1      |             |      Rank 2      |
|  (Coordinator)   |   Gloo P2P  |  (Middle Stage)  |   Gloo P2P  |   (Last Stage)   |
|   NVIDIA CUDA    | ----------->|     AMD ROCm     | ----------->|    Intel / CPU   |
|  (Embeddings +   |  Activations| (Hidden Layers)  |  Activations|  (Final Layers + |
|  Hidden Layers)  |             |                  |             | Classifier Head) |
+--------+---------+             +--------+---------+             +--------+---------+
         |                                |                                |
         | Log Metrics / Params           | Log Metrics / Params           | Log Metrics / Loss
         |                                |                                |
         +--------------------------------+--------------------------------+
                                          |
                                          v
                        +------------------------------------+
                        |       Central MLflow Server        |
                        |      http://100.x.y.z:5000         |
                        |   - Parameters & System Metadata   |
                        |   - Loss Curves & Throughput       |
                        |   - Comm Latency/Bandwidth Matrix  |
                        +------------------------------------+
```

- **Communication Backend**: CPU-staged tensor exchange over **Gloo** allows pipeline stages to run on completely heterogeneous hardware (e.g. NVIDIA + AMD + Intel + CPU) without vendor-specific interconnect conflicts.
- **Networking**: **Tailscale** establishes encrypted point-to-point mesh connectivity (`tailscale0`, IPv4 `100.x.y.z`) across physically distributed computers (laptops, desktops, cloud instances) even behind NATs or campus firewalls.
- **Pipeline Parallelism**: GPipe pipeline asynchronously shuttles micro-batch activations forward and gradients backward across ranks.
- **MLflow Tracking**: Centralized SQLite-backed tracking server running on the coordinator (or dedicated machine) on the Tailscale network. All distributed ranks log their hyperparameters, stage parameters, device profiles, real-time step loss, throughput, and communication matrices.

---

## 2. Environment & Installation

In your training Python environment on each machine (e.g. `conda activate dtfm`):

```bash
pip install -r requirements-mlflow.txt
```

Verify that MLflow is installed:

```bash
python -c "import mlflow; print('MLflow installed version:', mlflow.__version__)"
```

---

## 3. Starting the Central MLflow Tracking Server

Run the tracking server on the coordinator machine (Rank 0) or any machine accessible via Tailscale:

```bash
bash scripts/start_mlflow.sh
```

By default, the server:
- Binds to `0.0.0.0:5000` (listening on localhost and Tailscale IP).
- Uses a persistent SQLite database: `sqlite:///mlflow.db`.
- Saves run artifacts to `./mlartifacts`.
- Detects and prints your Tailscale IPv4 address (`100.x.y.z`).

### Verifying Connectivity from Remote Nodes
From any worker machine on Tailscale:

```bash
curl http://<COORDINATOR_TAILSCALE_IP>:5000/health
```
*(Should return `OK`)*

---

## 4. Preparing Data and Vocabulary

Before launching BERT training, ensure the GLUE QQP dataset and the BERT tokenizer vocabulary are downloaded:

```bash
bash scripts/download_data.sh
```

This ensures:
- `task_datasets/data/bert-large-cased-vocab.txt` (BERT WordPiece vocab)
- `task_datasets/data/QQP/train.tsv`, `dev.tsv`, `test.tsv`

*(Note: For rapid smoke testing before full training, set `SYNTHETIC_DATA=true`.)*

---

## 5. BERT Model Configurations

DT-FM supports two primary BERT profiles:

| Specification | `PROFILE=bert` (Base ~110M) | `PROFILE=bert-500m` (~510M Params) |
| :--- | :--- | :--- |
| **Total Parameters** | ~107.4 Million | **~509.5 Million** |
| **Hidden Dim (`embedding_dim`)** | 768 | **1280** |
| **Attention Heads (`num_heads`)** | 12 | **16** (80 dim/head) |
| **Total Layers** | 12 (4 layers / rank) | **24 (8 layers / rank)** |
| **Per-Rank Parameters** | ~36M params | **~170M params** |
| **Estimated VRAM per 3050** | ~1.5 GB | **~3.2 GB** *(cleanly fits in 6GB VRAM)* |
| **Sequence Length** | 128 | 128 |
| **Task** | `SeqClassification` (QQP) | `SeqClassification` (QQP) |

> [!TIP]
> On 3x laptops with **6GB RTX 3050 GPUs**, `PROFILE=bert-500m` partitions the 24 layers into 8 layers per laptop. In FP32 with gradient checkpointing, each laptop uses only ~3.2 GB of VRAM, leaving ~2.8 GB headroom.

---

## 6. Launching Distributed BERT Training with MLflow

> [!IMPORTANT]
> **Single MLflow Server Architecture**:
> - The MLflow tracking server runs **ONLY on Laptop 1 (Rank 0)**.
> - **Laptops 2 and 3 do NOT run an MLflow server**.
> - Laptops 2 and 3 only need Python with the `mlflow` client installed (`pip install -r requirements-mlflow.txt`). They send HTTP metric and artifact POST requests across Tailscale to `http://<MASTER_TAILSCALE_IP>:5000`.
> - Rank 0's central dashboard displays all runs, metrics, and logs from all 3 laptops in one place.

### Laptop 1 (Master / Rank 0, RTX 3050)
1. **Start the MLflow server**:
   ```bash
   bash scripts/start_mlflow.sh
   ```
2. **In a second terminal, launch Rank 0**:
   ```bash
   export MLFLOW_TRACKING_URI="http://100.125.135.116:5000"
   export MLFLOW_EXPERIMENT="DT-FM BERT-500M"
   export MLFLOW_GROUP="bert500m-launch-1"

   RANK=0 \
   WORLD_SIZE=3 \
   DEVICE=cuda \
   CUDA_ID=0 \
   PROFILE=bert-500m \
   BATCH=8 \
   MICRO=2 \
   EPOCHS=2 \
   STEPS_PER_EPOCH=50 \
   bash scripts/run_rank.sh
   ```

### Laptop 2 (Worker / Rank 1, RTX 3050)
```bash
export MLFLOW_TRACKING_URI="http://100.125.135.116:5000"
export MLFLOW_EXPERIMENT="DT-FM BERT-500M"
export MLFLOW_GROUP="bert500m-launch-1"

RANK=1 \
MASTER_IP=100.125.135.116 \
WORLD_SIZE=3 \
DEVICE=cuda \
CUDA_ID=0 \
PROFILE=bert-500m \
BATCH=8 \
MICRO=2 \
EPOCHS=2 \
STEPS_PER_EPOCH=50 \
bash scripts/run_rank.sh
```

### Laptop 3 (Worker / Rank 2, RTX 3050)
```bash
export MLFLOW_TRACKING_URI="http://100.125.135.116:5000"
export MLFLOW_EXPERIMENT="DT-FM BERT-500M"
export MLFLOW_GROUP="bert500m-launch-1"

RANK=2 \
MASTER_IP=100.125.135.116 \
WORLD_SIZE=3 \
DEVICE=cuda \
CUDA_ID=0 \
PROFILE=bert-500m \
BATCH=8 \
MICRO=2 \
EPOCHS=2 \
STEPS_PER_EPOCH=50 \
bash scripts/run_rank.sh

---

## 7. What MLflow Documents & Tracks

### Run Identification & Tags
- `launch_group`: Shared launch identifier grouping all pipeline ranks together.
- `pipeline_stage`: Position in pipeline (e.g. `0/3`, `1/3`, `2/3`).
- `pipeline_role`:
  - `first_stage`: Embedding layer + stage transformer layers.
  - `middle_stage`: Intermediate transformer layers.
  - `last_stage`: Final transformer layers + `SeqClassification` head + loss computation.
- `model_architecture`: Automatically tagged as `BERT`.
- `device_name`: Resolved device description (e.g. `NVIDIA GeForce RTX 3050 Laptop GPU`, `AMD ROCm device 0 (gfx1030)`, `Intel XPU device 0`, `CPU`).
- `network_interface`: Bound interface (e.g. `tailscale0`).

### Parameters
- **Architecture**: `seq_length`, `embedding_dim`, `num_heads`, `num_layers`, `total_layers`, `stage_parameters`, `tokenizer_type`.
- **Distributed Topology**: `rank`, `world_size`, `pipeline_group_size`, `data_group_size`, `tensor_comm`, `dist_backend`, `dist_url`.
- **Training Hyperparameters**: `lr`, `batch_size`, `micro_batch_size`, `gradient_accumulate_step`, `num_epochs`, `steps_per_epoch`, `seed`.

### Metrics (Logged Per Step)
- `last_microbatch_loss`: Final micro-batch cross-entropy loss (recorded by `last_stage`).
- `iter_s`: Iteration wall-clock duration in seconds.
- `samples_per_second`: Effective training throughput per rank (`batch_size / iter_s`).
- `forward_s`, `backward_s`, `optim_s`, `barrier_s`: Step component timing breakdown.
- `rebalance_seconds`, `assigned_layers`: Migration duration and layer reallocations (if dynamic scheduling is active).

### Artifacts (`measurements/metrics_rankN.json`)
At the end of training, each rank automatically uploads its measurements JSON artifact containing:
- Full step-by-step latency and throughput history.
- Compute vs. barrier synchronization efficiency percentage.
- **Inter-node communication matrices**:
  - `latency_ms`: Pairwise round-trip network latency measured across Tailscale peers.
  - `bandwidth_mbps`: Pairwise effective network bandwidth in Mbps.

---

## 8. Analyzing Results in the MLflow UI

1. Open your browser to `http://<COORDINATOR_IP>:5000`.
2. Select the experiment **BERT-Distributed**.
3. Filter runs by your launch group:
   ```
   tags.launch_group = 'bert-qqp-20260921-150000'
   ```
4. You will see 3 runs (one for each rank).
5. **Compare Runs**:
   - Select all runs in the group and click **Compare**.
   - Check `samples_per_second` and `iter_s` to compare execution speed across ranks.
   - Inspect `last_microbatch_loss` on Rank 2 to track model convergence over epochs.
6. **Inspect Artifacts**:
   - Click on Rank 0 or Rank 2, open **Artifacts** > `measurements` > `metrics_rankN.json`.
   - View the measured Tailscale latency and bandwidth matrix between all machines.
