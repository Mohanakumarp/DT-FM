# DT-FM cluster console

An initial local control service, browser dashboard, and Python worker agent.
The control service needs only Python's standard library. Training runs on
paired computers using the existing BERT QA runner, CPU/NVIDIA CUDA, FP32,
and CPU-staged Gloo communication.

## Start the dashboard

From the repository root:

```bash
python -m cluster_control.server --port 8080
```

Open <http://127.0.0.1:8080>, create an account, and create a cluster. Other users
create their own accounts, request membership, and wait for the owner to approve
them under **Members**. Only approved members can pair computers or read jobs.
Only the owner can choose model/dataset/topology settings, start jobs, stop jobs,
or decide membership requests.

The default database is `.cluster-data/control.sqlite3`. Accounts, memberships,
worker identities, immutable job specifications, status, and bounded logs survive
service restarts. This directory is ignored by Git. Do not commit or share worker
identity files: they contain bearer tokens.

To serve a trusted private network, bind to its IP, or use:

```bash
python -m cluster_control.server --host 0.0.0.0 --port 8080
```

Users and workers must then use the server's reachable private address, rather
than `127.0.0.1`. Authentication uses local password accounts, scrypt hashes,
expiring hashed sessions, HttpOnly/SameSite cookies, CSRF tokens, same-origin
requests, and basic authentication throttling. Email verification, password
recovery, OAuth, and production HTTP serving are not included. This standard
library server is intended for local/private-network experiments. For HTTPS
behind a reverse proxy, use both `--secure-cookie` and
`--public-origin https://your-exact-host` and configure that proxy yourself.

## Prepare a contributing computer

Every computer needs this checkout and its own compatible training Python
environment. Python 3.12 with PyTorch 2.8.0 CPU was used for the real integration
tests. The service was also tested with Python 3.14.

For a new CPU environment on Linux:

```bash
python3.12 -m venv .venv-cluster
source .venv-cluster/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-mt5.txt
```

For Windows PowerShell:

```powershell
py -3.12 -m venv .venv-cluster
.\.venv-cluster\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-cluster\Scripts\python.exe -m pip install -r requirements-mt5.txt
```

Use that environment's Python executable for all worker commands. NVIDIA
contributors should use an appropriate CUDA-enabled PyTorch environment instead
of the CPU installation above. All ranks need compatible PyTorch, Transformers,
and datasets versions; use the same checkout and dependency versions.

Inspect hardware and dependency availability:

```bash
python -m cluster_control.worker inspect
```

On the dashboard, choose **Connect computer**. Pair using its one-time code;
codes expire after ten minutes and can only be used once. The code belongs to
the approved user who generated it.

```bash
python -m cluster_control.worker pair --server http://127.0.0.1:8080 --address 127.0.0.1 --name "My computer" --devices cpu
python -m cluster_control.worker run
```

The pairing command prompts for the code without echoing it. `--devices` accepts
`cpu`, `cuda:0`, comma-separated devices such as `cuda:0,cuda:1`, or `all`. Pairing
records a local maximum set of devices the agent may use. Dashboard pause/resume
cannot expand this local consent. Pair a new agent to contribute additional
devices. The **Devices** page shows supported, unsupported, offline, shared,
reserved, and training devices. Only a computer's contributor can pause/resume
its sharing, including when the cluster owner is a different user.

Worker credentials default to `.cluster-data/worker.json`. POSIX token files are
created with mode `0600`; protect the directory using normal Windows account
permissions on Windows. Use `--state <path>` for multiple identities. Use
`worker run --output <directory>` to choose where logs and model files are saved.
Run one agent per shared physical GPU; duplicate reported GPU identities cannot
be scheduled concurrently. CPU agents on the same host are allowed, which also
supports a local multi-rank smoke demonstration.

## Training across computers

`--server` is the **control service** address. `--address` is the **worker's**
address that other training ranks can reach. They are distinct concepts.

Use a working private network such as the project's existing
[Tailscale setup](QA_TAILSCALE.md). Multi-computer jobs reject loopback worker
addresses. No automatic VPN installation, NAT traversal, firewall setup, or
network credential provisioning is included. Outbound HTTP polling connects
agents to the control service; tensor traffic goes directly between training
workers. Workers also need mutual connectivity for Gloo's peer connections,
not just access to the configured coordinator port.

For Linux workers that need an explicit Gloo interface:

```bash
python -m cluster_control.worker run --interface tailscale0
```

On native Windows, let Gloo choose the interface unless you have validated an
explicit interface for your environment. The launch code uses Python argument
lists, not Bash or shell execution. Windows process termination uses
`taskkill /T`; POSIX uses process groups. Physical Windows/CUDA/multi-host runs
still need separate validation; the completed verification was on Linux CPU.

## Configure and run

The owner uses **Training** to select:

- Offline tiny random BERT plus synthetic spans, or pretrained BERT QA.
- BERT Mini, Base, or Large and a Hugging Face dataset with SQuAD-compatible
  `train`/`validation` splits and question/context/answer fields. Dataset
  compatibility is ultimately verified by runtime preparation.
- Model and dataset revisions. Use commit hashes for immutable pins; `main`
  tracks the upstream branch and is not a reproducibility guarantee.
- Pipeline stages, pipeline replicas, batch/microbatch sizes, epochs, step cap,
  learning rate, example counts, sequence length, and checkpoint interval.
- Participating devices and their rank order. Select devices in the desired
  order; uncheck/recheck a device to move it to the end. One device is assigned
  to each rank. `ranks = stages × replicas`; rank 0 hosts the coordinator.

**Review assignments** validates consent, current device availability, runtime
support, rank counts, and model partition limits. The owner can save a draft or
start it. Starting atomically reserves devices and the coordinator endpoint.
All workers first run model/dataset preparation, including a rank-0 port check
and hostname resolution. Training starts only after every rank reports ready.
Actual Gloo communication is established when training starts; a successful
preparation phase does not prove multi-host collective connectivity.

The model is divided using the QA runner's existing contiguous, equal-unit
partitioning. The resource-aware scheduler used by the older classification
runner is not wired into this QA adapter. Memory estimates, automatic topology
optimization, unequal QA partitions, and GPU memory reservations are not
implemented. Preparation still loads the full model in host memory; jobs can
fail if a model does not fit.

The run view shows rank placement/state, current phase, completed steps, latest
loss, sampled loss curves, throughput, and worker logs. Throughput appears once
the runner produces its summary. The API keeps the last 2,000 bounded log
events per job; full logs stay on workers.

## Stop, failure, and recovery

The owner can stop a job. A contributor can pause sharing or stop their agent
with Ctrl+C. Revoking membership also disconnects the associated agents. A
failed/withdrawn rank fails the entire job and stops its peers. A reservation
remains held until the worker acknowledges termination, or its lease expires.
Membership revocation permanently invalidates existing worker tokens and unused
pairing codes. Reapproved contributors must pair new workers.

Agents stop local processes after losing the control connection for about ten
seconds (plus a bounded in-flight HTTP request and process termination). The
coordinator detects missing heartbeats after thirty seconds. Preparation has
the configured timeout. A worker restart during an active job fails that job;
it cannot silently start a rank from scratch.

Checkpoint files remain under each worker's
`<output>/<job-id>/rank<rank>/` directory. The underlying QA runner supports
manual `--resume`; this dashboard does not automatically resume failed jobs,
replace ranks, change membership mid-run, collect/download artifacts, sandbox
training inside containers, or verify the honesty of contributed computation.
Use trusted contributors for this initial version. Train with the approved
runner only: the API does not accept arbitrary scripts or shell commands.

## Verification

Control-service tests run without machine-learning dependencies:

```bash
python -m unittest discover -s tests -p 'test_cluster_*.py' -v
```

The two real worker integration tests skip if training dependencies are missing.
To include them, run the same command with the training environment's Python.
They pair real worker processes over HTTP, launch actual two-rank Gloo training,
check completed steps/finite losses/checkpoint files, stop an actively training
job, and verify that its devices can immediately train a new job.

Existing QA regression coverage:

```bash
python -m unittest discover -s tests -p 'test_qa_pipeline.py' -v
```

The dashboard requires no frontend build:

```bash
node --check cluster_control/web/app.js
```

Browser verification also exercised registration, cluster creation, membership
requests/approval, contributor permissions, pairing, sharing pause/resume,
owner configuration/review, a real training launch, and responsive layouts.
Layout checks used the 1280-pixel desktop preview and a 390-pixel same-origin
browser frame. Shared-preview resizing and screenshots were unreliable; the
responsive frame checked the actual dashboard without changing its source.
