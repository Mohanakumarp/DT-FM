# Lab experiment: setup, train, collect results

## Larger-model comparison

Use the updated source on both hosts and add `-Larger` to the rank 0 command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lab.ps1 -Rank 0 -Computers 2 -MasterIP 192.168.1.20 -Larger -Steps 20 -Warmup 3 -Repeats 3
```

This starts a 12-layer, width-512 model with sequence length 128, 8 heads,
batch size 16, microbatch size 2 and SGD learning rate 0.001, approximately 53 million parameters with
the QQP vocabulary. Each host reports physical CPU cores, available memory and
Windows GPU inventory. It checks a conservative memory estimate before any
large allocation and calibrates CPU threads using component forward/backward/
SGD measurements at 1, 2, 4, physical-core count and logical-core count.
The best measured setting is held fixed for all trials on that host. The peer
command automatically receives these settings; do not add `-Larger` there.

Both laptops run standalone baselines, one at a time. Reports compare distributed
throughput to the fastest measured standalone baseline as well as to equal
allocation. This is still CPU training; installed graphics hardware is not
automatically enabled. A preflight rejection is an estimated memory limit, not
an observed out-of-memory result. This preset does not deliberately exhaust RAM
or guarantee a distributed speedup. Increase further only after inspecting the
measured memory and throughput of this first larger configuration.

## Using two laptops

The launcher also supports two computers. Update the source files on both
computers to this version; keep the existing `.venv-lab` and downloaded data.
No setup reinstall is required if setup already succeeded.

On the first laptop, use its current LAN IPv4 address:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lab.ps1 -Rank 0 -Computers 2 -MasterIP 192.168.1.20 -Steps 10 -Warmup 2 -Repeats 1
```

Run the single worker command it prints on the second laptop. The worker
receives the topology from rank 0; it does not require `-Computers 2`.
Equal allocation uses 3 / 3 layers; resource allocation still chooses a split
from measured resources. Rank 0 then runs the single-computer reference and
generates the report. Both laptops need direct TCP access to each other.
The lab launcher explicitly binds Gloo peer sockets to the local IPv4 address
used to reach rank 0, avoiding Windows hostname resolution selecting link-local
IPv6. Initialization and individual collectives have a 90-second timeout.
The timeout does not impose a 90-second limit on the entire training run.
Omit the shortened step/repetition options for the full experiment.

The remaining instructions describe the default three-computer workflow.

Run setup once on each of three computers. Then start rank 0 and copy the two
commands it prints onto the other computers. Rank 0 coordinates every trial,
collects results automatically, and produces an offline report and a ZIP to
share with your mentor. No SSH, MLflow server, or manual copying of rank logs is
needed.

The default is a small CPU model trained from scratch on a fixed subset of real
Quora Question Pairs. This measures the training system. It is not a pretrained
foundation model, a full QQP training run, or a validation-accuracy benchmark.

## Before the lab

- Use the same source revision on all computers. These scripts must be present
  in the checkout; an older remote checkout will not contain local changes.
- Windows setup needs internet access and permission to install a per-user
  application. It installs Python if absent; no administrator account is
  normally needed. A managed lab policy may require the administrator's help.
- Use 64-bit Windows, or Linux with Bash and curl/wget. Allow a few GB of free
  disk space for Python, CPU PyTorch, data, and results. The default model is
  deliberately small for ordinary lab computers.
- Put the three computers on the same LAN, preferably wired Ethernet. They must
  be able to make TCP connections to each other. Campus Wi-Fi client isolation
  can prevent this even when all computers have internet access.

If these changes have been published to your Git remote, clone that revision:

```powershell
git clone https://github.com/Mohanakumarp/DT-FM.git
cd DT-FM
```

For an existing checkout, fetch/pull the revision containing the lab scripts
before running setup. Alternatively, copy the supplied lab source ZIP to each
computer and extract it. Run commands from the extracted project directory.
Do not copy a Windows virtual environment between computers; rerun setup.

## 1. Set up each Windows computer

Open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_lab.ps1
```

Setup performs these steps:

1. Reuses a working per-user Python 3.12 installation, or downloads the signed
   Python 3.12.10 installer from python.org and installs it for this user.
2. Creates `.venv-lab` and installs CPU PyTorch 2.13.0, NumPy 2.2.6, six 1.17.0,
   psutil 7.0.0, and Matplotlib 3.10.3.
3. Downloads QQP and the BERT cased vocabulary, selects the first 4,096 valid
   training rows, and records SHA-256 fingerprints and class counts.
4. Runs a two-process synthetic training smoke test on this computer.

The PowerShell execution-policy option applies only to this invocation. Setup
does not change the machine's persistent execution policy or firewall rules.

If Python is installed elsewhere:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_lab.ps1 -PythonPath "C:\path\to\python.exe"
```

Use matching Python patch versions on all computers. New installations use
3.12.10. Setup can be rerun after a failed download. `-SkipData -SkipSmoke` are
available for installation-only work; a real QQP run still requires the data.

## 2. Find the first computer's LAN address

On the computer that will be rank 0:

```powershell
ipconfig
```

Use its Ethernet or Wi-Fi IPv4 address, for example `192.168.1.20`. Do not use
`127.0.0.1` for a physical multi-computer run. Keep this computer awake and its
PowerShell terminal open until the report is written.

## 3. Start the experiment

On the first computer, replace the example address:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lab.ps1 -Rank 0 -MasterIP 192.168.1.20
```

It prints separate commands for rank 1 and rank 2, including a session code.
Copy the rank 1 command to the second computer and the rank 2 command to the
third. Each computer runs only its own command. The session code prevents an
unrelated client from joining the coordinator by accident; use this on a
trusted lab LAN, as HTTP traffic is not encrypted.

Once all computers join, the workflow runs:

| Configuration | What runs |
| --- | --- |
| Equal allocation | Six transformer layers, split 2 / 2 / 2 |
| Resource allocation | Same six layers; the existing scheduler measures available memory and compute time and chooses the split |
| Single computer | Same six layers on rank 0; the other computers wait |

There are three repetitions per configuration. Every trial uses 5 warmup steps
and 60 measured optimizer steps, batch size 8, microbatch size 2, sequence length
64, embedding width 128, and FP32 SGD. Data and model dimensions stay fixed.
The resource scheduler may choose an equal split if that fits the measurements.
It does not force a favorable result. Mid-training migration is not enabled.

For a short real-data rehearsal, change only the rank 0 command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lab.ps1 -Rank 0 -MasterIP 192.168.1.20 -Steps 10 -Warmup 2 -Repeats 1
```

For a network smoke test without QQP, add `-Synthetic` to rank 0. The other
computers automatically receive that setting. Synthetic reports are clearly
labeled and must not be presented as QQP results.

Useful rank 0 controls: `-Steps`, `-Warmup`, `-Repeats`, `-Layers` in multiples
of three, `-Timeout` in seconds per trial, `-ControlPort`, and `-TrainPort`.
`-Threads` is local to each computer and is recorded in the report; the default
is 2. Keep it fixed for comparisons on the same computer. Increasing model size
can exhaust RAM; begin with the defaults.

Training uses distinct batches from the fixed subset within each trial. The
requested batch count must fit the subset. For larger step budgets rerun setup
with `-Rows 16384` on all computers. The default 4,096 rows permit at most 512
total steps including warmup.

## 4. Show the results

Rank 0 prints the exact result directory under `logs/lab/` and these paths:

- `report.html`: self-contained report with embedded plots. Double-click it;
  no internet or server is needed. Browser Print can save it as PDF.
- `mentor-results.zip`: report, plots, CSVs, configuration, hardware metadata,
  rank results, commands, and log excerpts.
- `results.csv`: every trial's actual layer allocation, throughput and process
  wall time.
- `stage_results.csv`: per-stage timing and sampled process memory.
- `summary.json`: mean, sample standard deviation, min/max and repeat count.

Each computer also retains full `training.log` files in its local session
folder. The central result includes up to the last 200,000 characters per log.
The command prints a status update during long-running trials.

If report creation was interrupted after training finished, rebuild it on rank 0:

```powershell
.\.venv-lab\Scripts\python.exe scripts/lab_demo.py report "C:\path\to\logs\lab\SESSION-rank0"
```

The report refuses incomplete, failed, or inconsistent sessions. It does not
silently combine results from separate sessions.

For the review, show the question at the top, the hardware table, and the
throughput comparison. Explain the observed allocation and timing differences.
Show loss as evidence of the observed training objective, not as validation
accuracy. Do not promise a speedup before measuring one.

## What the measurements mean

- Throughput counts each example once across the pipeline. It divides measured
  examples by the slowest rank's training window after warmup. This includes
  batch loading, communication, waiting, and optimizer work.
- Process wall time also includes training-process initialization and resource
  profiling. This helps reveal scheduler overhead on short runs.
- Forward/backward/optimizer counters include communication and some waiting.
  They do not isolate pure compute. Barrier time overlaps those counters and
  is not stacked a second time.
- Memory is the sampled sum of process-tree RSS every 100 ms across startup,
  profiling and training, including the Windows venv redirector and actual
  interpreter child. Shared pages may be counted more than once. It is not
  exact peak model memory or GPU utilization. Older reports created before this
  fix may record only the small launcher process and should not support memory claims.
- Loss is the last microbatch's training loss. Partitioning can change initial
  weights even with the same seed, so this is not a convergence-equivalence
  experiment. No validation accuracy/F1 or model checkpoint is produced.
- A successful run proves completed work and records the measured performance.
  The report does not infer model quality, general speedup, or crash recovery.

## If something fails

- **Cannot join rank 0:** verify the address, session code, and that rank 0 is
  still running. On a worker, use `Test-NetConnection 192.168.1.20 -Port 8765`.
  The coordinator binds only to the rank 0 address supplied in the command.
- **All computers joined, training stalls:** examine the printed `training.log`.
  Gloo needs direct peer TCP access and uses dynamically selected ports in
  addition to the rendezvous port 29500. Allow the lab Python application on
  each computer for the lab network, with help from the administrator if
  necessary. Opening only 29500 may not be enough. Do not disable the firewall.
- **Windows firewall prompt:** allow Python on the intended trusted lab network
  if authorized by the lab. The setup script does not change firewall rules.
- **Code/runtime/data mismatch:** use the same source and setup on all computers.
  The launcher checks source bytes, Python, PyTorch, NumPy and dataset hashes.
- **CPU run is slow:** close unrelated work, use wired networking, rehearse with
  fewer steps, or increase `-Timeout` on rank 0. This does not imply a bug or a
  speedup. The printed stage timings help explain it.
- **Ctrl+C or worker failure:** the local training child is terminated; other
  workers receive a failure and stop their training children. If a launcher
  disappears entirely, remaining workers stop when the trial timeout expires.
  Logs remain, and no success report is created. Restart all three commands for
  a fresh session; this workflow does not resume checkpoints.

## Linux

```bash
bash scripts/setup_lab.sh
.venv-lab/bin/python scripts/lab_demo.py run --rank 0 --master 192.168.1.20
```

Setup installs a private uv binary and Python 3.12.10 without sudo. Copy the
Linux worker commands printed by rank 0 to the other computers. Use
`LAB_ROWS=16384 bash scripts/setup_lab.sh` for a larger subset. Linux installation
requires a supported CPU PyTorch wheel and system runtime libraries.

## Developer verification

```powershell
.\.venv-lab\Scripts\python.exe -m unittest tests.test_lab_demo -v
.\.venv-lab\Scripts\python.exe scripts/verify_lab_demo.py
.\.venv-lab\Scripts\python.exe scripts/verify_lab_demo.py --qqp
.\.venv-lab\Scripts\python.exe scripts/verify_lab_demo.py --qqp --powershell
```

The verifier launches three local workers and exercises real subprocesses,
HTTP coordination, training, automatic collection, and report generation.
Reports explicitly identify the single-host scope. This cannot verify a
physical three-computer network or the lab firewall configuration.

Validation on the development Windows machine: a fresh `.venv-lab` was created,
the setup script and its two-process smoke test completed, and the PowerShell
launcher completed real QQP equal/resource/single trials with automatic reports.
The Python launcher also completed two repetitions of each configuration. The
existing repository suite passed, as did the lab failure/collection tests.
The no-Python installer branch and Linux setup have not been executed on a bare
lab machine. Physical three-computer connectivity remains to be checked there.

Installer references: [Python Windows installation](https://docs.python.org/3.12/using/windows.html),
[Python 3.12.10](https://www.python.org/downloads/release/python-31210/),
[PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).
