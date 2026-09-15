# Browser demonstration

Install the optional dependency in the existing CPU environment:

```powershell
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements-mlflow.txt
```

Start the local server in one terminal:

```powershell
.\scripts\start_mlflow.ps1
```

Then run the demonstration in another:

```powershell
.\.venv-cpu\Scripts\python.exe scripts/run_mlflow_demo.py
```

Open http://127.0.0.1:5000 and select **DT-FM demo**. The demo records
20 actual forward/backward/optimizer iterations per rank. The first two runs
use identical seeded synthetic data and model settings with different learning
rates. Select them and use Compare to inspect `last_microbatch_loss`, `iter_s`,
and `samples_per_second`. Refresh during training to see newly logged points.

The third trial uses two local CPU ranks over Gloo and has one MLflow run per
rank. Filter by the shared `launch_group` tag. Only the last pipeline rank
records loss. Open Artifacts > measurements for each rank's JSON, including
the measured communication matrices. Missing links remain JSON null.

These are smoke tests, not validation accuracy or evidence of model quality.
Loss is the final microbatch's loss, not a full-batch average. Stage timings
include communication and synchronization. Samples per second counts the
input batch once, not each repetition used for gradient accumulation, and
must not be summed across pipeline ranks. Logging adds wall-clock overhead.
The two-rank trial has one transformer layer per stage, so its total model
depth differs from the one-rank trials. Do not use it to claim speedup.

Tracking is disabled by default. To track another existing training command,
append `--mlflow-tracking-uri http://127.0.0.1:5000`,
`--mlflow-experiment NAME`, `--mlflow-run-name NAME`, and a unique
`--mlflow-group GROUP`. Use the same group on every machine in one launch.
Each rank records its own status; a finished rank alone does not establish
that the entire distributed job succeeded. Tracking failures stop the run.

The provided server binds to this computer only. A multi-machine deployment
needs a server reachable by each rank and appropriate access controls.
The SQLite database and artifacts persist locally and are ignored by Git.
No checkpoints, validation metrics, or GPU utilization are collected by this
integration. GPU and physical multi-laptop execution need separate verification.

## Validation on 2026-09-15

Tested on Windows with Python 3.12.10, PyTorch 2.13.0+cpu, and MLflow 3.16.0.
The server was started with `scripts/start_mlflow.ps1` on port 5000.

- `python scripts/run_mlflow_demo.py` completed two single-rank trials and
  one two-rank CPU/Gloo trial, with 20 forward/backward/optimizer steps per rank.
- The HTTP client verified four `FINISHED` runs for launch prefix
  `20260915T054218427423Z`. Each had steps 1 through 20 for `iter_s`,
  `forward_s`, `backward_s`, `optim_s`, `barrier_s`, and `samples_per_second`.
  Throughput matched `batch_size / iter_s`; loss was present only on the last
  pipeline rank and matched the JSON loss history.
- All four artifacts downloaded through the server matched their local JSON
  files. Both pipeline ranks stored identical directed communication matrices
  with positive off-diagonal measurements.
- An invalid model configuration, embedding dimension 64 with 3 attention
  heads, exited nonzero and left an HTTP-tracked run with status `FAILED`.
- The disabled tracking context ran with `python -S`, without site-packages.
- `python scripts/verify_comm_probe.py --world-size 3 --timeout 120` passed.
  Repeating with `--occupy-probe-rank 1` also passed. Every rank completed
  two training steps; incoming links to the occupied listener remained null.
- `python -m unittest discover -s tests -v` passed all 25 tests in the working
  tree, including the separately pending launch-manifest tests. The real-store
  tracking test covers failed and finished status, step alignment, throughput,
  artifact download, and preservation of null probe links.
  An isolated export of the staged files also passed all 13 tests present in
  that snapshot, without the pending launch-manifest files.
- In the browser, filtering by the launch prefix returned four runs. Compare
  showed learning rates 0.01 and 0.001; both loss curves rendered through step
  20 with final values 0.5496012568 and 0.7366819382. The metric page initially
  stayed on its loading screen and loaded after a reload. The rank 1 overview
  showed Finished and 20 completed steps, and its artifact preview displayed
  the saved losses and communication matrices.

These checks used synthetic data on one CPU machine. They do not establish
QQP quality, GPU behavior, or physical multi-machine/WAN operation.
