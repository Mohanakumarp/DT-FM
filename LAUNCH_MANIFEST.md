# Profile to launch manifest

Generate a manifest from the `verification.json` written by
`scripts/verify_comm_probe.py`. The adapter uses only the Python standard library.

```bash
python scripts/profile_to_launch_manifest.py path/to/verification.json \
  --hosts 100.64.0.10 100.64.0.11 100.64.0.12 \
  --devices cpu rocm directml --output launch.json
```

Hosts and optional devices must follow the original measured rank order. Devices
default to CPU. Addresses are supplied explicitly because verification JSON does
not identify physical hosts. Multiple ranks may share a host. Run each printed
command on its indicated host from the repository root, with the existing
`scripts/run_rank.sh` environment and dataset prerequisites installed. Commands
are Bash commands, including when generated on Windows. Nothing is launched by
the adapter. The launcher's model, data, batch and iteration environment settings
still apply. Local Linux runs can set `DTFM_LOCAL=1` as usual.

The version 1 manifest preserves the entire input profile, including directed
`latency_ms` and `bandwidth_mbps` matrices in measured rank coordinates. Off-diagonal
null pairs remain unavailable; zeros are allowed only on the diagonal. Validation
rejects failed verification, malformed dimensions, mismatched availability,
booleans, nonpositive measurements and nonfinite numbers.

The adapter finds the minimum-cost path covering every measured rank, requiring
available links in both directions between adjacent stages for GPipe activations
and gradients. It sums `RTT_ms + payload_bytes * 8 / (Mbps * 1000)` over both
directions of every boundary, with a default 1 MiB payload in each direction.
Use `--payload-bytes` to change that assumption. This is a ranking proxy using
full measured RTT and echo-based bandwidth, not a prediction of iteration time
or a conversion to measured one-way latency. Equal scores use lexicographic
measured-rank order. Exact dynamic programming supports 2 through 12 ranks;
larger inputs and profiles with no full path fail explicitly.

`pipeline_order` lists measured ranks by stage. Each `ranks` entry records the
original measured rank, host, device, new launch rank, environment and command.
The first stage becomes rank 0 and its host becomes `MASTER_IP` everywhere.
Commands pin `WORLD_SIZE=PP_SIZE`, `DP_SIZE=1`, Gloo and probe skipping. Ports
default to 9000 and 9100 and can be changed with `--port` and `--log-port`.
No scheduler or runtime pipeline code is changed; existing equal stage sizing
still applies.

Before writing, the CLI serializes and checks the JSON by recomputing the selected
path, mapping and commands. Consumers can call
`utils.launch_manifest.validate_manifest` to perform the same check. This checks
internal consistency, not profile authenticity or freshness. The raw TCP probe
does not establish Gloo collective connectivity, port availability, accelerator
memory capacity or multi-host training success. Unavailable nonadjacent probe
links are retained and are not used as pipeline boundaries; Gloo initialization
and collectives still require a working cluster network.

Synthetic 2- and 3-rank fixture tests:

```bash
python -m unittest discover -s tests -p test_launch_manifest.py
```

## Run the emitted commands locally

The bounded smoke runner generates a fixture manifest through the CLI, validates
it, and executes its exact per-rank Bash commands with small synthetic CPU batches:

```bash
python scripts/verify_launch_manifest.py --world-size 2
python scripts/verify_launch_manifest.py --world-size 3
```

Use a Python environment with the DT-FM training dependencies installed. On
Windows, run these with `.venv-cpu\Scripts\python.exe`; the verifier selects Git
Bash from its standard installation location. `--bash` can specify another Git
Bash executable. On Linux, it uses Bash from PATH.

The launcher accepts `PYTHON` to use an explicit interpreter without activating
Conda, and `LOG_DIR` to isolate logs and metrics. Their defaults preserve the
existing launch behavior. Local Git Bash runs let Gloo select the Windows
interface rather than forcing Linux's `lo` interface.

Each verification run creates a fresh directory under `logs/smoke/` containing
`launch.json`, `commands.sh`, launcher logs, rank logs, metrics and a final
`verification.json`. Success requires every rank to exit successfully, complete
two steps, and record finite positive forward, backward and optimizer times.
The final stage must record two finite losses. The three-rank fixture exercises
the nonidentity order `[0, 2, 1]`. `--timeout` defaults to 120 seconds; failure or
timeout stops the launched process trees, retains diagnostic logs, and produces
no success report. This verifies local launch execution, not the fixture's
simulated network measurements or physical multi-host training.
