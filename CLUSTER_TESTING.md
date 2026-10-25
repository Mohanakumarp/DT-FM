# How to test the DT-FM cluster console

This guide covers automated tests and a manual browser test with real CPU
workers. Start with one computer. Test two physical computers only after the
local checks pass.

The reference version is `main` at merge commit
`5e8fd4b743e39e68453c197d5e42efa537980ece`, merged on 24 October 2026 through
[PR #1](https://github.com/Mohanakumarp/DT-FM/pull/1). Commands below use Windows
PowerShell and CPU training. The merged console supports CPU and NVIDIA CUDA
training; detecting another accelerator does not mean it can run console jobs.

## 1. Use a separate test checkout

The existing `C:\Projects\DT-FM` checkout contains uncommitted macOS and device
support work. A separate clone lets you test the merged code without changing
those files.

Choose a destination that does not already exist:

```powershell
git clone --branch main https://github.com/Mohanakumarp/DT-FM.git C:\Projects\DT-FM-cluster-test
Set-Location C:\Projects\DT-FM-cluster-test
git status --short --branch
git rev-parse HEAD
```

Record the revision. To reproduce the exact merged baseline rather than a
newer version of `main`, run this before installing or testing:

```powershell
git switch --detach 5e8fd4b743e39e68453c197d5e42efa537980ece
```

Use this test checkout in every terminal below. Leave your original checkout
alone. If the destination already exists, inspect it or choose another folder
instead of overwriting it.

## 2. Prepare the CPU environment

Install Git, Python 3.12, and Node.js if they are not already available. Node is
only needed for the JavaScript syntax check. The dashboard has no frontend build
or npm dependency installation.

Run once from the test checkout:

```powershell
py -3.12 -m venv .venv-cluster
.\.venv-cluster\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-cluster\Scripts\python.exe -m pip install -r requirements-mt5.txt
New-Item -ItemType Directory -Path .cluster-data\ui-test -Force | Out-Null
.\.venv-cluster\Scripts\python.exe -m pip freeze > .cluster-data\ui-test\python-packages.txt
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker inspect
```

Expected inspection output includes:

```json
{
  "runtime_ready": true,
  "runtime_error": "",
  "devices": [
    { "id": "cpu", "backend": "cpu" }
  ]
}
```

Additional fields are normal. Stop and resolve any `runtime_error` before
trying to train. Always use `.\.venv-cluster\Scripts\python.exe` for workers and
tests, so a different Python installation cannot hide missing dependencies.

The Windows PyTorch 2.8.0 CPU wheel failed Gloo transport creation during the
merge validation despite reporting Gloo as available. The Windows setup above
uses the verified 2.13.0 CPU version. See the
[upstream issue](https://github.com/pytorch/pytorch/issues/150381).

## 3. Run the automated cluster tests

These tests start their own temporary control servers and workers. You do not
need to start the dashboard, create accounts, or pair a worker first.

```powershell
.\.venv-cluster\Scripts\python.exe -m unittest discover -s tests -p 'test_cluster_*.py' -v 2>&1 | Tee-Object -FilePath .cluster-data\ui-test\cluster-tests.log
Write-Output "Test exit code: $LASTEXITCODE"
node --check cluster_control/web/app.js
Write-Output "JavaScript check exit code: $LASTEXITCODE"
```

At the reference revision, a complete cluster run ends with `Ran 21 tests` and
`OK`, with exit code `0`. The JavaScript check succeeds silently with exit code
`0`.

The cluster suite covers accounts, sessions, CSRF and origin checks, membership
approval and revocation, pairing, sharing consent, device reservations, job
lifecycle, and process shutdown. Its two worker integration tests also perform
real two-rank Gloo training, verify finite losses and checkpoints, cancel an
active job, and reuse its devices for another job. They force pairing codes
that start with `-` to cover the CLI argument edge case.

If those two tests are **skipped**, the run has not proved worker training.
Check `worker inspect` and the interpreter path, install the training
dependencies, and rerun the suite. Cold imports can take time; wait for the
final unittest summary rather than judging an empty terminal as a failure.

To rerun only the real worker tests after a failure:

```powershell
.\.venv-cluster\Scripts\python.exe -m unittest discover -s tests -p 'test_cluster_worker.py' -v
```

They use temporary directories and remove their training artifacts during
cleanup. Use the manual flow below when you want to inspect retained files.

## 4. Start an isolated dashboard

In **terminal 1**:

```powershell
Set-Location C:\Projects\DT-FM-cluster-test
.\.venv-cluster\Scripts\python.exe -m cluster_control.server --host 127.0.0.1 --port 8090 --database .cluster-data/ui-test/control.sqlite3
```

Leave that terminal open. In another terminal, check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/api/health
```

Expected result: `status` is `ok`.

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). The page should show the
sign-in form. The health endpoint confirms that the server responds; the next
steps check the actual UI and training behavior.

All manual-test accounts, worker identities, logs, and artifacts in this guide
use `.cluster-data/ui-test`. Restarting with the same database preserves them.
For a fresh test later, choose another test-data folder consistently in the
server and worker commands.

## 5. Test accounts and membership

Use two independent browser sessions, such as a normal window for the owner and
a private window for the contributor. Two tabs in the same browser profile
usually share the same login cookie.

| Step | Action | Pass condition |
|---|---|---|
| Owner account | Click **Create an account** and register a disposable test account. Use a password of at least 12 characters. | Registration opens the cluster directory. |
| Sign-in | Sign out, then sign back in with that account. | The same account and directory load. |
| Cluster | As owner, click **Create cluster**, name it `UI test lab`, and add a description. | The cluster opens and labels the account as owner. |
| Contributor | In the independent browser session, register a second disposable account. | The contributor can find `UI test lab` in the directory. |
| Request | As contributor, click **Request to join**. | The directory shows pending membership and awaiting approval. Cluster training details are not available yet. |
| Approval | As owner, open **Members** and approve the contributor. | The member changes from pending to approved. |
| Contributor access | Refresh or reopen the cluster as contributor, then open **Training**. | The contributor can view training history, but has no owner configuration or start controls. |

Keep both sessions available for the remaining checks.

## 6. Pair and run the first CPU worker

As the **owner**, click **Connect computer**. Generate one code for this worker.
Pairing codes expire after ten minutes and are single use.

In **terminal 2**:

```powershell
Set-Location C:\Projects\DT-FM-cluster-test
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker pair --server http://127.0.0.1:8090 --address 127.0.0.1 --name "Test CPU A" --devices cpu --state .cluster-data/ui-test/worker-a.json
```

Paste the code at the private prompt. Then start the worker:

```powershell
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker run --state .cluster-data/ui-test/worker-a.json --output .cluster-data/ui-test/artifacts-a
```

Leave it running. In **Devices**, `Test CPU A` should appear online and shared,
without a runtime error. As owner, pause and resume this computer's sharing.
The displayed contribution state should change accordingly.

The contributor should be able to see this computer but should not have a pause
control for it. Sharing belongs to the user who paired the worker.

Worker identity files contain bearer tokens. Keep them out of screenshots,
shared logs, and Git commits. If an identity already exists, run that worker
with its existing `--state`, or choose a new state filename and a fresh code
when pairing another identity.

## 7. Run a two-step training job from the UI

As owner, open **Training** and choose these settings:

| Setting | Value |
|---|---|
| Training workflow | Offline smoke · no downloads |
| Model | Tiny random BERT |
| Dataset | `synthetic` |
| Stages per pipeline | `1` |
| Pipeline replicas | `1` |
| Batch size per replica | `2` |
| Microbatch size | `1` |
| Epochs | `2` |
| Maximum optimizer steps | `2` |
| Learning rate | `0.00005` |
| Checkpoint interval | `1` |
| Advanced: sequence length | `16` |
| Advanced: document stride | `4` |
| Advanced: coordinator TCP port | `29500`, or another unused port |
| Advanced: preparation / communication timeout | `120` seconds |

The offline workflow generates its tiny model and synthetic data locally. It
does not download a pretrained model or dataset after dependency installation.

Select `Test CPU A`, click **Review assignments**, and verify that it is assigned
to rank 0. Click **Start training**.

Pass conditions:

- The job progresses through preparation and running to `completed`.
- Completed steps reaches `2` and the rank state is `succeeded`.
- The displayed loss is finite, and the loss chart contains optimizer-step data.
- Throughput appears after the summary is produced.
- Worker logs show completion, and checkpoint files exist locally.

Startup includes library imports and preparation. The actual optimizer steps
can finish quickly. Do not require a particular loss value, decreasing loss,
or throughput value for this synthetic functional test.

After the job completes, inspect the retained files in a spare terminal:

```powershell
$LatestJob = Get-ChildItem .cluster-data\ui-test\artifacts-a -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content -LiteralPath (Join-Path $LatestJob.FullName 'rank0\summary.json')
Get-ChildItem -LiteralPath (Join-Path $LatestJob.FullName 'rank0') -Filter 'checkpoint*.pt'
Get-Content -LiteralPath (Join-Path $LatestJob.FullName 'rank0\metrics.jsonl') | Select-String 'train_loss'
```

The summary should report `completed_steps: 2`, checkpoint files should be
nonempty, and metrics should contain numeric losses for the completed steps.
Each job has its own full ID as a directory name; the UI may show a shortened ID.

## 8. Test two local workers and rank assignment

As the **approved contributor**, click **Connect computer** and generate a new
code. In **terminal 3**:

```powershell
Set-Location C:\Projects\DT-FM-cluster-test
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker pair --server http://127.0.0.1:8090 --address 127.0.0.1 --name "Test CPU B" --devices cpu --state .cluster-data/ui-test/worker-b.json
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker run --state .cluster-data/ui-test/worker-b.json --output .cluster-data/ui-test/artifacts-b
```

Paste the contributor's code when the pairing command prompts. Both workers
must keep running. Multiple CPU agents on the same computer are allowed for
this local demonstration.

Repeat the training settings from section 7, changing **Stages per pipeline**
to `2` and keeping **Pipeline replicas** at `1`.

First select only one device. **Review assignments** should remain disabled
because two ranks require two devices. Then select CPU A followed by CPU B.
Review should show A at rank 0 and B at rank 1. Selection order determines
rank order; uncheck and recheck a device to move it to the end.

Start the job. Both ranks should succeed, each with a summary reporting two
completed steps and checkpoints under its own worker output directory:

```text
.cluster-data/ui-test/artifacts-a/<job-id>/rank0/
.cluster-data/ui-test/artifacts-b/<job-id>/rank1/
```

This proves two local worker processes can train together. It does not prove
connectivity between physical computers.

## 9. Test stopping, withdrawal, and recovery

Use only the disposable test cluster and workers from this guide. Before each
case, make sure the devices are online and sharing is enabled.

| Case | Action | Pass condition |
|---|---|---|
| Owner cancellation | Create a two-worker smoke job with epochs and step cap set to `10000`. Wait until an optimizer step is reported, then click **Stop run** as owner. | The job becomes `cancelled`, ranks stop, and reservations are released after worker acknowledgement. |
| Device reuse | Immediately configure another two-step job after cancellation finishes. | The same devices can be selected and the new job completes. |
| Contributor withdrawal | During another long job, pause CPU B as its contributor. | The job fails, its peer rank stops, and it does not continue indefinitely with a missing rank. Resume sharing before testing again. |
| Worker shutdown | During a long job, press Ctrl+C in one worker terminal. | That worker stops its local training. The cluster fails the affected job and stops its peer. Restart the worker using the same state file before another job. |
| Revocation | As owner, revoke the disposable contributor under **Members**. | That contributor loses cluster access and its old worker token no longer works. Any job using its contribution is stopped. |
| Reapproval | Reapprove that contributor and try the old worker identity. | Reapproval does not restore revoked tokens. Generate a fresh pairing code and use a new state filename to reconnect. |
| Server persistence | With all jobs idle, stop terminal 1 using Ctrl+C and restart it with the same database command. | Accounts, membership decisions, and job history remain. Sign in again if needed. |

A withdrawn worker may be detected through a heartbeat or lease timeout. Allow
for the documented timeout rather than expecting an immediate update. The
control server detects missing heartbeats after about 30 seconds; workers stop
local training after roughly ten seconds of lost control connectivity, plus
bounded request and process-termination time.

The dashboard does not automatically resume failed jobs or replace ranks.
Checkpoint presence alone is not evidence of automatic recovery.

## 10. Check desktop, mobile, and empty states

Use browser developer tools to inspect a desktop viewport of about 1280×900 and
a mobile viewport of about 390×844.

- Before pairing, the cluster explains how to connect a computer.
- Before any run, training history has a useful empty state.
- Offline and paused devices have distinct visible states.
- Form labels, actions, dialogs, and error messages remain readable on mobile.
- The page has no horizontal overflow. A wide rank table may scroll inside its
  own container.
- The run view shows configuration, ranks, progress, and logs without a blank
  state after navigation.
- The contributor view omits owner-only configuration and start actions.
- Record unexpected browser console errors and failed network requests.

## 11. Optional: reproduce the full Linux test suite with Docker

Use the clean test checkout and a running Linux Docker engine. This is optional
for the local UI walkthrough, but it reproduces the environment used for the
full regression suite.

Run these commands in a separate PowerShell terminal. The image and Compose
project names are dedicated to this test:

```powershell
Set-Location C:\Projects\DT-FM-cluster-test
$env:DTFM_IMAGE = 'dtfm-ui-test'
docker version
docker compose -p dtfm-ui-test build train
docker compose -p dtfm-ui-test run --rm --no-deps train test
```

At the reference revision, the full suite reports `Ran 108 tests` and `OK`.
The suite includes real multi-rank QA and summarization tests and can take
several minutes. Check Docker's exit code and the final test summary.

For the other Linux CI checks, follow
[the Docker workflow](.github/workflows/docker.yml) and
[Docker instructions](DOCKER.md). A successful CUDA image build proves that
the image builds; GPU execution needs a separate test on an NVIDIA machine.

When finished with this dedicated Compose project:

```powershell
docker compose -p dtfm-ui-test down
```

This command does not remove its named volumes. Keep evidence and datasets you
need for later comparison.

## 12. Optional: test two physical Windows computers

Run sections 1–3 on both computers with the same revision and dependency
versions. Use a working private network or the project's
[Tailscale setup](QA_TAILSCALE.md).

On the control-service computer, stop the local test server and restart it with
the same test database, bound to its reachable private IP:

```powershell
.\.venv-cluster\Scripts\python.exe -m cluster_control.server --host SERVER_PRIVATE_IP --port 8090 --database .cluster-data/ui-test/control.sqlite3
```

Replace `SERVER_PRIVATE_IP` before running the command. Each browser and worker
must now use `http://SERVER_PRIVATE_IP:8090`.

Pair new test worker identities on both computers. In each worker terminal,
replace the two values below with actual addresses:

```powershell
$ServerUrl = 'http://SERVER_PRIVATE_IP:8090'
$WorkerAddress = 'THIS_COMPUTER_PRIVATE_IP'
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker pair --server $ServerUrl --address $WorkerAddress --name "Network test CPU" --devices cpu --state .cluster-data/ui-test/network-worker.json
.\.venv-cluster\Scripts\python.exe -m cluster_control.worker run --state .cluster-data/ui-test/network-worker.json --output .cluster-data/ui-test/network-artifacts
```

The addresses serve different purposes:

| Value | Meaning |
|---|---|
| `--server` | The same control-service URL on every computer. |
| `--address` | This particular worker's address, reachable by the other training ranks. |
| Coordinator port | The training port on the rank-0 worker, such as `29500`. |

Do not use `127.0.0.1` as a physical worker's advertised address for this test.
Use only the two network-test workers in the job, with two stages and one
replica. Stop the old local agents if you no longer need them.

Check control-service reachability from the second computer:

```powershell
Invoke-RestMethod "$ServerUrl/api/health"
```

After starting a job, verify that both ranks prepare, establish Gloo
communication, complete optimizer steps, and save checkpoints on their own
computers. Reaching the dashboard or passing preparation does not prove tensor
communication works. Firewall and network rules must allow worker-to-worker
Gloo connections, including its peer connections, not just the dashboard and
coordinator ports.

Physical multi-computer and GPU training were not established by the October
24 merge validation. Record a separate result when performing this test.

## 13. Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named torch`, `transformers`, or `datasets` | Use the `.venv-cluster` executable and rerun the dependency installation and `worker inspect`. |
| `runtime_ready: false` | Read `runtime_error`. Hardware discovery alone is insufficient. |
| `unsupported gloo device` on Windows | Verify that this worker uses the documented PyTorch 2.13.0 CPU environment. An `is_gloo_available()` result alone does not prove transport creation works. |
| `argument --code: expected one argument` | Use the private pairing prompt. If scripting a disposable test code, pass it as `--code=<value>` so a leading dash is preserved as a value. |
| Pairing code rejected | Generate a fresh code from the intended approved account. Each code is single use and expires after ten minutes. |
| Worker identity already exists | Run the existing identity with `worker run --state ...`, or choose a new state path and fresh pairing code. |
| Worker is offline | Keep `worker run` open. Check the stored server URL, worker terminal errors, and network access. Pairing alone does not keep a worker online. |
| No eligible devices | Check approval, worker heartbeat, runtime readiness, sharing state, supported backend, and existing reservations. |
| Review is disabled | Select exactly `stages × replicas` eligible devices. |
| Address already in use | Pick an unused dashboard or coordinator port and use it consistently. Do not terminate an unrelated process to free the port. |
| Job prepares but fails at startup | Inspect the full worker log for Gloo, address resolution, firewall, or model errors. All workers being ready is not proof of collective connectivity. |
| Interface errors on native Windows | Leave `--interface` unset unless that interface was validated. Check for an inherited Linux-only `GLOO_SOCKET_IFNAME`, such as `tailscale0`. |
| Throughput is blank while training | The runner supplies throughput when it produces its summary. Check again after completion. |
| No checkpoint download button | Artifacts remain in each worker's output directory. The console has no artifact-download feature in this baseline. |

Full training output is stored under each worker's chosen output directory.
Look for `worker-rank*.log`, `rank*/metrics.jsonl`, `rank*/summary.json`, and
`rank*/checkpoint*.pt`. Preparation logs are kept under `preflight/`. The API
retains bounded logs, so local files are the source for a complete traceback.

## 14. Record the result and stop test services

Use this checklist for a test report:

- [ ] Revision and operating system recorded.
- [ ] Python and package versions saved.
- [ ] Cluster suite passed, including both real worker tests without skips.
- [ ] JavaScript syntax check passed.
- [ ] Owner and contributor accounts, membership, and permissions checked.
- [ ] Pairing and sharing pause/resume checked.
- [ ] One-worker and two-worker smoke jobs completed.
- [ ] Finite loss, completed steps, summaries, and checkpoint files verified.
- [ ] Cancellation and immediate device reuse checked.
- [ ] Withdrawal, revocation, and persistence checked, or explicitly marked untested.
- [ ] Desktop and mobile screenshots captured.
- [ ] Failures include the job ID, affected rank, full local traceback, and steps to reproduce.
- [ ] Physical multi-computer and GPU results are reported separately from local CPU and image-build results.

Stop active jobs first. Press Ctrl+C in both worker terminals, then stop the
control server. Keep the logs and checkpoints you need. Do not share worker
identity JSON files or database files as public test evidence.

The recorded baseline passed 21 Windows cluster tests, 108 Linux tests, browser
checks, CPU and two-container smoke training, the MLflow demo, and a CUDA image
build. See the
[successful pull-request workflow](https://github.com/Mohanakumarp/DT-FM/actions/runs/40061283961)
and [cluster console reference](CLUSTER_CONSOLE.md) for supporting detail.
