"""Exercise emitted Bash launch commands with bounded local CPU training."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.launch_manifest import validate_manifest


def check_rebalances(payload, output, rank, interval, steps):
    from scheduler.resource_scheduler import validate_resource_plan
    initial = validate_resource_plan(json.loads((output / ('resource_schedule_rank%d.json' % rank)).read_text()))
    active = initial['stage_layers']
    events = payload['rebalance_events']
    if [event['completed_steps'] for event in events] != list(range(interval, steps, interval)):
        raise ValueError('missing scheduled rebalance boundary')
    summary = []
    for event in events:
        if event['old_stage_layers'] != active or event['status'] not in ('committed', 'skipped', 'unchanged'):
            raise ValueError('invalid rebalance transition')
        if event['status'] == 'committed':
            plan = validate_resource_plan(event['resource_schedule'])
            if plan['total_layers'] != initial['total_layers'] or plan['stage_layers'] != event['new_stage_layers']:
                raise ValueError('rebalance changed model size or disagreed on assignment')
            if event['moved_layers'] <= 0 or len(event['state_checksums']) != event['moved_layers']:
                raise ValueError('migration lacks state verification')
            active = event['new_stage_layers']
        elif event['new_stage_layers'] != active:
            raise ValueError('uncommitted rebalance changed assignment')
        summary.append((event['completed_steps'], event['status'], event['old_stage_layers'], event['new_stage_layers']))
    if active != payload['resource_schedule']['stage_layers']:
        raise ValueError('final assignment differs from rebalance history')
    return summary


def check_metrics(manifest, output, steps=2):
    """Require each assigned rank to complete actual forward/backward/optimizer work."""
    checked = []
    dynamic_plan = None
    rebalance_summary = None
    for entry in manifest['ranks']:
        rank = entry['rank']
        payload = json.loads((output / ('metrics_rank%d.json' % rank)).read_text())
        if (payload['rank'] != rank or payload['world_size'] != manifest['world_size']
                or payload['pipeline_group_size'] != manifest['world_size']
                or payload['schedule']['completed_steps'] != steps):
            raise ValueError('rank %d has incorrect topology or incomplete steps' % rank)
        if 'assignment' in entry:
            model, assignment = payload['model'], entry['assignment']
            if (model['layers_per_stage'] != assignment['num_layers']
                    or model['layer_start'] != assignment['layer_start']
                    or model['layer_end'] != assignment['layer_end']
                    or model['total_layers'] != manifest['schedule']['total_layers']
                    or model['stage_layers'] != ','.join(map(str, manifest['schedule']['stage_layers']))):
                raise ValueError('rank %d did not execute its scheduled layer assignment' % rank)
        if 'dynamic_schedule' in manifest:
            from scheduler.resource_scheduler import validate_resource_plan
            plan = validate_resource_plan(payload['resource_schedule'])
            if dynamic_plan is not None and plan != dynamic_plan:
                raise ValueError('ranks did not agree on the live resource plan')
            dynamic_plan = plan
            model, stage = payload['model'], plan['stages'][rank]
            if (plan['total_layers'] != manifest['dynamic_schedule']['total_layers']
                    or len(plan['stages']) != manifest['world_size']
                    or any(model[key] != plan['model'][key] for key in ('seq_length', 'embedding_dim', 'num_heads'))
                    or model['layers_per_stage'] != stage['num_layers']
                    or model['layer_start'] != stage['layer_start']
                    or model['layer_end'] != stage['layer_end']
                    or model['total_layers'] != plan['total_layers']
                    or model['stage_layers'] != ','.join(map(str, plan['stage_layers']))):
                raise ValueError('rank %d did not execute its dynamic assignment' % rank)
            interval = manifest['dynamic_schedule'].get('rebalance_every', 0)
            if interval:
                summary = check_rebalances(payload, output, rank, interval, steps)
                if rebalance_summary is not None and summary != rebalance_summary:
                    raise ValueError('ranks disagree on rebalance history')
                rebalance_summary = summary
        for field in ('forward_incl_p2p', 'backward_incl_p2p', 'optimizer'):
            value = payload['training_time_s'][field]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('rank %d has no finite positive %s time' % (rank, field))
        if payload['latency_ms'] is not None or payload['bandwidth_mbps'] is not None:
            raise ValueError('launch unexpectedly reran the communication probe')
        losses = payload['losses']
        if (not isinstance(losses, list)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in losses)):
            raise ValueError('rank %d has invalid losses' % rank)
        if rank == manifest['world_size'] - 1 and len(losses) != steps:
            raise ValueError('final stage did not report one loss per step')
        checked.append(dict(rank=rank, measured_rank=entry['measured_rank'],
                            completed_steps=steps, losses=losses))
    return checked


def stop_workers(processes, output=None):
    for process in processes:
        if os.name == 'nt':
            if process.poll() is None:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        process.wait()
    if os.name == 'nt' and output is not None:
        # Git Bash can exit before taskkill sees its Python grandchildren.
        # Limit orphan cleanup to Python commands carrying this unique run path.
        run_path = str(output.resolve()).replace("'", "''")
        posix_path = output.resolve().as_posix().replace("'", "''")
        script = ("$workers = Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
                  "Where-Object { $_.CommandLine -and ($_.CommandLine.Contains('" + run_path +
                  "') -or $_.CommandLine.Contains('" + posix_path + "')) }; "
                  "$workers | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }")
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                       check=True, stdout=subprocess.DEVNULL, timeout=20)


def verify(args):
    root = Path(__file__).resolve().parents[1]
    parent = args.output_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='launch-%drank-' % args.world_size, dir=parent))
    print('Artifacts: %s' % output, flush=True)
    # Hold both sockets simultaneously so ephemeral ports cannot be duplicates.
    with socket.socket() as training, socket.socket() as logging:
        training.bind(('127.0.0.1', 0))
        logging.bind(('127.0.0.1', 0))
        port, log_port = training.getsockname()[1], logging.getsockname()[1]
    manifest_path = output / 'launch.json'
    schedule_args = []
    if getattr(args, 'dynamic', False):
        schedule_args = ['--dynamic', '--total-layers', str(args.total_layers)]
        if getattr(args, 'rebalance_every', 0):
            schedule_args += ['--rebalance-every', str(args.rebalance_every)]
    elif getattr(args, 'compute_profiles', None):
        schedule_args = ['--compute-profiles', *map(str, args.compute_profiles),
                         '--total-layers', str(args.total_layers)]
    elif getattr(args, 'scheduled', False):
        paths = []
        # Deliberately different capacities/timings make allocation and remapping
        # observable. These are fixtures, not claims about this CPU's performance.
        for rank in range(args.world_size):
            path = output / ('compute_rank%d.json' % rank)
            record = dict(schema_version=1,
                          model=dict(seq_length=32, embedding_dim=64, num_heads=4,
                                     batch_size=2, micro_batch_size=1, vocab_size=512),
                          rank=dict(measured_rank=rank, device='cpu', max_layers=rank + 1,
                                    layer_ms=6. / (rank + 1), first_stage_ms=1., last_stage_ms=1.))
            path.write_text(json.dumps(record), encoding='utf-8')
            paths.append(str(path))
        schedule_args = ['--compute-profiles', *paths, '--total-layers', str(sum(range(1, args.world_size + 1)))]
    result = subprocess.run([
        sys.executable, str(root / 'scripts/profile_to_launch_manifest.py'),
        str(root / 'tests/fixtures' / ('probe_%drank.json' % args.world_size)),
        '--hosts', *(['127.0.0.1'] * args.world_size),
        '--port', str(port), '--log-port', str(log_port), '--output', str(manifest_path), *schedule_args],
        check=True, capture_output=True, text=True, timeout=args.timeout)
    (output / 'commands.sh').write_text(result.stdout, encoding='utf-8')
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    env = os.environ.copy()
    for key in list(env):
        if key in ('STAGE_LAYERS', 'DYNAMIC_TOTAL_LAYERS', 'SCHEDULE_VOCAB_SIZE') or key.startswith(('SCHEDULER_', 'REBALANCE_')):
            env.pop(key)
    env.update(PYTHON=Path(sys.executable).as_posix(), LOG_DIR=output.as_posix(),
               DTFM_LOCAL='1', PROFILE='tiny', SYNTHETIC_DATA='true',
               SYNTHETIC_SAMPLES='8', SYNTHETIC_VOCAB_SIZE='512', SEQ='32',
               EMBED='64', LAYERS='1', HEADS='4', BATCH='2', MICRO='1',
               ITERS='2', EPOCHS='0', STEPS_PER_EPOCH='0', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    # Ambient shell startup files must not change the generated command.
    env.pop('BASH_ENV', None)
    env.pop('ENV', None)
    processes = []
    with ExitStack() as stack:
        try:
            for entry in manifest['ranks']:
                log = stack.enter_context((output / ('launcher_rank%d.log' % entry['rank'])).open('w'))
                processes.append(subprocess.Popen(
                    [args.bash, '--noprofile', '--norc', '-c', entry['command']],
                    cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=os.name != 'nt'))
            deadline = time.monotonic() + args.timeout
            while True:
                codes = [process.poll() for process in processes]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError('launch failed: %s; see %s' % (codes, output))
                if all(code == 0 for code in codes):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('launch exceeded timeout; see %s' % output)
                time.sleep(.1)
        finally:
            stop_workers(processes, output)
    checked = check_metrics(manifest, output)
    report = dict(passed=True, scope='local CPU synthetic training through emitted Bash commands',
                  pipeline_order=manifest['pipeline_order'], ranks=checked)
    if 'schedule' in manifest:
        report['schedule'] = manifest['schedule']
    if 'dynamic_schedule' in manifest:
        report['resource_schedule'] = json.loads((output / 'metrics_rank0.json').read_text())['resource_schedule']
    (output / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', type=int, choices=(2, 3), default=3)
    parser.add_argument('--scheduled', action='store_true', help='exercise unequal layer allocation and runtime integration')
    parser.add_argument('--dynamic', action='store_true', help='discover live CPU resources and assign stages at startup')
    parser.add_argument('--rebalance-every', type=int, default=0, help='exercise periodic resource checks between the two training steps')
    parser.add_argument('--compute-profiles', type=Path, nargs='+', help='use actual local CPU profiles instead of scheduling fixtures')
    parser.add_argument('--total-layers', type=int, help='required with --compute-profiles')
    git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
    default_bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
    parser.add_argument('--bash', default=default_bash, help='Git Bash on Windows; Bash on Linux')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output-dir', type=Path, default=root / 'logs/smoke')
    args = parser.parse_args()
    if args.rebalance_every < 0 or (args.rebalance_every and not args.dynamic):
        parser.error('--rebalance-every requires --dynamic and a nonnegative interval')
    if bool(args.compute_profiles or args.dynamic) != (args.total_layers is not None):
        parser.error('--total-layers is required with --compute-profiles or --dynamic')
    if sum(map(bool, (args.scheduled, args.compute_profiles, args.dynamic))) > 1:
        parser.error('choose --scheduled fixtures, --compute-profiles, or --dynamic')
    if not args.bash:
        parser.error('Bash is required')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    try:
        verify(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, '%s\n' % exc)


if __name__ == '__main__':
    main()
