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


def check_metrics(manifest, output, steps=2):
    """Require each assigned rank to complete actual forward/backward/optimizer work."""
    checked = []
    for entry in manifest['ranks']:
        rank = entry['rank']
        payload = json.loads((output / ('metrics_rank%d.json' % rank)).read_text())
        if (payload['rank'] != rank or payload['world_size'] != manifest['world_size']
                or payload['pipeline_group_size'] != manifest['world_size']
                or payload['schedule']['completed_steps'] != steps):
            raise ValueError('rank %d has incorrect topology or incomplete steps' % rank)
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


def stop_workers(processes):
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
    result = subprocess.run([
        sys.executable, str(root / 'scripts/profile_to_launch_manifest.py'),
        str(root / 'tests/fixtures' / ('probe_%drank.json' % args.world_size)),
        '--hosts', *(['127.0.0.1'] * args.world_size),
        '--port', str(port), '--log-port', str(log_port), '--output', str(manifest_path)],
        check=True, capture_output=True, text=True, timeout=args.timeout)
    (output / 'commands.sh').write_text(result.stdout, encoding='utf-8')
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    env = os.environ.copy()
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
            stop_workers(processes)
    checked = check_metrics(manifest, output)
    report = dict(passed=True, scope='local CPU synthetic training through emitted Bash commands',
                  pipeline_order=manifest['pipeline_order'], ranks=checked)
    (output / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', type=int, choices=(2, 3), default=3)
    git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
    default_bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
    parser.add_argument('--bash', default=default_bash, help='Git Bash on Windows; Bash on Linux')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output-dir', type=Path, default=root / 'logs/smoke')
    args = parser.parse_args()
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
