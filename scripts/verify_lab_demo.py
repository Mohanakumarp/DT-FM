"""Run the actual lab launcher with three local worker processes, bounded in time."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qqp', action='store_true')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--powershell', action='store_true', help='execute the Windows entry point on all ranks')
    parser.add_argument('--computers', type=int, choices=(2, 3), default=3)
    parser.add_argument('--all-single', action='store_true')
    parser.add_argument('--master', default='127.0.0.1', help='local bind address for the verification run')
    args = parser.parse_args()
    output = ROOT / 'logs' / 'lab' / ('verification-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    control, train = free_port(), free_port()
    while train == control:
        train = free_port()
    token = secrets.token_hex(12)
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    processes = []
    with ExitStack() as stack:
        try:
            for rank in range(args.computers):
                command = [sys.executable, '-u', str(ROOT / 'scripts/lab_demo.py'), 'run',
                           '--rank', str(rank), '--computers', str(args.computers), '--master', args.master, '--session-code', token,
                           '--control-port', str(control), '--train-port', str(train),
                           '--steps', str(args.steps), '--warmup', '1', '--repeats', str(args.repeats), '--timeout', '120']
                if not args.qqp:
                    command.append('--synthetic')
                if args.all_single:
                    command.append('--all-single')
                if args.powershell:
                    if args.all_single:
                        raise ValueError('--all-single verification uses the Python entry point')
                    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                               str(ROOT / 'scripts/start_lab.ps1'), '-PythonPath', sys.executable,
                               '-Rank', str(rank), '-Computers', str(args.computers), '-MasterIP', args.master, '-SessionCode', token,
                               '-ControlPort', str(control), '-TrainPort', str(train),
                               '-Steps', str(args.steps), '-Warmup', '1', '-Repeats', str(args.repeats), '-Timeout', '120']
                    if not args.qqp:
                        command.append('-Synthetic')
                stream = stack.enter_context((output / ('launcher%d.log' % rank)).open('w', encoding='utf-8'))
                processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT))
                if rank == 0:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        try:
                            with socket.create_connection((args.master, control), timeout=.5):
                                break
                        except OSError:
                            time.sleep(.2)
                    else:
                        raise RuntimeError('coordinator did not start')
            deadline = time.monotonic() + 180 + args.repeats * 180
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('worker failed; inspect ' + str(output))
                if time.monotonic() > deadline:
                    raise TimeoutError('launcher did not finish')
                time.sleep(.5)
            if any(process.returncode != 0 for process in processes):
                raise RuntimeError('worker failed; inspect ' + str(output))
        finally:
            for process in processes:
                if process.poll() is None:
                    # Launcher owns a training subprocess: terminate the whole tree in this test.
                    import psutil
                    try:
                        children = psutil.Process(process.pid).children(recursive=True)
                        for child in children:
                            child.terminate()
                        process.terminate()
                        _, alive = psutil.wait_procs(children, timeout=5)
                        for child in alive:
                            child.kill()
                        process.wait(timeout=10)
                    except psutil.NoSuchProcess:
                        pass
    log = (output / 'launcher0.log').read_text(encoding='utf-8')
    report_line = next(line for line in log.splitlines() if line.startswith('Report: '))
    report = Path(report_line.removeprefix('Report: '))
    assert report.is_file() and report.with_name('mentor-results.zip').is_file()
    summary = json.loads(report.with_name('summary.json').read_text())
    assert set(summary) == {'single', 'equal', 'resource'} | ({'single_rank%d' % r for r in range(1,args.computers)} if args.all_single else set())
    assert 'Only 1 distinct hostnames' in report.read_text(encoding='utf-8')
    print('PASS: actual local %d-rank %s training, equal/resource/single trials and report generation.' % (args.computers, 'QQP' if args.qqp else 'synthetic'))
    print('Report:', report)
    print('Launcher logs:', output)


if __name__ == '__main__':
    main()
