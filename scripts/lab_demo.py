"""One command per computer: coordinated CPU trials and automatic result collection.

The small authenticated HTTP coordinator carries configuration and results only.
Training activations and gradients use the existing Gloo pipeline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.lab_data import DATA, digest
from scripts.lab_hardware import model_fields, inventory, memory_preflight, calibrate_threads


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def code_digest():
    value = hashlib.sha256()
    paths = [ROOT / 'dist_runner.py']
    for folder in ('comm', 'modules', 'pipeline_parallel', 'data_parallel', 'utils', 'scheduler', 'task_datasets', 'scripts'):
        paths.extend((ROOT / folder).rglob('*.py'))
    for path in sorted(paths):
        value.update(path.relative_to(ROOT).as_posix().encode())
        value.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return value.hexdigest()


def fingerprint(synthetic):
    import torch
    import numpy
    import psutil
    data = {'dataset': 'synthetic'}
    if not synthetic:
        manifest = DATA / 'manifest.json'
        if not manifest.is_file():
            raise ValueError('QQP is missing. Run setup_lab.ps1 or lab_data.py first.')
        data = json.loads(manifest.read_text(encoding='utf-8'))
        for name, key in (('train.tsv', 'train_sha256'), ('vocab.txt', 'vocab_sha256')):
            if digest(DATA / name) != data[key]:
                raise ValueError('QQP file checksum differs from manifest: ' + name)
    return dict(code_sha256=code_digest(), python=platform.python_version(),
                torch=torch.__version__, numpy=numpy.__version__, data=data,
                hostname=socket.gethostname(), platform=platform.platform(), cpu=platform.processor(),
                logical_cpus=os.cpu_count(), ram_bytes=psutil.virtual_memory().total)


def compatible(first, other):
    return all(first[key] == other[key] for key in ('code_sha256', 'python', 'torch', 'numpy', 'data'))


def make_jobs(repeats, computers=3, all_single=False):
    jobs = []
    # Alternate order to reduce consistently favoring the later, warmer run.
    for repeat in range(repeats):
        modes = ['equal', 'resource'] if repeat % 2 == 0 else ['resource', 'equal']
        baselines = ['single'] + (['single_rank%d' % r for r in range(1, computers)] if all_single else [])
        for mode in modes + baselines:
            single_rank = 0 if mode == 'single' else int(mode.removeprefix('single_rank')) if mode.startswith('single_rank') else None
            jobs.append(dict(id=len(jobs), mode=mode, repeat=repeat + 1,
                             ranks=[single_rank] if single_rank is not None else list(range(computers))))
    return jobs


def validate_result(result, config, job, rank):
    metrics = result['metrics']
    size = len(job['ranks'])
    local_rank = job['ranks'].index(rank)
    budget = config['steps'] + config['warmup']
    if (metrics['rank'] != local_rank or metrics['world_size'] != size
            or metrics['pipeline_group_size'] != size
            or metrics['schedule']['completed_steps'] != budget
            or len(metrics['iterations']) != budget):
        raise ValueError('wrong rank/topology or incomplete optimizer steps')
    model = metrics['model']
    expected = model_fields(config)
    if (model['total_layers'] != config['layers'] or any(model[key] != expected[key]
            for key in ('embedding_dim', 'seq_length', 'num_heads'))):
        raise ValueError('training dimensions differ from experiment')
    counts = [int(n) for n in model['stage_layers'].split(',')]
    if len(counts) != size or min(counts) < 1 or sum(counts) != config['layers']:
        raise ValueError('invalid layer assignment')
    if job['mode'] != 'resource' and counts != [config['layers'] // size] * size:
        raise ValueError('actual layer allocation differs from the requested baseline')
    if job['mode'] == 'resource':
        plan = metrics.get('resource_schedule') or {}
        if plan.get('stage_layers') != counts or plan.get('total_layers') != config['layers']:
            raise ValueError('actual layer allocation differs from the resource plan')
    if model['layers_per_stage'] != counts[local_rank]:
        raise ValueError('rank did not train its assigned layers')
    for item in metrics['iterations']:
        for key in ('iter_s', 'forward_s', 'backward_s', 'optim_s', 'train_elapsed_s'):
            if not math.isfinite(item[key]) or item[key] <= 0:
                raise ValueError('missing or nonfinite training timing: ' + key)
    losses = metrics['losses']
    if local_rank == size - 1 and (len(losses) != budget or not all(math.isfinite(v) for v in losses)):
        raise ValueError('last stage did not produce a finite loss at every step')


class Coordinator:
    def __init__(self, config, output, token):
        self.config, self.output, self.token = config, output, token
        self.lock = threading.RLock()
        self.members, self.received, self.acks = {}, {}, set()
        self.computers = config.get('computers', 3)
        self.jobs = make_jobs(config['repeats'], self.computers, config.get('all_single', False))
        self.current = 0
        self.error = None
        self.changed = time.monotonic()
        write_json(output / 'config.json', config)

    def state(self):
        if self.error:
            return dict(status='failed', error=self.error)
        if self.current == len(self.jobs):
            return dict(status='complete')
        if len(self.members) != self.computers:
            return dict(status='waiting', connected=sorted(self.members))
        return dict(status='running', job=self.jobs[self.current])

    def fail(self, message):
        if not self.error:
            self.error = message
            write_json(self.output / 'failure.json', dict(error=message, job=self.current))
            print('FAILED:', message, flush=True)

    def dispatch(self, path, body):
        with self.lock:
            if path == '/config':
                return self.config
            if path == '/state':
                return self.state()
            rank = body.get('rank')
            if type(rank) is not int or rank not in range(self.computers):
                raise ValueError('rank is outside this session topology')
            if path == '/ack':
                self.acks.add(rank)
            elif path == '/fail':
                self.fail('rank %d: %s' % (rank, str(body.get('error'))[:2000]))
            elif path == '/register':
                member = body['member']
                if rank in self.members:
                    raise ValueError('this rank is already registered; use one terminal per rank')
                if self.members and not compatible(next(iter(self.members.values())), member):
                    self.fail('Code, Python, PyTorch, NumPy, or QQP differs between computers. Rerun the same setup on all three.')
                    raise ValueError(self.error)
                if member['data'].get('rows', 10**9) < model_fields(self.config)['batch_size'] * (self.config['steps'] + self.config['warmup']):
                    raise ValueError('QQP subset too small for step budget; rerun setup with more -Rows')
                self.members[rank] = member
                write_json(self.output / ('machine_rank%d.json' % rank), member)
                self.changed = time.monotonic()
                print('Connected rank %d: %s' % (rank, member['hostname']), flush=True)
                if 'hardware' in member:
                    print('Rank %d hardware: %s' % (rank, json.dumps(member['hardware'])), flush=True)
                    print('Rank %d selected CPU threads: %d' % (rank, member['threads']), flush=True)
            elif path == '/result':
                job_id = body['job_id']
                if job_id != self.current or self.current >= len(self.jobs):
                    raise ValueError('result does not belong to current job')
                job = self.jobs[self.current]
                if rank not in job['ranks'] or rank in self.received:
                    raise ValueError('unexpected or duplicate result')
                result = body['result']
                validate_result(result, self.config, job, rank)
                if self.received:
                    previous = next(iter(self.received.values()))['metrics']['model']['stage_layers']
                    if previous != result['metrics']['model']['stage_layers']:
                        raise ValueError('ranks disagree on actual layer allocation')
                folder = self.output / ('%02d-%s-%d' % (job_id, job['mode'], job['repeat']))
                write_json(folder / ('rank%d.json' % rank), result)
                self.received[rank] = result
                if set(self.received) == set(job['ranks']):
                    print('Completed %s, repetition %d' % (job['mode'], job['repeat']), flush=True)
                    self.current += 1
                    self.received = {}
                    self.changed = time.monotonic()
            else:
                raise ValueError('unknown endpoint')
            return self.state()


def handler_for(coordinator):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            # Serve only the explicitly prepared source patch, never directory
            # listings, arbitrary local paths, datasets, or environment files.
            self.connection.settimeout(15)
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + coordinator.token):
                self.send_error(403)
                return
            update = ROOT / 'logs' / 'lab' / 'source-update.zip'
            if self.path != '/source-update.zip' or not update.is_file():
                self.send_error(404)
                return
            payload = update.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self.connection.settimeout(15)
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + coordinator.token):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError('invalid request length')
                body = json.loads(self.rfile.read(length))
                answer = coordinator.dispatch(self.path, body)
                payload = json.dumps(answer, allow_nan=False).encode()
                self.send_response(200)
            except (ValueError, KeyError, TypeError) as exc:
                payload = json.dumps(dict(error=str(exc))).encode()
                self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    return Handler


class Client:
    def __init__(self, master, port, token):
        self.url = 'http://%s:%d' % (master, port)
        self.token = token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, path, body=None):
        request = urllib.request.Request(self.url + path, data=json.dumps(body or {}).encode(),
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('Coordinator rejected request: ' + exc.read().decode()) from exc


def local_ipv4_for(master, port):
    # UDP connect only asks the OS for its route; no packet is sent.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((master, port))
        address = route.getsockname()[0]
    if address == '0.0.0.0':
        raise ValueError('could not select a local IPv4 route to rank 0')
    return address


def command_for(config, job, rank, output):
    size = len(job['ranks'])
    model = model_fields(config)
    bind_address = config['master'] if rank == 0 else local_ipv4_for(config['master'], config['train_port'])
    command = [sys.executable, '-u', str(ROOT / 'dist_runner.py'),
        '--device', 'cpu', '--tensor-comm', 'gloo', '--dist-backend', 'gloo',
        '--gloo-bind-address', bind_address, '--dist-timeout-seconds', '90',
        '--dist-url', 'tcp://%s:%d' % (config['master'], config['train_port']),
        '--world-size', str(size), '--pipeline-group-size', str(size), '--data-group-size', '1',
        '--rank', str(job['ranks'].index(rank)), '--lr', str(config.get('lr', .01)),
        '--seed', str(100 + job['repeat']), '--num-iters', str(config['steps'] + config['warmup']),
        '--skip-comm-probe', 'true', '--use-offload', 'false', '--metrics-dir', str(output)]
    for key, value in model.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    if job['mode'] == 'resource':
        command += ['--dynamic-total-layers', str(config['layers'])]
    else:
        counts = [config['layers'] // size] * size
        command += ['--stage-layers', ','.join(map(str, counts))]
    if config['synthetic']:
        command += ['--synthetic-data', 'true', '--synthetic-vocab-size', '512',
                    '--synthetic-samples', str(model['batch_size'] * (config['steps'] + config['warmup']))]
    else:
        command += ['--train-data', str(DATA / 'train.tsv'), '--vocab-file', str(DATA / 'vocab.txt'),
                    '--tokenizer-type', 'BertWordPieceCase']
    return command


def execute_job(client, config, job, rank, output, threads):
    import psutil
    output.mkdir(parents=True, exist_ok=False)
    command = command_for(config, job, rank, output)
    write_json(output / 'command.json', command)
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), USE_LIBUV='0')
    start = time.monotonic()
    peak_rss = 0
    samples = 0
    last_check = last_print = start
    log = output / 'training.log'
    print('Starting %s repetition %d; log: %s' % (job['mode'], job['repeat'], log), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                now = time.monotonic()
                try:
                    # Windows venv python.exe can be a small redirector whose
                    # child owns the actual PyTorch allocations.
                    processes = [monitor] + monitor.children(recursive=True)
                    rss = 0
                    for child in processes:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak_rss = max(peak_rss, rss)
                    samples += 1
                except psutil.NoSuchProcess:
                    pass
                if now - start > config['timeout']:
                    raise TimeoutError('Training exceeded %ds. Check Gloo/firewall connectivity and training.log.' % config['timeout'])
                if now - last_check > 2:
                    if config.get('larger'):
                        from scheduler.resources import host_memory
                        if host_memory()['available_bytes'] < 256 * 1048576:
                            raise RuntimeError('Stopping because available memory/commit headroom fell below 256 MiB. '
                                               'Close unused applications or use a smaller model. This is not a measured OOM.')
                    state = client.call('/state')
                    if state['status'] == 'failed':
                        raise RuntimeError(state['error'])
                    last_check = now
                if now - last_print > 20:
                    print('%s repetition %d still running, %.0fs elapsed' % (job['mode'], job['repeat'], now - start), flush=True)
                    last_print = now
                time.sleep(.1)
        finally:
            if process.poll() is None:
                # Include the Windows interpreter redirector's descendants.
                try:
                    descendants = monitor.children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for child in descendants:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                _, alive = psutil.wait_procs(descendants, timeout=5)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
    if process.returncode != 0:
        raise RuntimeError('Training exited %d. %s\n%s' % (process.returncode, log,
                           log.read_text(encoding='utf-8', errors='replace')[-3000:]))
    result = dict(metrics=json.loads((output / ('metrics_rank%d.json' % job['ranks'].index(rank))).read_text()),
                  host_rank=rank,
                  peak_process_rss_bytes=peak_rss, memory_samples=samples, threads=threads,
                  memory_scope='sum of RSS across training process tree; includes Windows venv redirector',
                  process_wall_seconds=time.monotonic() - start, command=command,
                  log=log.read_text(encoding='utf-8', errors='replace')[-200000:])
    validate_result(result, config, job, rank)
    write_json(output / 'result.json', result)
    return result


def worker(client, rank, local_root, threads):
    config = client.call('/config')
    member = fingerprint(config['synthetic'])
    member['hardware'] = inventory()
    if config.get('larger'):
        threads = 0
        vocab_size = 512 if config['synthetic'] else len((DATA / 'vocab.txt').read_text(encoding='utf-8').splitlines())
        member['memory_preflight'] = memory_preflight(config, rank, config.get('computers', 3), vocab_size)
        if threads == 0:
            member['cpu_calibration'] = calibrate_threads(config, vocab_size)
            threads = member['cpu_calibration']['selected_threads']
    if threads == 0:
        import psutil
        threads = psutil.cpu_count(logical=False) or 1
    member['threads'] = threads
    client.call('/register', dict(rank=rank, member=member))
    done = set()
    last_message = None
    while True:
        state = client.call('/state')
        if state['status'] in ('complete', 'failed'):
            client.call('/ack', dict(rank=rank))
            if state['status'] == 'failed':
                raise RuntimeError(state['error'])
            print('All experiments finished. Rank 0 is generating the report.', flush=True)
            return
        job = state.get('job')
        if job and job['id'] not in done and rank in job['ranks']:
            result = execute_job(client, config, job, rank, local_root / ('job%02d' % job['id']), threads)
            client.call('/result', dict(rank=rank, job_id=job['id'], result=result))
            done.add(job['id'])
        else:
            message = 'Waiting for other computers: ' + json.dumps(state)
            if message != last_message:
                print(message, flush=True)
                last_message = message
            time.sleep(.5)


def run(args):
    address = ipaddress.ip_address(args.master)
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise ValueError('--master must be rank 0 IPv4 address, such as 192.168.1.20')
    if args.rank == 0 and args.layers % args.computers:
        raise ValueError('--layers must be divisible by --computers for the equal-split comparison')
    if args.control_port == args.train_port:
        raise ValueError('control and training ports must differ')
    if args.rank and not args.session_code:
        raise ValueError('copy --session-code from rank 0')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'logs' / 'lab' / (stamp + '-rank%d' % args.rank)
    output.mkdir(parents=True)
    print('Results:', output, flush=True)
    token = args.session_code or secrets.token_hex(12)
    client = Client(args.master, args.control_port, token)
    coordinator = server = None
    stop_watchdog = threading.Event()
    if args.rank == 0:
        config = {key: getattr(args, key) for key in ('master', 'steps', 'warmup', 'repeats', 'layers',
                                                    'synthetic', 'train_port', 'timeout', 'computers',
                                                    'larger', 'all_single', 'embedding_dim', 'seq_length',
                                                    'num_heads', 'batch_size', 'micro_batch_size', 'lr')}
        config['created_utc'] = stamp
        coordinator = Coordinator(config, output, token)
        server = ThreadingHTTPServer((args.master, args.control_port), handler_for(coordinator))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        for rank in range(1, args.computers):
            print('\nComputer %d, PowerShell:\npowershell -ExecutionPolicy Bypass -File scripts/start_lab.ps1 '
                  '-Rank %d -MasterIP %s -ControlPort %d -SessionCode %s'
                  % (rank + 1, rank, args.master, args.control_port, token), flush=True)
            print('Linux:\n.venv-lab/bin/python scripts/lab_demo.py run --rank %d --master %s '
                  '--control-port %d --session-code %s' % (rank, args.master, args.control_port, token), flush=True)

        def watchdog():
            while not stop_watchdog.wait(1):
                with coordinator.lock:
                    if coordinator.state()['status'] not in ('complete', 'failed') and time.monotonic() - coordinator.changed > args.timeout + 30:
                        coordinator.fail('Timed out waiting for all computers/results. Check rank numbers, network, and worker logs.')
        threading.Thread(target=watchdog, daemon=True).start()
    success = False
    try:
        worker(client, args.rank, output / 'local', args.threads)
        success = True
    except BaseException as exc:
        message = str(exc) or type(exc).__name__
        write_json(output / 'local_failure.json', dict(error=message))
        try:
            client.call('/fail', dict(rank=args.rank, error=message))
            client.call('/ack', dict(rank=args.rank))
        except Exception:
            pass
        raise
    finally:
        if coordinator:
            # Let waiting peers observe completion/failure before closing HTTP.
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                with coordinator.lock:
                    if set(coordinator.members).issubset(coordinator.acks):
                        break
                time.sleep(.2)
            stop_watchdog.set()
            server.shutdown()
            server.server_close()
            if success:
                from scripts.lab_report import generate
                generate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    launch = sub.add_parser('run')
    launch.add_argument('--rank', type=int, choices=(0, 1, 2), required=True)
    launch.add_argument('--master', required=True)
    launch.add_argument('--session-code')
    launch.add_argument('--computers', type=int, choices=(2, 3), default=3)
    launch.add_argument('--steps', type=int, default=60)
    launch.add_argument('--warmup', type=int, default=5)
    launch.add_argument('--repeats', type=int, default=3)
    launch.add_argument('--threads', type=int, default=2)
    launch.add_argument('--lr', type=float, default=.01)
    launch.add_argument('--larger', action='store_true', help='53M model, memory preflight and CPU thread calibration')
    launch.add_argument('--all-single', action='store_true', help='run a single-computer baseline on each host')
    for key, default in model_fields({}).items():
        launch.add_argument('--' + key.replace('_', '-'), type=int, default=default)
    launch.add_argument('--layers', type=int, default=6)
    launch.add_argument('--control-port', type=int, default=8765)
    launch.add_argument('--train-port', type=int, default=29500)
    launch.add_argument('--timeout', type=int, default=1800)
    launch.add_argument('--synthetic', action='store_true')
    report = sub.add_parser('report')
    report.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        from scripts.lab_report import generate
        generate(args.directory.resolve())
    else:
        if args.larger:
            args.layers, args.embedding_dim, args.seq_length, args.num_heads = 12, 512, 128, 8
            args.batch_size, args.micro_batch_size = 16, 2
            args.lr = .001
            args.all_single = True
            args.threads = 0
        if (not math.isfinite(args.lr) or args.lr <= 0 or args.embedding_dim < 1 or args.num_heads < 1 or args.embedding_dim % args.num_heads
                or args.seq_length < 1 or args.batch_size < 1 or args.micro_batch_size < 1
                or args.batch_size % args.micro_batch_size):
            parser.error('invalid model or batch dimensions')
        if (not 1 <= args.steps <= 5000 or not 0 <= args.warmup <= 1000
                or not 1 <= args.repeats <= 10 or not 3 <= args.layers <= 120
                or not 0 <= args.threads <= 128 or not 30 <= args.timeout <= 86400
                or not all(1 <= port <= 65535 for port in (args.control_port, args.train_port))):
            parser.error('invalid step/warmup/repetition/layer/thread/timeout/port value')
        run(args)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('ERROR:', str(error) or 'Interrupted', file=sys.stderr)
        sys.exit(1)
