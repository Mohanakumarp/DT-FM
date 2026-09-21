"""Collect wall-clock, compute vs sync, and comm-matrix metrics."""
import json
import os
import time


class RunMetrics:
    def __init__(self, args, tracker=None):
        self.args = args
        self.tracker = tracker
        self.t0 = time.time()
        self.marks = {'start': self.t0}
        self.iters = []
        self.losses = []
        self.latency_ms = None
        self.bandwidth_mbps = None
        self.peers = None
        self.rebalance_events = []

    def record_rebalance(self, event):
        self.rebalance_events.append(event)
        if self.tracker is not None:
            self.tracker.log_metrics(dict(
                assigned_layers=self.args.num_layers,
                rebalance_seconds=event['elapsed_s'],
                rebalance_committed=int(event['status'] == 'committed')),
                step=event['completed_steps'], synchronous=True)

    def mark(self, name):
        self.marks[name] = time.time()

    def elapsed(self, a, b=None):
        end = self.marks[b] if b else time.time()
        return max(0.0, end - self.marks[a])

    def record_iter(self, stats):
        if not isinstance(stats, dict):
            stats = {'iter_s': float(stats)}
        self.iters.append(stats)
        if self.tracker is not None:
            values = dict(stats)
            seconds = stats.get('iter_s', 0.0)
            if seconds > 0:
                values['samples_per_second'] = self.args.batch_size / seconds
            self.tracker.log_metrics(values, step=len(self.iters), synchronous=True)

    def record_loss(self, value):
        self.losses.append(float(value))
        if self.tracker is not None:
            self.tracker.log_metric('last_microbatch_loss', float(value),
                                    step=len(self.iters), synchronous=True)

    def set_comm_matrix(self, latency_ms, bandwidth_mbps, peers):
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.peers = peers

    def summary(self):
        train_s = self.elapsed('train_start', 'train_end') if 'train_end' in self.marks else 0.0
        fwd = sum(i.get('forward_s', 0.0) for i in self.iters)
        bwd = sum(i.get('backward_s', 0.0) for i in self.iters)
        opt = sum(i.get('optim_s', 0.0) for i in self.iters)
        barrier = sum(i.get('barrier_s', 0.0) for i in self.iters)
        iter_sum = sum(i.get('iter_s', 0.0) for i in self.iters)
        compute_s = fwd + bwd + opt
        # fwd/bwd include activation send/recv; barriers are explicit sync.
        efficient_s = max(0.0, iter_sum - barrier)
        return {
            'rank': self.args.rank,
            'world_size': self.args.world_size,
            'pipeline_group_size': self.args.pipeline_group_size,
            'model': {
                'seq_length': self.args.seq_length,
                'embedding_dim': self.args.embedding_dim,
                'layers_per_stage': self.args.num_layers,
                'stage_parameters': getattr(self.args, 'stage_parameters', None),
                'num_heads': self.args.num_heads,
                'total_layers': getattr(self.args, 'total_layers', self.args.num_layers * self.args.pipeline_group_size),
                'layer_start': getattr(self.args, 'layer_start', None),
                'layer_end': getattr(self.args, 'layer_end', None),
                'stage_layers': getattr(self.args, 'stage_layers', None),
            },
            'schedule': {
                'num_epochs': getattr(self.args, 'num_epochs', 0),
                'steps_per_epoch': getattr(self.args, 'steps_per_epoch', 0),
                'num_iters': self.args.num_iters,
                'completed_steps': len(self.iters),
            },
            'resource_schedule': getattr(self.args, 'resource_schedule', None),
            'rebalance_events': self.rebalance_events,
            'wall_clock_s': {
                'total': time.time() - self.t0,
                'comm_probe': self.elapsed('probe_start', 'probe_end') if 'probe_end' in self.marks else 0.0,
                'training': train_s,
            },
            'training_time_s': {
                'forward_incl_p2p': fwd,
                'backward_incl_p2p': bwd,
                'optimizer': opt,
                'barrier_sync': barrier,
                'iter_sum': iter_sum,
                'efficient_compute_est': efficient_s,
                'efficiency': (efficient_s / train_s) if train_s > 0 else 0.0,
            },
            'peers': self.peers,
            'latency_ms': self.latency_ms,
            'bandwidth_mbps': self.bandwidth_mbps,
            'losses': self.losses,
            # Keep individual steps so reports can exclude an explicit warmup.
            'iterations': self.iters,
        }

    def dump(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        payload = self.summary()
        with open(path, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print('[metrics] wrote', path)
        self.print_tables(payload)
        if self.tracker is not None:
            self.tracker.log_metrics({
                'completed_steps': len(self.iters),
                'training_wall_seconds': payload['wall_clock_s']['training'],
            }, synchronous=True)
            self.tracker.log_artifact(path, artifact_path='measurements')
        return payload

    def print_tables(self, payload):
        print('======== RUN METRICS rank', payload['rank'], '========')
        wc = payload['wall_clock_s']
        tr = payload['training_time_s']
        print('wall_clock total={:.2f}s  comm_probe={:.2f}s  training={:.2f}s'.format(
            wc['total'], wc['comm_probe'], wc['training']))
        print('efficient_compute_est={:.2f}s  barrier_sync={:.2f}s  efficiency={:.1%}'.format(
            tr['efficient_compute_est'], tr['barrier_sync'], tr['efficiency']))
        if payload['latency_ms']:
            print('latency_ms:')
            self._print_matrix(payload['latency_ms'], '{:8.2f}')
        if payload['bandwidth_mbps']:
            print('bandwidth_mbps:')
            self._print_matrix(payload['bandwidth_mbps'], '{:8.1f}')
        if payload['losses']:
            print('loss first={:.4f} last={:.4f} n={}'.format(
                payload['losses'][0], payload['losses'][-1], len(payload['losses'])))
        print('======== END METRICS ========')

    @staticmethod
    def _print_matrix(mat, fmt):
        header = '     ' + ''.join('{:>8d}'.format(j) for j in range(len(mat)))
        print(header)
        for i, row in enumerate(mat):
            cells = [
                fmt.format(x) if x is not None else '{:>8s}'.format('n/a')
                for x in row
            ]
            print('{:>4d} '.format(i) + ''.join(cells))
