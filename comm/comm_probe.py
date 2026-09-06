"""Pairwise latency/bandwidth over plain TCP (not Gloo send/recv).

Gloo point-to-point mixed with dist.barrier() hangs on Tailscale: the idle
rank enters a barrier while the other two are blocked in send/recv, and Gloo
never completes. Socket timeouts and shared listener status let healthy ranks
finish failed probe pairs. Process-group failures still require a restart.
"""
import os
import socket
import struct
import subprocess
import time

import torch.distributed as dist

from .probe_aggregation import aggregate_probe_results


def _guess_ipv4():
    try:
        if os.path.isdir('/sys/class/net/tailscale0'):
            out = subprocess.check_output(['tailscale', 'ip', '-4'], text=True)
            ip = out.strip().split('\n')[0]
            if ip:
                return ip
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('1.1.1.1', 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('peer closed during recv')
        buf += chunk
    return buf


def _ping_pong(conn, payload, rounds):
    rtts = []
    for i in range(rounds):
        t0 = time.perf_counter()
        conn.sendall(struct.pack('!I', len(payload)) + payload)
        hdr = _recv_exact(conn, 4)
        (n,) = struct.unpack('!I', hdr)
        _recv_exact(conn, n)
        rtts.append(time.perf_counter() - t0)
    return rtts


def _serve_pong(conn, rounds):
    for _ in range(rounds):
        hdr = _recv_exact(conn, 4)
        (n,) = struct.unpack('!I', hdr)
        data = _recv_exact(conn, n)
        conn.sendall(struct.pack('!I', len(data)) + data)


def measure_comm_matrix(comm, rank, world_size, device='cpu',
                        ping_bytes=64, ping_rounds=10,
                        bw_bytes=1 * 1024 * 1024, bw_rounds=3,
                        base_port=9200, timeout=20):
    """Collect TCP round-trip measurements initiated by each source rank.

    Latency is full RTT. Bandwidth is an echo-based estimate using RTT / 2,
    not an isolated one-way link measurement. Keep both source observations.
    """
    del comm, device
    measurements = []
    store = dist.distributed_c10d._get_default_store()
    my_ip = _guess_ipv4()
    store.set('probe-ip-%d' % rank, my_ip.encode('utf-8'))
    dist.barrier()
    ips = [store.get('probe-ip-%d' % i).decode('utf-8') for i in range(world_size)]
    print('[probe] TCP matrix rank=%d ip=%s peers=%s' % (rank, my_ip, ips))

    ping_payload = b'x' * ping_bytes
    bw_payload = b'y' * bw_bytes

    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                dist.barrier()
                dist.barrier()
                continue
            port = base_port + dst
            print('[probe] pair %d -> %d (%s:%d)' % (src, dst, ips[dst], port))
            srv = None
            ready_key = 'probe-ready-%d-%d' % (src, dst)
            if rank == dst:
                try:
                    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind(('0.0.0.0', port))
                    srv.listen(1)
                    srv.settimeout(timeout)
                except OSError as exc:
                    print('[probe] listener %d FAILED: %r' % (dst, exc))
                    if srv is not None:
                        srv.close()
                    srv = None
                store.set(ready_key, b'1' if srv is not None else b'0')
            dist.barrier()
            try:
                if rank == src:
                    if store.get(ready_key) != b'1':
                        raise ConnectionError('destination probe listener unavailable')
                    with socket.create_connection(
                            (ips[dst], port), timeout=timeout) as conn:
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        rtts = _ping_pong(conn, ping_payload, ping_rounds)
                        use = rtts[1:] if len(rtts) > 1 else rtts
                        latency_ms = 1000.0 * (sum(use) / len(use))
                        bws = _ping_pong(conn, bw_payload, bw_rounds)
                        useb = bws[1:] if len(bws) > 1 else bws
                        dt = sum(useb) / len(useb)
                        if dt <= 0:
                            raise ValueError(
                                'non-positive bandwidth round-trip time')
                        bandwidth_mbps = (
                            (bw_bytes * 8.0) / (dt / 2.0) / 1e6)
                    measurements.append({
                        'src': src,
                        'dst': dst,
                        'latency_ms': latency_ms,
                        'bandwidth_mbps': bandwidth_mbps,
                    })
                    print('[probe] %d->%d latency=%.2fms bw=%.1fMbps' % (
                        src, dst, latency_ms, bandwidth_mbps))
                elif rank == dst and srv is not None:
                    conn, _addr = srv.accept()
                    with conn:
                        conn.settimeout(timeout)
                        _serve_pong(conn, ping_rounds)
                        _serve_pong(conn, bw_rounds)
            except Exception as exc:
                print('[probe] pair %d->%d FAILED: %r' % (src, dst, exc))
                if rank == src:
                    measurements.append({
                        'src': src,
                        'dst': dst,
                        'latency_ms': None,
                        'bandwidth_mbps': None,
                    })
            finally:
                if srv is not None:
                    srv.close()
            dist.barrier()

    local_result = {'rank': rank, 'measurements': measurements}
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)
    latency, bandwidth = aggregate_probe_results(gathered, world_size)

    print('[probe] latency_ms', latency)
    print('[probe] bandwidth_mbps', bandwidth)
    return latency, bandwidth
