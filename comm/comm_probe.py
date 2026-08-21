"""Pairwise latency/bandwidth over plain TCP (not Gloo send/recv).

Gloo point-to-point mixed with dist.barrier() hangs on Tailscale: the idle
rank enters a barrier while the other two are blocked in send/recv, and Gloo
never completes. TCP sockets with timeouts cannot deadlock the process group.
"""
import os
import socket
import struct
import subprocess
import time

import torch
import torch.distributed as dist


def _empty_matrix(n, diag=0.0):
    return [[diag if i == j else 0.0 for j in range(n)] for i in range(n)]


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
        t0 = time.time()
        conn.sendall(struct.pack('!I', len(payload)) + payload)
        hdr = _recv_exact(conn, 4)
        (n,) = struct.unpack('!I', hdr)
        _recv_exact(conn, n)
        rtts.append(time.time() - t0)
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
    """Fill latency_ms[i][j] and bandwidth_mbps[i][j] (i -> j) via TCP."""
    del comm, device
    latency = _empty_matrix(world_size)
    bandwidth = _empty_matrix(world_size)
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
            if rank == dst:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(('0.0.0.0', port))
                srv.listen(1)
                srv.settimeout(timeout)
            dist.barrier()
            try:
                if rank == src:
                    conn = socket.create_connection((ips[dst], port), timeout=timeout)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    rtts = _ping_pong(conn, ping_payload, ping_rounds)
                    use = rtts[1:] if len(rtts) > 1 else rtts
                    latency[src][dst] = 1000.0 * (sum(use) / len(use))
                    bws = _ping_pong(conn, bw_payload, bw_rounds)
                    useb = bws[1:] if len(bws) > 1 else bws
                    dt = sum(useb) / len(useb)
                    if dt > 0:
                        bandwidth[src][dst] = (bw_bytes * 8.0) / (dt / 2.0) / 1e6
                    print('[probe] %d->%d latency=%.2fms bw=%.1fMbps' % (
                        src, dst, latency[src][dst], bandwidth[src][dst]))
                    conn.close()
                elif rank == dst:
                    conn, _addr = srv.accept()
                    conn.settimeout(timeout)
                    _serve_pong(conn, ping_rounds)
                    _serve_pong(conn, bw_rounds)
                    conn.close()
            except Exception as exc:
                print('[probe] pair %d->%d FAILED: %r' % (src, dst, exc))
            finally:
                if srv is not None:
                    srv.close()
            dist.barrier()

    packed = []
    for i in range(world_size):
        packed.extend(latency[i])
        packed.extend(bandwidth[i])
    tensor = torch.tensor(packed, dtype=torch.float64)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    row_w = world_size * 2
    for src in range(world_size):
        row = gathered[src].tolist()
        latency[src] = row[0:world_size]
        bandwidth[src] = row[world_size:row_w]

    print('[probe] latency_ms', latency)
    print('[probe] bandwidth_mbps', bandwidth)
    return latency, bandwidth
