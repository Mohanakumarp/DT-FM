"""Measure pairwise latency and bandwidth over the live tensor communicator."""
import time
import torch


def _empty_matrix(n, diag=0.0):
    return [[diag if i == j else 0.0 for j in range(n)] for i in range(n)]


def _make_tensor(numel, device):
    return torch.ones(numel, dtype=torch.float32, device=device)


def measure_comm_matrix(comm, rank, world_size, device='cpu',
                        ping_bytes=64, ping_rounds=20,
                        bw_bytes=4 * 1024 * 1024, bw_rounds=4):
    """Fill latency_ms[i][j] and bandwidth_mbps[i][j] (i -> j).

    Every rank must call this; it uses send/recv + barrier on `comm`.
    """
    latency = _empty_matrix(world_size)
    bandwidth = _empty_matrix(world_size, diag=0.0)
    ping_n = max(1, ping_bytes // 4)
    bw_n = max(1, bw_bytes // 4)
    ping = _make_tensor(ping_n, device)
    blob = _make_tensor(bw_n, device)
    ack = _make_tensor(ping_n, device)

    print('[probe] measuring comm matrix world_size=%d rank=%d' % (world_size, rank))
    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                comm.barrier()
                comm.barrier()
                continue
            rtts = []
            comm.barrier()
            for r in range(ping_rounds):
                if rank == src:
                    t0 = time.time()
                    comm.send(ping, dst)
                    comm.recv(ack, src=dst)
                    rtts.append(time.time() - t0)
                elif rank == dst:
                    comm.recv(ping, src=src)
                    comm.send(ack, dst=src)
            comm.barrier()
            if rank == src and rtts:
                # drop first warmup ping
                use = rtts[1:] if len(rtts) > 1 else rtts
                latency[src][dst] = 1000.0 * (sum(use) / len(use))

            bws = []
            comm.barrier()
            for r in range(bw_rounds):
                if rank == src:
                    t0 = time.time()
                    comm.send(blob, dst)
                    comm.recv(ack, src=dst)
                    dt = time.time() - t0
                    if dt > 0:
                        # one-way payload estimate: bytes / half-RTT
                        bws.append((bw_bytes * 8.0) / (dt / 2.0) / 1e6)
                elif rank == dst:
                    comm.recv(blob, src=src)
                    comm.send(ack, dst=src)
            comm.barrier()
            if rank == src and bws:
                use = bws[1:] if len(bws) > 1 else bws
                bandwidth[src][dst] = sum(use) / len(use)

    packed = []
    for i in range(world_size):
        packed.extend(latency[i])
        packed.extend(bandwidth[i])
    tensor = torch.tensor(packed, dtype=torch.float64)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    row_w = world_size * 2
    for src in range(world_size):
        row = gathered[src].tolist()
        latency[src] = row[0:world_size]
        bandwidth[src] = row[world_size:row_w]

    print('[probe] latency_ms rank', rank, latency)
    print('[probe] bandwidth_mbps rank', rank, bandwidth)
    return latency, bandwidth
