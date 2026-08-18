#!/usr/bin/env python3
"""Copy stdin to a local rank log and to the rank-0 log hub."""
import argparse
import os
import socket
import sys
import time


def connect_hub(host, port, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as exc:
            last = exc
            time.sleep(0.5)
    sys.stderr.write('log_tee: could not reach hub %s:%s (%s); local file only\n'
                     % (host, port, last))
    return None


def main():
    parser = argparse.ArgumentParser(description='Tee rank logs to file + hub')
    parser.add_argument('--rank', required=True)
    parser.add_argument('--hub', required=True, help='HOST:PORT of log_hub.py')
    parser.add_argument('--file', required=True)
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.file) or '.', exist_ok=True)
    host, port_s = args.hub.rsplit(':', 1)
    sock = connect_hub(host, int(port_s), args.timeout)
    if sock is not None:
        hello = ('HELLO rank=%s\n' % args.rank).encode('utf-8')
        try:
            sock.sendall(hello)
        except OSError:
            sock.close()
            sock = None

    with open(args.file, 'w', buffering=1) as local:
        while True:
            line = sys.stdin.readline()
            if line == '':
                break
            local.write(line)
            local.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if sock is not None:
                try:
                    sock.sendall(line.encode('utf-8'))
                except OSError:
                    sock.close()
                    sock = None
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        sock.close()


if __name__ == '__main__':
    main()
