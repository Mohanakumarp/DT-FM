#!/usr/bin/env python3
"""Collect stdout from every rank and print a single live stream.

Each client sends:
    HELLO rank=<n>\\n
    <log lines...>
"""
import argparse
import os
import socket
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description='DT-FM multi-rank log hub')
    parser.add_argument('--port', type=int, default=9100)
    parser.add_argument('--out', default='logs/all_ranks.log')
    parser.add_argument('--bind', default='0.0.0.0')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out = open(args.out, 'a', buffering=1)
    lock = threading.Lock()

    def emit(line):
        if not line.endswith('\n'):
            line = line + '\n'
        with lock:
            sys.stdout.write(line)
            sys.stdout.flush()
            out.write(line)
            out.flush()

    def handle(conn, addr):
        rank = '?'
        buf = b''
        try:
            conn.settimeout(None)
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    raw, buf = buf.split(b'\n', 1)
                    text = raw.decode('utf-8', 'replace')
                    if text.startswith('HELLO rank='):
                        rank = text.split('=', 1)[1].strip()
                        emit('[hub] rank %s connected from %s:%s' % (rank, addr[0], addr[1]))
                        continue
                    emit('[rank %s] %s' % (rank, text))
            if buf.strip():
                emit('[rank %s] %s' % (rank, buf.decode('utf-8', 'replace')))
        except OSError as exc:
            emit('[hub] rank %s disconnected (%s)' % (rank, exc))
        finally:
            conn.close()
            emit('[hub] rank %s closed' % rank)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(16)
    emit('[hub] listening on %s:%s -> %s' % (args.bind, args.port, args.out))
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        emit('[hub] stopped')
    finally:
        srv.close()
        out.close()


if __name__ == '__main__':
    main()
