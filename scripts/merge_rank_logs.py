#!/usr/bin/env python3
"""Print one or more rank log files with a [rank N] prefix."""
import os
import re
import sys


def rank_from_name(path):
    m = re.search(r'rank(\d+)', os.path.basename(path))
    return m.group(1) if m else '?'


def main():
    paths = sys.argv[1:]
    if not paths:
        print('usage: merge_rank_logs.py logs/rank0.log logs/rank1.log ...', file=sys.stderr)
        sys.exit(2)
    for path in paths:
        if not os.path.isfile(path):
            print('[missing] ' + path, file=sys.stderr)
            continue
        rank = rank_from_name(path)
        print('======== %s (rank %s) ========' % (path, rank))
        with open(path, 'r', errors='replace') as fh:
            for line in fh:
                if line.startswith('[rank ') or line.startswith('[hub]'):
                    sys.stdout.write(line)
                else:
                    sys.stdout.write('[rank %s] %s' % (rank, line))
        print('')


if __name__ == '__main__':
    main()
