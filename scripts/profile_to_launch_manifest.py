"""Generate a checked manifest and print one Bash command per measured rank."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.launch_manifest import build_manifest, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path, help='verification.json from verify_comm_probe.py')
    parser.add_argument('--hosts', nargs='+', required=True, help='hosts in measured rank order')
    parser.add_argument('--devices', nargs='+', help='devices in measured rank order; default CPU')
    parser.add_argument('--payload-bytes', type=int, default=1048576)
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--log-port', type=int, default=9100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        profile = json.loads(args.profile.read_text(encoding='utf-8'))
        manifest = build_manifest(profile, args.hosts, args.devices, args.payload_bytes,
                                  args.port, args.log_port)
        serialized = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
        validate_manifest(json.loads(serialized))
        if args.output.resolve() == args.profile.resolve():
            raise ValueError('output must not overwrite the input profile')
        args.output.write_text(serialized, encoding='utf-8')
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    for entry in manifest['ranks']:
        print('# Measured rank %d on %s becomes launch rank %d' % (
            entry['measured_rank'], entry['host'], entry['rank']))
        print(entry['command'])


if __name__ == '__main__':
    main()
