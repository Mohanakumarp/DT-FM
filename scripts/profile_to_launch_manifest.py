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
    parser.add_argument('--compute-profiles', nargs='+', type=Path,
                        help='per-rank JSON from profile_compute.py, in any order')
    parser.add_argument('--total-layers', type=int, help='total transformer layers across all stages')
    parser.add_argument('--dynamic', action='store_true', help='allocate stages from live resources at launch; no compute files needed')
    parser.add_argument('--rebalance-every', type=int, default=0, help='recheck resources and migrate every N completed steps')
    args = parser.parse_args()
    try:
        profile = json.loads(args.profile.read_text(encoding='utf-8'))
        compute_profile = None
        if args.compute_profiles:
            records = [json.loads(path.read_text(encoding='utf-8')) for path in args.compute_profiles]
            if any(type(record.get('schema_version')) is not int or record['schema_version'] != 1
                   or record.get('model') != records[0].get('model')
                   for record in records):
                raise ValueError('compute profiles must use schema 1 and identical model/batch dimensions')
            compute_profile = dict(schema_version=1, model=records[0]['model'],
                                   ranks=sorted([r['rank'] for r in records], key=lambda r: r['measured_rank']))
        manifest = build_manifest(profile, args.hosts, args.devices, args.payload_bytes,
                                  args.port, args.log_port, compute_profile, args.total_layers, args.dynamic,
                                  args.rebalance_every)
        serialized = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
        validate_manifest(json.loads(serialized))
        if args.output.resolve() in [p.resolve() for p in [args.profile, *(args.compute_profiles or [])]]:
            raise ValueError('output must not overwrite the input profile')
        args.output.write_text(serialized, encoding='utf-8')
    except (ValueError, OSError, OverflowError, KeyError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    for entry in manifest['ranks']:
        print('# Measured rank %d on %s becomes launch rank %d' % (
            entry['measured_rank'], entry['host'], entry['rank']))
        print(entry['command'])


if __name__ == '__main__':
    main()
