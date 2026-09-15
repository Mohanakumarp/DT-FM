import copy
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from utils.launch_manifest import build_manifest, validate_manifest, validate_profile

ROOT = Path(__file__).resolve().parents[1]


def fixture(size):
    return json.loads((ROOT / 'tests' / 'fixtures' / ('probe_%drank.json' % size)).read_text())


class LaunchManifestTests(unittest.TestCase):
    def test_two_rank_preserves_directions_and_units(self):
        profile = fixture(2)
        manifest = build_manifest(profile, ['a', 'b'])
        self.assertEqual(manifest['profile'], profile)
        self.assertEqual(manifest['pipeline_order'], [0, 1])
        self.assertAlmostEqual(manifest['selection']['cost_ms'],
                               1.25 + 2.5 + 1048576 * 8 / 125000 + 1048576 * 8 / 80000)
        validate_manifest(json.loads(json.dumps(manifest, allow_nan=False)))
        profile['latency_ms'][0][1] = 999
        self.assertEqual(manifest['profile']['latency_ms'][0][1], 1.25)

    def test_three_rank_avoids_null_and_remaps_devices(self):
        profile = fixture(3)
        manifest = build_manifest(profile, ['a', 'b', 'c'], ['cpu', 'rocm', 'directml'])
        self.assertEqual(manifest['pipeline_order'], [0, 2, 1])
        self.assertIsNone(manifest['profile']['latency_ms'][0][1])
        self.assertIsNone(manifest['profile']['bandwidth_mbps'][1][0])
        self.assertEqual([r['device'] for r in manifest['ranks']], ['cpu', 'directml', 'rocm'])
        for rank, entry in enumerate(manifest['ranks']):
            tokens = shlex.split(entry['command'])
            self.assertEqual(tokens[0], 'env')
            self.assertEqual(tokens[-2:], ['bash', 'scripts/run_rank.sh'])
            env = dict(token.split('=', 1) for token in tokens[1:-2])
            self.assertEqual(env, entry['env'])
            self.assertEqual(env['RANK'], str(rank))
            self.assertEqual((env['WORLD_SIZE'], env['PP_SIZE'], env['DP_SIZE']), ('3', '3', '1'))
            self.assertEqual(env['MASTER_IP'], 'a')

    def test_master_follows_remapped_first_rank(self):
        profile = fixture(3)
        # Rank 0 must be the middle stage, so rank 1 becomes the master.
        for field in ('latency_ms', 'bandwidth_mbps'):
            profile[field] = [[0, 2, 3], [4, 0, None], [5, None, 0]]
        manifest = build_manifest(profile, ['a', 'b', 'c'])
        self.assertEqual(manifest['pipeline_order'], [1, 0, 2])
        self.assertEqual(manifest['master_host'], 'b')
        self.assertTrue(all(r['env']['MASTER_IP'] == 'b' for r in manifest['ranks']))

    def test_minimizes_cost_against_exhaustive_oracle(self):
        profile = fixture(3)
        profile['latency_ms'] = [[0, 8, 2], [7, 0, 4], [1, 3, 0]]
        profile['bandwidth_mbps'] = [[0, 20, 200], [30, 0, 100], [150, 90, 0]]
        def score(order):
            return sum(profile['latency_ms'][a][b] + 8000 / profile['bandwidth_mbps'][a][b]
                       for x, y in zip(order, order[1:]) for a, b in ((x, y), (y, x)))
        expected = min(itertools.permutations(range(3)), key=lambda p: (score(p), p))
        manifest = build_manifest(profile, ['a', 'b', 'c'], payload_bytes=1000000)
        self.assertEqual(manifest['pipeline_order'], list(expected))
        self.assertAlmostEqual(manifest['selection']['cost_ms'], score(expected))

    def test_one_way_or_disconnected_pipeline_is_rejected(self):
        for size in (2, 3):
            profile = fixture(size)
            for field in ('latency_ms', 'bandwidth_mbps'):
                profile[field][0][size - 1] = None
            with self.assertRaisesRegex(ValueError, 'no full-rank pipeline'):
                build_manifest(profile, ['host'] * size)

    def test_invalid_profiles_fail_closed(self):
        for value in (True, '1', 0, -1, float('nan'), float('inf')):
            for field in ('latency_ms', 'bandwidth_mbps'):
                profile = fixture(2)
                profile[field][0][1] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_profile(profile)
        for field, value in [('passed', False), ('world_size', True), ('world_size', 13),
                             ('latency_ms', [[0, 1]]), ('latency_ms', [[0], [2, 0]]),
                             ('latency_ms', [[True, 1], [2, 0]]),
                             ('latency_ms', [[0, None], [2, 0]])]:
            profile = fixture(2)
            profile[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_profile(profile)

    def test_bad_launch_inputs_and_manifest_tampering(self):
        for kwargs in ({'hosts': ['a']}, {'hosts': ['a', 'bad;echo x']},
                       {'hosts': ['a', 'b'], 'devices': ['cpu']},
                       {'hosts': ['a', 'b'], 'port': 9100},
                       {'hosts': ['a', 'b'], 'payload_bytes': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_manifest(fixture(2), **kwargs)
        original = build_manifest(fixture(3), ['a', 'b', 'c'])
        for field, value in [('pipeline_order', [0, 1, 2]), ('world_size', True),
                             ('ranks', []), ('master_host', 'c'), ('schema_version', 2)]:
            manifest = copy.deepcopy(original)
            manifest[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_manifest(manifest)
        original['ranks'][0]['command'] += ' --unsafe'
        with self.assertRaises(ValueError):
            validate_manifest(original)

    def test_cli_emits_checked_json_and_commands_and_no_file_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'launch.json'
            for size in (2, 3):
                result = subprocess.run([
                    sys.executable, str(ROOT / 'scripts/profile_to_launch_manifest.py'),
                    str(ROOT / 'tests/fixtures' / ('probe_%drank.json' % size)),
                    '--hosts', *(['localhost'] * size), '--output', str(output)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = validate_manifest(json.loads(output.read_text()))
                for entry in manifest['ranks']:
                    self.assertIn(entry['command'], result.stdout)
            output.unlink()
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/profile_to_launch_manifest.py'),
                str(ROOT / 'tests/fixtures/probe_2rank.json'), '--hosts', 'a',
                '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
