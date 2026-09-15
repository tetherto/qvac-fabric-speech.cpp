#!/usr/bin/env python3
import copy
import importlib.util
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('pilot_summary', Path(__file__).with_name('summarize-music-pilot.py'))
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


RUN_SPEC = importlib.util.spec_from_file_location('pilot_runner', Path(__file__).with_name('run-music-pilot.py'))
runner = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(runner)


def fixture():
    return {'family': 'acestep', 'manifest_sha256': 'fixed',
            'configuration': {'checkpoint_identity': 'source-v1', 'generation_settings': {'duration': 20}},
            'manifest': {'prompts': [{'id': 'p1', 'subset': 'primary'}, {'id': 'p2', 'subset': 'primary'},
                                     {'id': 's1', 'subset': 'stress'}],
                         'seeds': [1, 2, 3], 'bootstrap': {'seed': 24846, 'replicates': 1000}},
            'records': [{'prompt_id': p, 'seed': s, 'status': 'ok', 'score': score, 'scorer_provenance': {'revision': 'v1'}}
                        for p, s, score in [('p1', 1, .2), ('p1', 2, .4), ('p1', 3, .6), ('p2', 1, .8), ('s1', 1, -1)]]}


class PilotTests(unittest.TestCase):
    def test_equal_prompt_weights_and_partial_coverage(self):
        result = summary.summarize(fixture())['subsets']
        self.assertAlmostEqual(result['primary']['equal_prompt_mean'], .6)
        self.assertEqual(result['primary']['successful'], 4)
        self.assertEqual(result['primary']['expected'], 6)
        self.assertTrue(result['primary']['partial'])
        self.assertEqual(result['stress']['equal_prompt_mean'], -1)
        self.assertEqual(len(result['primary']['prompts'][1]['missing']), 2)

    def test_pair_first_then_average_and_bootstrap(self):
        baseline = fixture()
        candidate = copy.deepcopy(baseline)
        candidate['records'][0]['score'] += .3
        candidate['records'][3]['score'] -= .2
        result = summary.summarize(candidate, baseline)['subsets']['primary']
        self.assertAlmostEqual(result['equal_prompt_paired_delta'], -.05)
        self.assertEqual(result['paired_count'], 4)
        self.assertEqual(result['paired_prompt_bootstrap_95'], summary.summarize(candidate, baseline)['subsets']['primary']['paired_prompt_bootstrap_95'])
        self.assertTrue(result['paired_partial'])

    def test_null_invalid_never_zero(self):
        pilot = fixture()
        for row in pilot['records']:
            row['score'] = float('nan')
        result = summary.summarize(pilot)['subsets']['primary']
        self.assertEqual(result['successful'], 0)
        self.assertIsNone(result['equal_prompt_mean'])

    def test_incompatible_rejected(self):
        for change in ['family', 'manifest_sha256', 'checkpoint_identity', 'scorer_provenance']:
            candidate = fixture()
            if change in candidate:
                candidate[change] = 'different'
            elif change == 'checkpoint_identity':
                candidate['configuration'][change] = 'different'
            else:
                candidate['records'][0][change] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.summarize(candidate, fixture())

    def test_mixed_provenance_rejected_without_baseline(self):
        pilot = fixture()
        pilot['records'][1]['scorer_provenance'] = {'revision': 'v2'}
        with self.assertRaises(ValueError):
            summary.summarize(pilot)

    def test_duplicate_and_unexpected_rejected(self):
        candidate = fixture()
        candidate['records'].append(candidate['records'][0])
        with self.assertRaises(ValueError):
            summary.summarize(candidate)
        candidate = fixture()
        candidate['records'][0]['seed'] = 999
        with self.assertRaises(ValueError):
            summary.summarize(candidate)

    def test_runner_applies_settings_and_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            config.write_text(json.dumps({'name': 'candidate', 'checkpoint_identity': 'source-v1',
                                          'backend': 'GPU', 'build_options': 'test'}))
            calls = []
            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                if '--family' in command:
                    env = kwargs['env']
                    self.assertEqual(env['MUSIC_DEVICE'], 'gpu')
                    self.assertEqual(env['MUSIC_DURATION'], '20')
                    self.assertEqual(env['MUSIC_SEED'], '17')
                    self.assertIn('acoustic guitar', env['MUSIC_CAPTION'])
                    destination = Path(command[-1])
                    artifacts = destination.with_suffix('.music.test') / 'run-1'
                    artifacts.mkdir(parents=True)
                    (artifacts / 'audio.wav').write_bytes(b'test WAV retained')
                    score = {'status': 'ok', 'score': .5, 'provenance': {'revision': 'v1'},
                             'policy_version': 'v1', 'artifact_dir': str(artifacts)}
                    destination.write_text(json.dumps({'status': 'ok', 'music_alignment': {'runs': [score]}}))
                else:
                    destination = Path(command[command.index('--json-out') + 1])
                    destination.write_text(json.dumps({'status': 'ok', 'score': .4}))
                return 0
            argv = ['run-music-pilot.py', '--family', 'acestep', '--configuration', str(config),
                    '--build-dir', str(root / 'build'), '--scorer-model-dir', str(root / 'clap'),
                    '--model-dir', str(root / 'models'), '--out-dir', str(root / 'out'),
                    '--limit', '1', '--controls', '--repeat-generation']
            with patch('sys.argv', argv), patch.object(runner, 'run_bounded', side_effect=fake_run):
                runner.main()
            pilot = json.loads((root / 'out/pilot.json').read_text())
            self.assertEqual(len(pilot['records']), 81)
            self.assertEqual(pilot['records'][0]['score'], .5)
            self.assertEqual(pilot['records'][1]['status'], 'not-run')
            self.assertEqual(len(pilot['controls']), 2)
            self.assertEqual(len(calls), 4)
            self.assertIn('benchmark_source_hashes', pilot['configuration'])
            self.assertIsNotNone(pilot['controls'][0]['mismatch_prompt_id'])

    def test_invalid_driver_score_rejected(self):
        for value in (True, float('nan'), float('inf'), 1.1, None, '0.2'):
            self.assertEqual(runner.normalize_score({'status': 'ok', 'score': value})['status'], 'scorer-error')
        self.assertIsNone(runner.normalize_score({'status': 'timeout', 'score': .2})['score'])

    def test_manifest_size(self):
        import json
        manifest = json.loads(Path(__file__).with_name('fixtures').joinpath('music-pilot-v1.json').read_text())
        self.assertEqual(sum(p['subset'] == 'primary' for p in manifest['prompts']), 24)
        self.assertEqual(len(manifest['seeds']), 3)
        self.assertEqual(len({p['id'] for p in manifest['prompts']}), len(manifest['prompts']))


if __name__ == '__main__':
    unittest.main()
