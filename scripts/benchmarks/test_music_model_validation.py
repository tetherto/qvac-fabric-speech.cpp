#!/usr/bin/env python3
"""Model-free checks for native MiniMax directory resolution and provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load('validate-music-models')
generation = load('music-generation-provenance')


class MusicModelValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def model(self, name, payload=b'GGUFfixture'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_all_native_tiers(self):
        for tier in validator.MINIMAX_TIERS:
            with self.subTest(tier=tier):
                directory = self.root / tier
                self.model(f'{tier}/mm3-lm-{tier}.gguf')
                self.model(f'{tier}/mm3-synth-{tier}.gguf')
                self.assertTrue(validator.validate(directory, 'minimax'))

    def test_unknown_or_empty_tier_rejected(self):
        for tier in ('q4_0', 'unknown', ''):
            with self.subTest(tier=tier):
                self.model(f'mm3-lm-{tier}.gguf')
                self.model(f'mm3-synth-{tier}.gguf')
                self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_split_directory_pair_and_case_insensitive_extension(self):
        lm = self.model('mm3/MM3-LM-Q4_K_M.GGUF')
        synth = self.model('Mm3-Synth-Q4_K_M.gguf')
        self.assertTrue(validator.validate(self.root, 'minimax'))
        self.assertEqual(validator.minimax_pair(self.root), (lm, synth))

    def test_mismatched_tiers_rejected(self):
        self.model('mm3-lm-q4_k_m.gguf')
        self.model('mm3-synth-f16.gguf')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_duplicates_across_directories_rejected_before_header_validation(self):
        self.model('mm3-lm-f16.gguf')
        self.model('mm3-synth-f16.gguf')
        self.model('mm3/MM3-LM-F16.gguf', b'broken')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_case_duplicate_rejected(self):
        self.model('mm3-lm-f16.gguf')
        second = self.model('MM3-LM-F16.gguf')
        self.model('mm3-synth-f16.gguf')
        if len(list(self.root.iterdir())) != 3:
            self.skipTest('filesystem is case insensitive')
        self.assertTrue(second.is_file())
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_unused_duplicate_tier_still_rejected(self):
        self.model('mm3-lm-f16.gguf')
        self.model('mm3-synth-f16.gguf')
        self.model('mm3-lm-unused.gguf')
        self.model('mm3/mm3-lm-unused.gguf')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_native_priority_does_not_skip_broken_preferred_pair(self):
        self.model('mm3-lm-f16.gguf')
        self.model('mm3-synth-f16.gguf')
        preferred = self.model('mm3-lm-q8_0.gguf', b'bad header')
        synth = self.model('mm3-synth-q8_0.gguf')
        self.assertEqual(validator.minimax_pair(self.root), (preferred, synth))
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_nested_non_mm3_directory_is_not_discovered(self):
        self.model('other/mm3-lm-f16.gguf')
        self.model('other/mm3-synth-f16.gguf')
        self.assertFalse(validator.validate(self.root, 'minimax'))

    def test_provenance_retains_preparation_and_allowlisted_build_options(self):
        model = self.model('mm3/MM3-LM-F16.GGUF')
        binary = self.model('build/bin/mm3-replay', b'binary')
        self.model('build/CMakeCache.txt', b'GGML_CUDA:BOOL=ON\nSECRET:STRING=private\nCMAKE_BUILD_TYPE:STRING=Release\n')
        metadata = {'source': {'revision': 'pinned'}, 'files': {'model': 'hash'}}
        self.model('provenance.json', json.dumps(metadata).encode())
        with patch.dict('os.environ', {'MM3_LM_FLASH': '1'}):
            result = generation.provenance(self.root, binary, 'gpu')
        self.assertEqual(result['preparation']['metadata'], metadata)
        self.assertEqual(result['model_files'], [generation.identity(model)])
        self.assertEqual(result['environment']['MM3_LM_FLASH'], '1')
        self.assertEqual(result['build_cache']['options'], {'GGML_CUDA': 'ON', 'CMAKE_BUILD_TYPE': 'Release'})
        self.assertEqual(result['requested_device'], 'gpu')

    def test_local_pair_does_not_require_preparation_or_build_metadata(self):
        binary = self.model('mm3-replay', b'binary')
        result = generation.provenance(self.root, binary, 'cpu')
        self.assertIsNone(result['preparation'])
        self.assertIsNone(result['build_cache'])


if __name__ == '__main__':
    unittest.main()
