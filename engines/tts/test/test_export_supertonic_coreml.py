#!/usr/bin/env python3
"""Model-free unit tests for scripts/export-supertonic-coreml.py.

A tiny vocoder is built from random tensors in the GGUF's own source names
and ONNX row-major layouts, so the checks need no checkpoint: the latent
unpack must equal supertonic_vocoder.cpp's reshape-permute chain; the whole
stack must be causal, which is what lets the engine drop each window's
leading context and zero-pad the last window on the right; the receptive
field must bound the context a window drop needs; and the sidecar stem rule
must match supertonic_coreml_path.cpp. Requires torch; skips without it,
mirroring the converter tests.
"""
import importlib.util
import pathlib
import unittest

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / 'export-supertonic-coreml.py'


def load_exporter():
    spec = importlib.util.spec_from_file_location('export_supertonic_coreml', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LATENT_DIM = 4
FACTOR = 3
CHANNELS = LATENT_DIM * FACTOR
WIDTH = 6
HIDDEN = 12
CHUNK = 5
FRAME = FACTOR * CHUNK  # samples per latent frame
KERNEL = 3


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def tiny_tensors(rng):
    """Random tensors in the model GGUF's source names and ONNX row-major
    layouts: full kernels [out, in, taps], depthwise kernels [C, 1, taps]."""
    t = {
        'vocoder:tts.ttl.normalizer.scale': np.array([1.5], dtype=np.float32),
        'vocoder:tts.ae.latent_mean': rand(rng, 1, LATENT_DIM, 1),
        'vocoder:tts.ae.latent_std': np.abs(rand(rng, 1, LATENT_DIM, 1)) + 0.5,
        'vocoder:node:/decoder/embed/net/Conv#1': rand(rng, WIDTH, LATENT_DIM, KERNEL),
        'vocoder:node:/decoder/embed/net/Conv#2': rand(rng, WIDTH),
        'vocoder:node:/decoder/head/act/PRelu#1': np.array([0.2], dtype=np.float32),
        'vocoder:tts.ae.decoder.final_norm.norm.weight': 1.0 + rand(rng, WIDTH),
        'vocoder:tts.ae.decoder.final_norm.norm.bias': rand(rng, WIDTH),
        'vocoder:tts.ae.decoder.final_norm.norm.running_mean': rand(rng, WIDTH),
        'vocoder:tts.ae.decoder.final_norm.norm.running_var': np.abs(rand(rng, WIDTH)) + 0.5,
        'vocoder:tts.ae.decoder.head.layer1.net.weight': rand(rng, HIDDEN, WIDTH, KERNEL),
        'vocoder:tts.ae.decoder.head.layer1.net.bias': rand(rng, HIDDEN),
        'vocoder:tts.ae.decoder.head.layer2.weight': rand(rng, CHUNK, HIDDEN, 1),
    }
    for i in range(10):
        prefix = f'vocoder:tts.ae.decoder.convnext.{i}'
        t[prefix + '.dwconv.net.weight'] = rand(rng, WIDTH, 1, KERNEL)
        t[prefix + '.dwconv.net.bias'] = rand(rng, WIDTH)
        t[prefix + '.norm.norm.weight'] = 1.0 + rand(rng, WIDTH)
        t[prefix + '.norm.norm.bias'] = rand(rng, WIDTH)
        t[prefix + '.pwconv1.weight'] = rand(rng, HIDDEN, WIDTH, 1)
        t[prefix + '.pwconv1.bias'] = rand(rng, HIDDEN)
        t[prefix + '.pwconv2.weight'] = rand(rng, WIDTH, HIDDEN, 1)
        t[prefix + '.pwconv2.bias'] = rand(rng, WIDTH)
        t[prefix + '.gamma'] = rand(rng, 1, WIDTH, 1)
    return t


def tiny_hparams():
    return {
        'latent_channels': CHANNELS,
        'base_chunk_size': CHUNK,
        'ttl_chunk_compress_factor': FACTOR,
        'sample_rate': 44100,
    }


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class VocoderTest(unittest.TestCase):
    def setUp(self):
        self.exporter = load_exporter()
        self.rng = np.random.default_rng(1234)
        self.model = self.exporter.Vocoder(tiny_tensors(self.rng), tiny_hparams()).eval()

    def test_output_shape(self):
        x = torch.randn(1, CHANNELS, 6)
        with torch.no_grad():
            self.assertEqual(tuple(self.model(x).shape), (1, 1, 6 * FRAME))

    def test_unpack_matches_ggml_chain(self):
        # supertonic_vocoder.cpp: x[c, t * factor + r] = latent[c * factor + r, t].
        frames = 5
        latent = torch.randn(1, CHANNELS, frames)
        got = self.model.unpack(latent)
        self.assertEqual(tuple(got.shape), (1, LATENT_DIM, frames * FACTOR))
        for c in range(LATENT_DIM):
            for t in range(frames):
                for r in range(FACTOR):
                    self.assertEqual(float(got[0, c, t * FACTOR + r]),
                                     float(latent[0, c * FACTOR + r, t]))

    def test_stack_is_causal(self):
        # A change at latent frame f must leave every sample before f * FRAME
        # alone, and zeros appended on the right must not change what came
        # before: both are what coreml_windows.h relies on.
        x = torch.randn(1, CHANNELS, 12)
        changed = x.clone()
        changed[..., 7:] += 1.0
        padded = F.pad(x, (0, 5))
        with torch.no_grad():
            base, moved, longer = self.model(x), self.model(changed), self.model(padded)
        self.assertTrue(torch.allclose(base[..., :7 * FRAME], moved[..., :7 * FRAME], atol=1e-4))
        self.assertFalse(torch.allclose(base[..., 7 * FRAME:], moved[..., 7 * FRAME:], atol=1e-3))
        self.assertTrue(torch.allclose(base, longer[..., :12 * FRAME], atol=1e-4))

    def test_context_drop_matches_full_pass(self):
        # A window starting mid-utterance reproduces the full pass once the
        # receptive field's worth of leading frames is dropped, while the very
        # first kept frame of a zero-dropped window must not (replicate
        # padding restarts at the window edge).
        context = self.exporter.receptive_field_frames(self.model, tiny_hparams())
        x = torch.randn(1, CHANNELS, 24 + context)
        with torch.no_grad():
            full = self.model(x)
            window = self.model(x[..., 4:])
        drop = context
        self.assertTrue(torch.allclose(full[..., (4 + drop) * FRAME:],
                                       window[..., drop * FRAME:], atol=1e-5))
        self.assertFalse(torch.allclose(full[..., 4 * FRAME:5 * FRAME],
                                        window[..., :FRAME], atol=1e-3))

    def test_squeezed_pointwise_matches_three_d(self):
        # requantize-gguf.py stores K=1 pointwise convs squeezed to [OC, IC]
        # (supertonic.pwconv_squeezed); the rebuild must re-expand them like
        # the C++ loader.
        tensors = tiny_tensors(np.random.default_rng(1234))
        squeezed = dict(tensors)
        for name, value in tensors.items():
            if ('.pwconv' in name or name.endswith('head.layer2.weight')) and value.ndim == 3:
                squeezed[name] = value[:, :, 0]
        model = self.exporter.Vocoder(squeezed, tiny_hparams()).eval()
        x = torch.randn(1, CHANNELS, 6)
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model(x), model(x)))

    def test_receptive_field(self):
        # embed + 10 dilated depthwise convs + head1 + pointwise head2, in
        # latent frames (rounded up): must match
        # supertonic_coreml_vocoder_context_frames in supertonic_vocoder.cpp.
        dilations = self.exporter.VOCODER_DILATIONS
        taps = (KERNEL - 1) * (1 + sum(dilations)) + (KERNEL - 1)
        want = -(-taps // FACTOR)
        self.assertEqual(self.exporter.receptive_field_frames(self.model, tiny_hparams()), want)


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class SidecarStemTest(unittest.TestCase):
    def test_matches_coreml_path_rule(self):
        stem = load_exporter().sidecar_stem
        self.assertEqual(stem('models/supertonic2.gguf'), 'supertonic2-vocoder')
        self.assertEqual(stem('models/supertonic2-q8_0.gguf'), 'supertonic2-vocoder')
        self.assertEqual(stem('supertonic3.F16.gguf'), 'supertonic3-vocoder')
        self.assertEqual(stem('supertonic-multilingual.gguf'), 'supertonic-multilingual-vocoder')


if __name__ == '__main__':
    unittest.main()
