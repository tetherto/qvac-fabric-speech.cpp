#!/usr/bin/env python3
"""Unit checks of coreml_template.py.

The pure parts (MILBlob parsing, blob matching, recipe drafting, header
refresh, exporter arguments, template.json merging) need only numpy. The spec
comparison needs coremltools' protobuf classes, and the end-to-end `suggest`
on a toy exporter also needs gguf-py and qvac-coreml-tool ($QVAC_COREML_TOOL).
Those cases are skipped when their requirements are missing, unless
QVAC_COREML_REQUIRE_COREMLTOOLS=1 (the template CI job), where a missing
requirement fails instead.
"""

import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreml_template as ctpl  # noqa: E402

REQUIRE = os.environ.get("QVAC_COREML_REQUIRE_COREMLTOOLS") == "1"
HAVE_CT = importlib.util.find_spec("coremltools") is not None
HAVE_GGUF = importlib.util.find_spec("gguf") is not None
TOOL = os.environ.get("QVAC_COREML_TOOL", "")


def needs(cond, what):
    if cond:
        return lambda f: f
    if REQUIRE:
        def fail(self):
            self.fail(f"requirement missing: {what}")
        return lambda f: fail
    return unittest.skip(f"requirement missing: {what}")


def weight_bin(arrays):
    """A MILBlob v2 file holding `arrays` (fp16 or fp32), coremltools' layout."""
    out = bytearray(64)
    struct.pack_into("<II", out, 0, len(arrays), 2)
    for a in arrays:
        dtype = ctpl.MIL_FP16 if a.dtype == np.float16 else ctpl.MIL_FP32
        data = a.astype("<f2" if dtype == ctpl.MIL_FP16 else "<f4").tobytes()
        meta = len(out)
        hdr = bytearray(64)
        struct.pack_into("<IIQQ", hdr, 0, 0xDEADBEEF, dtype, len(data), meta + 64)
        out += hdr + data
        out += bytes((-len(out)) % 64)
    return bytes(out)


class WeightBinTest(unittest.TestCase):
    def test_round_trip(self):
        a = np.arange(5, dtype=np.float16)
        b = np.linspace(-1, 1, 7).astype(np.float32)
        blobs = ctpl.read_weight_bin(weight_bin([a, b]))
        self.assertEqual([x.offset for x in blobs], [64, 192])
        self.assertEqual([x.dtype_name for x in blobs], ["fp16", "fp32"])
        self.assertEqual([x.count for x in blobs], [5, 7])
        np.testing.assert_array_equal(blobs[0].array(), a)
        np.testing.assert_array_equal(blobs[1].array(), b)

    def test_malformed(self):
        good = bytearray(weight_bin([np.ones(4, np.float16)]))
        for mutate, msg in [
            (lambda d: struct.pack_into("<I", d, 4, 3), "version 3"),
            (lambda d: struct.pack_into("<I", d, 64, 0xCAFEBABE), "bad sentinel"),
            (lambda d: struct.pack_into("<Q", d, 64 + 8, 1 << 20), "has data offset"),
            (lambda d: struct.pack_into("<I", d, 0, 2), "past the end"),
            (lambda d: struct.pack_into("<I", d, 64 + 4, 7), "MIL dtype 7"),
        ]:
            d = bytearray(good)
            mutate(d)
            with self.assertRaisesRegex(ctpl.TemplateError, msg):
                ctpl.read_weight_bin(bytes(d))
        with self.assertRaises(ctpl.TemplateError):
            ctpl.read_weight_bin(b"\0" * 10)


def blob_of(arr, dtype=np.float16):
    return ctpl.read_weight_bin(weight_bin([np.asarray(arr).reshape(-1).astype(dtype)]))[0]


class ExplainTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = rng.standard_normal((4, 6)).astype(np.float32)
        self.v = rng.standard_normal(24).astype(np.float32)
        self.tensors = {"w": ("f32", self.w), "v": ("f32", self.v), "q": ("q8_0", None)}

    def test_cast(self):
        self.assertEqual(ctpl.explain_blob(blob_of(self.w), self.tensors)[0], 'f16(t("w"))')
        self.assertEqual(ctpl.explain_blob(blob_of(self.w, np.float32), self.tensors)[0], 'f32(t("w"))')

    def test_transpose(self):
        self.assertEqual(ctpl.explain_blob(blob_of(self.w.T), self.tensors)[0], 'f16(permute(t("w"), [1, 0]))')

    def test_scalar_multiple_and_negation(self):
        self.assertEqual(ctpl.explain_blob(blob_of(self.v * np.float32(0.5)), self.tensors)[0],
                         'f16(mul(t("v"), 0.5))')
        self.assertEqual(ctpl.explain_blob(blob_of(-self.w), self.tensors)[0], 'f16(mul(t("w"), -1))')
        k = np.float32(0.31830987)  # 1/pi in f32, as an exporter's constant
        got = ctpl.explain_blob(blob_of(self.v * k), self.tensors)[0]
        self.assertTrue(got.startswith('f16(mul(t("v"), '), got)
        lit = float(got.split(", ")[1].rstrip("))"))
        np.testing.assert_array_equal((self.v * np.float32(lit)).astype(np.float16),
                                      (self.v * k).astype(np.float16))

    def test_constants(self):
        self.assertTrue(ctpl.explain_blob(blob_of(np.zeros(9)), self.tensors)[0].startswith("f16(zeros([9]))"))
        self.assertTrue(ctpl.explain_blob(blob_of(np.ones(9)), self.tensors)[0].startswith("f16(ones([9]))"))

    def test_unexplained(self):
        self.assertEqual(ctpl.explain_blob(blob_of(np.sin(self.v)), self.tensors), [])

    def test_literal_round_trips(self):
        for k in (0.5, 1 / 3, 9.8672466, -2.5e-07, 60000.0):
            self.assertEqual(np.float32(float(ctpl._fmt_num(k))), np.float32(k))


class DraftTest(unittest.TestCase):
    def test_draft(self):
        rng = np.random.default_rng(3)
        w = rng.standard_normal((4, 6)).astype(np.float32)
        tensors = {"enc.w": ("f32", w), "enc.w_copy": ("f32", w.copy())}
        data = weight_bin([w.reshape(-1).astype(np.float16), np.sin(w).reshape(-1).astype(np.float16)])
        blobs = ctpl.read_weight_bin(data)
        manifest = {"tensors": [{"name": "enc.w", "type": "q8_0"}]}
        text, flagged, ambiguous = ctpl.draft_recipe("toy", "ab" * 32, len(data), blobs, tensors, manifest)
        lines = text.splitlines()
        self.assertEqual(lines[:5], ["mlrecipe 1", "template toy", f"spec_sha256 {'ab' * 32}",
                                     f"weight_bin_size {len(data)}", "blobs 2"])
        self.assertIn('source "enc.w" f32,q8_0 [4,6]', lines)
        self.assertIn('blob 64 fp16 24 f16(t("enc.w"))', lines)
        self.assertEqual(ambiguous, [64])  # two equal tensors: twin B decides
        self.assertEqual(flagged, [blobs[1].offset])
        self.assertIn(f"blob {blobs[1].offset} fp16 24 ???", lines)
        self.assertEqual(lines[-1], "end")

    def test_refresh_header(self):
        text = ("mlrecipe 1\ntemplate old\nspec_sha256 00\nweight_bin_size 1\nblobs 1\n"
                'source "a" f32 [2]\n# blobs follow\nblob 64 fp16 2 f16(t("a"))\nend\n')
        out = ctpl.refresh_recipe_header(text, "new", "ff" * 32, 192, 1)
        self.assertIn("template new\n", out)
        self.assertIn(f"spec_sha256 {'ff' * 32}\n", out)
        self.assertIn("weight_bin_size 192\n", out)
        self.assertTrue(out.endswith('source "a" f32 [2]\n# blobs follow\nblob 64 fp16 2 f16(t("a"))\nend\n'))
        with self.assertRaisesRegex(ctpl.TemplateError, "lacks blobs"):
            ctpl.refresh_recipe_header("mlrecipe 1\ntemplate a\nspec_sha256 0\nweight_bin_size 1\nend\n", "a", "0", 1, 1)


class TemplateJsonTest(unittest.TestCase):
    def test_expand_args(self):
        self.assertEqual(ctpl.expand_args(["--gguf", "{gguf}", "--output", "{out}/x", "--w", "128"], "/m.gguf", "/o"),
                         ["--gguf", "/m.gguf", "--output", "/o/x", "--w", "128"])
        with self.assertRaises(ctpl.TemplateError):
            ctpl.expand_args([128], "a", "b")

    def test_merge_keeps_author_fields(self):
        existing = {"id": "old", "family": "parler", "stage": "dac", "match": {"architecture": "parler",
                    "metadata": {"parler.dac.hop": 512}}, "default_compute_units": "cpu_and_ane",
                    "weight_bin_size": 1}
        computed = {"schema": 1, "id": "parler-dac-w128", "weight_bin_size": 4096,
                    "functions": [{"name": "main", "inputs": [], "outputs": []}], "stateful": False}
        j = ctpl.merge_template_json(existing, computed)
        self.assertEqual(j["id"], "parler-dac-w128")
        self.assertEqual(j["weight_bin_size"], 4096)
        self.assertEqual(j["family"], "parler")
        self.assertEqual(j["match"]["metadata"], {"parler.dac.hop": 512})
        self.assertEqual(j["default_compute_units"], "cpu_and_ane")
        self.assertEqual(j["expected_warm_load_ms"], {"main": 0})
        self.assertEqual(j["patches"], [])
        self.assertTrue(j["shareable"])

    def test_merge_defaults_match_from_manifest(self):
        manifest = {"metadata": [{"key": "general.architecture", "value": "qvac-toy"}]}
        j = ctpl.merge_template_json({}, {"stateful": True, "functions": []}, manifest)
        self.assertEqual(j["match"], {"architecture": "qvac-toy", "metadata": {}})
        self.assertFalse(j["shareable"])  # a stateful model gets one instance per owner


class ManifestTest(unittest.TestCase):
    def test_equivalent_modulo_identifiers(self):
        def manifest(ids, path="com.apple.CoreML/model.mlmodel"):
            return json.dumps({"fileFormatVersion": "1.0.0", "rootModelIdentifier": ids[1], "itemInfoEntries": {
                ids[0]: {"author": "com.apple.CoreML", "name": "weights", "path": "com.apple.CoreML/weights"},
                ids[1]: {"author": "com.apple.CoreML", "name": "model.mlmodel", "path": path}}})
        a = manifest(["A1", "A2"])
        self.assertTrue(ctpl.manifest_equivalent(a, manifest(["B1", "B2"])))
        self.assertFalse(ctpl.manifest_equivalent(a, manifest(["B1", "B2"], "com.apple.CoreML/other.mlmodel")))
        wrong_root = json.loads(manifest(["B1", "B2"]))
        wrong_root["rootModelIdentifier"] = "B1"  # the weights item
        self.assertFalse(ctpl.manifest_equivalent(a, json.dumps(wrong_root)))
        self.assertFalse(ctpl.manifest_equivalent(a, "not json"))


class SpecDiffTest(unittest.TestCase):
    @needs(HAVE_CT, "coremltools")
    def test_spec_diff(self):
        from coremltools.proto import Model_pb2

        a = Model_pb2.Model()
        a.specificationVersion = 9
        a.description.metadata.userDefined["k"] = "1"
        a.description.metadata.userDefined["same"] = "x"
        b = Model_pb2.Model()
        b.CopyFrom(a)
        self.assertEqual(ctpl.spec_diff(a, b), [])
        b.description.metadata.userDefined["k"] = "2"
        b.description.metadata.userDefined["extra"] = "y"
        b.specificationVersion = 8
        d = ctpl.spec_diff(a, b)
        self.assertIn("specificationVersion", d)
        self.assertIn("description.metadata.userDefined['k']", d)
        self.assertIn("description.metadata.userDefined['extra'] (only in B)", d)
        # coremltools' conversion date is not a difference.
        b.CopyFrom(a)
        a.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"] = "2026-11-13"
        b.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"] = "2026-11-14"
        self.assertEqual(ctpl.spec_diff(a, b), [])
        del a.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"]
        self.assertEqual(ctpl.spec_diff(a, b), [])
        del b.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"]
        # Deterministic serialisation is independent of map insertion order.
        c = Model_pb2.Model()
        c.specificationVersion = 9
        c.description.metadata.userDefined["same"] = "x"
        c.description.metadata.userDefined["k"] = "1"
        self.assertEqual(a.SerializeToString(deterministic=True), c.SerializeToString(deterministic=True))


class ToySuggestTest(unittest.TestCase):
    """`suggest` on the toy exporter explains every blob, and the draft equals
    the committed toy recipe (whose blobs were all suggested)."""

    @needs(HAVE_CT and HAVE_GGUF and TOOL and Path(TOOL).is_file(), "coremltools, gguf and $QVAC_COREML_TOOL")
    def test_suggest_stateless_toy(self):
        toy = HERE.parent / "test" / "toy-templates" / "toy-stateless"
        exporter = HERE.parent / "test" / "toy" / "export_toy.py"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "recipe.txt"
            p = subprocess.run([sys.executable, str(HERE / "coreml_template.py"), "--tool", TOOL, "suggest",
                                "--manifest", str(toy / "reference-gguf.json"), "--exporter", str(exporter),
                                "--exporter-arg=--kind", "--exporter-arg=stateless", "--exporter-arg=--gguf",
                                "--exporter-arg={gguf}", "--exporter-arg=--output", "--exporter-arg={out}",
                                "--id", "toy-stateless", "-o", str(out)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("0 unexplained, 0 ambiguous", p.stderr)
            # spec_sha256 hashes the exporter's spec, which carries the
            # conversion date: everything else must match.
            def body(text):
                return [line for line in text.splitlines() if not line.startswith("spec_sha256 ")]
            self.assertEqual(body(out.read_text()), body((toy / "recipe.txt").read_text()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
