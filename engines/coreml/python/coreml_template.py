#!/usr/bin/env python3
"""Make and check Core ML generation templates.

A template (engines/<engine>/coreml-templates/<id>/) is the weight-free
model.mlmodel and Manifest.json of an unchanged exporter's .mlpackage, plus a
hand-completed recipe.txt that says how every weight.bin blob is computed from
GGUF tensors, plus template.json. This script orchestrates the exporter and
qvac-coreml-tool; it never reimplements the generator (every weight.bin it
compares is written by the C++ interpreter through `qvac-coreml-tool`).

  suggest   Runs the exporter on twin A and content-matches every weight.bin
            blob against f16(t(name)), transposes and simple compiler forms
            (scalar multiples, negation, zeros, ones). Writes a draft
            recipe.txt; blobs it cannot explain are flagged for the author.
  extract   Runs the exporter on twins A and B, checks that the two specs are
            structurally equal and that their weight.bin layouts agree, then
            writes model.mlmodel (deterministic serialisation of A's spec),
            Manifest.json and template.json, and refreshes the header of an
            existing recipe.txt.
  report    The extraction and cross-seed gates: for each template, runs the
            exporter on twin A (f32), twin B (f32) and twin Q (quantised; the
            exporter sees `qvac-coreml-tool f32 Q`), generates weight.bin with
            the C++ generator and compares blob by blob (`qvac-coreml-tool
            verify`); the exporter's spec must equal the template's.

Twins come from the template's reference-gguf.json (`qvac-coreml-tool dump`
of a real GGUF) through `qvac-coreml-tool twin`. The exporter is described by
template.json `exporter`: {"path", "args"}; in `args`, {gguf} is replaced by
the input GGUF and {out} by an output directory, which must then hold exactly
one .mlpackage.

Requirements: numpy, gguf (gguf-py) and coremltools, pinned by
engines/parakeet/scripts/requirements-coreml.txt; the exporter's own
requirements; a built qvac-coreml-tool (--tool, $QVAC_COREML_TOOL or PATH).
"""

import argparse
import hashlib
import itertools
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED_A, SEED_B, SEED_Q = 1, 2, 3
SCHEMA = 1
MIL_FP16, MIL_FP32 = 1, 2
REQUIRED_FILES = ("Manifest.json", "model.mlmodel", "recipe.txt", "template.json")
# Keys of template.json that `extract` recomputes; everything else is the author's.
COMPUTED_KEYS = ("schema", "id", "exporter", "toolchain", "functions", "stateful", "weight_bin_size",
                 "generation_peak_bytes")


class TemplateError(RuntimeError):
    pass


# ---- weight.bin (MILBlob v2) --------------------------------------------------------


@dataclass
class Blob:
    offset: int        # metadata offset (the recipe's blob offset)
    dtype: int         # MIL_FP16 | MIL_FP32
    nbytes: int
    data_offset: int
    raw: bytes = field(repr=False, default=b"")

    @property
    def dtype_name(self):
        return "fp16" if self.dtype == MIL_FP16 else "fp32"

    @property
    def count(self):
        return self.nbytes // (2 if self.dtype == MIL_FP16 else 4)

    def array(self):
        return np.frombuffer(self.raw, dtype="<f2" if self.dtype == MIL_FP16 else "<f4")


def read_weight_bin(data):
    """Parses a MILBlob v2 weight.bin (bytes). Raises TemplateError when malformed."""
    if len(data) < 64:
        raise TemplateError("weight.bin: shorter than its header")
    count, version = struct.unpack_from("<II", data, 0)
    if version != 2:
        raise TemplateError(f"weight.bin: version {version}, expected 2")
    blobs, off = [], 64
    for i in range(count):
        if off + 64 > len(data):
            raise TemplateError(f"weight.bin: blob {i} metadata at {off} is past the end")
        sentinel, dtype, nbytes, doff = struct.unpack_from("<IIQQ", data, off)
        if sentinel != 0xDEADBEEF:
            raise TemplateError(f"weight.bin: bad sentinel at {off}")
        if doff != off + 64 or doff + nbytes > len(data):
            raise TemplateError(f"weight.bin: blob at {off} has data offset {doff} and {nbytes} bytes")
        if dtype not in (MIL_FP16, MIL_FP32):
            raise TemplateError(f"weight.bin: blob at {off} has MIL dtype {dtype} (only fp16 and fp32 are generated)")
        blobs.append(Blob(off, dtype, nbytes, doff, bytes(data[doff:doff + nbytes])))
        off = (doff + nbytes + 63) // 64 * 64
    return blobs


def layout_of(blobs):
    return [(b.offset, b.dtype, b.nbytes) for b in blobs]


# ---- packages, specs ------------------------------------------------------------------


def package_parts(pkg):
    pkg = Path(pkg)
    data = pkg / "Data" / "com.apple.CoreML"
    parts = {"spec": data / "model.mlmodel", "weights": data / "weights" / "weight.bin",
             "manifest": pkg / "Manifest.json"}
    for k, p in parts.items():
        if not p.is_file():
            raise TemplateError(f"{pkg}: no {p.relative_to(pkg)}")
    return parts


def load_spec(path):
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString(Path(path).read_bytes())
    return spec


def canonical_spec_bytes(path):
    """Deterministic serialisation (sorted map fields): the template's model.mlmodel."""
    return load_spec(path).SerializeToString(deterministic=True)


# Spec metadata coremltools stamps on every conversion that does not depend
# on the weights: ignored when two specs are compared (an export on another
# day is the same model).
VOLATILE_METADATA = ("com.github.apple.coremltools.conversion_date",)


def manifest_equivalent(a, b):
    """Two Manifest.json texts that differ at most in their random item identifiers."""
    def norm(text):
        j = json.loads(text)
        items = j.get("itemInfoEntries", {})
        root = items.get(j.get("rootModelIdentifier"), {})
        return (j.get("fileFormatVersion"), root.get("path"),
                sorted(tuple(sorted(v.items())) for v in items.values()))
    try:
        return norm(a) == norm(b)
    except (ValueError, AttributeError):
        return False


def spec_diff(a, b, path="", out=None, limit=50):
    """Paths at which two protobuf messages differ (field by field, maps by key).

    The user metadata keys in VOLATILE_METADATA are ignored.
    """
    from google.protobuf.descriptor import FieldDescriptor

    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if a.DESCRIPTOR is not b.DESCRIPTOR:
        out.append(path or "<root>")
        return out
    for fd in a.DESCRIPTOR.fields:
        name = f"{path}.{fd.name}" if path else fd.name
        va, vb = getattr(a, fd.name), getattr(b, fd.name)
        is_msg = fd.type == FieldDescriptor.TYPE_MESSAGE
        repeated = fd.is_repeated if hasattr(fd, "is_repeated") else fd.label == FieldDescriptor.LABEL_REPEATED
        if repeated:
            if is_msg and fd.message_type.GetOptions().map_entry:
                ka, kb = set(va.keys()), set(vb.keys())
                if name.endswith("metadata.userDefined"):
                    ka -= set(VOLATILE_METADATA)
                    kb -= set(VOLATILE_METADATA)
                for k in sorted(ka ^ kb, key=str):
                    out.append(f"{name}[{k!r}] (only in {'A' if k in ka else 'B'})")
                vf = fd.message_type.fields_by_name["value"]
                for k in sorted(ka & kb, key=str):
                    if vf.type == FieldDescriptor.TYPE_MESSAGE:
                        spec_diff(va[k], vb[k], f"{name}[{k!r}]", out, limit)
                    elif va[k] != vb[k]:
                        out.append(f"{name}[{k!r}]")
            elif len(va) != len(vb):
                out.append(f"{name} (length {len(va)} vs {len(vb)})")
            elif is_msg:
                for i, (x, y) in enumerate(zip(va, vb)):
                    spec_diff(x, y, f"{name}[{i}]", out, limit)
            elif list(va) != list(vb):
                out.append(name)
        elif is_msg:
            ha, hb = a.HasField(fd.name), b.HasField(fd.name)
            if ha != hb:
                out.append(f"{name} (set in {'A' if ha else 'B'} only)")
            elif ha:
                spec_diff(va, vb, name, out, limit)
        elif va != vb:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def spec_summary(spec):
    """functions, their inputs/outputs and whether the model is stateful (template.json)."""
    from coremltools.proto import FeatureTypes_pb2  # noqa: F401  (registers the types)

    desc = spec.description

    def feats(items):
        res = []
        for f in items:
            t = f.type
            kind = t.WhichOneof("Type")
            if kind == "stateType":
                t = t.stateType
                kind = t.WhichOneof("Type")
            if kind == "multiArrayType":
                arr = getattr(t, kind)
                dt = {65552: "fp16", 65568: "fp32", 65600: "fp64", 131104: "int32"}.get(arr.dataType, str(arr.dataType))
                res.append({"name": f.name, "dtype": dt, "shape": [int(s) for s in arr.shape]})
            else:
                res.append({"name": f.name, "dtype": kind or "unknown", "shape": []})
        return res

    functions = []
    if len(desc.functions):
        for fn in desc.functions:
            functions.append({"name": fn.name, "inputs": feats(fn.input), "outputs": feats(fn.output),
                              "states": feats(fn.state)})
    else:
        functions.append({"name": "main", "inputs": feats(desc.input), "outputs": feats(desc.output),
                          "states": feats(desc.state)})
    stateful = any(f["states"] for f in functions)
    for f in functions:
        f.pop("states")
    return functions, stateful


# ---- qvac-coreml-tool and GGUFs ---------------------------------------------------------


def find_tool(explicit=None):
    cand = explicit or os.environ.get("QVAC_COREML_TOOL") or shutil.which("qvac-coreml-tool")
    if not cand or not Path(cand).is_file():
        raise TemplateError("qvac-coreml-tool not found: pass --tool or set QVAC_COREML_TOOL")
    return str(Path(cand).resolve())


def run(cmd, what, capture=True, env=None):
    p = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True, env=env)
    if p.returncode != 0:
        detail = (p.stderr or p.stdout or "").strip() if capture else ""
        raise TemplateError(f"{what} failed (exit {p.returncode}): {' '.join(map(str, cmd))}\n{detail}")
    return p.stdout if capture else ""


def make_twin(tool, manifest, seed, types, out, sets=()):
    cmd = [tool, "twin", "--manifest", manifest, "--seed", str(seed), "--types", types, "-o", out]
    for s in sets:
        cmd += ["--set", s]
    run(cmd, f"twin seed {seed}")
    return out


def gguf_tensors(path):
    """name -> (ggml type name, numpy-order f32 array or None when quantised)."""
    import gguf

    r = gguf.GGUFReader(str(path))
    out = {}
    for t in r.tensors:
        tname = t.tensor_type.name.lower()
        arr = None
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            arr = np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(list(t.shape))])
        out[t.name] = (tname, arr)
    return out


# ---- the exporter ---------------------------------------------------------------------------


def repo_root():
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "engines").is_dir() and (p / ".git").exists():
            return p
    return here.parents[3]


def expand_args(args, gguf, out):
    res = []
    for a in args:
        if not isinstance(a, str):
            raise TemplateError(f"exporter args must be strings: {a!r}")
        res.append(a.replace("{gguf}", str(gguf)).replace("{out}", str(out)))
    return res


def run_exporter(exporter, gguf, out_dir, python=None, root=None):
    """Runs template.json-style `exporter` {path, args} on `gguf`; returns the .mlpackage."""
    root = Path(root or repo_root())
    script = Path(exporter["path"])
    if not script.is_absolute():
        script = root / script
    if not script.is_file():
        raise TemplateError(f"exporter {script} not found")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python or sys.executable, str(script)] + expand_args(exporter.get("args", []), gguf, out_dir)
    run(cmd, f"exporter {script.name}")
    pkgs = sorted(p for p in out_dir.rglob("*.mlpackage") if p.is_dir())
    if len(pkgs) != 1:
        raise TemplateError(f"exporter {script.name} left {len(pkgs)} .mlpackage under {out_dir}, expected 1")
    return pkgs[0]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- suggest: content matching ----------------------------------------------------------------


def _fmt_num(k):
    """A literal that strtod parses back to the same f32 value."""
    s = repr(float(np.float32(k)))
    return s[:-2] if s.endswith(".0") else s


def _cast(arr, dtype):
    return arr.astype(np.float16) if dtype == MIL_FP16 else arr.astype(np.float32)


def _equal(blob_arr, cand, dtype):
    if cand.size != blob_arr.size:
        return False
    c = _cast(cand.reshape(-1), dtype)
    if not np.all(np.isfinite(c)):
        return False
    return c.view(np.uint16 if dtype == MIL_FP16 else np.uint32).tobytes() == \
        blob_arr.view(np.uint16 if dtype == MIL_FP16 else np.uint32).tobytes()


def _layouts(arr, max_rank=4):
    """(permutation or None, array) for the identity and every axis permutation (rank <= 4)."""
    yield None, arr
    if 2 <= arr.ndim <= max_rank:
        for perm in itertools.permutations(range(arr.ndim)):
            if list(perm) != list(range(arr.ndim)):
                yield list(perm), np.transpose(arr, perm)


def _expr_t(name, perm):
    e = f't("{name}")'
    return e if perm is None else f"permute({e}, {perm})"


def _scalar_multiple(b, flat, dtype):
    """The shortest decimal k with cast(flat * f32(k)) == b bit for bit, or None.

    Every element bounds k: the output cast maps an interval of products to
    b[i]. The intersection of those intervals holds the exporter's constant;
    its shortest decimal (up to 9 significant digits, then the nearest f32
    values) is suggested. The cast hides the constant's last digits, so the
    author confirms a suggested literal against the exporter.
    """
    x = flat.astype(np.float64)
    y = b.astype(np.float64)
    if not np.all(np.isfinite(y)):
        return None
    nz = x != 0
    if not np.any(nz) or np.any(y[~nz] != 0):
        return None
    ftype = np.float16 if dtype == MIL_FP16 else np.float32
    half = np.spacing(np.abs(b.astype(ftype))).astype(np.float64) / 2
    a1 = (y[nz] - half[nz]) / x[nz]
    a2 = (y[nz] + half[nz]) / x[nz]
    lo, hi = float(np.max(np.minimum(a1, a2))), float(np.min(np.maximum(a1, a2)))
    if not (lo <= hi) or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    mid = (lo + hi) / 2
    if mid == 0:
        return None
    cands = []
    for digits in range(1, 10):
        scale = 10.0 ** (int(np.floor(np.log10(abs(mid)))) - digits + 1)
        base = round(mid / scale)
        for c in sorted({base, base - 1, base + 1}, key=lambda c: abs(c * scale - mid)):
            v = float(f"{c * scale:.{digits}g}")
            if lo <= v <= hi:
                cands.append(np.float32(v))
    f = np.float32(mid)
    cands += [f, np.nextafter(f, np.float32(np.inf)), np.nextafter(f, np.float32(-np.inf))]
    for k in cands:
        if k in (0, 1, -1) or not np.isfinite(k):
            continue
        if _equal(b, flat * k, dtype):
            return k
    return None


def explain_blob(blob, tensors, max_candidates=8):
    """Recipe expressions (without the outer cast) that reproduce `blob` bit for bit."""
    b = blob.array()
    cast = "f16" if blob.dtype == MIL_FP16 else "f32"
    out = []
    # Compiler-synthesised constants.
    if not np.any(b.view(np.uint16 if blob.dtype == MIL_FP16 else np.uint32)):
        out.append(f"{cast}(zeros([{blob.count}]))  # needs zeros(), a later generator version")
    elif np.all(b == 1):
        out.append(f"{cast}(ones([{blob.count}]))  # needs ones(), a later generator version")
    for name, (_, arr) in tensors.items():
        if arr is None or arr.size != blob.count:
            continue
        for perm, lay in _layouts(arr):
            if _equal(b, lay, blob.dtype):
                out.append(f"{cast}({_expr_t(name, perm)})")
            elif _equal(b, -lay, blob.dtype):
                out.append(f"{cast}(mul({_expr_t(name, perm)}, -1))")
            else:
                k = _scalar_multiple(b, lay.reshape(-1).astype(np.float32), blob.dtype)
                if k is not None:
                    out.append(f"{cast}(mul({_expr_t(name, perm)}, {_fmt_num(k)}))")
            if len(out) >= max_candidates:
                return out
    return out


def draft_recipe(template_id, spec_sha256, weight_bin, blobs, tensors, manifest=None):
    """Draft recipe.txt text and the list of unexplained blob offsets."""
    ref_types = {}
    if manifest:
        for t in manifest.get("tensors", []):
            ref_types[t["name"]] = t["type"]
    body, used, flagged, ambiguous = [], [], [], []
    for b in blobs:
        cands = explain_blob(b, tensors)
        if not cands:
            flagged.append(b.offset)
            body.append(f"# UNEXPLAINED: complete this blob by hand (see the recipe language in include/qvac_coreml/recipe.h)")
            body.append(f"blob {b.offset} {b.dtype_name} {b.count} ???")
            continue
        if len(cands) > 1:
            ambiguous.append(b.offset)
            body.append("# AMBIGUOUS on this twin; other candidates: " + " | ".join(cands[1:]))
        body.append(f"blob {b.offset} {b.dtype_name} {b.count} {cands[0]}")
        for c in cands[:1]:
            for name in tensors:
                if f't("{name}")' in c and name not in used:
                    used.append(name)
    head = ["mlrecipe 1", f"template {template_id}", f"spec_sha256 {spec_sha256}",
            f"weight_bin_size {weight_bin}", f"blobs {len(blobs)}"]
    for name in used:
        _, arr = tensors[name]
        types = ["f32"]
        rt = ref_types.get(name)
        if rt and rt not in types:
            types.append(rt)
        shape = ",".join(str(d) for d in arr.shape)
        head.append(f'source "{name}" {",".join(types)} [{shape}]')
    text = "\n".join(head + body + ["end", ""])
    return text, flagged, ambiguous


def refresh_recipe_header(text, template_id, spec_sha256, weight_bin_size, blobs):
    """Rewrites the four header values of an existing recipe; the body is the author's."""
    want = {"template": template_id, "spec_sha256": spec_sha256, "weight_bin_size": str(weight_bin_size),
            "blobs": str(blobs)}
    out, seen = [], set()
    for line in text.splitlines():
        key = line.split(" ", 1)[0] if line and not line.startswith(("#", " ", "\t")) else ""
        if key in want and key not in seen:
            out.append(f"{key} {want[key]}")
            seen.add(key)
        else:
            out.append(line)
    missing = set(want) - seen
    if missing:
        raise TemplateError(f"recipe.txt: header lacks {', '.join(sorted(missing))}")
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


# ---- template.json ---------------------------------------------------------------------------------


def toolchain_versions():
    res = {}
    for mod in ("coremltools", "torch", "numpy"):
        try:
            res[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - an exporter may not need torch
            pass
    return res


def merge_template_json(existing, computed, manifest=None):
    """The author's fields of `existing` (family, stage, match, memory, ...) plus `computed`."""
    j = dict(existing or {})
    j.update(computed)
    j.setdefault("family", "")
    j.setdefault("stage", "")
    j.setdefault("min_os", {"macos": "15.0"})
    j.setdefault("multifunction_ane", False)
    j.setdefault("default_compute_units", "all")
    j.setdefault("shareable", not computed.get("stateful", False))
    j.setdefault("pad_safe", False)
    j.setdefault("export_kind", "")
    j.setdefault("state_bytes", 0)
    j.setdefault("expected_warm_load_ms", {f["name"]: 0 for f in computed.get("functions", [])})
    j.setdefault("memory", {"runtime_weight_bytes": computed.get("weight_bin_size", 0), "workspace_bytes": 0,
                            "compile_peak_bytes": None, "ane_compile_reserve_bytes": None,
                            "load_peak_bytes": {"cpu_and_ane": None, "all": None, "cpu_and_gpu": None,
                                                "cpu_only": None},
                            "margin": 1.25, "coreml_bound": "conservative", "validated_for": []})
    j.setdefault("patches", [])
    j.setdefault("prefer_ggml_on", [])
    if "match" not in j:
        arch = ""
        for m in (manifest or {}).get("metadata", []):
            if m.get("key") == "general.architecture":
                arch = m.get("value", "")
        j["match"] = {"architecture": arch, "metadata": {}}
    j.setdefault("literals", [])
    j.setdefault("provided", [])
    return j


# ---- commands --------------------------------------------------------------------------------------


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if m.get("schema") != 1:
        raise TemplateError(f"{path}: not a schema-1 reference-gguf.json")
    return m


def exporter_from_args(a):
    return {"path": a.exporter, "args": list(a.exporter_arg or [])}


def cmd_suggest(a):
    tool = find_tool(a.tool)
    manifest = load_manifest(a.manifest)
    with tempfile.TemporaryDirectory(prefix="qvac-coreml-suggest-") as tmp:
        twin = make_twin(tool, a.manifest, SEED_A, "f32", Path(tmp) / "twin-a.gguf")
        pkg = run_exporter(exporter_from_args(a), twin, Path(tmp) / "export-a", a.python)
        parts = package_parts(pkg)
        data = parts["weights"].read_bytes()
        blobs = read_weight_bin(data)
        spec_sha = hashlib.sha256(canonical_spec_bytes(parts["spec"])).hexdigest()
        text, flagged, ambiguous = draft_recipe(a.id, spec_sha, len(data), blobs, gguf_tensors(twin), manifest)
    Path(a.output).write_text(text)
    print(f"suggest: {len(blobs)} blobs, {len(flagged)} unexplained, {len(ambiguous)} ambiguous -> {a.output}",
          file=sys.stderr)
    return 0


def _export_twin(tool, tdir_manifest, exporter, seed, types, work, python, root, label, sets=()):
    twin = make_twin(tool, tdir_manifest, seed, types, work / f"twin-{label}.gguf", sets)
    feed = twin
    if types != "f32":
        feed = work / f"twin-{label}-f32.gguf"
        run([tool, "f32", twin, "-o", feed], f"f32 of twin {label}")
    pkg = run_exporter(exporter, feed, work / f"export-{label}", python, root)
    return twin, pkg


def cmd_extract(a):
    tool = find_tool(a.tool)
    out = Path(a.template_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(a.manifest) if a.manifest else out / "reference-gguf.json"
    manifest = load_manifest(manifest_path)
    if manifest_path.resolve() != (out / "reference-gguf.json").resolve():
        shutil.copyfile(manifest_path, out / "reference-gguf.json")
    existing = json.loads((out / "template.json").read_text()) if (out / "template.json").is_file() else {}
    exporter = exporter_from_args(a) if a.exporter else existing.get("exporter")
    if not exporter or "path" not in exporter:
        raise TemplateError("extract: no exporter (pass --exporter or keep template.json `exporter`)")
    root = Path(a.root) if a.root else repo_root()
    template_id = a.id or existing.get("id") or out.name
    with tempfile.TemporaryDirectory(prefix="qvac-coreml-extract-") as tmp:
        work = Path(tmp)
        _, pkg_a = _export_twin(tool, manifest_path, exporter, SEED_A, "f32", work, a.python, root, "a")
        _, pkg_b = _export_twin(tool, manifest_path, exporter, SEED_B, "f32", work, a.python, root, "b")
        pa, pb = package_parts(pkg_a), package_parts(pkg_b)
        spec_a, spec_b = load_spec(pa["spec"]), load_spec(pb["spec"])
        diff = spec_diff(spec_a, spec_b)
        if diff:
            raise TemplateError("extract: the exporter's spec differs between twin A and twin B at\n  " +
                                "\n  ".join(diff) + "\nValues in the spec that depend on the weights need "
                                "patches, which this generator version does not support.")
        wa, wb = pa["weights"].read_bytes(), pb["weights"].read_bytes()
        la, lb = layout_of(read_weight_bin(wa)), layout_of(read_weight_bin(wb))
        if la != lb or len(wa) != len(wb):
            raise TemplateError("extract: weight.bin layout differs between twin A and twin B "
                                "(blob count, offsets, dtypes or sizes)")
        spec_bytes = spec_a.SerializeToString(deterministic=True)
        # Keep an unchanged spec and manifest byte for byte (the manifest's
        # item identifiers and the conversion date are random per export), so
        # re-extracting an unchanged template leaves its template_sha256 alone.
        old_spec = out / "model.mlmodel"
        if old_spec.is_file() and not spec_diff(load_spec(old_spec), spec_a):
            spec_bytes = old_spec.read_bytes()
        else:
            old_spec.write_bytes(spec_bytes)
        new_manifest = pa["manifest"].read_text()
        old_manifest = out / "Manifest.json"
        if not (old_manifest.is_file() and manifest_equivalent(old_manifest.read_text(), new_manifest)):
            old_manifest.write_text(new_manifest)
        functions, stateful = spec_summary(spec_a)
        spec_sha = hashlib.sha256(spec_bytes).hexdigest()
        script = Path(exporter["path"])
        exp_abs = script if script.is_absolute() else root / script
        computed = {"schema": SCHEMA, "id": template_id,
                    "exporter": {"path": exporter["path"], "sha256": file_sha256(exp_abs),
                                 "args": exporter.get("args", [])},
                    "toolchain": toolchain_versions(), "functions": functions, "stateful": stateful,
                    "weight_bin_size": len(wa)}
        recipe = out / "recipe.txt"
        if recipe.is_file():
            recipe.write_text(refresh_recipe_header(recipe.read_text(), template_id, spec_sha, len(wa), len(la)))
        j = merge_template_json(existing, computed, manifest)
        j.setdefault("generation_peak_bytes", 0)
        (out / "template.json").write_text(json.dumps(j, indent=2) + "\n")
        if recipe.is_file():
            # Generate from twin A once: the recipe must parse and match; the
            # generator's peak memory goes into template.json.
            gen = json.loads(run([tool, "generate", "--template", out, "--gguf", work / "twin-a.gguf",
                                  "-o", work / "gen-a.mlpackage"], "generate (twin A)"))
            j["generation_peak_bytes"] = int(gen["generation_peak_bytes"])
            (out / "template.json").write_text(json.dumps(j, indent=2) + "\n")
        else:
            print(f"extract: no recipe.txt in {out}; run `suggest`, complete it, then extract again",
                  file=sys.stderr)
    print(f"extract: {template_id}: spec {spec_sha[:16]}, {len(la)} blobs, weight.bin {len(wa)} bytes -> {out}",
          file=sys.stderr)
    return 0


def check_template(tool, tdir, python=None, root=None, quant_types="q8_0", work=None):
    """The extraction (A) and cross-seed (B, Q) gates for one template. Returns result dicts."""
    tdir = Path(tdir)
    for f in REQUIRED_FILES + ("reference-gguf.json",):
        if not (tdir / f).is_file():
            raise TemplateError(f"{tdir}: missing {f}")
    tj = json.loads((tdir / "template.json").read_text())
    exporter = tj.get("exporter")
    if not exporter:
        raise TemplateError(f"{tdir}: template.json has no exporter")
    template_spec = load_spec(tdir / "model.mlmodel")
    results = []
    cases = [("A", SEED_A, "f32"), ("B", SEED_B, "f32"), ("Q", SEED_Q, quant_types)]
    owned = work is None
    work = Path(work or tempfile.mkdtemp(prefix="qvac-coreml-report-"))
    try:
        for label, seed, types in cases:
            r = {"template": tj.get("id", tdir.name), "twin": label, "seed": seed, "types": types}
            twin, pkg = _export_twin(tool, tdir / "reference-gguf.json", exporter, seed, types, work, python, root,
                                     label.lower())
            parts = package_parts(pkg)
            r["spec_diff"] = spec_diff(template_spec, load_spec(parts["spec"]))
            p = subprocess.run([tool, "verify", "--template", str(tdir), "--gguf", str(twin), "--ref",
                                str(parts["weights"])], capture_output=True, text=True)
            r["verify_exit"] = p.returncode
            r["verify"] = p.stdout.strip().splitlines()
            r["verify_stderr"] = p.stderr.strip()
            r["pass"] = p.returncode == 0 and not r["spec_diff"]
            results.append(r)
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)
    return results


def find_templates(root):
    root = Path(root)
    return sorted(p.parent for p in root.glob("engines/*/coreml-templates/*/template.json"))


def cmd_report(a):
    tool = find_tool(a.tool)
    root = Path(a.root) if a.root else repo_root()
    dirs = [Path(d) for d in (a.template_dir or [])]
    if a.all:
        dirs += find_templates(root)
    if not dirs and not a.allow_empty:
        raise TemplateError("report: no templates (pass --template-dir or --all)")
    all_results, ok = [], True
    for d in dirs:
        try:
            res = check_template(tool, d, a.python, root, a.quant_types)
        except TemplateError as e:
            res = [{"template": d.name, "twin": "-", "pass": False, "error": str(e)}]
        for r in res:
            ok &= r["pass"]
            status = "PASS" if r["pass"] else "FAIL"
            extra = r.get("error") or (f"spec differs at {r['spec_diff'][:3]}" if r.get("spec_diff") else
                                       (r["verify"][-1] if r.get("verify") else r.get("verify_stderr", "")))
            print(f"{status} {r['template']} twin {r['twin']} ({r.get('types', '-')}): {extra}")
            if not r["pass"] and r.get("verify"):
                for line in r["verify"]:
                    if "FAIL" in line or "error" in line.lower():
                        print(f"    {line}")
        all_results += res
    if a.json:
        Path(a.json).write_text(json.dumps(all_results, indent=1) + "\n")
    print(f"report: {len(dirs)} template(s), {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", help="qvac-coreml-tool (default: $QVAC_COREML_TOOL, then PATH)")
    ap.add_argument("--python", help="interpreter for the exporter (default: this one)")
    ap.add_argument("--root", help="repository root for relative exporter paths (default: found from this file)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suggest", help="draft recipe.txt from twin A")
    s.add_argument("--manifest", required=True, help="reference-gguf.json")
    s.add_argument("--exporter", required=True, help="exporter script")
    s.add_argument("--exporter-arg", action="append", help="one exporter argument ({gguf}, {out}); repeatable")
    s.add_argument("--id", required=True, help="template id")
    s.add_argument("-o", "--output", required=True, help="draft recipe.txt")
    s.set_defaults(fn=cmd_suggest)

    e = sub.add_parser("extract", help="write model.mlmodel, Manifest.json and template.json")
    e.add_argument("--template-dir", required=True)
    e.add_argument("--manifest", help="reference-gguf.json (default: the template directory's)")
    e.add_argument("--exporter", help="exporter script (default: template.json `exporter`)")
    e.add_argument("--exporter-arg", action="append")
    e.add_argument("--id", help="template id (default: template.json `id`, then the directory name)")
    e.set_defaults(fn=cmd_extract)

    r = sub.add_parser("report", help="extraction and cross-seed gates against the exporter")
    r.add_argument("--template-dir", action="append", help="repeatable")
    r.add_argument("--all", action="store_true", help="every engines/*/coreml-templates/* under --root")
    r.add_argument("--allow-empty", action="store_true", help="succeed when there is no template")
    r.add_argument("--quant-types", default="q8_0", help="twin Q block types (default q8_0)")
    r.add_argument("--json", help="also write the results as JSON")
    r.set_defaults(fn=cmd_report)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except TemplateError as e:
        print(f"coreml_template: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
