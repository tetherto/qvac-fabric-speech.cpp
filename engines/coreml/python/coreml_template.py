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
            existing recipe.txt. Every tensor must differ between A and B
            (one equal in both would hide whether the exporter folds it into
            the program). Inline spec constants whose values differ
            between A and B (weights folded into the spec) become patch sites:
            model.mlmodel holds a sentinel there and template.json `patches`
            records it; the recipe's patch lines compute the values. The same
            holds for a spec metadata string that differs between A and B: for
            every `meta "<key>" str` line of the recipe, twins A and B carry
            different values of that GGUF string, so a spec string the
            exporter copies from it (a checkpoint hash) becomes a `str` patch
            site the recipe fills with meta_s("<key>"). Literals
            declared in template.json with an `exporter_blob` offset are copied
            from that weight.bin blob into literals.bin, once A and B agree on it.
  report    The extraction and cross-seed gates: for each template, runs the
            exporter on twin A (f32), twin B (f32) and twin Q (quantised; the
            exporter sees `qvac-coreml-tool f32 Q`), generates the package with
            the C++ generator (`qvac-coreml-tool generate`) and compares its
            weight.bin blob by blob (`qvac-coreml-tool verify`); the generated
            spec, patches applied, must equal the exporter's. Each twin
            carries its own value of every GGUF string the recipe declares
            (`meta "<key>" str`), so a string baked into the spec fails B and Q.
  real      The real-weights gate on one GGUF (a production model): the
            exporter runs on `qvac-coreml-tool f32` of it, the C++ generator
            reads the GGUF itself, and the two packages are compared as in
            `report`.

Twins come from the template's reference-gguf.json (`qvac-coreml-tool dump`
of a real GGUF) through `qvac-coreml-tool twin`. The exporter is described by
template.json `exporter`: {"path", "args"[, "gguf_name"]}; in `args`, {gguf}
is replaced by the input GGUF and {out} by an output directory, which must
then hold exactly one .mlpackage. With `gguf_name`, the exporter always sees
the input GGUF under that file name (for exporters that record it in the
spec), so every export of the template carries the same name.

Requirements: numpy, gguf (gguf-py) and coremltools, pinned by
engines/parakeet/scripts/requirements-coreml.txt; the exporter's own
requirements; a built qvac-coreml-tool (--tool, $QVAC_COREML_TOOL or PATH).
"""

import argparse
import hashlib
import itertools
import json
import os
import re
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
                 "generation_peak_bytes", "patches")
# MIL DataType values of the inline constants a patch can rewrite.
MIL_TYPE_FP16, MIL_TYPE_FP32 = 10, 11
SENTINEL_TAG = b"qvac-coreml-sentinel\0"


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


# ---- inline constants and patch sites ------------------------------------------------------


def inline_constants(spec):
    """(path, attribute) of every op attribute whose tensor value is inline in an ML program spec.

    `path` names the function, block specialisation, op output and attribute,
    so it is the same in two exports of one graph.
    """
    out = []
    if spec.WhichOneof("Type") != "mlProgram":
        return out

    def walk(block, where):
        for i, op in enumerate(block.operations):
            name = op.outputs[0].name if len(op.outputs) else f"#{i}"
            for k in sorted(op.attributes):
                a = op.attributes[k]
                if a.WhichOneof("value") == "immediateValue" and a.immediateValue.WhichOneof("value") == "tensor":
                    out.append((f"{where}/{name}.{k}", a))
            for j, b in enumerate(op.blocks):
                walk(b, f"{where}/{name}[{j}]")

    prog = spec.mlProgram
    for fname in sorted(prog.functions):
        fn = prog.functions[fname]
        for bname in sorted(fn.block_specializations):
            walk(fn.block_specializations[bname], f"{fname}:{bname}")
    return out


def _tensor_payload(attr):
    t = attr.immediateValue.tensor
    kind = t.WhichOneof("value")
    if kind == "bytes":
        return kind, bytes(t.bytes.values)
    if kind == "floats":
        return kind, struct.pack(f"<{len(t.floats.values)}f", *t.floats.values)
    return kind, getattr(t, kind).SerializeToString(deterministic=True) if kind else b""


def differing_constants(spec_a, spec_b):
    """The inline constants of A whose values differ in B: [(path, attribute of A, 'fp16'|'fp32')].

    Raises TemplateError when one of them is not an fp16 or fp32 tensor (a
    patch rewrites only those) or when the two specs do not list the same
    inline constants.
    """
    ca, cb = inline_constants(spec_a), inline_constants(spec_b)
    if [p for p, _ in ca] != [p for p, _ in cb]:
        raise TemplateError("extract: twin A and twin B do not have the same inline constants")
    sites = []
    for (path, va), (_, vb) in zip(ca, cb):
        ka, pa = _tensor_payload(va)
        kb, pb = _tensor_payload(vb)
        if ka == kb and pa == pb:
            continue
        dt = va.type.tensorType.dataType
        if ka == "bytes" and kb == "bytes" and dt == MIL_TYPE_FP16 and len(pa) == len(pb):
            sites.append((path, va, "fp16"))
        elif ka == "floats" and kb == "floats" and dt == MIL_TYPE_FP32 and len(pa) == len(pb):
            sites.append((path, va, "fp32"))
        else:
            raise TemplateError(f"extract: the weight-dependent inline constant {path} (MIL dtype {dt}, {ka}) "
                                "cannot be patched (only fp16 and fp32 tensors can)")
    return sites


def sentinel_bytes(path, n, dtype):
    """Deterministic, path-specific bytes for a patch site; finite values for fp32."""
    out, c = b"", 0
    while len(out) < n:
        out += hashlib.sha256(SENTINEL_TAG + path.encode() + struct.pack("<I", c)).digest()
        c += 1
    out = bytearray(out[:n])
    if dtype == "fp32":
        for i in range(3, n, 4):
            if out[i] & 0x7F == 0x7F and out[i - 1] & 0x80:
                out[i] &= 0xBF  # an all-ones exponent would be Inf or NaN
    return bytes(out)


def _set_payload(attr, data, dtype):
    t = attr.immediateValue.tensor
    if dtype == "fp16":
        t.bytes.values = data
    else:
        del t.floats.values[:]
        t.floats.values.extend(struct.unpack(f"<{len(data) // 4}f", data))


def str_sentinel(path, n):
    """A deterministic, path-specific ASCII string of n characters for a string patch site."""
    out, c = "", 0
    while len(out) < n:
        out += hashlib.sha256(SENTINEL_TAG + b"str\0" + path.encode() + struct.pack("<I", c)).hexdigest()
        c += 1
    return out[:n].encode()


# Every twin label a template's gates use: extract makes A and B, report A,
# B and Q. The string table is built over all of them at once, so A's value
# is the same in both.
TWIN_LABELS = ("a", "b", "q")
_TWIN_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _twin_candidate(key, label, n, counter):
    """n characters from _TWIN_ALPHABET, derived from (key, label, counter)."""
    out, block = [], 0
    while len(out) < n:
        digest = hashlib.sha256(b"qvac-coreml-twin-string\0" + label.encode() + b"\0" + key.encode() +
                                struct.pack("<II", counter, block)).digest()
        out += [_TWIN_ALPHABET[b % len(_TWIN_ALPHABET)] for b in digest]
        block += 1
    return "".join(out[:n])


def twin_string_table(refs):
    """{label: {key: value}} for the GGUF strings `refs` ({key: reference
    value}): for each key an ASCII value of the reference's UTF-8 byte length
    (a `patch str` site's length is in bytes) per twin label. Deterministic,
    and never ambiguous: a key's values differ between every two labels and
    from the reference value (so extraction sees every declared string vary,
    and a template that baked one twin's or the reference's value fails the
    others), and within one label no two keys share a value (so a spec string
    maps to one key). Raises TemplateError when the length leaves too few
    distinct values."""
    table = {label: {} for label in TWIN_LABELS}
    for key in sorted(refs):
        ref = refs[key]
        n = len(ref.encode("utf-8"))
        taken = {ref}
        for label in TWIN_LABELS:
            for counter in range(4096):
                v = _twin_candidate(key, label, n, counter)
                if v not in taken and v not in table[label].values():
                    break
            else:
                raise TemplateError(f"twin strings: no distinct {n}-byte value left for GGUF string {key} "
                                    f"(twin {label.upper()})")
            taken.add(v)
            table[label][key] = v
    return table


def recipe_logical_lines(recipe_text):
    """The recipe's logical lines, as the C++ parser reads them: `#` comments
    removed (outside strings), a newline inside an open `(` or `[` is
    whitespace, each line trimmed of spaces, tabs and CRs, blank lines dropped."""
    out, cur, depth, in_str, in_comment = [], [], 0, False, False

    def flush():
        line = "".join(cur).strip(" \t\r")
        if line:
            out.append(line)
        cur.clear()

    for c in recipe_text or "":
        if c == "\n":
            in_comment = False
            if depth > 0:
                cur.append(" ")
            else:
                flush()
            continue
        if in_comment:
            continue
        if in_str:
            in_str = c != '"'
            cur.append(c)
            continue
        if c == "#":
            in_comment = True
            continue
        if c == '"':
            in_str = True
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth = max(depth - 1, 0)
        cur.append(c)
    flush()
    return out


def recipe_str_metas(recipe_text):
    """The keys of the recipe's `meta "<key>" str` declarations, in any
    spacing the recipe grammar accepts (indented, tabs, a trailing comment)."""
    keys = []
    for line in recipe_logical_lines(recipe_text):
        m = re.fullmatch(r'meta[ \t]+"([^"]+)"[ \t]+str', line)
        if m:
            keys.append(m.group(1))
    return keys


def recipe_source_types(recipe_text):
    """name -> the ggml types the recipe's `source` line accepts, in order."""
    out = {}
    for line in recipe_logical_lines(recipe_text):
        m = re.match(r'source[ \t]+"([^"]+)"[ \t]+([A-Za-z0-9_,]+)[ \t]*\[', line)
        if m:
            out[m.group(1)] = [t.lower() for t in m.group(2).split(",")]
    return out


# Block sizes of the types twin Q may take (elements per block along ne[0]).
_BLOCK = {"f32": 1, "f16": 1, "bf16": 1}


def _block_size(t):
    if t in _BLOCK:
        return _BLOCK[t]
    import gguf

    return gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[t.upper()]][0]


def q_twin_tensor_types(manifest, source_types, quant_types):
    """`twin --tensor-type` entries for twin Q: each tensor the reference
    holds in f32 whose recipe source accepts a lower precision takes the first
    of `quant_types` the recipe accepts, else the recipe's first type below
    f32 (so a template whose sources accept f16 has its f16 conversion
    exercised, not only f32). Tensors below f32 in the reference keep the
    twin writer's own rotation; f32-only sources stay f32."""
    wanted = [t.strip().lower() for t in quant_types.split(",") if t.strip()]
    entries = {}
    for t in manifest.get("tensors", []):
        name, ref = t.get("name"), str(t.get("type", "")).lower()
        allowed = source_types.get(name)
        if ref != "f32" or not allowed:
            continue
        lower = [x for x in allowed if x != "f32" and t["ne"][0] % _block_size(x) == 0]
        pick = next((x for x in wanted if x in lower), lower[0] if lower else None)
        if pick:
            entries[name] = pick
    return entries


def verified_types(path):
    """name -> ggml type name of every tensor the GGUF at `path` holds."""
    import gguf

    return {t.name: t.tensor_type.name.lower() for t in gguf.GGUFReader(str(path)).tensors}


def type_summary(types):
    """'f16 x119, f32 x8' for a name -> type map."""
    counts = {}
    for t in types.values():
        counts[t] = counts.get(t, 0) + 1
    return ", ".join(f"{t} x{n}" for t, n in sorted(counts.items()))


def twin_sets(manifest, keys, label):
    """`twin --set` values giving each GGUF string in `keys` its value in
    twin `label` from twin_string_table (distinct per label and per key)."""
    ref = {m.get("key"): m.get("value") for m in manifest.get("metadata", [])}
    refs = {}
    for k in keys:
        v = ref.get(k)
        if not isinstance(v, str) or not v:
            raise TemplateError(f"extract: the recipe reads GGUF string {k}, which the reference manifest lacks")
        refs[k] = v
    table = twin_string_table(refs)[label]
    return [f"{k}={table[k]}" for k in keys]


def differing_strings(spec_a, spec_b, strings=()):
    """Spec user metadata strings copied from a twin GGUF string: [(path, spec key, GGUF key)].

    `strings` lists (GGUF key, twin A value, twin B value) of the strings the
    twins vary (twin_sets). A spec string that holds A's value in A and B's in
    B becomes a site; any other difference is left to spec_diff, which fails
    the extraction.
    """
    ua, ub = spec_a.description.metadata.userDefined, spec_b.description.metadata.userDefined
    out = []
    for k in sorted(set(ua.keys()) & set(ub.keys())):
        if k in VOLATILE_METADATA or ua[k] == ub[k]:
            continue
        keys = [gguf_key for gguf_key, va, vb in strings if ua[k] == va and ub[k] == vb]
        if len(keys) > 1:
            raise TemplateError(f"extract: spec string {k!r} matches GGUF strings {keys} in both twins; "
                                "the mapping is ambiguous")
        if keys:
            out.append((f"description.metadata.userDefined[{k!r}]", k, keys[0]))
    return out


def find_sites(spec_bytes, sentinels):
    """[{spec_offset, len, sentinel_hex}] for (path, sentinel) pairs; each must occur once."""
    sites = []
    for path, sent in sentinels:
        at = spec_bytes.find(sent)
        if at < 0 or spec_bytes.find(sent, at + 1) >= 0:
            raise TemplateError(f"extract: the sentinel of {path} occurs {'never' if at < 0 else 'twice'} in the spec")
        sites.append({"spec_offset": at, "len": len(sent), "sentinel_hex": sent.hex()})
    return sorted(sites, key=lambda p: p["spec_offset"])


def patch_spec(spec_a, spec_b, strings=()):
    """A's spec with a sentinel at every weight-dependent inline constant and metadata string.

    Returns (spec message, [(path, sentinel bytes, dtype, A's payload)]), dtype
    'fp16', 'fp32' or 'str' (for a str site the payload is A's string bytes).
    The sentinel spec must equal B structurally once B gets the same
    sentinels; otherwise the graphs differ and TemplateError is raised.
    """
    from coremltools.proto import Model_pb2

    sites = differing_constants(spec_a, spec_b)
    strings = differing_strings(spec_a, spec_b, strings)
    sa, sb = Model_pb2.Model(), Model_pb2.Model()
    sa.CopyFrom(spec_a)
    sb.CopyFrom(spec_b)
    ca, cb = dict(inline_constants(sa)), dict(inline_constants(sb))
    out = []
    for path, attr, dtype in sites:
        _, payload = _tensor_payload(attr)
        sent = sentinel_bytes(path, len(payload), dtype)
        _set_payload(ca[path], sent, dtype)
        _set_payload(cb[path], sent, dtype)
        out.append((path, sent, dtype, payload))
    for path, key, gguf_key in strings:
        payload = sa.description.metadata.userDefined[key].encode()
        sent = str_sentinel(path, len(payload))
        sa.description.metadata.userDefined[key] = sent.decode()
        sb.description.metadata.userDefined[key] = sent.decode()
        out.append((path, sent, "str:" + gguf_key, payload))
    diff = spec_diff(sa, sb)
    if diff:
        raise TemplateError("extract: the exporter's spec differs between twin A and twin B at\n  " +
                            "\n  ".join(diff) + "\n(beyond inline fp16/fp32 constants, which become patches)")
    plain = spec_a.SerializeToString(deterministic=True)
    patched = sa.SerializeToString(deterministic=True)
    if len(plain) != len(patched):
        raise TemplateError("extract: a sentinel changed the spec's framing")
    return sa, out


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


def make_twin(tool, manifest, seed, types, out, sets=(), tensor_types=None):
    cmd = [tool, "twin", "--manifest", manifest, "--seed", str(seed), "--types", types, "-o", out]
    for s in sets:
        cmd += ["--set", s]
    for name, t in sorted((tensor_types or {}).items()):
        cmd += ["--tensor-type", f"{name}={t}"]
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


def coinciding_tensors(twin_a, twin_b):
    """Names of the tensors whose bytes are the same in two twins. Extraction
    finds the weights an exporter folds into its program by the differences
    between twins A and B, so a tensor equal in both is a blind spot: a
    program constant computed from it would be taken for a fixed one."""
    import gguf

    def raw(path):
        return {t.name: np.asarray(t.data).tobytes() for t in gguf.GGUFReader(str(path)).tensors}

    a, b = raw(twin_a), raw(twin_b)
    return sorted(n for n in a if n in b and a[n] == b[n])


def check_twins_differ(twin_a, twin_b, what):
    same = coinciding_tensors(twin_a, twin_b)
    if same:
        raise TemplateError(f"{what}: twin A and twin B hold the same values for {len(same)} tensor(s), so the "
                            "exporter's use of them cannot be checked: " + ", ".join(same[:20]) +
                            (" ..." if len(same) > 20 else ""))


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
    """Runs template.json-style `exporter` {path, args[, gguf_name]} on `gguf`; returns the .mlpackage."""
    root = Path(root or repo_root())
    script = Path(exporter["path"])
    if not script.is_absolute():
        script = root / script
    if not script.is_file():
        raise TemplateError(f"exporter {script} not found")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = exporter.get("gguf_name")
    if name:
        if Path(name).name != name or not name.endswith(".gguf"):
            raise TemplateError(f"exporter gguf_name {name!r} must be a plain *.gguf file name")
        feed_dir = out_dir.parent / (out_dir.name + "-input")
        feed_dir.mkdir(parents=True, exist_ok=True)
        feed = feed_dir / name
        if feed.exists() or feed.is_symlink():
            feed.unlink()
        feed.symlink_to(Path(gguf).resolve())
        gguf = feed
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
        out.append(f"{cast}(zeros([{blob.count}]))")
    elif np.all(b == 1):
        out.append(f"{cast}(ones([{blob.count}]))")
    for name, (_, arr) in tensors.items():
        if arr is None or arr.size != blob.count:
            continue
        for perm, lay in _layouts(arr):
            if _equal(b, lay, blob.dtype):
                out.append(f"{cast}({_expr_t(name, perm)})")
            elif _equal(b, -lay, blob.dtype):
                out.append(f"{cast}(neg({_expr_t(name, perm)}))")
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
    e = {"path": a.exporter, "args": list(a.exporter_arg or [])}
    if getattr(a, "exporter_gguf_name", None):
        e["gguf_name"] = a.exporter_gguf_name
    return e


def cmd_suggest(a):
    tool = find_tool(a.tool)
    manifest = load_manifest(a.manifest)
    with tempfile.TemporaryDirectory(prefix="qvac-coreml-suggest-") as tmp:
        twin = make_twin(tool, a.manifest, SEED_A, "f32", Path(tmp) / "twin-a.gguf")
        pkg = run_exporter(exporter_from_args(a), twin, Path(tmp) / "export-a", a.python, a.root)
        parts = package_parts(pkg)
        data = parts["weights"].read_bytes()
        blobs = read_weight_bin(data)
        spec_sha = hashlib.sha256(canonical_spec_bytes(parts["spec"])).hexdigest()
        text, flagged, ambiguous = draft_recipe(a.id, spec_sha, len(data), blobs, gguf_tensors(twin), manifest)
    Path(a.output).write_text(text)
    print(f"suggest: {len(blobs)} blobs, {len(flagged)} unexplained, {len(ambiguous)} ambiguous -> {a.output}",
          file=sys.stderr)
    return 0


def _export_twin(tool, tdir_manifest, exporter, seed, types, work, python, root, label, sets=(), tensor_types=None):
    twin = make_twin(tool, tdir_manifest, seed, types, work / f"twin-{label}.gguf", sets, tensor_types)
    feed = twin
    if types != "f32" or tensor_types:
        feed = work / f"twin-{label}-f32.gguf"
        run([tool, "f32", twin, "-o", feed], f"f32 of twin {label}")
    pkg = run_exporter(exporter, feed, work / f"export-{label}", python, root)
    return twin, pkg


def extract_literals(declared, blobs_a, blobs_b):
    """literals.bin bytes and the template.json `literals` entries, or None when none is declared.

    Each declared literal {id, dtype, shape, exporter_blob} is copied from the
    exporter's weight.bin blob at metadata offset `exporter_blob`. The blob
    must be byte-identical in twins A and B: a literal never depends on the
    weights. Literals are laid out in declaration order, 64-byte aligned.
    """
    if not declared:
        return None
    by_a = {b.offset: b for b in blobs_a}
    by_b = {b.offset: b for b in blobs_b}
    data, entries = bytearray(), []
    for lit in declared:
        lid, src = lit.get("id"), lit.get("exporter_blob")
        if not isinstance(lid, str) or not isinstance(src, int):
            raise TemplateError(f"extract: literal {lid!r} needs an id and an `exporter_blob` weight.bin offset")
        ba, bb = by_a.get(src), by_b.get(src)
        if ba is None or bb is None:
            raise TemplateError(f"extract: literal {lid}: no weight.bin blob at offset {src}")
        if ba.raw != bb.raw:
            raise TemplateError(f"extract: literal {lid}: blob {src} differs between twin A and twin B, "
                                "so it depends on the weights")
        if lit.get("dtype") != ba.dtype_name:
            raise TemplateError(f"extract: literal {lid}: dtype {lit.get('dtype')!r}, blob {src} is {ba.dtype_name}")
        shape = lit.get("shape") or []
        if int(np.prod(shape)) != ba.count:
            raise TemplateError(f"extract: literal {lid}: shape {shape} has {int(np.prod(shape))} elements, "
                                f"blob {src} has {ba.count}")
        data += bytes((-len(data)) % 64)
        entry = dict(lit)
        entry["offset"] = len(data)
        data += ba.raw
        entries.append(entry)
    return bytes(data), entries


def patch_drafts(sites, patch_sites, tensors):
    """Draft patch lines, explained against twin A like `suggest`: [(spec_offset, line)]."""
    by_sent = {p["sentinel_hex"]: p for p in patch_sites}
    drafts = []
    for path, sent, dtype, payload in sites:
        site = by_sent[sent.hex()]
        if dtype.startswith("str:"):
            key = dtype[4:]
            drafts.append((site["spec_offset"], f'patch {site["spec_offset"]} str {len(payload)} meta_s("{key}")', path))
            continue
        blob = Blob(site["spec_offset"], MIL_FP16 if dtype == "fp16" else MIL_FP32, len(payload), 0, payload)
        cands = explain_blob(blob, tensors)
        expr = cands[0] if cands else "???"
        drafts.append((site["spec_offset"], f"patch {site['spec_offset']} {dtype} {blob.count} {expr}", path))
    return sorted(drafts)


def add_patch_drafts(text, drafts):
    """The recipe with a drafted line for every patch site it lacks (explained drafts only), before `end`."""
    have = {int(m.group(1)) for m in re.finditer(r"^patch (\d+) ", text, re.M)}
    add = [(off, line, path) for off, line, path in drafts if off not in have and not line.endswith("???")]
    if not add:
        return text, []
    lines = text.rstrip("\n").split("\n")
    ends = [i for i, l in enumerate(lines) if l.strip() == "end"]
    if not ends:
        raise TemplateError("recipe.txt has no `end` line, so the drafted patch lines have nowhere to go")
    end = ends[-1]
    ins = []
    for _, line, path in add:
        ins += [f"# drafted by extract from twin A ({path}): confirm against the exporter", line]
    lines[end:end] = ins
    return "\n".join(lines) + "\n", add


def _promote(out, writes):
    """Writes the staged template files into `out`, each through a temporary file and a rename."""
    out.mkdir(parents=True, exist_ok=True)
    for name, data in writes.items():
        tmp = out / f".{name}.extract-tmp"
        tmp.write_bytes(data)
        os.replace(tmp, out / name)


def cmd_extract(a):
    tool = find_tool(a.tool)
    out = Path(a.template_dir)
    manifest_path = Path(a.manifest) if a.manifest else out / "reference-gguf.json"
    manifest = load_manifest(manifest_path)
    existing = json.loads((out / "template.json").read_text()) if (out / "template.json").is_file() else {}
    exporter = exporter_from_args(a) if a.exporter else existing.get("exporter")
    if not exporter or "path" not in exporter:
        raise TemplateError("extract: no exporter (pass --exporter or keep template.json `exporter`)")
    root = Path(a.root) if a.root else repo_root()
    template_id = a.id or existing.get("id") or out.name
    recipe_path = out / "recipe.txt"
    # GGUF strings the recipe reads (meta "<key>" str): twin-specific values,
    # so a spec string copied from one of them becomes a str patch site.
    str_keys = recipe_str_metas(recipe_path.read_text() if recipe_path.is_file() else "")
    with tempfile.TemporaryDirectory(prefix="qvac-coreml-extract-") as tmp:
        work = Path(tmp)
        # Every output is staged first: `writes` holds the files to replace,
        # `stage` the template as it will be once they are written (the
        # twin-A generate runs on it). The template directory is written only
        # after every check has passed, so a rejected extraction leaves its
        # files as they were.
        writes = {}
        stage = work / "stage" / (out.name or "template")
        if out.is_dir():
            shutil.copytree(out, stage)
        else:
            stage.mkdir(parents=True)
        if manifest_path.resolve() != (out / "reference-gguf.json").resolve():
            writes["reference-gguf.json"] = manifest_path.read_bytes()
        sets_a, sets_b = twin_sets(manifest, str_keys, "a"), twin_sets(manifest, str_keys, "b")
        twin_a, pkg_a = _export_twin(tool, manifest_path, exporter, SEED_A, "f32", work, a.python, root, "a", sets_a)
        twin_b, pkg_b = _export_twin(tool, manifest_path, exporter, SEED_B, "f32", work, a.python, root, "b", sets_b)
        strings = [(k, va.split("=", 1)[1], vb.split("=", 1)[1]) for k, va, vb in zip(str_keys, sets_a, sets_b)]
        check_twins_differ(twin_a, twin_b, "extract")
        pa, pb = package_parts(pkg_a), package_parts(pkg_b)
        spec_a, spec_b = load_spec(pa["spec"]), load_spec(pb["spec"])
        # Weight-dependent inline constants become patch sites (sentinels in
        # the template's spec); any other difference fails the extraction.
        spec_a, sites = patch_spec(spec_a, spec_b, strings)
        wa, wb = pa["weights"].read_bytes(), pb["weights"].read_bytes()
        blobs_a, blobs_b = read_weight_bin(wa), read_weight_bin(wb)
        la, lb = layout_of(blobs_a), layout_of(blobs_b)
        if la != lb or len(wa) != len(wb):
            raise TemplateError("extract: weight.bin layout differs between twin A and twin B "
                                "(blob count, offsets, dtypes or sizes)")
        spec_bytes = spec_a.SerializeToString(deterministic=True)
        # Keep an unchanged spec and manifest byte for byte (the manifest's
        # item identifiers and the conversion date are random per export), so
        # re-extracting an unchanged template leaves its template_sha256 alone.
        old_spec = out / "model.mlmodel"
        sentinels = [(path, sent) for path, sent, _, _ in sites]
        reuse = False
        if old_spec.is_file() and not spec_diff(load_spec(old_spec), spec_a):
            try:
                find_sites(old_spec.read_bytes(), sentinels)
                reuse = True
            except TemplateError:
                reuse = False
        if reuse:
            spec_bytes = old_spec.read_bytes()
        else:
            writes["model.mlmodel"] = spec_bytes
        patch_sites = find_sites(spec_bytes, sentinels)
        literals_bin = extract_literals(existing.get("literals", []), blobs_a, blobs_b)
        drafts = patch_drafts(sites, patch_sites, gguf_tensors(work / "twin-a.gguf")) if sites else []
        new_manifest = pa["manifest"].read_text()
        old_manifest = out / "Manifest.json"
        if not (old_manifest.is_file() and manifest_equivalent(old_manifest.read_text(), new_manifest)):
            writes["Manifest.json"] = new_manifest.encode()
        functions, stateful = spec_summary(spec_a)
        spec_sha = hashlib.sha256(spec_bytes).hexdigest()
        script = Path(exporter["path"])
        exp_abs = script if script.is_absolute() else root / script
        exporter_out = {"path": exporter["path"], "sha256": file_sha256(exp_abs), "args": exporter.get("args", [])}
        if exporter.get("gguf_name"):
            exporter_out["gguf_name"] = exporter["gguf_name"]
        computed = {"schema": SCHEMA, "id": template_id, "exporter": exporter_out,
                    "toolchain": toolchain_versions(), "functions": functions, "stateful": stateful,
                    "weight_bin_size": len(wa), "patches": patch_sites}
        recipe = out / "recipe.txt"
        recipe_text = None
        if recipe.is_file():
            recipe_text = refresh_recipe_header(recipe.read_text(), template_id, spec_sha, len(wa), len(la))
        j = merge_template_json(existing, computed, manifest)
        j.setdefault("generation_peak_bytes", 0)
        if literals_bin is not None:
            data, entries = literals_bin
            writes["literals.bin"] = data
            j["literals"] = entries
        added = []
        if sites:
            print(f"extract: {len(sites)} weight-dependent inline constant(s) become patch sites:", file=sys.stderr)
            for _, line, path in drafts:
                print(f"  {line}   # {path}", file=sys.stderr)
            if recipe_text is not None:
                recipe_text, added = add_patch_drafts(recipe_text, drafts)
        if recipe_text is not None:
            writes["recipe.txt"] = recipe_text.encode()
        writes["template.json"] = (json.dumps(j, indent=2) + "\n").encode()
        for name, data in writes.items():
            (stage / name).write_bytes(data)
        if recipe_text is not None:
            # Generate from twin A once, on the staged template: the recipe must
            # parse and match; the generator's peak memory goes into template.json.
            gen = json.loads(run([tool, "generate", "--template", stage, "--gguf", work / "twin-a.gguf",
                                  "-o", work / "gen-a.mlpackage"], "generate (twin A)"))
            j["generation_peak_bytes"] = int(gen["generation_peak_bytes"])
            writes["template.json"] = (json.dumps(j, indent=2) + "\n").encode()
        # Every check has passed: write the template.
        _promote(out, writes)
        if added:
            print(f"extract: added {len(added)} drafted patch line(s) to {recipe}", file=sys.stderr)
        if recipe_text is None:
            print(f"extract: no recipe.txt in {out}; run `suggest`, complete it, then extract again",
                  file=sys.stderr)
    print(f"extract: {template_id}: spec {spec_sha[:16]}, {len(la)} blobs, weight.bin {len(wa)} bytes -> {out}",
          file=sys.stderr)
    return 0


def check_template(tool, tdir, python=None, root=None, quant_types=None, work=None):
    """The extraction (A) and cross-seed (B, Q) gates for one template. Returns result dicts."""
    tdir = Path(tdir)
    for f in REQUIRED_FILES + ("reference-gguf.json",):
        if not (tdir / f).is_file():
            raise TemplateError(f"{tdir}: missing {f}")
    tj = json.loads((tdir / "template.json").read_text())
    exporter = tj.get("exporter")
    if not exporter:
        raise TemplateError(f"{tdir}: template.json has no exporter")
    results = []
    types_q = quant_types if quant_types is not None else tj.get("report_quant_types", "q8_0")
    cases = [("A", SEED_A, "f32"), ("B", SEED_B, "f32"), ("Q", SEED_Q, types_q)]
    # Each twin also gets its own value of every GGUF string the recipe
    # declares (as `extract` does for A and B), so a template that baked one
    # checkpoint's string into its spec instead of patching it fails B and Q.
    recipe = tdir / "recipe.txt"
    recipe_text = recipe.read_text()
    str_keys = recipe_str_metas(recipe_text)
    manifest = load_manifest(tdir / "reference-gguf.json")
    # Twin Q holds the lower precisions the recipe accepts; the exporter sees
    # its f32 expansion.
    q_types = q_twin_tensor_types(manifest, recipe_source_types(recipe_text), types_q)
    owned = work is None
    work = Path(work or tempfile.mkdtemp(prefix="qvac-coreml-report-"))
    try:
        twins = {}
        for label, seed, types in cases:
            r = {"template": tj.get("id", tdir.name), "twin": label, "seed": seed, "types": types}
            forced = q_types if label == "Q" else None
            twin, pkg = _export_twin(tool, tdir / "reference-gguf.json", exporter, seed, types, work, python, root,
                                     label.lower(), twin_sets(manifest, str_keys, label.lower()), forced)
            twins[label] = twin
            # The types the twin really holds, checked against the plan: a
            # gate that claims a precision must have exercised it.
            held = verified_types(twin)
            r["verified_types"] = type_summary(held)
            wrong = sorted(n for n, t in (forced or {}).items() if held.get(n) != t)
            if label != "Q":
                wrong += sorted(n for n, t in held.items() if t != "f32")
            r.update(compare_packages(tool, tdir, twin, pkg, work, label.lower()))
            if wrong:
                r["wrong_types"] = wrong
                r["pass"] = False
            if label == "B":
                # A tensor equal in twins A and B is one these gates cannot
                # check (the exporter may fold it into the program).
                r["coinciding"] = coinciding_tensors(twins["A"], twin)
                r["pass"] = r["pass"] and not r["coinciding"]
            results.append(r)
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)
    return results


def compare_packages(tool, tdir, gguf, pkg, work, label):
    """Generates `tdir` from `gguf` and compares it with the exporter's package `pkg`."""
    parts = package_parts(pkg)
    gen = Path(work) / f"gen-{label}.mlpackage"
    run([tool, "generate", "--template", tdir, "--gguf", gguf, "-o", gen], f"generate ({label})")
    gen_parts = package_parts(gen)
    r = {"spec_diff": spec_diff(load_spec(gen_parts["spec"]), load_spec(parts["spec"]))}
    p = subprocess.run([tool, "verify", "--template", str(tdir), "--ours", str(gen_parts["weights"]),
                        "--ref", str(parts["weights"])], capture_output=True, text=True)
    r["verify_exit"] = p.returncode
    r["verify"] = p.stdout.strip().splitlines()
    r["verify_stderr"] = p.stderr.strip()
    r["pass"] = p.returncode == 0 and not r["spec_diff"]
    return r


def check_real(tool, tdir, gguf, python=None, root=None, work=None):
    """The real-weights gate: the exporter on f32(gguf) against the generator on gguf."""
    tdir = Path(tdir)
    tj = json.loads((tdir / "template.json").read_text())
    owned = work is None
    work = Path(work or tempfile.mkdtemp(prefix="qvac-coreml-real-"))
    try:
        feed = work / "real-f32.gguf"
        run([tool, "f32", gguf, "-o", feed], "f32 of the real GGUF")
        pkg = run_exporter(tj["exporter"], feed, work / "export-real", python, root)
        r = compare_packages(tool, tdir, gguf, pkg, work, "real")
        r.update({"template": tj.get("id", tdir.name), "twin": "real", "types": Path(gguf).name})
        return r
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)


def cmd_real(a):
    tool = find_tool(a.tool)
    root = Path(a.root) if a.root else repo_root()
    ok, results = True, []
    for d in a.template_dir:
        try:
            r = check_real(tool, d, a.gguf, a.python, root)
        except TemplateError as e:
            r = {"template": Path(d).name, "twin": "real", "pass": False, "error": str(e)}
        ok &= r["pass"]
        extra = r.get("error") or (f"spec differs at {r['spec_diff'][:3]}" if r.get("spec_diff") else
                                   (r["verify"][-1] if r.get("verify") else r.get("verify_stderr", "")))
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['template']} on {Path(a.gguf).name}: {extra}")
        for line in r.get("verify", []):
            if "FAIL" in line:
                print(f"    {line}")
        results.append(r)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1) + "\n")
    print(f"real: {len(results)} template(s), {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


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
            extra = r.get("error") or (f"tensors not in their planned type: {r['wrong_types'][:5]}"
                                       if r.get("wrong_types") else
                                       f"spec differs at {r['spec_diff'][:3]}" if r.get("spec_diff") else
                                       f"tensors equal in twins A and B: {r['coinciding'][:5]}"
                                       if r.get("coinciding") else
                                       (r["verify"][-1] if r.get("verify") else r.get("verify_stderr", "")))
            held = f"; holds {r['verified_types']}" if r.get("verified_types") else ""
            print(f"{status} {r['template']} twin {r['twin']} ({r.get('types', '-')}{held}): {extra}")
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
    s.add_argument("--exporter-gguf-name", help="file name the exporter sees its input GGUF under")
    s.add_argument("--id", required=True, help="template id")
    s.add_argument("-o", "--output", required=True, help="draft recipe.txt")
    s.set_defaults(fn=cmd_suggest)

    e = sub.add_parser("extract", help="write model.mlmodel, Manifest.json and template.json")
    e.add_argument("--template-dir", required=True)
    e.add_argument("--manifest", help="reference-gguf.json (default: the template directory's)")
    e.add_argument("--exporter", help="exporter script (default: template.json `exporter`)")
    e.add_argument("--exporter-arg", action="append")
    e.add_argument("--exporter-gguf-name", help="file name the exporter sees its input GGUF under")
    e.add_argument("--id", help="template id (default: template.json `id`, then the directory name)")
    e.set_defaults(fn=cmd_extract)

    r = sub.add_parser("report", help="extraction and cross-seed gates against the exporter")
    r.add_argument("--template-dir", action="append", help="repeatable")
    r.add_argument("--all", action="store_true", help="every engines/*/coreml-templates/* under --root")
    r.add_argument("--allow-empty", action="store_true", help="succeed when there is no template")
    r.add_argument("--quant-types", help="twin Q types (overrides the template default of q8_0 or report_quant_types)")
    r.add_argument("--json", help="also write the results as JSON")
    r.set_defaults(fn=cmd_report)

    g = sub.add_parser("real", help="real-weights gate: the exporter on f32(GGUF) against the generator on the GGUF")
    g.add_argument("--template-dir", action="append", required=True, help="repeatable")
    g.add_argument("--gguf", required=True, help="a production GGUF the templates match")
    g.add_argument("--json", help="also write the results as JSON")
    g.set_defaults(fn=cmd_real)
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
