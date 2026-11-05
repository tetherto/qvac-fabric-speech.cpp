#!/usr/bin/env python3
"""Write tiny Pocket FlowLM Core ML step sidecars for the runtime checks.

`good/flow-lm-step.mlmodelc` is the real step of a random-weight FlowLM
(scripts/pocket_coreml_models.random_checkpoint, source "tiny-checkpoint")
with a CONTEXT-position state; `good/reference/` holds its configuration,
prefix KV rows in the ggml cache layout, and the hidden rows and next latents
a float32 causal forward gives for a few frames after that prefix.
`probe/` declares exactly the features the runtime expects but only sums its
inputs; every `<feature>/` declares one of them with a doubled leading
dimension, `schema/` and `source/` carry the wrong schema or checkpoint hash
and `stateless/` keeps no state. test-pocket-coreml --shape-fixture
(POCKET_COREML_TEST_SHAPES=<out>) must reproduce the reference with the good
sidecar, accept the probe and reject every other one before predicting.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pocket_coreml_models as pm  # noqa: E402

CONTEXT, PREFIX, FRAMES, SOURCE, SCHEMA = 12, 5, 4, "tiny-checkpoint", "1"


def features(d):
    latent, half = (1, d["latent_dim"]), (1, d["head_dim"] // 2)
    inputs = {"latent": latent, "noise": latent, "cos": half, "sin": half,
              "mask": (1, 1, 1, CONTEXT), "write_mask": (1, 1, CONTEXT, 1)}
    outputs = {"hidden": (1, d["dim"]), "next_latent": latent}
    state = (d["layers"], d["heads"], CONTEXT, d["head_dim"])
    return inputs, outputs, {"keys": state, "values": state}


class Probe(torch.nn.Module):
    def __init__(self, outputs, states):
        super().__init__()
        self.outputs, self.states = list(outputs), list(states)
        for name, shape in outputs.items():
            self.register_buffer("base_" + name, torch.zeros(shape))
        for name, shape in states.items():
            self.register_buffer(name, torch.zeros(shape, dtype=torch.float16))

    def forward(self, *inputs):
        total = sum(x.mean() for x in inputs)
        for name in self.states:
            state = getattr(self, name)
            state[0:1] = state[0:1] * 0.5 + total
            total = total + state.mean()
        return tuple(getattr(self, "base_" + name) + total for name in self.outputs)


def convert(model, inputs, outputs, states, metadata):
    import coremltools as ct

    examples = tuple(torch.zeros(shape) for shape in inputs.values())
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), examples, check_trace=False)
    converted = ct.convert(
        traced, convert_to="mlprogram",
        inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in inputs.items()],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in outputs],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=s, dtype=np.float16), name=n) for n, s in states.items()],
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
    for key, value in metadata.items():
        converted.user_defined_metadata[key] = value
    return converted


def compile_into(model, directory, scratch):
    directory.mkdir(parents=True, exist_ok=True)
    package = Path(scratch) / directory.name / "flow-lm-step.mlpackage"
    package.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(package))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(directory)],
                   check=True, stdout=subprocess.DEVNULL)


def write_reference(out, tensors, metadata):
    d = pm.dimensions(metadata)
    rng = np.random.default_rng(5)
    model = pm.FlowLM(tensors, metadata).eval()
    prefix = torch.from_numpy(rng.normal(0, 1, (PREFIX, d["dim"])).astype(np.float32))
    latents = [torch.from_numpy(tensors["bos_emb"])]
    noises = torch.from_numpy(rng.normal(0, 0.5, (FRAMES, d["latent_dim"])).astype(np.float32))
    hidden, nxt = [], []
    with torch.no_grad():
        _, keys, values = model.causal(prefix)
        time_condition = model.time_condition(0.0, 1.0)
        rows = prefix
        for f in range(FRAMES):  # each frame's latent is the previous frame's next latent
            rows = torch.cat((rows, model.linear(latents[-1][None], "input_linear", False)))
            h = model.causal(rows)[0][-1:]
            n = noises[f:f + 1] + model.velocity(h, noises[f:f + 1], time_condition)
            hidden.append(h[0])
            nxt.append(n[0])
            latents.append(n[0])
    out.mkdir(parents=True, exist_ok=True)
    # ggml cache layout: per layer [position][head][head_dim]
    np.save(out / "prefix_keys.npy", np.stack([k.transpose(0, 1).numpy() for k in keys]).astype(np.float32))
    np.save(out / "prefix_values.npy", np.stack([v.transpose(0, 1).numpy() for v in values]).astype(np.float32))
    np.save(out / "latents.npy", torch.stack(latents[:FRAMES]).numpy().astype(np.float32))
    np.save(out / "noises.npy", noises.numpy())
    np.save(out / "hidden.npy", torch.stack(hidden).numpy().astype(np.float32))
    np.save(out / "next.npy", torch.stack(nxt).numpy().astype(np.float32))
    config = {k: int(metadata["pocket." + k]) for k in
              ("dim", "heads", "layers", "ff_dim", "latent_dim", "flow_dim", "flow_depth", "vocab_size")}
    config.update(rope_base=float(metadata["pocket.rope_base"]), context=CONTEXT, prefix=PREFIX, frames=FRAMES,
                  source=SOURCE)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def doubled(shape):
    return (shape[0] * 2,) + tuple(shape[1:])


def write_fixture(out):
    import coremltools as ct

    tensors, metadata = pm.random_checkpoint(11, source=SOURCE)
    d = pm.dimensions(metadata)
    stamp = {"pocket.coreml.schema": SCHEMA, "pocket.source_sha256": SOURCE, "pocket.coreml.context": str(CONTEXT)}
    inputs, outputs, states = features(d)
    with tempfile.TemporaryDirectory() as scratch:
        step = pm.FlowStep(tensors, metadata, CONTEXT).eval()
        compile_into(convert(step, inputs, outputs, states, stamp), out / "good", scratch)
        write_reference(out / "good" / "reference", tensors, metadata)
        print("[good] real step, context", CONTEXT)
        compile_into(convert(Probe(outputs, states), inputs, outputs, states, stamp), out / "probe", scratch)
        for name in list(inputs) + list(outputs) + list(states):
            bad = [{n: doubled(s) if n == name else s for n, s in group.items()} for group in (inputs, outputs, states)]
            compile_into(convert(Probe(bad[1], bad[2]), bad[0], bad[1], bad[2], stamp), out / name, scratch)
            print(f"[{name}] declared {doubled({**inputs, **outputs, **states}[name])}")
        for variant, changed in (("schema", {"pocket.coreml.schema": "0"}),
                                 ("source", {"pocket.source_sha256": "another-checkpoint"})):
            compile_into(convert(Probe(outputs, states), inputs, outputs, states, {**stamp, **changed}),
                         out / variant, scratch)
            print(f"[{variant}] {changed}")
        stateless = Probe(outputs, {})
        compile_into(ct.convert(torch.jit.trace(stateless.eval(), tuple(torch.zeros(s) for s in inputs.values()),
                                                check_trace=False),
                                convert_to="mlprogram",
                                inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in inputs.items()],
                                outputs=[ct.TensorType(name=n, dtype=np.float32) for n in outputs],
                                minimum_deployment_target=ct.target.iOS18), out / "stateless", scratch)
        print("[stateless] no state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    write_fixture(parser.parse_args().out)
