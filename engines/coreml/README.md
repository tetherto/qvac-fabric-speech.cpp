# Shared Core ML runtime

`engines/coreml` holds the Core ML code that the tts, parakeet and audiogen
engines share:

- the load scheduler and model manager (`src/scheduler.cpp`,
  `src/manager.cpp`), compiled into each engine library under the engine's
  own namespace by `qvac_coreml_add_to()` in `qvac_coreml.cmake`;
- the portable weight generator: the recipe parser and interpreter
  (`src/recipe.cpp`, `src/interp.cpp`), the MILBlob v2 writer
  (`src/milblob.cpp`), the GGUF tensor source (`src/gguf_source.cpp`), the
  source digest (`src/digest.cpp`) and the on-disk artifact cache
  (`src/cache.cpp`, `src/failed.cpp`). Given a template directory and a GGUF,
  it writes the `weight.bin` of an `.mlpackage` byte for byte the same on
  Linux and macOS;
- `qvac-coreml-tool`, the host CLI for that generator (below);
- on Apple, the on-device compile of a generated package
  (`src/apple/compile.mm`) and the templates embedded in each engine library
  (`src/templates.h`);
- `python/coreml_template.py`, which makes templates from an unchanged
  exporter and checks them against it, and `python/golden.py`, the
  generator tripwire.

No library is installed: the headers under `include/` are build-interface
only.

## Engine build options

`qvac_coreml_add_to(<engine-lib> NAMESPACE <ns> PREFIX <P> ENABLE <bool> [TEMPLATES <dirs>])`
in `qvac_coreml.cmake` folds the runtime into an engine library when the
engine's Core ML option (`TTS_CPP_COREML`, `PARAKEET_COREML`,
`AUDIOGEN_COREML`) is on. With it come these cache options:

| Option | Default | Effect |
|---|---|---|
| `<P>_COREML_GENERATE` (`TTS_CPP_COREML_GENERATE`, `PARAKEET_COREML_GENERATE`, `AUDIOGEN_COREML_GENERATE`) | `ON` on macOS, `OFF` on iOS | Compiles the generator (recipe interpreter, MILBlob writer, `GgufSource`, cache, on-device compile) into the library and embeds the template directories passed as `TEMPLATES`. Needs the `ggml` target. With `OFF`, only the runtime half (background loads of prebuilt sidecars) is compiled and no template is embedded. |
| `QVAC_COREML_TEMPLATE_BUDGET_MB` | `32` | The most template bytes (uncompressed) one library may embed. Configure fails above it with `the embedded Core ML templates take N bytes, over QVAC_COREML_TEMPLATE_BUDGET_MB=…`. |
| `QVAC_COREML_DEV` | `OFF` | Compiles the template override (below) into a release build. Test builds (`<P>_BUILD_TESTS=ON`) always have it. |

**Embedding.** Each file of each template (`Manifest.json`, `model.mlmodel`,
`recipe.txt`, `template.json` and an optional `literals.bin`; never the
test-only `reference-gguf.json`) becomes one generated assembler stub that
places it uncompressed in `__TEXT,__const` with `.incbin`, so the bytes are
paged in only when a package is written. A generated table lists
`{template id, file, begin, end, sha256}`; CMake computes each `sha256`, so
`template_sha256` (part of every cache key) is known without reading the
files. Editing a template file reconfigures the build. A template directory
must be named after its `template.json` `id`. Linux builds embed nothing.

**Template override.** Where compiled in, the `template_dir` option
(`Options::template_dir`) or the `QVAC_COREML_TEMPLATE_DIR` environment
variable makes the generator read `<dir>/<template-id>/` instead of the
embedded copy. Elsewhere `template_dir` is ignored with a warning.

**On-device compile.** `apple::compile_package()` calls
`+[MLModel compileModelAtURL:completionHandler:]`. Core ML writes the bundle
to `$TMPDIR/<package name>.mlmodelc`; the pipeline names packages
`qvac-coreml-<key>.mlpackage`, so a process killed mid-compile leaves
`$TMPDIR/qvac-coreml-<key>.mlmodelc`, which a routine prune removes after an
hour. The bundle is then renamed into staging, or copied when `$TMPDIR` is on
another volume. `artifact.json` records the OS build and the `coremlc`
version read from the compiled `model.mil`. Generation needs macOS 15.

## Building the tests and the tool

The portable tests and `qvac-coreml-tool` build on Linux and macOS (not on
Windows) with CMake 3.20 or newer and a C++17 compiler. The tool and the
GGUF test cases also need ggml (`ggml-base` for GGUF parsing and
`ggml_get_type_traits()->to_float` dequantisation).

Standalone, against a ggml install (for example the `ggml-speech` vcpkg
port):

```sh
cmake -S engines/coreml -B build-coreml -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build-coreml
ctest --test-dir build-coreml -L coreml-runtime
./build-coreml/qvac-coreml-tool dump model.gguf
```

Without ggml on `CMAKE_PREFIX_PATH` the standalone build still configures,
builds the scheduler, recipe, MILBlob and cache tests without their GGUF
cases, and prints
`qvac-coreml-tool and the GGUF cases of the generator tests: not built (no ggml target)`.

Inside an engine build the tests and the tool come with the engine's test
option: `TTS_CPP_BUILD_TESTS`, `PARAKEET_BUILD_TESTS` or
`AUDIOGEN_BUILD_TESTS` (or `SPEECH_BUILD_TESTS` in the umbrella build). There
ggml is required: configure fails with
`qvac_coreml_add_tests(REQUIRE_GGML): no ggml target` instead of silently
dropping the tool and the GGUF cases.

```sh
cmake -S engines/tts -B build-tts -DTTS_CPP_BUILD_TESTS=ON
cmake --build build-tts --target qvac-coreml-tool test-coreml-recipe
ctest --test-dir build-tts -L coreml-runtime
```

The tool and the generator are compiled with `-ffp-contract=off`, so no
floating-point expression is fused into an FMA and the output does not
depend on the host CPU (x86-64 or arm64).

## Making a template

Templates live in `engines/<engine>/coreml-templates/<template-id>/`. The
exporter (`engines/*/scripts/export-*-coreml.py`) is never modified; it stays
the numerical reference. `python/coreml_template.py` needs the coremltools
venv pinned by `engines/parakeet/scripts/requirements-coreml.txt` (numpy,
gguf-py, coremltools), the exporter's own requirements and a built
`qvac-coreml-tool` (`--tool`, or `$QVAC_COREML_TOOL`).

1. Write the shape manifest of a real GGUF:
   `qvac-coreml-tool dump model.gguf -o <dir>/reference-gguf.json`.
2. Draft the recipe from the exporter run on twin A (seed 1, f32):

   ```sh
   python engines/coreml/python/coreml_template.py --tool build/qvac-coreml-tool suggest \
       --manifest <dir>/reference-gguf.json --id <template-id> -o <dir>/recipe.txt \
       --exporter engines/tts/scripts/parler/export-dac-coreml.py \
       --exporter-arg=--gguf --exporter-arg={gguf} --exporter-arg=--output --exporter-arg={out}
   ```

   Every `weight.bin` blob is matched against `f16(t(name))`, transposes,
   scalar multiples, negation, `zeros` and `ones`. A blob it cannot explain is
   written as `blob <offset> <dtype> <count> ???` under an `UNEXPLAINED`
   comment; a blob that several tensors explain gets an `AMBIGUOUS` comment.
   Suggested constants come from the twin's values: confirm them against the
   exporter.
3. Complete `recipe.txt` by hand (the recipe language: `include/qvac_coreml/recipe.h`).
4. Write `template.json` with the fields only the author knows (`family`,
   `stage`, `export_kind`, `match.metadata`, `exporter` with `{gguf}` and
   `{out}` in its `args`, `default_compute_units`, `memory`), then extract:

   ```sh
   python engines/coreml/python/coreml_template.py --tool build/qvac-coreml-tool extract --template-dir <dir>
   ```

   `extract` runs the exporter on twins A and B (seeds 1 and 2), fails when
   their specs differ anywhere (a weight-dependent value in the spec needs
   patches, which this generator version lacks) or when their `weight.bin`
   layouts differ, then writes `model.mlmodel` (A's spec, deterministically
   serialised), `Manifest.json` and the computed `template.json` fields
   (`functions`, `stateful`, `weight_bin_size`, `generation_peak_bytes`, the
   exporter's sha256, the toolchain versions), and refreshes the header of
   `recipe.txt`.
5. Check it against the exporter:

   ```sh
   python engines/coreml/python/coreml_template.py --tool build/qvac-coreml-tool report --template-dir <dir>
   ```

   For twins A and B (f32) and Q (`--quant-types`, default `q8_0`; the
   exporter gets `qvac-coreml-tool f32` of it), `report` compares the
   exporter's `weight.bin` blob by blob with the C++ generator's
   (`qvac-coreml-tool verify`) and the exporter's spec with the template's.
   `--all` checks every template under `engines/*/coreml-templates/`.
6. Add the template's golden line:
   `python3 engines/coreml/python/golden.py update --tool build/qvac-coreml-tool`.

### Gates in CI

`.github/workflows/coreml-templates-ci.yml` runs on changes to the exporters,
the templates, `engines/coreml` and the venv pin, on the self-hosted
Apple-silicon lane: the `coreml_template.py` unit checks, `report` on the toy
templates (a harness self-check), `report --all` (the extraction and
cross-seed gates) and `golden.py check`.

`engines/coreml/test/golden.txt` holds, per template, the generator version,
`template_sha256` and the sha256 of the `weight.bin` generated from twin A.
`golden.py check` (also the `test-coreml-golden` ctest on Linux and macOS)
fails when the generated bytes change while the generator version and the
template stay the same: a generator change that alters output must bump
`GENERATOR_VERSION` (`include/qvac_coreml/recipe.h`). A version or template
change, a template without a line or a line without a template also fail
until `golden.py update` is run and `golden.txt` committed.

### Toy templates

`test/toy-templates/` holds three test-only templates made by
`test/toy/export_toy.py` with coremltools' MIL builder: `toy-stateless`,
`toy-multi` (two functions sharing weights) and `toy-state` (an MLState
accumulator). They are routing and lifetime fixtures, never a gate on a
model. `test-coreml-toy` (label `coreml-toy`, macOS 15 or later) embeds them
in a test library through `qvac_coreml_add_to(... TEMPLATES ...)` and checks,
on the device: the embedded table against the source files; generate,
compile, load and predict for each toy, compared with the same arithmetic on
the GGUF's weights (CPU, and all compute units for the stateless toy);
`artifact.json`; the user metadata of an on-device-compiled bundle; the
memo hit on a second preparation; the compile output naming and the copy
fallback; and the repair of an unusable artifact (moved to `quarantine/`,
regenerated, compiled again). It prints `compile_peak_bytes` per toy.

```sh
ctest --test-dir build-coreml -L coreml-toy --output-on-failure
```

To rebuild the toys after changing `export_toy.py`:

```sh
python engines/coreml/test/toy/export_toy.py --kind stateless --init seed.gguf
qvac-coreml-tool dump seed.gguf -o engines/coreml/test/toy-templates/toy-stateless/reference-gguf.json
python engines/coreml/python/coreml_template.py --tool qvac-coreml-tool extract \
    --template-dir engines/coreml/test/toy-templates/toy-stateless
```

## qvac-coreml-tool

```
qvac-coreml-tool <command> [arguments]
```

Every command exits 0 on success and 2 on a usage error or a failure (the
message goes to stderr, prefixed `qvac-coreml-tool:`). `verify` exits 1 when a
blob does not match. JSON results go to stdout unless `-o` names a file.

A template directory, as `generate`, `digest` and `verify` take with
`--template`, holds `recipe.txt`, `template.json`, `model.mlmodel`,
`Manifest.json` and optionally `literals.bin`. Whenever one of these commands
reads a GGUF, it first checks that the GGUF matches the template (tensor
names, types and shapes as the recipe's `source` lines declare them) and fails
with `no_template: <reason>` when it does not.

### dump

```
qvac-coreml-tool dump <model.gguf> [-o <manifest.json>]
```

Writes a shape manifest: every metadata key with its GGUF type and value, and
every tensor's name, type, `n_dims`, `ne` and value statistics (min, max,
mean, std). `twin` reads it.

```sh
qvac-coreml-tool dump models/mimi.gguf -o mimi-manifest.json
```

### twin

```
qvac-coreml-tool twin --manifest <manifest.json> --seed <N> --types <type[,type...]> -o <out.gguf>
                      [--set key=value ...]
```

Writes a random GGUF with the same metadata and tensor geometry as the
manifest. Each tensor's values are drawn from a normal distribution with the
recorded mean and standard deviation, clamped to the recorded range. Tensors
that are not f32 in the original (quantised, f16, bf16) take the `--types`
entries in turn, quantised with ggml's own quantisers, when their row length is
a multiple of that type's block size; the rest stay f32. `--set` sets a string
metadata value (overriding the manifest's) and may be repeated. The same manifest,
seed and types give the same file on every host, so tests can use a twin in
place of a production model they cannot ship.

```sh
qvac-coreml-tool twin --manifest mimi-manifest.json --seed 7 --types q8_0,f16 -o mimi-twin.gguf
```

### f32

```
qvac-coreml-tool f32 <in.gguf> -o <out.gguf>
```

Writes a copy whose tensors are dequantised to f32 by ggml
(`ggml_get_type_traits(type)->to_float`). Tensors of a type ggml cannot
convert are copied unchanged. Use it to give a Python exporter the same
values the generator reads from a quantised GGUF.

```sh
qvac-coreml-tool f32 mimi-twin.gguf -o mimi-twin-f32.gguf
```

### generate

```
qvac-coreml-tool generate --template <dir> --gguf <model.gguf> -o <out.mlpackage> [--options <opts>]
```

Runs the recipe interpreter and the MILBlob writer and writes an
`.mlpackage` (no Core ML compile, so it works on Linux). Prints the template
id, `template_sha256`, `source_digest`, `artifact_key`, the SHA-256 and size
of the written `weight.bin`, the generator's peak memory
(`generation_peak_bytes`) and `generator_version`. `--options` only feeds
the printed `artifact_key` (default `out=fp16`).

```sh
qvac-coreml-tool generate --template templates/mimi-decoder --gguf models/mimi.gguf -o mimi.mlpackage
```

### digest

```
qvac-coreml-tool digest --template <dir> --gguf <model.gguf> [--options <opts>]
```

Prints the cache identities of a GGUF for a template without generating
anything: `template_sha256`, `source_digest` (with the number of tensors and
bytes it covers), `artifact_key`, `memo_id`, the GGUF's real path and the
options (default `out=fp16`). A runtime with the same inputs looks up the same
cache entry.

```sh
qvac-coreml-tool digest --template templates/mimi-decoder --gguf models/mimi.gguf
```

### verify

```
qvac-coreml-tool verify --template <dir> (--gguf <model.gguf> | --ours <weight.bin>) --ref <weight.bin>
```

Compares a `weight.bin` with a reference one (usually the coremltools export
of the same weights), blob by blob. The left side is either generated from
`--gguf` in memory or read from `--ours`. For each blob it prints the offset,
the comparison class (`exact`, or `ulp` for blobs whose recipe uses
operations that may differ in the last place), the dtype, the number of
differing elements, the largest ULP distance and `ok` or `FAIL`; layout
errors (header, metadata, offsets, padding) are reported first. The last line
is `PASS` or `FAIL`.

```sh
qvac-coreml-tool verify --template templates/mimi-decoder --gguf models/mimi.gguf \
    --ref export/mimi.mlpackage/Data/com.apple.CoreML/weights/weight.bin
```
