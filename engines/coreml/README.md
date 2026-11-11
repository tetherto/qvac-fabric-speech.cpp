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
- `qvac-coreml-tool`, the host CLI for that generator (below).

No library is installed: the headers under `include/` are build-interface
only.

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
