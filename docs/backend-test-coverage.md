# Backend Test Coverage

This document summarizes automated coverage across Neptune's exported operators. It distinguishes
structural lowering, backend compilation, and numerical execution because success at one stage does
not establish the next.

- **HTile lowering** checks exporter and schedule output structurally.
- **Backend compilation** translates Kernel HTile and invokes the backend compiler. The cuTile path
  launches once because compilation occurs at launch.
- **Runtime correctness** compares device output with a reference implementation.

Validated on an NVIDIA GeForce RTX 5080 (sm120) with Triton 3.7.1, cuTile 1.6.0,
TileLang 0.1.9, and apache-tvm-ffi 0.1.10. The full Python suite passed all 472 tests
(including 90 pipeline runtime cases); the MLIR lit suite passed all 47 tests.
The table uses ✅ for covered, ⚠️ for partial coverage, and ❌ for missing coverage.

| Stage / backend | Dense attention | Packed varlen attention | Mamba selective scan |
|---|---|---|---|
| HTile lowering | ✅ Global, causal, windowed, ALiBi, and KV-FP8; MHA/GQA/MQA layouts | ✅ | ✅ |
| Triton compilation | ✅ | ✅ | ✅ |
| cuTile compilation | ✅ | ✅ | ✅ |
| TileLang compilation | ✅ | ✅ | ✅ |
| Triton runtime correctness | ✅ Export through backend; all variants and head layouts | ✅ Export through backend | ✅ Export through backend; dtype/shape/tile matrix |
| cuTile runtime correctness | ✅ Export through backend; all variants and head layouts | ✅ Export through backend | ✅ Export through backend; dtype/shape/tile matrix |
| TileLang runtime correctness | ✅ Export through backend; all variants and head layouts | ✅ Export through backend | ✅ Export through backend; dtype/shape/tile matrix |

## Test Locations

- [`test/python/test_pipeline.py`](../test/python/test_pipeline.py) covers exporter-to-HTile
  structure, backend compilation, and end-to-end correctness for all three operators.
- [`test/python/test_translators.py`](../test/python/test_translators.py) covers backend translator
  primitives and runs checked-in causal-attention Kernel HTile on all three backends.
- [`test/Pipeline`](../test/Pipeline) contains Transform-dialect and HTile structural tests.
- [`test/Loop/fuse_reduction_initial_value.mlir`](../test/Loop/fuse_reduction_initial_value.mlir)
  checks the finite running-max initialization needed for initially masked softmax rows.

The translator-level dense-attention tests still start from
[`test/python/data/causal_attention_htile.mlir`](../test/python/data/causal_attention_htile.mlir).
The pipeline correctness tests instead export, schedule, translate, and execute fresh kernels:

- **Dense attention:** 17 cases per backend. All five variants cross MHA/GQA/MQA layouts
  with batch size two, sequence length 256, and head dimension 64. A 73-token window,
  distinct ALiBi slopes, and non-unit KV-FP8 scales exercise variant-specific semantics.
  Two rectangular causal GQA cases cover both query/KV length directions, head dimension
  128, and custom 64×32 tiles. References are the existing eager Torch operator modules.
- **Packed varlen attention:** four cases per backend. Unequal document lengths include
  singleton, partial-tile, full-length, and empty documents, with both int32 and int64 offsets,
  head dimensions 64/128, and default/custom tiles. Each document is checked independently
  with the shared FP32 dense-attention reference (noncausal); the JAX packed-window custom
  calls themselves are compiler markers, not executable reference implementations.
  Tests assert the exported offset ABI, so an int64 request cannot silently become int32.
- **Mamba:** nine cases per backend, crossing FP16/BF16/FP32 with three shape/tile cases:
  batch sizes 2/3, sequence lengths 3/8/33, state dimensions 8/16/32, expansion factors 1/2,
  and channel blocks 64/128/256. The shared FP32 recurrent reference checks every output.

Runtime tests compare all output elements, check shape/dtype/finiteness, and initialize output
buffers to NaN to detect missing stores. Launch bounds come from generated kernel metadata.

Kernel timing and optional Nsight capture are available separately through
[`neptune-bench`](benchmarking.md); these are not performance regression tests.

## Scope and Execution Requirements

The previously listed operator-level runtime gaps are covered. This is a bounded regression matrix,
not exhaustive coverage of all shapes, tail dimensions, hardware architectures, or tile configurations.
Dense attention uses the exporter's FP16 activation contract (with FP8 K/V for that variant).
Mamba's current schedule fails on degenerate cases with one timestep, one batch item, or one
channel block (loop canonicalization/fusion limitations). The runtime matrix avoids these cases;
supporting them remains separate compiler work.

Backend execution requires the corresponding Python package and an Nvidia CUDA runtime; unavailable
backends are skipped. cuTile KV-FP8 execution additionally requires an sm100-or-newer GPU.
Exporter tests also require Torch-MLIR (dense attention) or JAX (varlen attention and Mamba).

Run the 90-case end-to-end numerical matrix against a freshly built compiler (rather than an
older `neptune-opt` installed in the venv) with:

```sh
source .venv/bin/activate
cmake --build build
NEPTUNE_MLIR_OPT="$PWD/build/neptune-opt" \
  python -m pytest test/python/test_pipeline.py -k output_correctness
```
