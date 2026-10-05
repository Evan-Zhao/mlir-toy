# Backend Test Coverage

This document summarizes automated coverage across Neptune's exported operators. It distinguishes
structural lowering, backend compilation, and numerical execution because success at one stage does
not establish the next.

- **HTile lowering** checks exporter and schedule output structurally.
- **Full-pipeline correctness** exports, lowers, translates, compiles, launches, and compares device
  output with a reference implementation. It replaces the separate compilation-only suites;
  their shape matrices are retained alongside the numerical regression cases.

The test environment is an NVIDIA GeForce RTX 5080 (sm120) with Triton 3.7.1, cuTile 1.6.0,
TileLang 0.1.9, and apache-tvm-ffi 0.1.10. The 144-case pipeline runtime matrix includes
windowed-attention regressions for initially masked rows in the rolling softmax.
The table uses ✅ for covered, ⚠️ for partial coverage, and ❌ for missing coverage.

| Stage / backend | Dense attention | Packed varlen attention | Mamba selective scan |
|---|---|---|---|
| HTile lowering | ✅ Global, causal, windowed, ALiBi, and KV-FP8; MHA/GQA/MQA layouts | ✅ | ✅ |
| Triton compilation | ✅ | ✅ | ✅ |
| cuTile compilation | ✅ | ✅ | ✅ |
| TileLang compilation | ✅ | ✅ | ✅ |
| Triton runtime correctness | ✅ All variants/layouts, including windowed attention | ✅ Export through backend | ✅ Export through backend; dtype/shape/tile matrix |
| cuTile runtime correctness | ✅ All variants/layouts, including windowed attention | ✅ Export through backend | ✅ Export through backend; dtype/shape/tile matrix |
| TileLang runtime correctness | ✅ All variants/layouts, including windowed attention | ✅ Export through backend | ✅ Export through backend; dtype/shape/tile matrix |

## Test Locations

- [`test/python/test_pipeline.py`](../test/python/test_pipeline.py) covers exporter-to-HTile
  structure, backend compilation, and end-to-end correctness for all three operators.
- [`test/python/test_translators.py`](../test/python/test_translators.py) covers backend translator
  primitives and runs checked-in causal-attention Kernel HTile on all three backends.
- [`test/Pipeline`](../test/Pipeline) contains Transform-dialect and HTile structural tests.

The translator-level dense-attention tests still start from
[`test/python/data/causal_attention_htile.mlir`](../test/python/data/causal_attention_htile.mlir).
The pipeline correctness tests instead export, schedule, translate, and execute fresh kernels:

- **Dense attention:** 32 cases per backend. The former compilation matrix covers all five
  variants at MHA (batch 1, sequence 512, head dimension 128), GQA (batch 2, sequence 512,
  head dimension 64), and MQA (batch 1, sequence 16384, head dimension 64). Another 15 cases
  cross all variants/layouts at batch 2, sequence 256, and head dimension 64. A 73-token window,
  distinct ALiBi slopes, and non-unit KV-FP8 scales exercise variant-specific semantics.
  Two rectangular causal GQA cases cover both query/KV length directions, head dimension
  128, and custom 64×32 tiles. The shared FP32 `reference_attn` checks every query row in
  bounded-memory chunks, including the 16K cases, with window and ALiBi semantics checked
  independently against Torch SDPA. GQA uses the exporter's head mapping; FP8 inputs are
  dequantized into FP16 as in the operator. Kernel probability rounding and approximate
  scale motion are allowed by the numerical tolerance.
- **Packed varlen attention:** seven cases per backend. The three former compilation shapes
  retain 2/4/8 documents, 512/1024 total tokens, 2/4 heads, and max document bounds 256/512.
  Four additional cases use unequal document lengths including
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

## Scope and Execution Requirements

The nine windowed-attention cases (three head layouts on three backends) previously produced
NaNs from `-inf - (-inf)` on initially masked rows. The built-in Torch attention StableHLO
exporter now sets the FP32 softmax max initializer to `-FLT_MAX` (`0xFF7FFFFF`), leaving masked
scores at `-inf`. Explicit TA initialization carries this finite seed into the running max.
These regressions now pass without skips or xfails. Export tests separately verify that the
max initializer changes while the mask's shared `-inf` constant and sum's zero seed do not.
This policy addresses empty prefixes, not a defined output for an entirely masked final row.

This is a bounded regression matrix, not exhaustive coverage of all shapes, tail dimensions,
hardware architectures, or tile configurations.
Dense attention uses the exporter's FP16 activation contract (with FP8 K/V for that variant).
Mamba's current schedule fails on degenerate cases with one timestep, one batch item, or one
channel block (loop canonicalization/fusion limitations). The runtime matrix avoids these cases;
supporting them remains separate compiler work.

Backend execution requires the corresponding Python package and an Nvidia CUDA runtime; unavailable
backends are skipped. cuTile KV-FP8 execution additionally requires an sm100-or-newer GPU.
Exporter tests also require Torch-MLIR (dense attention) or JAX (varlen attention and Mamba).

Run the 144-case end-to-end numerical matrix against a freshly built compiler (rather than an
older `neptune-opt` installed in the venv) with:

```sh
source .venv/bin/activate
cmake --build build
NEPTUNE_MLIR_OPT="$PWD/build/neptune-opt" \
  python -m pytest test/python/test_pipeline.py -k output_correctness
```
