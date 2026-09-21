# Backend Test Coverage

This document summarizes automated coverage across Neptune's exported operators. It distinguishes
structural lowering, backend compilation, and numerical execution because success at one stage does
not establish the next.

- **HTile lowering** checks exporter and schedule output structurally.
- **Backend compilation** translates Kernel HTile and invokes the backend compiler. The cuTile path
  launches once because compilation occurs at launch.
- **Runtime correctness** compares device output with a reference implementation.

The backend tests are currently validated with TileLang 0.1.9 and apache-tvm-ffi 0.1.10.
The table uses ✅ for covered, ⚠️ for partial coverage, and ❌ for missing coverage.

| Stage / backend | Dense attention | Packed varlen attention | Mamba selective scan |
|---|---|---|---|
| HTile lowering | ✅ Global, causal, windowed, ALiBi, and KV-FP8; MHA/GQA/MQA layouts | ✅ | ✅ |
| Triton compilation | ✅ | ✅ | ✅ |
| cuTile compilation | ✅ | ✅ | ✅ |
| TileLang compilation | ✅ | ✅ | ✅ |
| Triton runtime correctness | ⚠️ Causal from checked-in Kernel HTile | ❌ | ✅ Export through backend |
| cuTile runtime correctness | ⚠️ Causal from checked-in Kernel HTile | ❌ | ✅ Export through backend |
| TileLang runtime correctness | ⚠️ Causal from checked-in Kernel HTile | ❌ | ✅ Export through backend |

## Test Locations

- [`test/python/test_pipeline.py`](../test/python/test_pipeline.py) covers exporter-to-HTile
  structure, backend compilation, and Mamba end-to-end correctness.
- [`test/python/test_translators.py`](../test/python/test_translators.py) covers backend translator
  primitives and runs checked-in causal-attention Kernel HTile on all three backends.
- [`test/Pipeline`](../test/Pipeline) contains Transform-dialect and HTile structural tests.

The dense-attention correctness tests start from
[`test/python/data/causal_attention_htile.mlir`](../test/python/data/causal_attention_htile.mlir).
They validate backend translation and execution, but not the attention exporter or scheduling path.
Mamba correctness tests do exercise the generated exporter-to-backend path.

## Missing Coverage

The main gaps are:

1. exporter-to-backend correctness for dense attention, including global, windowed, ALiBi, KV-FP8,
   GQA, and MQA cases,
2. runtime correctness for packed variable-length attention on all three backends, and
3. broader Mamba correctness coverage across activation dtypes, shapes, and channel block sizes.

When adding coverage, keep compilation and correctness cases small enough for the default Python
test suite and gate backend execution on the corresponding Python package and Nvidia CUDA runtime.
