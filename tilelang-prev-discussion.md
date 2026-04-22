# TileLang Translator Design Summary

## TileLang language model (from `example_amd_flash_attn_fwd.py`)

Key constructs:
- `@T.prim_func` — kernel function decorator
- `with T.Kernel(grid_x, grid_y, threads=128) as (bx, by)` — defines grid dims AND thread count; block indices are the `as` bindings
- `T.alloc_fragment([M, N], dtype)` → register tile (HTile `#local`)
- `T.alloc_shared([M, N], dtype)` → shared memory tile (HTile `#shared`)
- `T.fill(tile, val)` → initialize tile
- `T.copy(global[bz, bx*BM:, by, :], shared)` → global→shared load (HTile `htile.load`)
- `T.copy(local, global[...])` → store back (HTile `htile.store`)
- `T.gemm(A, B, C, transpose_B=True, policy=GemmWarpPolicy.FullRow)` → matmul accumulating into C in-place
- `T.reduce_max(src, dst, dim=1, clear=False)` / `T.reduce_sum(...)` → reductions
- `for i, j in T.Parallel(M, N): tile[i,j] = expr` → elementwise ops
- `for k in T.Pipelined(N_tiles, num_stages=2):` → software-pipelined loop

## HTile MLIR → TileLang mapping

| HTile MLIR | TileLang |
|---|---|
| `htile.load %q[%bz, %by, %i, %c0]` | `Q_shared = T.alloc_shared(...)` + `T.copy(Q[bz, i:i+BM, by, :], Q_shared)` |
| `htile.store %tile, %out[...]` | `T.copy(tile, Output[bz, i:i+BM, by, :])` |
| `htile.full %val : f32 -> tensor<...>` | `acc = T.alloc_fragment(...)` + `T.fill(acc, val)` |
| `htile.dot %a, %b {transpose_b}` | `T.gemm(a, b, c, transpose_B=True)` |
| `htile.reduce axis 1 kind "max"` | `T.reduce_max(src, dst, dim=1)` |
| `htile.reduce axis 1 kind "sum"` | `T.reduce_sum(src, dst, dim=1)` |
| `arith.*` on tensors | `for i, j in T.Parallel(M, N): buf[i,j] = expr` |
| `linalg.broadcast` | **Eliminated** — implicit in `T.Parallel` indexing (use `m_i[i]` instead of broadcasting to 2D) |
| `scf.for` with `iter_args` | Accumulators become pre-allocated buffers; loop body mutates them in-place; `scf.yield` becomes no-op |
| `scf.for {htile.pipeline_stages = 2}` | `T.Pipelined(N, num_stages=2)` |

## Fundamental difference from Triton translator

**Memory model**: Triton is functional (SSA → new value per op), TileLang is imperative (pre-allocated mutable buffers). This requires a **two-pass structure**:

1. **Allocation pass** — scan the function body first, emit `T.alloc_shared`/`T.alloc_fragment` declarations at the top of the kernel before any ops
2. **Emission pass** — emit operations as in-place buffer mutations

## What's missing from `flash_attention_htile.mlir` for TileLang

| Info needed | Current state | Solution |
|---|---|---|
| Thread count | `threads=(%c128, %c1, %c1)` — now fixed in file | Read from `gpu.launch` block size operands |
| Pipeline depth | Not present | Discardable attr `htile.pipeline_stages = 2 : i32` on `scf.for` — BUT: MLIR parses `{...}` between `step` and `iter_args` as an unknown op, not an attr-dict; placement needs to be resolved |
| Warp layout / MMA policy | Not present | Long-term: extend `#htile.encoding` to carry layout (like Triton GPU's `#ttg.nvidia_mma<...>`); short-term: optional attr on `htile.dot` |

## `linalg.broadcast` elimination strategy

Rather than emitting a broadcast op, the TileLang translator should detect the pattern `broadcast(vec) → 2D` and instead emit the downstream elementwise op directly with indexed access:

```python
# Instead of: score - broadcast(m_i)  [two ops]
# Emit:
for i, j in T.Parallel(BM, BN):
    score[i, j] -= m_i[i]   # m_i indexed only on row dim, no broadcast needed
```

This requires the translator to look ahead at the broadcast's use site, not emit it independently.

## Proposed translator architecture

```
translate_common.py   — MLIR IR utilities shared by both backends:
                        _op_type_name, _module_top_ops, _tensor_shape,
                        _memref_shape, dispatch skeleton, scf.for traversal

translate_triton.py   — Triton-specific emission (functional, pointer arithmetic)
translate_tilelang.py — TileLang-specific emission (imperative, buffer mutation)
```

## MLIR passes identified as worth building (relevant to TileLang)

| Pass | What it does | Why TileLang needs it |
|---|---|---|
| `htile.permute + htile.dot → htile.dot{transpose_b}` | Already done manually | `T.gemm(transpose_B=True)` vs separate transpose |
| `tensor.empty + linalg.broadcast → htile.broadcast` | Canonicalize into one op | Simpler to detect and eliminate in translator |
| `htile.copy` elimination | Remove placement-only copies | No-op in both translators; noise in SSA |
| Batch-dim folding | Fold 4D memref + batch indices into flat ptr | Both translators need it; currently 30-line ad-hoc logic in Triton translator |
