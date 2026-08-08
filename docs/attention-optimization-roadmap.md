# Attention Optimization Roadmap

This note tracks measured attention optimizations that are not yet implemented. Each item records
its intended transformation, supporting PTX and timing evidence, required compiler work, and known
applicability constraints. Operator feature coverage is tracked separately in the
[attention variant roadmap](attention-variants.md).

## 1. Fuse GQA Query Groups to Reuse K/V Tiles

**Status:** Measured with a manual Triton prototype; not implemented.

For GQA, let `R = Q_heads / KV_heads` be the number of query heads that share each K/V head. The
current operator layout is `(B, R, H_kv, S, D)`, and the generated grid parallelizes all of
`(B, R, H_kv, S_tile)`. Consequently, the `R` programs for one K/V head load the same K/V tiles.

Instead, parallelize `(B, H_kv, S_tile)` and evaluate all `R` query groups in one program. Logically,
the program can collapse the group and query-row dimensions around the matrix operations:

```text
Q:      R x M x D  -> (R*M) x D
scores: R x M x N  -> (R*M) x N
acc:    R x M x D  -> (R*M) x D
```

The max and sum reductions remain over `N`, independently for every collapsed row. K and V retain
shapes `N x D` and are loaded only once for all `R` query groups.

### Measured Evidence

Measurements used an RTX 6000 Ada, `B=1`, `Q_heads=4`, `KV_heads=2`, `D=64`, `BLOCK_M=128`, and
`BLOCK_N=64`. The prototype fused both query groups and used eight warps to keep per-thread work
comparable to the existing four-warp kernel.

| Sequence length | Existing | Group-fused | Change |
| ---: | ---: | ---: | ---: |
| 8192 | 355.14 us | 313.36 us | 11.8% faster |
| 16384 | 1.699 ms | 1.553 ms | 8.6% faster |

For `S=8192`, the generated PTX and resource usage changed as follows:

| Metric | Existing | Group-fused |
| --- | ---: | ---: |
| CTAs | 256 | 128 |
| PTX lines | 1845 | 1803 |
| `cp.async` instructions | 32 | 20 |
| Shared memory | 49,152 bytes | 65,536 bytes |
| Registers per thread | 250 | 251 |
| Spill loads/stores | 0 / 0 | 0 / 0 |

The prototype differed from the existing kernel by at most `3.05e-5`, with a mean difference of
`2.80e-8`.

### Required Compiler Work

Changing the GQA tile sizes directly from `[1, 1, 1, M, N, 0]` to `[1, R, 1, M, N, 0]` is not
sufficient. Tiling succeeds, but `transform.htile.linalg_to_semantic` rejects the resulting grouped
`linalg.generic`. Supporting the optimization requires:

1. Keep `R` out of `scf.forall` while tiling its full extent inside each program.
2. Represent the shared-RHS dot as `(R*M) x D` by `D x N`, or teach HTile lowering the equivalent
   grouped-dot form.
3. Preserve independent online-softmax state for every `(R, M)` row.
4. Handle the strided Q and output head mapping for a fixed K/V head.
5. Restore the logical `(R, M, D)` output layout before storing.

No new online-softmax algebra is required. The existing max-scale rewrite and rolling-update repair
apply independently to the collapsed rows.

### Applicability

The transformation reduces grid parallelism. At `S=4096`, the prototype regressed from 85.59 us to
127.40 us because only 64 fused CTAs remained. It should therefore be selected only when the
remaining batch, K/V-head, and query-tile grid is large enough. The selection policy is deferred to
future schedule selection or autotuning work.


## 2. Normalize Causal Masks to Mixed-Loop-Local Coordinates

**Status:** Measured with manual Triton prototypes; not implemented.

Dead-tile specialization splits causal attention into a mask-free live prefix, a mixed diagonal
suffix, and an omitted dead suffix. For `BLOCK_M=128` and `BLOCK_N=64`, every query tile has exactly
two mixed K/V tiles. The mixed loop currently reconstructs global token positions before comparing
them:

```text
query = query_block * 128 + row
key   = key_block * 64 + col
mask  = key <= query
```

The specialization has already proved that the mixed loop starts at
`key_block = query_block * 2`. The same predicate can therefore use local coordinates:

```text
diagonal_offset = key_block - mixed_lower
mask = diagonal_offset * 64 + col <= row
```

This removes tensor-wide i64 fills, global-position additions, and 64-bit comparisons from the two
masked iterations. It does not change the matrix operations or memory pipeline.

At `S=4096`, specialization already reduces the number of K/V block iterations per head from 2048
to 1056. Only 64 of the remaining iterations, about 6.1%, evaluate a mask. The local predicate
therefore targets a small but irreducible part of the current causal schedule.

### Measured Evidence

Measurements used an RTX 6000 Ada, `B=1`, `H=4`, `BLOCK_M=128`, `BLOCK_N=64`, four warps, and three
stages. Timings are medians from interleaved runs.

| Shape | Existing predicate | Local predicate | Change |
| --- | ---: | ---: | ---: |
| `S=1024, D=64` | 23.24 us | 22.81 us | 1.83% faster |
| `S=4096, D=64` | 74.36 us | 72.96 us | 1.88% faster |
| `S=8192, D=64` | 227.07 us | 223.82 us | 1.43% faster |
| `S=4096, D=128` | 156.36 us | 155.05 us | 0.84% faster |

For `S=4096` and `D=64`, PTX and resource usage changed as follows:

| Metric | Existing predicate | Local predicate |
| --- | ---: | ---: |
| PTX lines | 3747 | 3405 |
| PTX bytes | 164,146 | 154,410 |
| PTX instruction lines | 2597 | 2249 |
| `setp` instructions | 124 | 75 |
| `selp` instructions | 122 | 72 |
| PTX b64 virtual registers | 261 | 181 |
| Registers per thread | 255 | 252 |
| `mma.sync` instructions | 256 | 256 |
| `cp.async` instructions | 48 | 48 |

The local-coordinate prototype was bit-identical to the existing generated kernel. Narrowing only
the global mask arithmetic from i64 to i32 also improved `S=4096` by 1.74%, but produced less PTX
simplification and introduced a small stack frame. Local normalization is the preferred form.

### Required Compiler Work

`transform.loop.specialize_dead_tile` already owns the information needed for this rewrite: the
matched comparison, the possible-live interval, the mixed-loop lower bound, and the affine
relationship between the producer indices and loop IV. When cloning the masked producer into the
mixed loop, it should:

1. Prove that the query-tile origin equals the mixed-loop lower bound multiplied by `BLOCK_N`.
2. Replace the global key and query expressions with an affine expression relative to the mixed
   lower bound.
3. Preserve the existing predicate when the origins, tile-size ratio, or integer range cannot be
   proved.
4. Keep fully-live producer bypass and dead-suffix truncation unchanged.

A more general implementation may also normalize the two inequalities in a sliding-window mask,
but causal comparison support is sufficient for the measured optimization.

### Alternatives Tested

Several other ways to specialize the two diagonal tiles regressed on the same `S=4096`, `D=64`
case:

| Experiment | Change |
| --- | ---: |
| Statically unroll the two mixed iterations | 5.38% slower |
| Statically unroll after localizing the mask | 2.20% slower |
| Replace the clamped upper bound with `mixed_lower + 2` | 3.48% slower |
| Combine the two N=64 diagonal tiles into one N=128 tile | 18.46% slower |
| Hoist the Q tile across the live and mixed loops | 2.12% slower |

Static unrolling duplicated the loop body, weakened async pipelining, and introduced a stack frame.
The N=128 diagonal tile produced a 240-byte stack frame and lost half of the `cp.async`
instructions. These forms should not be pursued without corresponding pipeline and register-pressure
improvements.

Using `BLOCK_M=64` reduced diagonal waste and was 28.3% faster at `S=1024`, but was 8.4% slower at
`S=8192`. This remains a shape-dependent tile-selection decision rather than a causal-mask rewrite.
