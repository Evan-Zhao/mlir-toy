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
