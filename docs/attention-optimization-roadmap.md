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

## 3. Remove the Row-Constant Part of ALiBi After Tiling

**Status:** Measured with manual Triton prototypes; not implemented.

The generated ALiBi kernel constructs a two-dimensional bias tile from absolute query and key
positions:

```text
query = query_tile_origin + row
key   = key_tile_origin + col
bias  = (key - query) * slope
logit = dot * scale + bias
```

For a fixed query row, `-query * slope` is constant across the softmax reduction. Softmax is
invariant to this row-wise translation:

```text
softmax_j(x_ij + key_j * slope - query_i * slope)
  = softmax_j(x_ij + key_j * slope)
```

Dropping the full query position would make the remaining bias grow with the absolute sequence
position. A tile-relative form keeps values smaller while eliminating the row dimension:

```text
bias = (key - query_tile_origin) * slope
```

This differs from the original bias by `row * slope`, which is still constant across each softmax
row. The causal predicate must continue to use the original absolute query and key positions.

### Measured Evidence

Measurements used an RTX 6000 Ada, four warps, three stages, and `BLOCK_N=64`. The current kernels
included positive-scale sinking after max and the revised exp-to-exp2 handling for biased logits.
Timings are medians from repeated interleaved runs.

| Shape | `BLOCK_M` | Existing bias | Tile-relative bias | Change |
| --- | ---: | ---: | ---: | ---: |
| `S=1024, D=64` | 64 | 23.40 us | 17.51 us | 25.2% faster |
| `S=4096, D=64` | 128 | 91.69 us | 78.85 us | 14.0% faster |
| `S=4096, D=128` | 128 | 190.78 us | 176.25 us | 7.6% faster |

For `S=1024` and `D=64`, selected PTX instruction counts changed as follows:

| Metric | Existing bias | Tile-relative bias |
| --- | ---: | ---: |
| `mul.f32` instructions | 132 | 100 |
| `cvt.rn.f32.s64` instructions | 16 | 0 |
| `sub.f32` instructions | 64 | 0 |
| Local spill loads/stores | 0 / 0 | 0 / 0 |

At `S=4096` and `D=64`, `mul.f32` decreased from 264 to 168, while the integer-to-float
conversions and floating-point subtractions were again eliminated. The optimized kernels differed
from the existing FP16 outputs by at most `9.77e-4`. For `S=1024`, both kernels had the same
`9.77e-4` maximum error against a PyTorch reference and nearly identical mean error.

### Required Compiler Work

This is a softmax-invariance transformation, not a generally valid arithmetic canonicalization. A
schedule-aware implementation should:

1. Identify the ALiBi expression `(key - query) * slope` feeding the softmax logits.
2. Apply the rewrite after query tiling, when `query_tile_origin` and the row offset are explicit.
3. Prove that the removed term depends only on non-reduction axes and that the max state is internal
   to the softmax update.
4. Preserve the original absolute positions in causal or window predicates.
5. Rewrite both the mask-free live loop and the masked mixed loop without changing dead-suffix
   specialization.

A more general formulation could hoist any row-only additive term through max and cancel it from
centered logits:

```text
max_j(x_ij + r_i) = max_j(x_ij) + r_i
(x_ij + r_i) - (max_j(x_ij) + r_i) = x_ij - max_j(x_ij)
```

The targeted ALiBi form is a smaller first implementation and preserves better numerical range than
replacing the relative distance with the absolute key position.

### Secondary Experiments

After removing the row-constant term, precomputing `col * slope` outside the K/V loop and adding a
scalar block offset improved timings by only about another 1%. Row-wise reciprocal normalization
was slightly slower, and duplicate probability `exp2` expressions in generated Triton were already
eliminated in PTX. Tile and warp changes remain shape-dependent autotuning decisions.

## 4. Add cuTile-Specific FMHA Scheduling and Arithmetic Hints

**Status:** Measured with manual cuTile prototypes; not implemented.

The generated cuTile kernel is sensitive to memory-latency hints and division semantics that do not
have direct equivalents in the current HTile schedule. Measurements used an RTX 6000 Ada,
`B=1`, `H=4`, `S=4096`, `D=64`, and `BLOCK_N=64`. Timings are medians from interleaved runs.

### 4.1 Mark K and V Loads with Asymmetric Latency

The official cuTile FMHA kernel assigns different expected DRAM latencies to its loop-carried
loads:

```python
k = ct.load(K, ..., latency=2)
v = ct.load(V, ..., latency=4)
```

Adding only these hints to the generated `BLOCK_M=128` kernel improved latency from 122.21 us to
116.16 us, or 5.0%. The cubin changed from a single loop body to a deeper software-pipelined form:

| Metric | Inferred latency | K=2, V=4 |
| --- | ---: | ---: |
| Registers per thread | 255 | 255 |
| Stack per thread | 64 bytes | 24 bytes |
| Shared memory | 49,200 bytes | 49,184 bytes |
| Static `HMMA` instructions | 128 | 256 |
| Static `MUFU.EX2` instructions | 72 | 144 |

The doubled static compute counts reflect loop unrolling rather than additional runtime work. Using
the same hint for both loads was neutral at low values and regressed at high values; the asymmetric
K/V hints are material.

The cuTile translator should attach backend-specific latency hints based on the semantic role of a
memory operation. The values remain candidates for shape-dependent tuning rather than universal
constants.

### 4.2 Use Approximate Division for FP16 Output Normalization

The generated epilogue uses precise elementwise division after broadcasting the row sum. The
maintained NVIDIA cuTile kernel instead requests approximate division:

```python
out = ct.truediv(
    acc,
    row_sum,
    rounding_mode=ct.RoundingMode.APPROX,
    flush_to_zero=True,
)
```

Changing only the rounding mode improved the same kernel from 122.21 us to 118.07 us, or 3.4%.
`flush_to_zero=True` alone had no effect. Selected SASS counts changed as follows:

| Metric | Precise | Approximate |
| --- | ---: | ---: |
| Reciprocal helper calls | 65 | 1 |
| `FCHK` correction instructions | 64 | 0 |
| `FFMA` instructions | 341 | 12 |
| Stack per thread | 64 bytes | 40 bytes |
| Cubin size | 148,256 bytes | 132,512 bytes |

The optimized output differed in 31 of 1,048,576 FP16 elements, with maximum difference
`6.10e-5` and mean difference `4.62e-10`. Selection of approximate division must therefore be tied
to the operator's numerical policy rather than applied as a general arithmetic canonicalization.

### 4.3 Tune Query Tile Size After Applying the Hints

Combining the latency hints and approximate division reduced the `BLOCK_M=128` kernel to
114.72 us. Selecting `BLOCK_M=64` after those changes reduced it further to 110.53 us, an
additional 3.7% and a total 9.6% improvement over the generated baseline. The M=64 cubin used 238
registers, no stack, and 40,968 bytes of shared memory.

This does not establish M=64 as a universal cuTile default. Tile-size changes alter grid
parallelism and repeated K/V work, and prior Triton measurements favored M=128 at this sequence
length. cuTile tile selection should be tuned independently by backend and shape.

### Operation Reordering Experiments

Operation reordering is not yet a standalone roadmap item. Explicitly reusing the probability tile
increased stack use from 64 to 392 bytes per thread, while reusing both probability and rescaling
tiles increased it to 1,016 bytes and regressed latency to 553 us. Preserving reduction dimensions,
tracking a scaled maximum, and forcing `occupancy=2` also regressed when applied independently.

cuTile and `tileiras` already perform code motion, loop splitting, software pipelining, and register
allocation. Source transformations still change the dataflow graph and tile live ranges, but no
isolated operation-order rewrite has yet improved this kernel. The maintained NVIDIA kernel's
ordering may be useful only together with its tile shape, latency hints, approximate division, and
occupancy target.
