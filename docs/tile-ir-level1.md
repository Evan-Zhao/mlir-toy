# Level-1 Scheduled Tile IR

Level 1 is the scheduled, target-independent tile form of `tileir`. It records
which tensor tiles are computed, which loops are parallel or sequential, and
which tile-level recurrences are carried across loops. It does not record GPU
hierarchy, memory spaces, layouts, warp roles, async copies, or barriers.

The intended contract is:
- Tile sizes and grid structure are explicit.
- Parallel output tiles are explicit.
- Sequential reduction or streaming loops are explicit.
- Loop-carried tile state is explicit.
- Tile bodies remain structured tensor computations.
- Hardware placement is still absent.

The canonical example is `tests/data/flash_attention_l1.mlir`. It describes a
FlashAttention schedule in a form that is close to Triton, while still leaving
room for later lowering to more hardware-specific languages.

## Core Representation

Level 1 uses upstream MLIR operations rather than custom iterator-defining ops.
Tile values are plain `tensor<...>` values, and tile bodies use upstream ops
such as `linalg`, `arith`, `math`, and `tensor`.

Tile structure is encoded directly in tensor and loop operations:
- `tensor.extract_slice` forms input tiles.
- Static slice sizes define tile shapes.
- Loop bounds define the tile grid.
- `scf.forall` marks independent output tiles.
- `tensor.parallel_insert_slice` publishes each output tile.
- `affine.for` or `scf.for` carries sequential tile state with `iter_args`.
- Tensor element types and explicit casts define the numerical policy.

This keeps L1 valid upstream MLIR while preserving enough structure for later
Triton, TileLang, CuTe, ThunderKittens, or GPU-specific lowering.

## FlashAttention Shape

The FlashAttention L1 example makes these schedule decisions explicit:
- The output is partitioned into `(B, H, M-block)` tiles with
  `scf.forall (%b, %h, %i_block) in (1, 32, 32)`.
- Each parallel instance computes one `128 x 128` output tile.
- The query tile is extracted once per output tile.
- The K/V sequence dimension is streamed by an inner `affine.for` over `64`
  blocks of size `64`.
- The online-softmax recurrence is represented by loop-carried tensor values:
  row normalizer `l`, accumulator `acc`, and row max `m`.
- The body keeps tile math in structured/value form: `linalg.matmul`,
  `linalg.reduce`, `linalg.broadcast`, `linalg.map`, `arith`, and `math`.
- The final tile is normalized, truncated, and inserted into the shared output
  tensor with `tensor.parallel_insert_slice`.

This is intentionally close to a Triton-style FlashAttention program: one
program instance per output tile, one sequential loop over K/V blocks, and
explicit tile-local online-softmax state.

## Structured Tile Bodies

Level 1 should preserve high-level tile computations instead of immediately
lowering them to scalar loops or hardware fragments.

For example, `Q @ K^T` is represented as `linalg.matmul` with explicit
`indexing_maps` on the K tile:

```mlir
linalg.matmul
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (n, k)>,
    affine_map<(m, n, k) -> (m, n)>]
```

This keeps the contraction recognizable while avoiding a non-upstream
`matmul_transpose_b` op spelling. Later lowering can choose whether this
becomes a Triton `tl.dot`, a tiled MMA operation, or a target-specific dot op.

Elementwise tile operations may use `linalg.map` or tensor `arith`/`math` ops.
In the FlashAttention example, the softmax uses `math.exp2`, so the score scale
is `log2(e) / sqrt(D)` rather than just `1 / sqrt(D)`.

## Schedule Hints

Level 1 may carry backend-relevant schedule hints as attributes when they do
not force a hardware hierarchy, memory space, or layout. The FlashAttention
example attaches:

```mlir
{pipeline_stages = 2 : i32}
```

to the inner K/V loop. This records pipeline intent for later lowering, but it
does not specify shared memory allocation, async copy mechanics, warp roles, or
barriers. Those belong to a more physical level.

If these hints become part of the dialect contract, prefer namespaced
attributes such as `tileir.pipeline_stages` over unqualified attributes.

## Loop Operator Rule

Level 1 normally uses two iterator-defining ops:
- `scf.forall` for the outer parallel grid that produces disjoint output tiles.
- `affine.for` for static, regular sequential loops that carry tile state.

Use `scf.for` instead of `affine.for` when the sequential loop has non-affine
bounds, such as dynamic sequence lengths, causal early exits, or other values
that cannot satisfy affine validity rules.

Avoid `affine.parallel` for the main output grid in value-based tensor IR.
`affine.parallel` models reduction-style aggregation, not destination-passing
parallel insertion of disjoint tensor slices.

| Loop kind in the IR                                                         | Op           | Reason                                                                                             |
| --------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| Outer parallel grid producing disjoint output tiles                         | `scf.forall` | Supports `shared_outs` and `tensor.parallel_insert_slice`.                                         |
| Static sequential reduction or streaming loop carrying tile state           | `affine.for` | Supports `iter_args`; affine bounds keep tile-level scheduling passes available.                    |
| Sequential loop with non-affine bounds                                      | `scf.for`    | Same basic recurrence shape as `affine.for`, without affine validity constraints.                   |
| Reduction-only parallel loop                                                | `affine.parallel` | Useful in general MLIR, but not the default for L1 output-tile decomposition.                  |

## What L1 Does Not Encode

The FlashAttention L1 example deliberately does not encode:
- GPU blocks, warps, lanes, warpgroups, or thread mappings.
- Shared/register/fragment memory spaces.
- Tile layouts, swizzles, MMA operand layouts, or vector register shapes.
- Async copy operations, barriers, or producer/consumer role specialization.
- Boundary masks or causal masks; the example assumes the fixed shapes divide
  evenly by the chosen tile sizes.

Those decisions can be introduced by later lowering passes while preserving the
same scheduled tile dataflow.

## Affine Details

`affine.for` is useful for static inner loops because affine bounds make the
body legible to polyhedral analysis: dependence checking, loop tiling, fusion,
interchange, vectorization, software pipelining, and unroll-and-jam.

The cost is restriction:
- The step must be a positive integer constant.
- Bounds must be affine maps over valid affine operands.
- SSA index values from non-affine sources cannot directly drive affine
  indexing.

Valid affine operands include induction variables of enclosing affine loops,
symbols defined outside affine scope, constants, and results of `affine.apply`
over valid affine operands. That set is closed under composition, which is what
makes affine analyses sound.

For static, regular sequential loops such as the K/V streaming loop in
FlashAttention, `affine.for` is a strict upgrade over `scf.for`:
- `iter_args` work for tensors, scalars, and other SSA values.
- The body can still contain non-affine ops such as `linalg`, `arith`, `math`,
  and `tensor`.
- Only operations that require affine analysis, such as `affine.load`,
  `affine.store`, and `affine.apply`, need to obey affine validity rules.

Do not use affine loops everywhere by default. The outer output grid needs
parallel destination-passing tensor writes, which is exactly the role of
`scf.forall` with `shared_outs` and `tensor.parallel_insert_slice`.
