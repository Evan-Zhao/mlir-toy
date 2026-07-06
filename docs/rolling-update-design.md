# Attention Rolling Update Design

This note describes the rolling-update operator fusion transformation in its current MLIR form.
Integrated attention examples using it live under `test/Pipeline`, starting with
`test/Pipeline/tm_global_attention.mlir`.

Rolling update is a generalized form of operator fusion that can put pairs of reductions together
under the same loop. Since that breaks the usual producer-consumer dependencies,
rolling update performs extra _repair_ steps to restore correctness.

## Pipeline

The current MLIR pipeline has three steps:

1. `transform.fusion.find_next_reduction`
   Starting from a loop-local producer, run a forward BFS on the def-use graph
   and return the nearest reduction plus the ordered elementwise chain between
   the producer and that reduction.

2. `transform.fusion.clone_fuse_elemwise`
   Clone that elementwise chain into a sidecar chain, fuse it under the
   streaming loop, and publish the sidecar tensors as extra loop results. The
   original out-of-loop chain is left untouched.

3. `transform.fusion.repair_reduction_frontier`
   Repair the chosen reduction frontier by deriving a repair term `H`,
   materializing it as a pointwise tensor update, and rebuilding the frontier
   reduction to use that repaired init tensor.

This keeps the payload IR verifier-valid throughout the schedule while still
matching the shape of TVM's rolling update.

## Transform Details

### `transform.fusion.find_next_reduction`

Signature:

```text
(producer_op) -> (next_reduce_op, elemwise_ops)
```

This is a pure analysis op using `NavigationTransformOpTrait` (so the input handle is not
consumed). It returns:

- `next_reduce_op`: the first reduction-like op reachable from `producer_op` by BFS,
- `elemwise_ops`: the elementwise ops strictly between the producer and that reduction,
  sorted in producer-to-consumer order.

The op fails if no reduction is reachable or if any op on the path is not a
supported single-result elementwise `linalg.generic`.
Tensor `arith` / `math` elementwise ops must be normalized first, e.g. with
`convert-elementwise-to-linalg`.

### `transform.fusion.clone_fuse_elemwise`

Signature:

```text
(elemwise_ops, parallel_loop, streaming_loop) -> (sidecar_ops)
```

This transform clones each elementwise op, fuses the clone under the loop nest,
and returns the cloned sidecar ops in the same order. It rebuilds both loops:

- the inner `scf.for` gets one extra iter_arg/result per sidecar tensor,
- the outer `scf.forall` gets matching extra shared_out/result slots.

The loop handles are preserved and remapped to the rebuilt loops. The original elementwise
chain remains unchanged until a later repair step rewires a specific reduction use.

### `transform.fusion.repair_reduction_frontier`

Signature:

```text
(producer_reduce_ops, reduce_op, elemwise_orig_ops, elemwise_sidecar_ops, outer_loop, inner_loop)
  -> (repaired_reduce_op)
```

This transform implements the _repair_ on the reduction frontier (`reduce_op`).
It performs the following steps:

1. Fuse `reduce_op` into `inner_loop` (via `outer_loop`) so that it stays next
   to the fused `sidecar_chain_ops`.
1. Inline the sidecars into `reduce_op` with repeated `linalg::fuseElementwiseOps`,
   until the frontier becomes a single `linalg.generic`.
1. Inspect that fused body and classify scalar inputs:
   - `r0`, `r1`, ... for values produced by earlier reductions `producer_reduce_ops`,
   - `c0`, `c1`, ... for ordinary values,
1. Extract:
   - `f_expr` from the `reduce_op`, which should be a simple reduction combiner, such as `a + b`
   - `g_expr` from the non-accumulator side of the reduction combiner,
     which should coincide with the combined computation of `elemwise_sidecar_ops`.
1. Call the Python/SymPy solver on these two expressions to derive a repair term `H`,
   and prove that `H` is valid for the combiner `f`.
1. Materialize `H` as an elementwise `linalg.generic` (call it `update_op`),
   which updates the DPS init argument of `reduce_op`;
   then clone `reduce_op` with its DPS init remapped to the output of `update_op`.

The result is a repaired reduction whose accumulator is no longer the original
partial state, but the repaired state computed from `H`.

## Current Assumptions

The current implementation assumes:

- a single-result, single-init, one-reduction-dim `linalg.generic` frontier,
- the frontier can be fused under the inner loop,
- the fused frontier still matches a self-reduction `out = f(out, g(...))`,
- producer reductions are destination-style ops whose previous value is their init operand,
- the solved repair term eliminates all `c*` variables before re-materialization.

## Attention Example

For the current attention schedule, the intended flow is:

1. build the outer `scf.forall` over output tiles and the inner streaming loop over K/V blocks,
2. fuse QK and score scaling under that streaming loop,
3. find the nearest reduction frontier from the loop-local score tile,
4. clone and fuse the elementwise chain under the loop as sidecar ops,
5. repair the frontier reduction by deriving and applying `H`,
6. repeat for later frontiers such as `P @ V` accumulation and row sum.

For row sum and output accumulation this yields the usual FlashAttention
recurrences:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

For decode-input attention, the K/V dimension should remain parallel across splits rather than
becoming a serial streaming loop. That schedule uses split-k rfactor/write-back forms instead of
rolling update; see [the SplitK update design](split-k-update-design.md).

## Transform-Dialect Notes

- `transform.fusion.find_next_reduction` is analysis-only and does not consume its input handle.
- `clone_fuse_elemwise` and `repair_reduction_frontier` rewrite explicit ops in functional style,
  while loop handles are read-only inputs remapped to rebuilt loops.
- Ordered multi-op handles are part of the contract: the elementwise chain is
  not treated as an unordered set.
- The fusion helpers depend on loop results being traceable through
  `tensor.insert_slice` / `tensor.parallel_insert_slice` relays. The integrated
  schedules run canonicalization at points where those relays are still
  recoverable by later fusion steps.

## Testing

Useful coverage is:

1. analysis-only tests for `transform.fusion.find_next_reduction`,
2. synthetic tests for cloned and fused elementwise sidecar chains,
3. reduction-frontier repair tests for supported single-result reduction cases,
4. end-to-end attention tests that check the repaired loop-carried recurrence shape.
