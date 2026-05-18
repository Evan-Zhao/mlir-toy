# Attention Rolling Update Design

This note describes the rolling-update operator fusion transformation in its current MLIR form.
An example of it applied to Attention is in `test/python/data/attention_l0_to_l1.transform.mlir`.

Rolling update is a generalized form of operator fusion that can put pairs of reductions together
under the same loop. Since that breaks the usual producer-consumer dependencies,
rolling update performs extra _repair_ steps to restore correctness.

## Pipeline

The current MLIR pipeline has three steps:

1. `transform.match.loop_ru.rolling_update_next_reduction`
   Starting from a loop-local producer, run a forward BFS on the def-use graph
   and return the nearest reduction plus the ordered elementwise chain between
   the producer and that reduction.

2. `transform.loop_ru.clone_fuse_elemwise`
   Clone that elementwise chain into a sidecar chain, fuse it under the
   streaming loop, and publish the sidecar tensors as extra loop results. The
   original out-of-loop chain is left untouched.

3. `transform.loop_ru.repair_reduction_frontier`
   Repair the chosen reduction frontier by deriving a repair term `H`,
   materializing it as a pointwise tensor update, and rebuilding the frontier
   reduction to use that repaired init tensor.

This keeps the payload IR verifier-valid throughout the schedule while still
matching the shape of TVM's rolling update.

## Transform Details

### `transform.match.loop_ru.rolling_update_next_reduction`

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
supported single-result elementwise op.

### `transform.loop_ru.clone_fuse_elemwise`

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

### `transform.loop_ru.repair_reduction_frontier`

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
      which should coinside with the combined computation of `elemwise_sidecar_ops`.
1. Call the Python/SymPy solver on these two expressions to derive a repair term `H`.
1. Apply the solution `H` to produce a new `linalg.generic` operation to fix the result.
   This operation updates the accumulator of `reduce_op` before `reduce_op` runs.

The result is a repaired reduction whose accumulator is no longer the original
partial state, but the repaired state computed from `H`.

## Current Assumptions

The current implementation assumes:

- a unary single-result `linalg.generic` reduction frontier,
- the fused frontier still matches a self-reduction `out = f(out, g(...))`,
- producer reductions are destination-style ops whose previous value is their
  init operand,
- the solved repair term eliminates all `c*` variables before
  re-materialization,
- the repaired init tensor is pointwise on the reduction-result domain, so the
  accumulator and producer reductions agree on that result shape.

## Attention Example

For the current attention schedule, the intended flow is:

1. build the outer `scf.forall` over output tiles and the inner streaming loop over K/V blocks,
2. fuse QK and score scaling under that streaming loop,
3. find the nearest reduction frontier from the loop-local score tile,
4. force-fuse the elementwise chain under the loop as sidecar ops,
5. repair the frontier reduction by deriving and applying `H`,
6. repeat for later frontiers as more rolling-update cases are implemented.

For row sum and output accumulation this yields the usual FlashAttention
recurrences:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

## Transform-Dialect Notes

- `rolling_update_next_reduction` is analysis-only and does not consume its input handle.
- `clone_fuse_elemwise` and `repair_reduction_frontier` rewrite explicit ops in functional style,
  while loop handles are read-only inputs remapped to rebuilt loops.
- Ordered multi-op handles are part of the contract: the elementwise chain is
  not treated as an unordered set.
- TODO: upstream `scf::tileAndFuseConsumer` expects loop results to be
  published through `tensor.insert_slice` / `tensor.parallel_insert_slice`.
  If canonicalization removes those relays, rebuild minimal slices just before
  fusion or call `tileAndFuseConsumerOfSlices` with explicit slices.

## Testing

Useful coverage is:

1. analysis-only tests for `rolling_update_next_reduction`,
2. synthetic tests for force-fused elementwise chains,
3. reduction-frontier repair tests for the supported unary reduction case,
4. end-to-end attention tests that check the repaired loop-carried recurrence shape.
