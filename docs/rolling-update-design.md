# Attention Rolling Update Design

## Goal

This note describes the rolling-update part of the attention L0-to-L1
schedule in its current MLIR form.

In TVM terms, rolling update is the reduction-side analogue of
`reverse_compute_at`: move a consumer chain under a streaming loop, then repair
the incomplete reduction so the loop-local update becomes globally correct.

For FlashAttention, this is the mechanism behind:

- running row max under the K/V streaming loop,
- building probability tiles from the updated max,
- updating row sum online,
- updating `P @ V` online,
- normalizing after the loop.

The main schedule remains in `test/python/data/attention_l0_to_l1.transform.mlir`.

## Core Idea

The current design has three stages:

1. find the nearest reduction consumer from a loop-local value via forward BFS,
   returning both that reduction and the ordered elementwise chain between them,
2. force-fuse that ordered elementwise chain under the streaming loop as a
   sidecar computation,
3. checkpoint at the reduction frontier and rebuild it as a repaired
   recurrence.

This keeps the payload IR verifier-valid throughout the schedule while still
matching the shape of TVM's rolling update.

## Transform Split

### `transform.match.linalg_ext.rolling_update_next_reduction`

Pure analysis. Uses `NavigationTransformOpTrait`, so the input handle is not
consumed. Signature:

```text
(producer_op) -> (next_reduce_op, elemwise_ops)
```

Performs a forward BFS on the def-use graph starting from `producer_op`, where
each edge has unit length. Returns the first reduction-like op encountered —
i.e., the reduction at minimum BFS distance — along with all elementwise ops
strictly between `producer_op` and that reduction.

Return:

- `next_reduce_op`: a handle containing the single nearest reduction op,
- `elemwise_ops`: a handle containing the elementwise ops on the path from
  `producer_op` to that reduction, sorted in producer-to-consumer order.

BFS naturally handles branching use-def graphs: when the producer has multiple
direct consumers, the one that is a reduction is found at the current BFS
level before any of its transitive successors. In the attention graph
(`loop → red1(max) + elem1(exp_shift, uses loop+red1) → red2(sum) → …`),
BFS returns `red1` at distance 1 and an empty `elemwise_ops` handle, because
`red1` directly consumes the loop output even though `elem1` does too.

The op fails if no reduction is reachable from `producer_op`, or if any op
between `producer_op` and the nearest reduction is not a supported elementwise
(all-parallel `linalg.generic` or `linalg.map` with one result).

### `transform.linalg_ext.force_fuse_elemwise_chain_into_loop`

This is the MLIR equivalent of the "naive fusion" in Neptune-TVM rolling update.
The main difference is this version clones the element-wise operations
to create a "sidecar" chain of ops whose results are not used yet,
so technically no program semantic is violated.

Signature:

```text
(elemwise_chain_ops, streaming_loop) -> (sidecar_chain_ops, new_streaming_loop)
```

Given:

- the ordered elementwise handle from `transform.match.linalg_ext.rolling_update_next_reduction`,
- a streaming loop handle,

the transform should clone each elementwise op, fuse it under the loop,
and return the cloned sidecar ops in strictly the same order.

The sidecar chain is intentionally incomplete at this point.
It may compute values that are different from their out-of-loop counterparts.
That is acceptable as long as those sidecar values are not published to the original users yet.

### Reduction Checkpoint And Repair

This is the atomic step where rolling update becomes semantically complete.

Signature:

```text
(frontier_op, sidecar_chain_ops, streaming_loop)
  -> (repaired_frontier_op, repaired_sidecar_chain_ops, new_streaming_loop)
```

Inputs:

- the frontier reduction,
- the ordered sidecar chain already fused under the loop,
- the refreshed loop handle.

The transform should:

- convert the frontier reduction into loop-carried state,
- derive the repair term implied by the sidecar chain,
- rewrite the frontier and dependent sidecar ops to use the repaired
  recurrence,
- return refreshed handles for the repaired chain and loop.

This is the place to build recurrences such as:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
```

The important design constraint is that this step stays atomic. Before the
repair is applied, the sidecar chain is not yet a correct replacement for the
original reduction.

## Attention Example

For the current attention schedule, the intended flow is:

1. Build the outer `scf.forall` over output tiles and the inner streaming loop
   over K/V blocks.
2. Fuse QK and score scaling under that streaming loop.
3. Run `transform.match.linalg_ext.rolling_update_next_reduction` starting from
   the loop-local score tile. This finds the nearest reduction frontier and
   returns the ordered elementwise chain leading into it.
4. Force-fuse that elementwise chain under the streaming loop as sidecar ops.
5. Checkpoint at the frontier reduction and rebuild it as the online
   recurrence.
6. Continue to the next frontier until the loop carries the full online
   softmax state.

The end result is the usual FlashAttention recurrence:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

## Transform-Dialect Caveats

- `rolling_update_next_reduction` uses `NavigationTransformOpTrait` and does
  not consume its input handle; the producer handle remains valid after the
  call. Transforms that rewrite the loop (`force_fuse_elemwise_chain_into_loop`,
  checkpoint repair) use `FunctionalStyleTransformOpTrait` and do consume their
  inputs — always use the refreshed handles they return.
- Rewriting the streaming loop invalidates nested handles. Any transform in the
  force-fusion or checkpoint stage must return a refreshed loop handle.
- The sidecar chain should remain separate from the original chain until a
  final publish step. That is what makes intermediate states safe.
- Ordered multi-op handles are part of the contract here. The later transforms
  should not treat the elementwise chain as an unordered set.

## Testing Guidance

Rolling update should be tested in layers:

1. analysis-only tests for `rolling_update_next_reduction`,
2. small synthetic tests for force-fused elementwise chains,
3. reduction-checkpoint tests for max, sum, and matmul-style rolling updates,
4. end-to-end attention tests that check the final loop-carried recurrence
   shape.

## Autoscheduler Note

This split is also the right granularity for tuning. The autoscheduler can vary
tile sizes and fusion choices independently without having to treat rolling
update as one monolithic primitive.
