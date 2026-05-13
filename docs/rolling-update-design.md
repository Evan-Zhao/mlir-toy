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
   sidecar computation and publish it as additional loop results,
3. checkpoint at the reduction frontier and rebuild the first repaired
   recurrence state under the streaming loop.

This keeps the payload IR verifier-valid throughout the schedule while still
matching the shape of TVM's rolling update.

## Transform Split

### `transform.match.loop_ru.rolling_update_next_reduction`

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

### `transform.loop_ru.clone_fuse_elemwise`

This is the MLIR equivalent of the "naive fusion" in Neptune-TVM rolling update.
The main difference is this version clones the element-wise operations
to create a "sidecar" chain of ops whose results are not used yet,
so technically no program semantic is violated.

Signature:

```text
(elemwise_ops, parallel_loop, streaming_loop) -> (sidecar_ops)
```

The transform clones each elementwise op, fuses it under the loop,
and returns the cloned sidecar ops in strictly the same order.

The input loop handles are preserved rather than consumed. If the transform
rebuilds either loop, transform tracking remaps the existing handles to the
replacement loops.

The transform rebuilds both loops:

- the inner `scf.for` gets one extra iter_arg/result per sidecar tensor,
- the outer `scf.forall` gets matching extra shared_out/result slots.

Each sidecar op is cloned under the inner loop, tiled from loop-local producer tiles,
inserted into its full-tensor sidecar destination, and relayed through the outer loop.

Crucially, this still does not change the uses of the original out-of-loop
elementwise chain. The sidecar values are published as additional loop results,
but they are not yet substituted into the original computation except by the
later repair step.

### `transform.loop_ru.repair_reduction_frontier`

This is the step that makes a chosen rolling-update frontier semantically correct.

Signature:

```text
(producer_reduce_ops, reduce_op, elemwise_orig_ops, sidecar_chain_ops, outer_loop, inner_loop)
  -> (repaired_reduce_op, repaired_sidecar_chain_ops)
```

Inputs:

- the producer reductions feeding the chosen frontier,
- the frontier reduction,
- the original ordered elementwise chain outside the loop,
- the ordered sidecar chain already fused under the loop,
- the enclosing `scf.forall`,
- the refreshed inner `scf.for`.

What TVM does here:

- mock-inline the elementwise chain into the frontier reduction,
- match the inlined block as a self-reduction `out = f(out, g(...))`,
- classify RHS loads into:
    - incomplete frontier values `x`,
    - updated loop-local values `x'`,
    - ordinary operands `u`,
- derive a repair updater `H(r, x, x')` such that:

```text
reduce_j g(x', u_j) = H(reduce_j g(x, u_j), x, x')
```

- validate that `H` distributes over the reducer:

```text
H(f(y1, y2), x, x') = f(H(y1, x, x'), H(y2, x, x'))
```

- run rfactor, then rewrite the write-back block to use `H` with:
    - `r` = the loop-carried partial reduction state,
    - `x` = previous frontier values,
    - `x'` = repaired current frontier values.

For attention, this is the step that produces repair terms such as
`exp2(m_prev - m_next)`.

To implement the same in MLIR:

- [x] rebuilds the inner `scf.for` and outer `scf.forall` to carry an extra reduction state,
- [x] clones the frontier reduction under the inner loop on the sidecar tile,
- [x] rewires the frontier reduction to consume the relayed sidecar value,
- [ ] inline or otherwise normalize the sidecar chain into a single scalar reduction
      RHS expression,
- [ ] match the reduction as `out = f(out, g(...))`,
- [ ] derive the updater `H(r, x, x')` from that RHS,
- [ ] prove or conservatively check that `H` is valid for the reducer,
- [ ] apply `H` when rebuilding the loop-carried/write-back reduction state instead
      of only substituting the sidecar result directly.

The current implementation makes these assumptions:

- a unary single-result `linalg.generic` reduction, and
- exactly one original elementwise result to feed that reduction.

## Attention Example

For the current attention schedule, the intended flow is:

1. Build the outer `scf.forall` over output tiles and the inner streaming loop
   over K/V blocks.
2. Fuse QK and score scaling under that streaming loop.
3. Run `transform.match.loop_ru.rolling_update_next_reduction` starting from
   the loop-local score tile. This finds the nearest reduction frontier and
   returns the ordered elementwise chain leading into it.
4. Force-fuse that elementwise chain under the streaming loop as sidecar ops
   and publish them as extra loop results.
5. Run `transform.loop_ru.repair_reduction_frontier` to turn the chosen
   reduction frontier into loop-carried state under the streaming loop.
6. Repeat this pattern for later frontiers as more repair steps are implemented.

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
  not consume its input handle; the producer handle remains valid after the call.
- `clone_fuse_elemwise` and `repair_reduction_frontier` use
  `FunctionalStyleTransformOpTrait` for the explicit op handles they rewrite,
  but the loop handles themselves are read-only inputs. Payload tracking remaps
  those loop handles to replacement loops when the transform rebuilds them.
- The sidecar chain should remain separate from the original chain until a
  targeted repair step rewires a specific frontier use. That is what makes the
  intermediate states safe.
- Ordered multi-op handles are part of the contract here. The later transforms
  should not treat the elementwise chain as an unordered set.

## Testing Guidance

Rolling update should be tested in layers:

1. analysis-only tests for `rolling_update_next_reduction`,
2. small synthetic tests for force-fused elementwise chains,
3. reduction-frontier repair tests for the currently supported unary reduction pattern,
4. end-to-end attention tests that check the final loop-carried recurrence shape.

## Autoscheduler Note

This split is also the right granularity for tuning. The autoscheduler can vary
tile sizes and fusion choices independently without having to treat rolling
update as one monolithic primitive.
