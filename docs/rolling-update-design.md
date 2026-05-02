# Attention Rolling Update Design

## Goal

This note isolates the `rolling_update` part of the attention L0-to-L1
schedule.

In TVM terms, this is the transformation that turns a reduction under a
streaming loop into an online recurrence with repaired writeback. For
FlashAttention, it is the mechanism that fuses and synchronizes:

- row max,
- probability tile construction,
- row sum,
- `P @ V`,
- accumulator repair.

The main schedule remains in
`test/python/data/attention_l0_to_l1.transform.mlir`. This document is the
detailed design reference for the custom rolling-update transforms needed there.

## What `rolling_update` Means

At a high level, `rolling_update` is the TVM-style primitive that:

1. finds reductions that are incomplete under a streaming loop,
2. fuses the necessary producer chain under that loop,
3. converts those reductions into loop-carried state,
4. derives the algebra needed to repair the partial update,
5. publishes a globally correct result.

For attention, this produces the online-softmax recurrences:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

This is the reduction counterpart to TVM `reverse_compute_at`: it does not just
move a consumer under a loop, it also changes the meaning of the reduction so
that the loop-local partial computation becomes globally correct.

## Why MLIR Should Decompose It

The TVM primitive is intentionally large because it mutates the only live TIR
buffer graph. Stopping halfway can expose invalid intermediate states.

In MLIR, tensor SSA gives a better structure:

- the original algorithmic chain can remain live,
- a new scheduled sidecar chain can be built in parallel,
- only a final publish step has to replace the original result.

That makes it practical to decompose `rolling_update` into smaller transform
ops while keeping the payload IR verifier-valid throughout the schedule.

## TVM Reference Behavior

The current TVM implementation in
`/Users/evanzhao/workspace/nautilus/src/tir/schedule/primitive/reduction.cc`
bundles these phases:

1. frontier discovery,
2. unsafe fusion under the target loop,
3. reduce-to-scan conversion,
4. mock inlining for analysis,
5. algebraic repair derivation,
6. generalized rfactor,
7. writeback repair.

The dangerous phases are the ones that mutate live producers or expose partial
results before repair.

## MLIR Decomposition

The intended MLIR decomposition is:

### `transform.match.linalg_ext.rolling_frontier`

Pure analysis op.

Input:

- target reduction-like op,
- target streaming loop,
- optional mode such as `scan` or `split_k`.

Output:

- frontier reduction handles,
- spatial producer-chain handles in topological order,
- metadata describing incomplete values and reduction dimensions.

This replaces TVM frontier discovery and should fail silenceably if the graph
is not a supported rolling-update pattern.

### `transform.linalg_ext.clone_chain_under_loop`

Clone or tile the spatial producer chain under the target loop.

This is the MLIR replacement for TVM’s unsafe reverse-compute-at phase. The
original producer chain stays live until a final publish step.

For attention, this is where loop-local score tiles and score elementwise
transforms are created under the streaming loop.

### `transform.linalg_ext.reduction_to_scan`

Convert one frontier reduction into explicit loop-carried state.

For L1, the preferred representation is usually `affine.for` or `scf.for`
`iter_args`, not an expanded scan buffer:

```text
m_next = max(m_prev, row_max(score_tile))
l_next = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
```

### `transform.linalg_ext.derive_rolling_repair`

Pure analysis op.

Given:

- reducer kind,
- cloned producer chain,
- incomplete-value metadata,
- previous/current value mapping,

derive the algebraic repair plan needed to turn partial results into the global
result.

For FlashAttention, this should derive or recognize:

```text
scale = exp2(m_prev - m_next)
l_next = scale * l_prev + row_sum
acc_next = broadcast(scale) * acc_prev + pv
```

### `transform.linalg_ext.rfactor_reduction`

Factor the target reduction over the streaming loop.

In MLIR, this should prefer constructing a sidecar partial computation instead
of replacing original uses immediately. For attention, this is the step that
turns a full `P @ V` into per-K/V-block `p_tile @ v_tile`.

### `transform.linalg_ext.apply_rolling_repair`

Apply the repair algebra to the factored computation.

For attention, this creates the loop-carried accumulator recurrence:

```text
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
```

At this point, the sidecar computation is mathematically complete, but it still
does not have to replace the original result.

### `transform.linalg_ext.publish_scheduled_result`

Replace the original algorithmic result with the completed scheduled result.

This is the only externally visible step. After publication, canonicalization
and DCE can remove the old L0 chain.

## FlashAttention-Specific Pipeline

For the current attention example, the decomposed pipeline is:

1. Match the attention chain structurally.
2. Create the outer `scf.forall` over `(B, H, M-block)`.
3. Extract `q_tile` once per output tile.
4. Create the inner streaming loop over K/V blocks.
5. Clone or tile QK and score scaling inside the streaming loop.
6. Add row-max recurrence:

   ```text
   row_max = reduce_max(score_tile)
   m_next = max(m_prev, row_max)
   ```

7. Add probability tile:

   ```text
   p_tile = exp2(score_tile - broadcast(m_next))
   ```

8. Add row-sum recurrence:

   ```text
   row_sum = reduce_sum(p_tile)
   l_next = exp2(m_prev - m_next) * l_prev + row_sum
   ```

9. Add `P @ V` partial:

   ```text
   pv = p_tile @ v_tile
   ```

10. Add accumulator recurrence:

   ```text
   acc_next = broadcast(exp2(m_prev - m_next)) * acc_prev + pv
   ```

11. Normalize after the loop:

   ```text
   out_tile = acc_final / broadcast(l_final)
   ```

12. Insert the output tile into the `scf.forall` shared output.
13. Publish the scheduled result and DCE the original chain.

## Atomicity Boundaries

These steps can be independent because they do not expose incorrect values to
original users:

- frontier matching and analysis,
- sidecar cloning and tile extraction,
- repair derivation,
- row-max recurrence construction,
- row-sum recurrence construction,
- `P @ V` partial construction,
- accumulator recurrence construction.

These should remain atomic internally:

- any step that replaces live uses with a value that is only correct after
  repair,
- any step that mutates the only live producer chain instead of cloning it,
- final publish plus required result checks.

The preferred design is to avoid the first two cases whenever possible.

## Testing Guidance

Rolling update should be tested in layers:

1. matcher-only tests for frontier discovery,
2. small synthetic tests for max, sum, and matmul rolling updates,
3. end-to-end attention structure tests checking:
   - outer `scf.forall`,
   - inner streaming loop with `iter_args`,
   - row-max and row-sum recurrences,
   - `P @ V` partial,
   - final normalization.

## Autoscheduler Implications

The autoscheduler should tune smaller semantic steps rather than a single
monolithic `rolling_update`.

Useful knobs include:

- outer M tile size,
- streaming K/V block size,
- scan-buffer versus loop-carried-state representation,
- accumulator element type and truncation points,
- where probability tiles are materialized or fused,
- whether to use `exp` or `exp2`.

A high-level convenience transform can still exist, for example:

```mlir
transform.linalg_ext.flash_attention_schedule %attention
  {m_tile = 128, kv_tile = 64}
```

but it should lower internally to the smaller rolling-update transform family
above.
