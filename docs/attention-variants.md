# Attention Operator Feature Roadmap

This note records attention operator features that are useful targets for the Neptune attention
pipeline. The goal is to organize the roadmap around operator behavior rather than kernel schedules.
Scheduling techniques such as rolling update, SplitK, persistent decode, or work queues
are tracked separately, even when a feature strongly motivates a particular schedule.

A concrete attention variant is usually a composition of several axes: for example,
paged KV-cache decode with GQA, RoPE, causal masking, FP8 K/V, and cache append.

## Feature-Axis Model

Attention variants should be described using the axes below. The checklist is ordered roughly by
roadmap priority.
Checked items have current Neptune pipeline coverage; unchecked items are useful targets.

- [x] **Static sequence geometry**: full-sequence attention, rectangular Q-versus-K/V sequence shapes,
      and decode-shaped inputs where Q has length 1 and K/V are long.
- [x] **Head mapping**: MHA, GQA, and MQA. MQA is the special case of GQA with one K/V head.
- [x] **Basic score domains**: dense global attention, triangular causal attention,
      and contiguous sliding-window causal attention.
- [x] **Basic score value modifiers**: ALiBi head-dependent additive bias.
- [x] **Basic input numeric representation**: FP16 attention, plus causal attention with FP8 K/V inputs
      and fused per-head K/V dequantization scales.
- [ ] **Runtime sequence and segment bounds**: logical Q and K/V lengths should be carried by
      runtime metadata for variable-length requests or packed document segments.
- [ ] **KV-cache decode ABI**: K/V may come from persistent cache storage rather than freshly
      materialized dense tensors, with explicit request slots, cache lengths, cache positions,
      and batch indirection.
- [ ] **Cache mutation semantics**: the operator may append or write new K/V into the cache,
      or expose a paired cache-update operation with compatible metadata.
- [ ] **K/V physical layout**: logical K/V token coordinates may map to head-major, token-major,
      blocked, or other serving-oriented memory layouts.
- [ ] **Page-table K/V indirection**: logical K/V positions may map through a block table
      into paged cache storage.
- [ ] **Quantized KV-cache representation**: persistent K/V may be stored in low-precision
      formats such as FP8, with explicit scale metadata and scale granularity.
- [ ] **Position-dependent Q/K transforms**: Q and/or K may be transformed before the QK dot,
      usually using token position and channel index. RoPE is the primary target.
- [ ] **Compressed or latent K/V representation**: the cache may store latent vectors or
      compressed factors that must be reconstructed before attention consumes per-head K/V values.

For the features that Neptune already supports, the static Transform-dialect examples live under
[`test/Pipeline`](../test/Pipeline).
The Python pipeline tests also exercise multiple shapes, including GQA with one K/V head.
Decode-input scheduling support is tracked in the [SplitK update design](split-k-update-design.md).

## Missing Feature Axes

### Runtime Sequence and Segment Bounds

Runtime sequence and segment bounds cover batches where each request, sequence, or packed document
segment has different valid Q and K/V lengths. The operator should not infer validity only from
static tensor extents or from a padded square score matrix. Instead, valid row and K/V ranges should
come from metadata such as per-request lengths, cumulative sequence lengths, or segment boundaries.

The first target should be **variable-length packed prefill attention**. Q, K, and V are flattened
across the batch, and `cu_seqlens`-style metadata maps each token range back to a request or
document segment. This removes padding work and exercises nonuniform tile liveness without requiring
KV-cache storage.

This axis composes with causal and document masking. Bounds say which tokens exist; the score domain
says which valid Q/K pairs may interact. For packed causal training, the effective mask is usually
“same document segment and not in the future.”

### KV-Cache Decode ABI

KV-cache decode distinguishes ordinary dense K/V tensors from persistent K/V storage owned by a
serving runtime. A cache-ready operator should accept metadata for request slots, current cache
lengths, logical cache positions, and optional batch indirection. It should not assume that K/V are
freshly materialized dense tensors for exactly the current call.

The first target should be **contiguous KV-cache decode attention**. Q has length 1 or a small
number of tokens, K/V are read from a persistent contiguous cache, and each request has a runtime
cache length that bounds the valid K/V range. For GQA and MQA, Q-head-to-KV-head mapping should be
explicit rather than implicit broadcasting.

The ABI should also include position metadata before RoPE is implemented. Decode attention needs to
know the absolute position of the new query token and the logical positions of cached K/V tokens.

### Cache Mutation Semantics

Cache mutation covers attention calls that write newly computed K/V values into the cache, or a
paired cache-update operation that uses the same metadata as attention. This is distinct from
read-only cache attention because it introduces side effects, aliasing constraints, and ordering
requirements.

The first target should be **decode with cache append**. The operator receives new K/V for the
current token, writes them into cache at `cache_position`, and attends over the valid cache prefix.
The roadmap should define whether attention reads the cache before or after the append; serving
paths usually want “append then attend.”

Even if the update lowers as a separate kernel, representing mutation at the operator level helps
the compiler preserve correctness around cache slots, positions, and request isolation.

### K/V Physical Layout

K/V physical layout is an operator feature when it changes how logical token coordinates map to
memory coordinates. Ordinary dense tensors are only one layout. Serving-oriented kernels often use
head-major, token-major, or blocked layouts chosen for vectorized cache loads.

The first target should be **blocked contiguous KV-cache decode**. Logical K/V positions are
contiguous for each request, but the physical tensor groups token, head, and head-dimension
coordinates into cache-load-friendly blocks. This exercises explicit layout indexing without adding
page-table indirection.

The operator should separate logical coordinates from physical layout: logical coordinates identify
the request, token, and KV head; physical layout determines the address.

### Page-Table K/V Indirection

Paged KV-cache support separates logical sequence coordinates from physical storage coordinates. K/V
tile loads are indirect: a logical token range is mapped through a block table into one or more
physical cache pages. This is K/V storage indirection, not score sparsity.

The first target should be **PagedAttention-style decode**. The operator takes a block table,
per-request cache lengths, page size, and paged K/V tensors. For each request, logical K/V positions
are translated through the block table before memory is loaded.

The compiler representation should preserve logical attention positions, page-table lookup, and
physical layout inside a page as separate concepts.

### Quantized K/V Representation

Quantized K/V representation covers cases where K/V are stored in a lower-precision format and
interpreted using scale metadata. This is broader than the current FP8 K/V input path because
persistent KV caches need explicit scale layout and scale granularity.

The first target should be **FP8 KV-cache decode with explicit scales**. The operator should model
cache storage dtype, compute dtype, and scale tensors. Scale granularity should be explicit:
per-tensor, per-head, per-token, or layout-dependent scales imply different indexing and fusion choices.

Quantized cache support should compose with both contiguous and paged cache addressing.

### Position-Dependent Q/K Transforms

Position-dependent Q/K transforms modify Q and K before the QK dot, usually based on token position
and channel index. The compiler target is to fuse the transform into tile loads or immediately
before the dot, without materializing transformed Q/K tensors.

The first target should be **RoPE-fused attention**. RoPE rotates paired Q and K channels using
position-dependent sin/cos values. In prefill, positions usually come from token indices within each
sequence or segment. In decode, positions come from cache-position metadata.

RoPE should be represented as a Q/K transform rather than a score modifier because it changes the
vectors entering the dot product, not the score after the dot.

### Compressed or Latent K/V Representation

Compressed or latent K/V representations change the mathematical source of K/V, not only their
layout or dtype. The cache may store latent vectors, compressed factors, or split positional /
non-positional components that must be reconstructed before attention consumes per-head K/V values.

The first target should be **multi-head latent attention (MLA)**. In MLA-style decode, the cache
stores compressed latent state instead of ordinary per-head K/V tensors.
The attention loop reconstructs the K/V values it needs and places that projection work
relative to QK, softmax, and PV.

This feature should remain separate from quantized KV cache. Quantization changes how values are
stored numerically; MLA changes what the stored values mean mathematically.
