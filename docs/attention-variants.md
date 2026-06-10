# Attention Variant Roadmap

This note records attention variants that are useful targets for the Neptune attention pipeline.
The goal is to prioritize variants that are both common in modern models and interesting for a
tiled compiler: masks that change tile liveness, layouts that change memory access, and score
transforms that should fuse into the FlashAttention-like loop.

## Current Coverage

The pipeline currently covers these variants:

- **Global multi-head attention**: dense unmasked scaled dot-product attention.
- **Causal multi-head attention**: triangular score masking with dead-tile specialization.
- **Global grouped-query attention (GQA)**: Q heads grouped over fewer K/V heads.
- **Global multi-query attention (MQA)**: the GQA case where all Q heads share one K/V head.

The static Transform-dialect examples live under [`test/Pipeline`](../test/Pipeline). The Python
pipeline tests also exercise multiple shapes, including GQA with one K/V head, which is the MQA
case.

## Prioritized Variants

### Sliding-Window Causal Attention

Sliding-window causal attention should be the next target. It keeps the causal ordering but limits
each query row to a fixed-width local history window. This is common in long-context decoder
models and is a direct extension of the current causal-mask support.

Compiler pressure points:

- The K/V streaming loop has both a causal upper bound and a local-window lower bound.
- Many score tiles are completely dead, while boundary tiles are partially live.
- The mask predicate is still affine/index-based, so it should reuse much of the causal
  dead-tile specialization machinery.
- The resulting kernel should avoid both future tokens and tokens older than the window instead
  of relying on `-inf` masking after loading all K/V tiles.

### Variable-Length Packed Attention

Packed attention handles batches where each sequence has a different length and padding should not
consume tile work. This is important for real training and inference batches.

Compiler pressure points:

- Per-sequence metadata drives row and K/V bounds.
- Tile liveness becomes data-dependent at the batch/sequence level.
- The schedule needs clean handling for empty or partial tiles at sequence boundaries.

### Decode Attention With KV Cache

Decode attention computes one or a few new query positions against a growing K/V cache. This is the
inference-critical form for autoregressive serving, and MQA/GQA make it especially memory-layout
sensitive.

Compiler pressure points:

- Q has a very small sequence length while K/V can be long.
- K/V cache layout dominates performance more than the QK and PV math shape.
- The schedule should expose cache loads, head grouping, and cache position arithmetic explicitly.

### PagedAttention-Style KV Cache

PagedAttention is primarily a memory-layout variant: logical K/V positions are mapped through a
block table into non-contiguous physical cache pages.

Compiler pressure points:

- K/V tile loads are indirect through a page table rather than simple affine slices.
- The schedule needs to separate logical sequence coordinates from physical memory coordinates.
- It is a good test for whether HTile can represent tile loads that are not simple strided slices.

### RoPE-Fused Attention

Rotary position embedding applies a position-dependent rotation to Q and K before the QK dot. The
useful compiler target is not a standalone RoPE kernel, but fusing those transforms into the
attention loop without materializing rotated Q/K tensors.

Compiler pressure points:

- Q and K elementwise transforms should be fused into their tile loads or immediately before dot.
- The transform depends on token position and channel parity/pairing.
- The Q-side and K-side transforms must remain structurally visible through TA/linalg matching.

### ALiBi-Fused Attention

ALiBi adds a head-dependent linear bias to attention scores based on query/key distance. It is a
small score-side transform, but common enough to deserve a direct test.

Compiler pressure points:

- The bias should fuse into the score tile before row-max.
- The bias uses query/key indices and per-head slopes, so it exercises tiled index materialization.
- It is a good minimal test for score-bias fusion independent of masking.

### Block-Sparse Attention With Global Tokens

Longformer/BigBird-style attention combines local blocks with selected global or random blocks.
This is more general than sliding-window attention because the active K/V tiles are no longer a
single contiguous interval per query tile.

Compiler pressure points:

- The K/V streaming loop may need sparse tile lists instead of affine lower/upper bounds.
- Global tokens create non-local dependencies that should still share the online-softmax state.
- A practical implementation needs a representation for sparse block patterns before lowering to
  backend-specific launch code.

### Cross-Attention

Cross-attention uses Q from one sequence and K/V from another. It is not as exotic as sparse or
paged attention, but it is common in encoder-decoder and multimodal models.

Compiler pressure points:

- Q and K/V sequence lengths differ.
- Masks may be non-causal but still batch- or modality-dependent.
- Shape matching should not assume square score matrices.

### Multi-Head Latent Attention

Multi-head latent attention (MLA) compresses K/V cache state into latent vectors and reconstructs
the per-head values needed by attention. This is a larger departure from ordinary GQA/MQA.

Compiler pressure points:

- The K/V cache representation is no longer plain per-token per-head K/V tensors.
- Extra projection/reconstruction work must be placed relative to the attention loop.
- It is a good long-term target after cache-layout and decode-attention support are stable.

## Suggested Order

1. Sliding-window causal attention.
1. Variable-length packed attention.
1. Decode attention with contiguous GQA/MQA KV cache.
1. PagedAttention-style KV cache.
1. RoPE and ALiBi score/QK fusion.
1. Block-sparse attention with global tokens.
1. Cross-attention.
1. Multi-head latent attention.
