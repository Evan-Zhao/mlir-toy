"""Print a JAX DeepSeek Sparse Attention program as StableHLO MLIR.

DeepSeek Sparse Attention (DSA) has two stages. First, a small "lightning
indexer" scores every query/key-token pair:

    index_score[q, k] = sum_j w[q, j] * relu(index_q[q, j] @ index_k[k])

The indexer uses 64 query heads of dimension 128 in DeepSeek-V3.2-Exp, but only
one shared key per token. The top 2048 causal key positions under this score are
then used by the much larger core attention. Selection is fine-grained per
query token and shared by all core-attention heads. Thus the indexer remains
quadratic but cheap, while core attention changes from O(Q*K) to O(Q*top_k).

The file contains both stages as separate functions, but ``main`` lowers only
the sparse core attention. Selected token indices are an input, leaving top-k
selection out of the generated program for now. Input projections,
normalization, RoPE, FP8 quantization, cache management, and output projections
are deliberately omitted. The core uses the MQA form of Multi-head Latent
Attention (MLA): one latent key/value entry is shared across all 128 query
heads. With the default DeepSeek-V3.2-Exp dimensions, a core key concatenates a
512-wide KV latent with a 64-wide RoPE key, and the value is the 512-wide latent
before its omitted per-head output projection.
"""

import argparse
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.jit, static_argnames=("top_k",))
def deepseek_lightning_indexer(
    index_query: jax.Array,
    index_key: jax.Array,
    index_weights: jax.Array,
    *,
    top_k: int,
) -> jax.Array:
    """Return the top-k causal token indices selected by the DSA indexer.

    Queries are treated as the final ``queries`` positions in the KV sequence.
    This function is retained for later use, but is not lowered by ``main``.
    """
    if index_query.ndim != 4 or index_key.ndim != 3:
        raise ValueError("Expected rank-4 index queries and rank-3 index keys")
    batch, num_queries, index_heads, index_dim = index_query.shape
    kv_tokens = index_key.shape[1]
    if index_key.shape != (batch, kv_tokens, index_dim):
        raise ValueError(
            f"Expected index_key shape {(batch, kv_tokens, index_dim)}; got {index_key.shape}"
        )
    if index_weights.shape != (batch, num_queries, index_heads):
        raise ValueError(
            f"Expected index_weights shape {(batch, num_queries, index_heads)}; "
            f"got {index_weights.shape}"
        )
    if not 0 < top_k <= kv_tokens:
        raise ValueError(f"Expected 0 < top_k <= kv_tokens ({kv_tokens}); got {top_k}")
    if num_queries > kv_tokens:
        raise ValueError(
            f"Expected queries ({num_queries}) <= kv_tokens ({kv_tokens}) for causal attention"
        )

    # [B, Q, HI, K]. Scaling matches the reference implementation's
    # head_dim**-0.5 and n_index_heads**-0.5 factors. DeepSeek executes this
    # inexpensive indexer in FP8; use float32 here to keep the example simple.
    index_dot = jnp.einsum(
        "bqhd,bkd->bqhk", index_query, index_key, preferred_element_type=jnp.float32
    )
    index_scale = jnp.asarray(1.0 / math.sqrt(index_heads * index_dim), dtype=jnp.float32)
    index_scores = jnp.sum(
        jax.nn.relu(index_dot) * index_weights.astype(jnp.float32)[..., None], axis=2
    )
    index_scores = index_scores * index_scale

    key_positions = jnp.arange(kv_tokens, dtype=jnp.int32)
    query_positions = kv_tokens - num_queries + jnp.arange(num_queries, dtype=jnp.int32)
    causal = key_positions[None, :] <= query_positions[:, None]
    index_scores = jnp.where(causal[None, :, :], index_scores, -jnp.inf)
    _, selected_indices = lax.top_k(index_scores, top_k)
    return selected_indices


@jax.jit
def deepseek_sparse_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    selected_indices: jax.Array,
) -> jax.Array:
    """Apply DSA's sparse MQA core using preselected token indices.

    Args:
        query: Core-attention queries, ``[batch, queries, heads, qk_dim]``.
        key: Core-attention MQA keys, ``[batch, kv_tokens, qk_dim]``.
        value: Core-attention MQA values, ``[batch, kv_tokens, value_dim]``.
        selected_indices: Valid, causal KV indices selected for each query,
            ``[batch, queries, selected_tokens]``.

    Returns:
        Sparse attention output, ``[batch, queries, heads, value_dim]``.
    """
    if query.ndim != 4 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Expected rank-4 queries and rank-3 MQA keys/values")
    batch, num_queries, _, qk_dim = query.shape
    kv_tokens = key.shape[1]
    if key.shape != (batch, kv_tokens, qk_dim):
        raise ValueError(f"Expected key shape {(batch, kv_tokens, qk_dim)}; got {key.shape}")
    if value.shape[:2] != (batch, kv_tokens):
        raise ValueError(f"Expected value shape [batch, kv_tokens, value_dim]; got {value.shape}")
    if selected_indices.ndim != 3 or selected_indices.shape[:2] != (batch, num_queries):
        raise ValueError(
            "Expected selected_indices shape [batch, queries, selected_tokens]; "
            f"got {selected_indices.shape}"
        )
    if not jnp.issubdtype(selected_indices.dtype, jnp.integer):
        raise ValueError(f"Expected integer selected_indices; got {selected_indices.dtype}")

    # Gather [B, Q, selected_tokens, D] MQA entries. Indices are supplied by
    # the indexer and are promised to be in bounds and causal. Expressing the
    # batched gather directly avoids Python indexing's negative-index fixup.
    def gather_selected(x):
        return lax.gather(
            x,
            selected_indices[..., None],
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(3,),
                collapsed_slice_dims=(1,),
                start_index_map=(1,),
                operand_batching_dims=(0,),
                start_indices_batching_dims=(0,),
            ),
            slice_sizes=(1, 1, x.shape[2]),
            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        )

    selected_key = gather_selected(key)
    selected_value = gather_selected(value)

    attention_scores = jnp.einsum(
        "bqhd,bqtd->bqht",
        query,
        selected_key,
        preferred_element_type=jnp.float32,
    )
    attention_scores *= jnp.asarray(1.0 / math.sqrt(qk_dim), dtype=jnp.float32)
    probabilities = jax.nn.softmax(attention_scores, axis=-1)
    output = jnp.einsum(
        "bqht,bqtv->bqhv",
        probabilities,
        selected_value,
        preferred_element_type=jnp.float32,
    )
    return output.astype(query.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    def positive_int(value_str: str) -> int:
        value = int(value_str)
        if value <= 0:
            parser.error(f"Expected positive integer, got {value}")
        return value

    # DeepSeek-V3.2-Exp dimensions, with a practical chunked-prefill shape.
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--query-tokens", type=positive_int, default=128)
    parser.add_argument("--kv-tokens", type=positive_int, default=16384)
    parser.add_argument("--top-k", type=positive_int, default=2048)
    parser.add_argument("--attention-heads", type=positive_int, default=128)
    parser.add_argument("--kv-lora-rank", type=positive_int, default=512)
    parser.add_argument("--rope-head-dim", type=positive_int, default=64)
    parser.add_argument(
        "--activation-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = args.batch
    queries = args.query_tokens
    kv_tokens = args.kv_tokens
    heads = args.attention_heads
    kv_rank = args.kv_lora_rank
    qk_dim = kv_rank + args.rope_head_dim
    dtype = jnp.dtype(args.activation_dtype)

    query = jax.ShapeDtypeStruct((batch, queries, heads, qk_dim), dtype)
    key = jax.ShapeDtypeStruct((batch, kv_tokens, qk_dim), dtype)
    value = jax.ShapeDtypeStruct((batch, kv_tokens, kv_rank), dtype)
    selected_indices = jax.ShapeDtypeStruct((batch, queries, args.top_k), jnp.int32)

    stablehlo_module = deepseek_sparse_attention.lower(query, key, value, selected_indices)
    print(stablehlo_module.as_text())


if __name__ == "__main__":
    main()
