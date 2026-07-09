"""Print a JAX packed variable-length attention program as StableHLO MLIR.

This variant expresses documents as a parallel batch using ``vmap`` instead of a
sequential ``fori_loop``. It gathers fixed-size, padded document windows from
packed Q/K/V, applies regular masked attention over the dense
``[num_docs, max_doc_tokens, heads, head_dim]`` view, and scatters valid document
results back to packed output. Runtime document lengths come from
``cu_seqlens``; the maximum document length must be provided statically via
``--max-doc-tokens``.
"""

import argparse
import math
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("max_doc_len"))
def doc_offset_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, offsets: jax.Array, *, max_doc_len: int
) -> jax.Array:
    """
    Document-offset attention over packed Q/K/V.
    q, k, v: [T, H, D]
    offsets: [N + 1], where document d occupies
              [offsets[d], offsets[d + 1]) in the packed token axis.
    Returns:
        out: [T, H, D]

    This computes per-document attention:
        out[offsets[d] : offsets[d+1], :, :] = softmax(q_d @ k_d.T / sqrt(Dq)) @ v_d

    but avoids dynamically sized slices by using a fixed max_doc_len.
    The source intentionally uses ``vmap`` and a ``unique_indices=True`` scatter to expose
    document-level parallelism to the compiler.
    `offsets` must describe a monotonic, non-overlapping partition of the packed token axis,
    with every document length <= max_doc_len.
    """

    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"Expected q, k, v to have the same shape; got {q.shape}, {k.shape}, {v.shape}"
        )
    LT, H, D = q.shape
    L0 = max_doc_len
    N = offsets.shape[0] - 1
    starts = offsets[:-1]
    lengths = offsets[1:] - starts
    token_offsets = jnp.arange(L0, dtype=offsets.dtype)
    scale = jnp.asarray(1.0 / math.sqrt(D), dtype=jnp.float32)

    def load_doc(x, start):
        # This is the JAX idiom closest to a masked vector load. Unlike
        # dynamic_slice, gather/take with mode="fill" does not clamp the window
        # start when start + L0 exceeds LT; out-of-bounds lanes are filled with 0.
        token_indices = start + token_offsets
        return jnp.take(x, token_indices, axis=0, mode="fill", fill_value=0)

    def one_doc_attention(start, doc_len):
        """Dense masked attention for one logical document window."""

        q_doc, k_doc, v_doc = [load_doc(x, start) for x in (q, k, v)]
        scores = jnp.einsum("ihd,jhd->hij", q_doc, k_doc, preferred_element_type=jnp.float32)
        scores = scores * scale
        valid_tokens = token_offsets < doc_len
        key_valid = valid_tokens.reshape(1, 1, L0)
        query_valid = valid_tokens.reshape(1, L0, 1)
        scores = jnp.where(jnp.logical_and(query_valid, key_valid), scores, -jnp.inf)
        probs = jax.nn.softmax(scores, axis=-1)
        out_doc_f32 = jnp.einsum("hij,jhd->ihd", probs, v_doc, preferred_element_type=jnp.float32)
        return out_doc_f32.astype(q.dtype)

    # [N, M, H, D]. `vmap` makes the whole per-document attention computation a
    # batched, document-parallel map rather than a loop-carried sequential loop.
    out_docs = jax.vmap(one_doc_attention)(starts, lengths)

    # Scatter valid document tokens back to packed layout. Invalid padded tokens
    # are sent to unique out-of-bounds sink positions so the scatter can be marked
    # unique and the OOB updates can be dropped instead of materializing sinks.
    doc_token_indices = starts.reshape(N, 1) + token_offsets.reshape(1, L0)
    valid_doc_tokens = token_offsets.reshape(1, L0) < lengths.reshape(N, 1)
    sink_indices = LT + jnp.arange(N * L0, dtype=offsets.dtype).reshape(N, L0)
    scatter_indices = jnp.where(valid_doc_tokens, doc_token_indices, sink_indices).reshape(N * L0)
    scatter_updates = out_docs.reshape(N * L0, H, D)
    out = jnp.zeros((LT, H, D), dtype=q.dtype)
    return out.at[scatter_indices, :, :].set(scatter_updates, unique_indices=True, mode="drop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    def positive_int(value_str: str) -> int:
        value = int(value_str)
        if value <= 0:
            parser.error(f"Expected positive integer, got {value}")
        return value

    parser.add_argument("--batch", type=positive_int, default=8)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--total-tokens", type=positive_int, default=1024)
    parser.add_argument("--max-doc-tokens", type=positive_int, default=None)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--index-dtype", choices=("int32", "int64"), default="int32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_doc_tokens = args.max_doc_tokens
    if max_doc_tokens is None:
        max_doc_tokens = (args.total_tokens + args.batch - 1) // args.batch
    index_dtype = jnp.dtype(args.index_dtype)
    shape = (args.total_tokens, args.heads, args.head_dim)
    q, k, v = [jax.ShapeDtypeStruct(shape, jnp.float16) for _ in range(3)]
    offsets = jax.ShapeDtypeStruct((args.batch + 1,), index_dtype)
    attn_fn = partial(doc_offset_attention, max_doc_len=max_doc_tokens)
    stablehlo_module = jax.jit(attn_fn).lower(q, k, v, offsets)
    print(stablehlo_module.as_text())


if __name__ == "__main__":
    main()
