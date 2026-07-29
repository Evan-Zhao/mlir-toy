"""Print a JAX packed variable-length attention program as StableHLO MLIR.

This variant expresses documents as a parallel batch using ``vmap`` instead of a sequential
``fori_loop``. Neptune custom calls preserve contiguous masked window reads and insertion around a
regular masked attention computation over the dense
``[num_docs, max_doc_tokens, heads, head_dim]`` view. Runtime document lengths come from
``offsets``; the maximum document length must be provided statically via ``--max-doc-tokens``.
"""

import argparse
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def packed_window_extract(
    packed: jax.Array,
    starts: jax.Array,
    lengths: jax.Array,
    other: jax.Array,
    *,
    max_doc_len: int,
) -> jax.Array:
    """Extract contiguous masked windows from ``packed[T, ...]``.

    ``starts`` and ``lengths`` have shape ``[N]``, ``other`` is scalar, and the result has shape
    ``[N, max_doc_len, ...]``. Element ``[d, i, ...]`` reads
    ``packed[starts[d] + i, ...]`` when ``i < lengths[d]``; otherwise it produces ``other`` without
    accessing ``packed``. Each range must satisfy
    ``0 <= lengths[d] <= max_doc_len`` and ``0 <= starts[d] <= T - lengths[d]``.

    ``max_doc_len`` is static and encoded by the result type. The StableHLO custom call is currently
    a compiler marker, not an executable FFI implementation.
    """

    if starts.shape != lengths.shape or starts.ndim != 1:
        raise ValueError("Expected starts and lengths to be rank-one arrays with the same shape")
    if other.shape or other.dtype != packed.dtype:
        raise ValueError("Expected other to be a scalar with the packed element dtype")
    result = jax.ShapeDtypeStruct((starts.shape[0], max_doc_len, *packed.shape[1:]), packed.dtype)
    return jax.ffi.ffi_call("neptune.packed_window_extract", result, has_side_effect=False)(
        packed, starts, lengths, other
    )


def packed_window_insert(
    windows: jax.Array,
    destination: jax.Array,
    starts: jax.Array,
    lengths: jax.Array,
) -> jax.Array:
    """Insert masked ``windows[N, W, ...]`` into ``destination[T, ...]``.

    The result has the destination shape. When ``i < lengths[d]``, element ``[d, i, ...]`` is
    inserted at ``starts[d] + i``; otherwise no memory access occurs. Unwritten destination
    elements are preserved. Each range must satisfy ``0 <= lengths[d] <= W`` and
    ``0 <= starts[d] <= T - lengths[d]``, and valid ranges must not overlap.

    The StableHLO custom call is pure and currently serves as a compiler marker, not an executable
    FFI implementation.
    """

    if starts.shape != lengths.shape or starts.ndim != 1:
        raise ValueError("Expected starts and lengths to be rank-one arrays with the same shape")
    if windows.ndim < 2 or windows.shape[0] != starts.shape[0]:
        raise ValueError("Expected one window per start")
    if destination.shape[1:] != windows.shape[2:] or destination.dtype != windows.dtype:
        raise ValueError("Expected destination and window element shapes and dtypes to match")
    result = jax.ShapeDtypeStruct(destination.shape, destination.dtype)
    return jax.ffi.ffi_call("neptune.packed_window_insert", result, has_side_effect=False)(
        windows, destination, starts, lengths
    )


@partial(jax.jit, static_argnames=("max_doc_len",))
def doc_offset_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    offsets: jax.Array,
    *,
    max_doc_len: int,
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

    but avoids dynamically sized tensors by using a fixed max_doc_len. The source intentionally
    uses ``vmap`` and semantic packed-window calls to expose document-level parallelism while
    preserving contiguous access information for the compiler.
    `offsets` must describe a monotonic, non-overlapping partition of the packed token axis,
    with every document length <= max_doc_len.
    """

    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"Expected q, k, v to have the same shape; got {q.shape}, {k.shape}, {v.shape}"
        )
    LT, H, D = q.shape
    L0 = max_doc_len
    starts = offsets[:-1]
    lengths = offsets[1:] - starts
    token_offsets = jnp.arange(L0, dtype=offsets.dtype)
    scale = jnp.asarray(1.0 / math.sqrt(D), dtype=jnp.float32)

    def one_doc_attention(q_doc, k_doc, v_doc, doc_len):
        """Dense masked attention for one logical document window."""

        # Compute ihd @ jhd -> hij.
        # The `dimension_numbers` argument is (
        #   (lhs_contracting_dims, rhs_contracting_dims),
        #   (lhs_batch_dims, rhs_batch_dims)).
        # And the output order is determined: batch dims first, then lhs non-contracting dims,
        # then rhs non-contracting dims.
        # We are using lax.dot over jnp.einsum because lax.dot lowers directly to stablehlo.dot_general,
        # while jnp.einsum may favor a different (equivalent) order for the contraction.
        scores_hij = lax.dot(
            q_doc,
            k_doc,
            dimension_numbers=(([2], [2]), ([1], [1])),
            preferred_element_type=jnp.float32,
        )
        scores_hij = scores_hij * scale

        # Apply an OOB mask to both the key and the query, so it looks like a sub-square within a square.
        # We have passes to convert this mask to actual reduction in number of iterations during lowering.
        valid_tokens = token_offsets < doc_len
        query_valid = valid_tokens.reshape(1, L0, 1)
        key_valid = valid_tokens.reshape(1, 1, L0)
        scores_hij = jnp.where(query_valid & key_valid, scores_hij, -jnp.inf)

        probs_hij = jax.nn.softmax(scores_hij, axis=-1).astype(q_doc.dtype)
        # Compute hij @ jhd -> hid.
        out_hid = lax.dot(
            probs_hij,
            v_doc,
            dimension_numbers=(([2], [0]), ([0], [1])),
            preferred_element_type=jnp.float32,
        )
        out_ihd = out_hid.transpose(1, 0, 2)
        return out_ihd.astype(q.dtype)

    q_docs, k_docs, v_docs = [
        packed_window_extract(x, starts, lengths, jnp.zeros((), dtype=x.dtype), max_doc_len=L0)
        for x in (q, k, v)
    ]

    # [N, M, H, D]. `vmap` makes the whole per-document attention computation a
    # batched, document-parallel map rather than a loop-carried sequential loop.
    out_docs = jax.vmap(one_doc_attention)(q_docs, k_docs, v_docs, lengths)

    destination = jnp.zeros((LT, H, D), dtype=q.dtype)
    return packed_window_insert(out_docs, destination, starts, lengths)


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
    parser.add_argument("--max-doc-tokens", type=positive_int, default=512)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--index-dtype", choices=("int32", "int64"), default="int32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_doc_tokens = args.max_doc_tokens
    index_dtype = jnp.dtype(args.index_dtype)
    shape = (args.total_tokens, args.heads, args.head_dim)
    q, k, v = [jax.ShapeDtypeStruct(shape, jnp.float16) for _ in range(3)]
    offsets = jax.ShapeDtypeStruct((args.batch + 1,), index_dtype)
    attn_fn = partial(doc_offset_attention, max_doc_len=max_doc_tokens)
    stablehlo_module = jax.jit(attn_fn).lower(q, k, v, offsets)
    print(stablehlo_module.as_text())


if __name__ == "__main__":
    main()
