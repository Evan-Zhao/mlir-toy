"""Print a JAX packed variable-length attention program as StableHLO MLIR.

This variant expresses documents as a parallel batch using ``vmap`` instead of a
sequential ``fori_loop``. It pads the packed token axis and loads fixed-size document windows
with ranged gathers, applies regular masked attention over the dense
``[num_docs, max_doc_tokens, heads, head_dim]`` view,
and scatters valid document results back to packed output.
Runtime document lengths come from ``offsets``;
the maximum document length must be provided statically via ``--max-doc-tokens``.
"""

import argparse
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


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
    token_indices = starts[:, None] + token_offsets[None, :]

    # StableHLO has no masked slicing operation. A ranged load `[start, end)` in StableHLO
    # clamps `start` to make the whole window in bound, which is surprising and not what we intend.
    # Instead, pad the packed token axis first to prevent OOB before we gather.
    # L0 rows (rather than L0 - 1) also cover a zero-length final document whose start equals LT.
    # The padding value is unobservable: key positions beyond each document length are masked,
    # while padded query rows are dropped by the final scatter.
    def masked_load_docs(x):
        padded = jnp.pad(x, ((0, L0), (0, 0), (0, 0)), constant_values=0)
        return lax.gather(
            padded,
            starts[:, None],
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1, 2, 3), collapsed_slice_dims=(), start_index_map=(0,)
            ),
            slice_sizes=(L0, H, D),
            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        )

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
        valid_tokens = token_offsets < doc_len
        key_valid = valid_tokens.reshape(1, 1, L0)
        # Padded query rows are scattered to out-of-bounds sink indices and
        # dropped, so only key positions need to be masked for valid outputs.
        scores_hij = jnp.where(key_valid, scores_hij, -jnp.inf)
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

    q_docs, k_docs, v_docs = [masked_load_docs(x) for x in (q, k, v)]

    # [N, M, H, D]. `vmap` makes the whole per-document attention computation a
    # batched, document-parallel map rather than a loop-carried sequential loop.
    out_docs = jax.vmap(one_doc_attention)(q_docs, k_docs, v_docs, lengths)

    # Scatter valid document tokens back to packed layout. Invalid padded tokens
    # are sent to unique out-of-bounds sink positions so the scatter can be marked
    # unique and the OOB updates can be dropped instead of materializing sinks.
    doc_ids = jnp.arange(N, dtype=offsets.dtype)
    valid_doc_tokens = lengths[:, None] > token_offsets[None, :]
    sink_indices = LT + doc_ids[:, None] * L0 + token_offsets[None, :]
    scatter_indices = jnp.where(valid_doc_tokens, token_indices, sink_indices)
    return lax.scatter(
        jnp.zeros((LT, H, D), dtype=q.dtype),
        scatter_indices[..., None],  # [N, L0, 1]
        out_docs,  # [N, L0, H, D]
        dimension_numbers=lax.ScatterDimensionNumbers(
            update_window_dims=(2, 3),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        ),
        indices_are_sorted=False,
        unique_indices=True,
        mode=lax.GatherScatterMode.FILL_OR_DROP,
    )


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
