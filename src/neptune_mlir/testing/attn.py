"""Dense/packed attention inputs, kernel ABI arguments and FP32 Torch references."""

import math
from itertools import accumulate, pairwise

from neptune_mlir.operator.variants import AttentionVariant


def make_attn_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype=None,
    device="cuda",
    seed: int = 0,
    scale: float = 1.0,
):
    """Return Q/K/V in [batch, heads, seq_len, head_dim] order, without an output buffer.

    ``dtype`` is a Torch dtype (default FP16). A local generator preserves the caller's
    global RNG state. Scaling happens in the storage dtype, as in the original tests.
    """
    import torch

    dtype = torch.float16 if dtype is None else dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    return tuple(
        torch.randn(shape, dtype=dtype, device=device, generator=generator) * scale
        for _ in range(3)
    )


def reference_attn(
    q,
    k,
    v,
    *,
    causal: bool = True,
    rows=None,
    block_rows: int = 128,
    window_size: int | None = None,
    alibi_slopes=None,
):
    """Return FP32 attention for all query rows, or selected rows in the supplied order.

    Inputs use [batch, heads, seq_len, head_dim] layout. Causal masking uses absolute
    query/key positions (upper-left alignment). Query chunking bounds score storage;
    selected rows use the same math and preserve their absolute positions in the mask.
    Optional ALiBi slopes use key-minus-query distance; a causal window keeps the
    current query and its preceding window_size - 1 keys. Both use absolute positions
    even for selected rows. The output is not rounded back to activation storage dtype.
    Numerical assertions and backend ABI handling stay with callers.
    """
    import torch

    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    if window_size is not None and (window_size <= 0 or not causal):
        raise ValueError("window_size must be positive and requires causal attention")
    if rows is None:
        row_positions = torch.arange(q.shape[-2], device=q.device)
        selected_q = q
    else:
        row_positions = torch.as_tensor(rows, dtype=torch.long, device=q.device)
        if row_positions.ndim != 1:
            raise ValueError("rows must be a one-dimensional sequence of query positions")
        selected_q = q.index_select(-2, row_positions)
    output = torch.empty(
        (*selected_q.shape[:-1], v.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    k_t = k.transpose(-1, -2).float()
    v_f32 = v.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    key_positions = torch.arange(k.shape[-2], device=q.device)
    for start in range(0, selected_q.shape[-2], block_rows):
        end = min(start + block_rows, selected_q.shape[-2])
        scores = torch.matmul(selected_q[..., start:end, :].float(), k_t) * scale
        query_positions = row_positions[start:end, None]
        if alibi_slopes is not None:
            distance = (key_positions[None, :] - query_positions).float()
            scores = scores + distance * alibi_slopes[..., None, None].float()
        if causal:
            mask = key_positions[None, :] <= query_positions
            if window_size is not None:
                mask = mask & (key_positions[None, :] > query_positions - window_size)
            scores = scores.masked_fill(~mask, float("-inf"))
        output[..., start:end, :] = torch.matmul(torch.softmax(scores, dim=-1), v_f32)
    return output


def make_dense_arguments(options, *, device="cuda", seed=17, scale=1.0):
    import torch

    o = options
    batch, qh = o["batch"], o["q_heads"]
    kh = o.get("kv_heads") or qh
    qs, ks = o["seq_len"], o.get("kv_seq_len") or o["seq_len"]
    q = make_attn_inputs(batch, qh, qs, o["head_dim"], device=device, seed=seed, scale=scale)[0]
    _, k, v = make_attn_inputs(
        batch, kh, ks, o["head_dim"], device=device, seed=seed + 12, scale=scale
    )
    inputs = [q, k, v]
    variant = AttentionVariant(o["variant"])
    if variant == AttentionVariant.ALIBI_CAUSAL_ATTN:
        inputs.append(torch.linspace(0.01, 0.3, qh, device=device))
    elif variant == AttentionVariant.KV_FP8_CAUSAL_ATTN:
        inputs[1:3] = [x.to(torch.float8_e4m3fn) for x in (k, v)]
        shape = (1,) if kh == 1 else (1, kh, 1, 1)
        inputs.extend(
            [
                torch.linspace(0.5, 1.25, kh, device=device).reshape(shape),
                torch.linspace(1.5, 0.75, kh, device=device).reshape(shape),
            ]
        )
    return [*inputs, torch.full_like(q, float("nan"))]


def reference_dense(options, arguments, *, rows=None):
    import torch

    q, k, v = arguments[:3]
    variant = AttentionVariant(options["variant"])
    slopes = arguments[3] if variant == AttentionVariant.ALIBI_CAUSAL_ATTN else None
    if variant == AttentionVariant.KV_FP8_CAUSAL_ATTN:
        # The exporter dequantizes to FP16 before the dot products.
        k, v = [(x.float() * s).half() for x, s in zip((k, v), arguments[3:5])]
    # Exporter GQA groups use query_head % kv_heads, not repeat_interleave.
    indices = torch.arange(q.shape[1], device=q.device) % k.shape[1]
    k, v = [x.index_select(1, indices) for x in (k, v)]
    return reference_attn(
        q,
        k,
        v,
        rows=rows,
        causal=variant != AttentionVariant.GLOBAL_ATTN,
        window_size=(
            options.get("window_size", 128)
            if variant == AttentionVariant.WINDOWED_CAUSAL_ATTN
            else None
        ),
        alibi_slopes=slopes,
    )


def document_lengths(options, lengths=None):
    """Default to a balanced deterministic packing; also accept uneven/empty docs."""
    if lengths is None:
        n, remainder = divmod(options["total_tokens"], options["num_docs"])
        lengths = [n + (i < remainder) for i in range(options["num_docs"])]
    lengths = tuple(lengths)
    if len(lengths) != options["num_docs"] or sum(lengths) != options["total_tokens"]:
        raise ValueError("doc_lengths must match num_docs and total_tokens")
    if any(n < 0 or n > options["max_doc_tokens"] for n in lengths):
        raise ValueError("document lengths must be between zero and max_doc_tokens")
    return lengths


def make_varlen_arguments(options, *, lengths=None, device="cuda", seed=41, scale=1.0):
    import torch

    lengths = document_lengths(options, lengths)
    q, k, v = [
        x.squeeze(0).transpose(0, 1).contiguous()
        for x in make_attn_inputs(
            1,
            options["heads"],
            options["total_tokens"],
            options["head_dim"],
            device=device,
            seed=seed,
            scale=scale,
        )
    ]
    offsets = torch.tensor(
        list(accumulate(lengths, initial=0)),
        dtype=getattr(torch, options["index_dtype"]),
        device=device,
    )
    return [q, k, v, offsets, torch.full_like(q, float("nan"))]


def reference_varlen(arguments):
    import torch

    q, k, v, offsets = arguments[:4]
    reference = torch.empty_like(q, dtype=torch.float32)
    for start, end in pairwise(offsets.tolist()):
        if start == end:
            continue
        doc = [x[start:end].transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        reference[start:end] = reference_attn(*doc, causal=False).squeeze(0).transpose(0, 1)
    return reference
