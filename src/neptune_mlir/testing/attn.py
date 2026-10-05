"""Dense attention input generation and an FP32 Torch correctness reference."""

import math


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
