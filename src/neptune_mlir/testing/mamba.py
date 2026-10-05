"""Mamba-1 selective-scan inputs and an FP32 recurrent Torch reference."""


def make_mamba_inputs(
    batch: int,
    seq_len: int,
    channels: int,
    state_dim: int,
    dtype=None,
    device="cuda",
    seed: int = 0,
    scale: float = 0.2,
):
    """Return the seven operator inputs: u, delta, A, B, C, D, delta_bias.

    Activations use ``dtype`` (default BF16); A, D and delta_bias are FP32. A is
    nonpositive for stable recurrence. ``scale`` applies to normal draws, not A.
    A local generator preserves global RNG state. ABI buffers are the caller's job.
    """
    import torch

    dtype = torch.bfloat16 if dtype is None else dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dt=dtype):
        return torch.randn(shape, dtype=dt, device=device, generator=generator) * scale

    shape = (batch, seq_len, channels)
    state_shape = (channels, state_dim)
    bc_shape = (batch, seq_len, state_dim)
    return (
        randn(shape),
        randn(shape),
        -torch.rand(state_shape, dtype=torch.float32, device=device, generator=generator),
        randn(bc_shape),
        randn(bc_shape),
        randn((channels,), torch.float32),
        randn((channels,), torch.float32),
    )


def reference_mamba(u, delta, a, b, c, d, delta_bias):
    """Return selective scan from zero state, with FP32 recurrence and u's output dtype.

    u/delta use [batch, seq_len, channels], A uses [channels, state_dim], B/C use
    [batch, seq_len, state_dim], and D/delta_bias use [channels]. State remains FP32;
    each output token is rounded to the activation storage dtype. No kernel ABI
    scratch buffers, initial-state interface, or validation policy are involved.
    """
    import torch

    state = torch.zeros((u.shape[0], u.shape[2], a.shape[1]), dtype=torch.float32, device=u.device)
    output = torch.empty_like(u)
    for t in range(u.shape[1]):
        u_t = u[:, t].float()
        dt = torch.nn.functional.softplus(delta[:, t].float() + delta_bias[None, :])
        state = (
            torch.exp(dt[..., None] * a[None, :, :]) * state
            + dt[..., None] * b[:, t].float()[:, None, :] * u_t[..., None]
        )
        y_t = (state * c[:, t].float()[:, None, :]).sum(dim=2)
        output[:, t] = (y_t + d[None, :] * u_t).to(output.dtype)
    return output
