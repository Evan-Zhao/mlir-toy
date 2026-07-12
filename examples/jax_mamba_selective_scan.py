"""Print a JAX Mamba selective scan program as StableHLO MLIR.

Mamba makes the SSM input-dependent: each token supplies a positive step size
``delta`` and state-space input/output vectors ``b`` and ``c``. For every batch
item and expanded model channel, the selective scan applies the recurrence

    dt_t = softplus(delta_t + delta_bias)
    h_t  = exp(dt_t * A) * h_{t-1} + dt_t * b_t * u_t
    y_t  = sum(c_t * h_t) + D * u_t

where ``A`` is a negative diagonal state-transition parameter. ``delta``, ``b``,
and ``c`` select how the current token is written, retained, and read, while the
scan preserves linear complexity in sequence length. This is the real-valued
Mamba-1 reference discretization: the transition uses the exponential exactly,
and the input term uses ``dt * b * u``.

Only the scan is represented here. A full Mamba block's input projections,
depthwise convolution, generation of delta/b/c, output gate, and output
projection are deliberately omitted. Shapes use batch/sequence/channel order,
which is natural for JAX; the original Mamba kernels commonly use channel-first
``[batch, channels, sequence]`` storage.
"""

import argparse

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def selective_scan(
    u: jax.Array,
    delta: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d_skip: jax.Array,
    delta_bias: jax.Array,
) -> jax.Array:
    """Apply the Mamba selective SSM recurrence.

    Args:
        u: Scan input, ``[batch, sequence, channels]``.
        delta: Token- and channel-dependent step sizes before softplus, with
            the same shape as ``u``.
        a: Negative diagonal continuous-time transitions,
            ``[channels, state_dim]``. Mamba parameterizes this as
            ``a = -exp(a_log)`` outside the scan.
        b: Token-dependent input vectors, ``[batch, sequence, state_dim]``.
        c: Token-dependent output vectors, ``[batch, sequence, state_dim]``.
        d_skip: Per-channel direct feedthrough, ``[channels]``.
        delta_bias: Per-channel step-size bias, ``[channels]``.

    Returns:
        Output ``[batch, sequence, channels]``, with the dtype of ``u``.

    The recurrent state has shape ``[batch, channels, state_dim]``. As in the
    reference Mamba implementation, recurrence arithmetic is float32 even when
    activation storage is bfloat16 or float16.
    """
    if u.shape != delta.shape or u.ndim != 3:
        raise ValueError(
            f"Expected u and delta to have equal rank-3 shapes; got {u.shape}, {delta.shape}"
        )
    batch, sequence, channels = u.shape
    state_dim = a.shape[1]
    if a.shape != (channels, state_dim):
        raise ValueError(f"Expected a.shape[0] == channels ({channels}); got {a.shape}")
    if not (b.shape == c.shape == (batch, sequence, state_dim)):
        raise ValueError(
            "Expected b and c to have shape "
            f"{(batch, sequence, state_dim)}; got {b.shape}, {c.shape}"
        )
    if not (d_skip.shape == delta_bias.shape == (channels,)):
        raise ValueError(
            f"Expected d_skip and delta_bias to have shape {(channels,)}; "
            f"got {d_skip.shape}, {delta_bias.shape}"
        )

    # Index the sequence axis in place rather than transposing it for lax.scan.
    # The fori_loop induction variable is always in [0, sequence_length), so the
    # dynamic slices' specified out-of-bounds clamping is never exercised.
    sequence_length = u.shape[1]
    a_f32 = a.astype(jnp.float32)
    d_skip_f32 = d_skip.astype(jnp.float32)
    delta_bias_f32 = delta_bias.astype(jnp.float32)

    def step(t, carry):
        state, output = carry
        u_t = lax.dynamic_index_in_dim(
            u, t, axis=1, keepdims=False, allow_negative_indices=False
        ).astype(jnp.float32)
        delta_t = lax.dynamic_index_in_dim(
            delta, t, axis=1, keepdims=False, allow_negative_indices=False
        ).astype(jnp.float32)
        b_t = lax.dynamic_index_in_dim(
            b, t, axis=1, keepdims=False, allow_negative_indices=False
        ).astype(jnp.float32)
        c_t = lax.dynamic_index_in_dim(
            c, t, axis=1, keepdims=False, allow_negative_indices=False
        ).astype(jnp.float32)
        dt_t = jax.nn.softplus(delta_t + delta_bias_f32[None, :])

        # transition: [B, C, N]; input update broadcasts b_t across channels.
        transition = jnp.exp(dt_t[..., None] * a_f32[None, :, :])
        state = transition * state + dt_t[..., None] * b_t[:, None, :] * u_t[..., None]
        y_t = jnp.einsum("bcn,bn->bc", state, c_t)
        y_t = y_t + d_skip_f32[None, :] * u_t
        output = lax.dynamic_update_index_in_dim(
            output,
            y_t.astype(u.dtype),
            t,
            axis=1,
            allow_negative_indices=False,
        )
        return state, output

    initial_state = jnp.zeros((batch, channels, state_dim), dtype=jnp.float32)
    initial_output = jnp.zeros_like(u)
    _, output = lax.fori_loop(0, sequence_length, step, (initial_state, initial_output))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    def positive_int(value_str: str) -> int:
        value = int(value_str)
        if value <= 0:
            parser.error(f"Expected positive integer, got {value}")
        return value

    # Mamba-130M-like dimensions: d_model=768, expand=2, d_state=16.
    parser.add_argument("--batch", type=positive_int, default=8)
    parser.add_argument("--sequence-length", type=positive_int, default=2048)
    parser.add_argument("--model-dim", type=positive_int, default=768)
    parser.add_argument("--expand", type=positive_int, default=2)
    parser.add_argument("--state-dim", type=positive_int, default=16)
    parser.add_argument(
        "--activation-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = args.batch
    length = args.sequence_length
    channels = args.expand * args.model_dim
    state_dim = args.state_dim
    activation_dtype = jnp.dtype(args.activation_dtype)

    u = jax.ShapeDtypeStruct((batch, length, channels), activation_dtype)
    delta = jax.ShapeDtypeStruct((batch, length, channels), activation_dtype)
    a = jax.ShapeDtypeStruct((channels, state_dim), jnp.float32)
    b = jax.ShapeDtypeStruct((batch, length, state_dim), activation_dtype)
    c = jax.ShapeDtypeStruct((batch, length, state_dim), activation_dtype)
    d_skip = jax.ShapeDtypeStruct((channels,), jnp.float32)
    delta_bias = jax.ShapeDtypeStruct((channels,), jnp.float32)

    stablehlo_module = selective_scan.lower(u, delta, a, b, c, d_skip, delta_bias)
    print(stablehlo_module.as_text())


if __name__ == "__main__":
    main()
