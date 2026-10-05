"""CPU checks for shared operator inputs and references (no GPU/compiler required)."""

import subprocess
import sys

import pytest

from neptune_mlir.testing import (
    make_attn_inputs,
    make_mamba_inputs,
    reference_attn,
    reference_mamba,
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_testing_imports_are_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import neptune_mlir.testing; "
                "assert not {'torch', 'pytest', 'mlir'} & sys.modules.keys()"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_shared_input_factories(torch, dtype):
    dtype = getattr(torch, dtype)
    before = torch.random.get_rng_state().clone()
    attn_kwargs = {
        "batch": 2,
        "heads": 3,
        "seq_len": 7,
        "head_dim": 4,
        "dtype": dtype,
        "device": "cpu",
        "seed": 19,
    }
    q, k, v = make_attn_inputs(**attn_kwargs)
    assert q.shape == k.shape == v.shape == (2, 3, 7, 4)
    assert q.dtype == k.dtype == v.dtype == dtype
    for actual, repeated, scaled in zip(
        (q, k, v), make_attn_inputs(**attn_kwargs), make_attn_inputs(**attn_kwargs, scale=0.2)
    ):
        assert torch.equal(actual, repeated)
        assert torch.equal(actual * 0.2, scaled)
    mamba_kwargs = {
        "batch": 2,
        "seq_len": 7,
        "channels": 4,
        "state_dim": 3,
        "dtype": dtype,
        "device": "cpu",
        "seed": 19,
    }
    inputs = make_mamba_inputs(**mamba_kwargs)
    assert len(inputs) == 7  # Semantic inputs only, no output or outlining buffers.
    assert [tuple(x.shape) for x in inputs] == [
        (2, 7, 4),
        (2, 7, 4),
        (4, 3),
        (2, 7, 3),
        (2, 7, 3),
        (4,),
        (4,),
    ]
    assert [x.dtype for x in inputs] == [
        dtype,
        dtype,
        torch.float32,
        dtype,
        dtype,
        torch.float32,
        torch.float32,
    ]
    assert bool((inputs[2] <= 0).all())
    for actual, repeated in zip(inputs, make_mamba_inputs(**mamba_kwargs)):
        assert torch.equal(actual, repeated)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("causal", [False, True])
def test_attn_reference_full_and_selected_rows(torch, causal):
    inputs = make_attn_inputs(
        batch=2, heads=2, seq_len=7, head_dim=4, device="cpu", dtype=torch.float32
    )
    q, k, v = inputs
    # Compare against Torch's own attention, not another hand-written softmax baseline.
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    actual = reference_attn(q, k, v, causal=causal, block_rows=3)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    rows = [6, 0, 3, 3, 1]  # Absolute mask positions, arbitrary order, and duplicates.
    subset = reference_attn(q, k, v, causal=causal, rows=rows, block_rows=2)
    torch.testing.assert_close(subset, actual[:, :, rows], rtol=1e-5, atol=1e-6)
    assert reference_attn(q, k, v, rows=[]).shape == (2, 2, 0, 4)
    for original, regenerated in zip(
        inputs,
        make_attn_inputs(
            batch=2,
            heads=2,
            seq_len=7,
            head_dim=4,
            device="cpu",
            dtype=torch.float32,
        ),
    ):
        assert torch.equal(original, regenerated)


def test_attn_reference_is_fp32_for_low_precision_inputs(torch):
    inputs = make_attn_inputs(batch=1, heads=2, seq_len=7, head_dim=4, device="cpu")
    actual = reference_attn(*inputs)
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(x.float() for x in inputs),
        is_causal=True,
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attn_reference_validates_row_config(torch):
    inputs = make_attn_inputs(batch=1, heads=1, seq_len=4, head_dim=4, device="cpu")
    with pytest.raises(ValueError, match="block_rows"):
        reference_attn(*inputs, block_rows=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        reference_attn(*inputs, rows=[[0, 1]])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_mamba_reference_keeps_fp32_state(torch, dtype):
    dtype = getattr(torch, dtype)
    # A=0, delta=bias=0, u=B=C=1: state_t=t*log(2), y_t=3*t*log(2)+D.
    # A long enough scan catches accidental rounding of the recurrent state to BF16/FP16.
    u = torch.ones((2, 64, 4), dtype=dtype)
    delta = torch.zeros_like(u)
    a = torch.zeros((4, 3))
    b = c = torch.ones((2, 64, 3), dtype=dtype)
    d = torch.full((4,), 0.25)
    bias = torch.zeros((4,))
    actual = reference_mamba(u, delta, a, b, c, d, bias)
    expected = torch.arange(1, 65).float() * (3 * torch.tensor(2.0).log()) + 0.25
    expected = expected[None, :, None].expand_as(u).to(dtype)
    assert actual.shape == u.shape
    assert actual.dtype == dtype
    assert actual.device == u.device
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_mamba_reference_is_repeatable_and_does_not_mutate_inputs(torch):
    inputs = make_mamba_inputs(batch=2, seq_len=7, channels=4, state_dim=3, device="cpu")
    originals = tuple(x.clone() for x in inputs)
    actual = reference_mamba(*inputs)
    repeated = reference_mamba(*inputs)
    assert torch.equal(actual, repeated)
    for original, current in zip(originals, inputs):
        assert torch.equal(original, current)
