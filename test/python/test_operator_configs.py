"""Export/benchmark option parity and shared runtime inputs, without a GPU."""

import json

import pytest

from neptune_mlir.benchmarks.cases import Case
from neptune_mlir.cli.bench import parse_args as bench_args
from neptune_mlir.cli.export import parse_args as export_args
from neptune_mlir.operator.variants import VARIANTS
from neptune_mlir.testing import (
    document_lengths,
    make_arguments,
    reference_output,
)


@pytest.mark.parametrize(
    "argv",
    [
        [
            "dense",
            "--variant",
            str(variant),
            "--q-heads",
            "4",
            "--kv-heads",
            "2",
            "--seq-len",
            "256",
            "--kv-seq-len",
            "128",
            "--window-size",
            "73",
        ]
        for variant in VARIANTS
    ]
    + [
        ["varlen", "--num-docs", "4", "--total-tokens", "512", "--index-dtype", "int64"],
        ["mamba", "--model-dim", "256", "--expand", "2", "--activation-dtype", "float32"],
    ],
)
def test_export_benchmark_share_options(argv):
    exported = vars(export_args(argv))
    exported.pop("stage")
    case = bench_args([*argv, "--profiler", "cudaevent"]).cases[0]
    config = case.config()
    config.pop("doc_lengths", None)
    assert config == exported
    assert Case(**json.loads(json.dumps(case.config()))).config() == case.config()


@pytest.mark.parametrize("operator", ["dense", "varlen", "mamba"])
def test_lower_uses_export_dispatch(monkeypatch, operator):
    from neptune_mlir.cli import export

    calls = []
    monkeypatch.setattr(export, "export_at_stage", lambda args: calls.append(vars(args)) or "htile")
    case = Case(operator, func_name="custom_name")
    assert case.lower() == "htile"
    assert calls == [{"operator": operator, "stage": "htile", **case.options}]


def test_aliases_and_case_identity():
    a = bench_args(["attn", "--heads", "8", "--profiler", "cudaevent"]).cases[0]
    b = bench_args(["dense", "--q-heads", "8", "--profiler", "cudaevent"]).cases[0]
    assert a.config() == b.config()
    assert a.name == b.name
    assert a.name != Case("dense", q_heads=8, block_m=64).name
    with pytest.raises(ValueError, match="unknown dense options"):
        Case("dense", typo=1)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_dense_inputs_and_sampled_reference(variant, kv_heads):
    torch = pytest.importorskip("torch")
    case = Case(
        "dense",
        variant=variant,
        batch=2,
        q_heads=4,
        kv_heads=kv_heads,
        seq_len=32,
        kv_seq_len=48,
        head_dim=16,
        window_size=17,
    )
    # Tiny CPU shapes exercise reference semantics, not GPU schedule divisibility.
    arguments = make_arguments("dense", case.options, device="cpu", seed=17)
    assert arguments[0].shape == (2, 4, 32, 16)
    assert arguments[1].shape == (2, kv_heads, 48, 16)
    assert torch.isnan(arguments[-1]).all()
    count = 6 if str(variant) == "kv-fp8-causal" else 5 if str(variant) == "alibi-causal" else 4
    assert len(arguments) == count
    if str(variant) == "kv-fp8-causal":
        assert arguments[1].dtype == torch.float8_e4m3fn
        assert arguments[3].shape == ((1,) if kv_heads == 1 else (1, kv_heads, 1, 1))
    full = reference_output("dense", case.options, arguments)
    sampled = reference_output("dense", case.options, arguments, rows=[0, 15, 31])
    torch.testing.assert_close(sampled, full[:, :, [0, 15, 31]])
    assert torch.isfinite(full).all()


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_varlen_runtime_packing(index_dtype):
    torch = pytest.importorskip("torch")
    lengths = (0, 1, 7, 0, 3)
    case = Case(
        "varlen",
        num_docs=5,
        total_tokens=11,
        max_doc_tokens=8,
        heads=2,
        head_dim=16,
        index_dtype=index_dtype,
        doc_lengths=lengths,
    )
    case.validate()
    arguments = make_arguments("varlen", case.options, device="cpu", lengths=lengths)
    assert arguments[3].tolist() == [0, 0, 1, 8, 8, 11]
    assert arguments[3].dtype == getattr(torch, index_dtype)
    ref = reference_output("varlen", case.options, arguments)
    assert torch.isfinite(ref).all()
    # A singleton document's attention is exactly V, with no cross-document leakage.
    torch.testing.assert_close(ref[0], arguments[2][0].float())
    arguments[-1].copy_(ref)
    assert case.check(torch, arguments)["scope"] == "full"
    assert Case(**json.loads(json.dumps(case.config()))).config() == case.config()


def test_default_packing_and_invalid_lengths():
    options = Case("varlen", num_docs=4, total_tokens=7).options
    assert document_lengths(options) == (2, 2, 2, 1)
    for lengths in [(7,), (0, 0, 0, 6), (-1, 2, 3, 3)]:
        with pytest.raises(ValueError):
            document_lengths(options, lengths)
    with pytest.raises(ValueError):
        document_lengths({**options, "max_doc_tokens": 1})


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_mamba_runtime_abi(dtype):
    torch = pytest.importorskip("torch")
    case = Case(
        "mamba",
        batch=2,
        sequence_length=3,
        model_dim=32,
        expand=2,
        state_dim=8,
        block_channels=32,
        activation_dtype=dtype,
    )
    case.validate()
    args = make_arguments("mamba", case.options, device="cpu")
    assert len(args) == 10
    assert args[0].shape == (2, 3, 64)
    assert args[7].shape == (2, 64, 8)
    assert args[8].shape == (1,)
    args[-1].copy_(reference_output("mamba", case.options, args))
    assert case.check(torch, args)["max_abs_error"] == 0
