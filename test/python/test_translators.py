#!/usr/bin/env python3
import ast
import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from neptune_mlir.plugin import find_plugin_path  # noqa: E402
from neptune_mlir.translators.common import translate_file_with  # noqa: E402
from neptune_mlir.translators.cutile import (
    translate_file as translate_cutile,  # noqa: E402
)
from neptune_mlir.translators.tilelang import (  # noqa: E402
    translate_file as translate_tilelang,
)
from neptune_mlir.translators.triton import Translator as TritonTranslator  # noqa: E402
from neptune_mlir.translators.triton import (
    translate_file as translate_triton,  # noqa: E402
)

PARENT_DIR = Path(__file__).resolve().parent
PLUGIN = find_plugin_path()
if PLUGIN is None:
    pytest.exit("MLIR plugin path not found")
GOLDEN_DIR = PARENT_DIR / "golden"
MLIR_FILE = PARENT_DIR / "data" / "flash_attention_htile.mlir"
DOT_TRANSPOSE_PASS_PIPELINE = (
    "builtin.module(htile-dot-transpose-to-load-order,cse,canonicalize)"
)
FLASH_GRID = (32, 32, 1)
FLASH_SHAPE = (1, 32, 4096, 128)
FLASH_REF_BLOCK_ROWS = 128


def require_translator_deps():
    if shutil.which("mlir-opt") is None:
        pytest.skip("mlir-opt is required for translator tests")
    if PLUGIN is None or not PLUGIN.exists():
        pytest.skip(f"HTile dialect plugin is missing: {PLUGIN}")


def require_cuda_torch():
    if not _module_available("torch"):
        pytest.skip("torch is required for functional translator tests")
    import torch

    if not torch.cuda.is_available():
        pytest.skip(
            "A working CUDA runtime is required for functional translator tests"
        )
    try:
        torch.cuda.init()
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        pytest.skip(f"CUDA initialization failed: {exc}")
    return torch


def require_tilelang_runtime():
    if not _module_available("tilelang"):
        pytest.skip("tilelang is required for TileLang functional translator tests")
    import tilelang

    return tilelang


def require_cutile_runtime():
    if not _module_available("cuda.tile"):
        pytest.skip("cuda.tile is required for cuTile functional translator tests")
    import cuda.tile as ct

    return ct


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _exec_translated_module(module_ast: ast.Module, module_name: str):
    source = ast.unparse(module_ast) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"{module_name}_",
        delete=False,
    ) as f:
        f.write(source)
        module_path = Path(f.name)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load translated module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_attention_inputs(torch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    q = torch.randn(
        FLASH_SHAPE, dtype=torch.float16, device="cuda", generator=generator
    )
    k = torch.randn(
        FLASH_SHAPE, dtype=torch.float16, device="cuda", generator=generator
    )
    v = torch.randn(
        FLASH_SHAPE, dtype=torch.float16, device="cuda", generator=generator
    )
    out = torch.empty(FLASH_SHAPE, dtype=torch.float16, device="cuda")
    return q, k, v, out


def _reference_attention(torch, q, k, v):
    ref = torch.empty(FLASH_SHAPE, dtype=torch.float32, device=q.device)
    k_t = k.transpose(-1, -2).float()
    v_f = v.float()
    scale = 1.0 / math.sqrt(q.shape[-1])

    for row_start in range(0, q.shape[2], FLASH_REF_BLOCK_ROWS):
        row_end = min(row_start + FLASH_REF_BLOCK_ROWS, q.shape[2])
        q_block = q[:, :, row_start:row_end, :].float()
        scores = torch.matmul(q_block, k_t)
        probs = torch.softmax(scores * scale, dim=-1)
        ref[:, :, row_start:row_end, :] = torch.matmul(probs, v_f)

    return ref


def _assert_attention_output_close(torch, out, q, k, v):
    assert tuple(out.shape) == FLASH_SHAPE
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all())
    ref = _reference_attention(torch, q, k, v)
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=1e-3)


def _launch_cutile_kernel(ct, torch, kernel, args):
    stream = torch.cuda.current_stream()
    try:
        ct.launch(stream, FLASH_GRID, kernel, args)
    except TypeError:
        ct.launch(stream, FLASH_GRID, kernel, *args)


def assert_matches_golden(actual_module: ast.Module, golden_name: str):
    actual = ast.unparse(actual_module) + "\n"
    expected = (GOLDEN_DIR / golden_name).read_text()
    assert actual == expected


def test_triton_translator_matches_golden():
    require_translator_deps()
    assert_matches_golden(
        translate_triton(str(MLIR_FILE), str(PLUGIN)),
        "flash_attention_triton.py",
    )


def test_tilelang_translator_matches_golden():
    require_translator_deps()
    assert_matches_golden(
        translate_tilelang(str(MLIR_FILE), str(PLUGIN)),
        "flash_attention_tilelang.py",
    )


def test_cutile_translator_matches_golden():
    require_translator_deps()
    assert_matches_golden(
        translate_cutile(str(MLIR_FILE), str(PLUGIN)),
        "flash_attention_cutile.py",
    )


def test_triton_translator_functional():
    require_translator_deps()
    torch = require_cuda_torch()
    triton_module = _exec_translated_module(
        translate_triton(str(MLIR_FILE), str(PLUGIN)),
        "translated_flash_attention_triton",
    )
    q, k, v, out = _make_attention_inputs(torch)
    triton_module.flash_attention_htile[FLASH_GRID](q, k, v, out)
    torch.cuda.synchronize()
    _assert_attention_output_close(torch, out, q, k, v)


def test_tilelang_translator_functional():
    require_translator_deps()
    torch = require_cuda_torch()
    tilelang = require_tilelang_runtime()
    tilelang_module = _exec_translated_module(
        translate_tilelang(str(MLIR_FILE), str(PLUGIN)),
        "translated_flash_attention_tilelang",
    )
    kernel = tilelang.compile(
        tilelang_module.flash_attention_htile,
        out_idx=[3],
        execution_backend="tvm_ffi",
        target="cuda",
    )
    q, k, v, _ = _make_attention_inputs(torch)
    out = kernel(q, k, v)
    if isinstance(out, (list, tuple)):
        out = out[0]
    torch.cuda.synchronize()
    _assert_attention_output_close(torch, out, q, k, v)


def test_cutile_translator_functional():
    require_translator_deps()
    torch = require_cuda_torch()
    ct = require_cutile_runtime()
    cutile_module = _exec_translated_module(
        translate_cutile(str(MLIR_FILE), str(PLUGIN)),
        "translated_flash_attention_cutile",
    )
    q, k, v, out = _make_attention_inputs(torch)
    _launch_cutile_kernel(
        ct, torch, cutile_module.flash_attention_htile, (q, k, v, out)
    )
    torch.cuda.synchronize()
    _assert_attention_output_close(torch, out, q, k, v)


def test_dot_transpose_to_load_order_pass():
    require_translator_deps()
    result = subprocess.run(
        [
            "mlir-opt",
            f"--load-dialect-plugin={PLUGIN}",
            f"--load-pass-plugin={PLUGIN}",
            f"--pass-pipeline={DOT_TRANSPOSE_PASS_PIPELINE}",
            str(MLIR_FILE),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "transpose_b" not in result.stdout
    assert "htile.permute" not in result.stdout
    assert "dimension_order = array<i64: 1, 0>" in result.stdout
    assert (
        'htile.dot %1, %10 {warp_policy = "full_row"} : '
        "tensor<128x128xf16, #htile.encoding<placement = shared>>, "
        "tensor<128x64xf16, #htile.encoding<placement = shared>>"
    ) in result.stdout


def test_triton_rejects_unfissioned_dot_transpose():
    require_translator_deps()
    with pytest.raises(NotImplementedError, match="htile-dot-transpose-to-load-order"):
        translate_file_with(str(MLIR_FILE), TritonTranslator, str(PLUGIN))
