#!/usr/bin/env python3
"""Golden tests for HTile Python translators."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "neptune"))

from translate_to_tilelang import translate_file as translate_tilelang  # noqa: E402
from translate_common import translate_file_with  # noqa: E402
from translate_to_cutile import translate_file as translate_cutile  # noqa: E402
from translate_to_triton import translate_file as translate_triton  # noqa: E402
from translate_to_triton import Translator as TritonTranslator  # noqa: E402

PLUGIN = ROOT / "build" / "libHTileDialectPlugin.dylib"
GOLDEN_DIR = ROOT / "tests" / "golden"
MLIR_FILE = ROOT / "tests" / "flash_attention_htile.mlir"
DOT_TRANSPOSE_PASS_PIPELINE = (
    "builtin.module(htile-dot-transpose-to-load-order,cse,canonicalize)"
)


def require_translator_deps():
    if shutil.which("mlir-opt") is None:
        pytest.skip("mlir-opt is required for translator tests")
    if not PLUGIN.exists():
        pytest.skip(f"HTile dialect plugin is missing: {PLUGIN}")


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
        "htile.dot %1, %10 {warp_policy = \"full_row\"} : "
        "tensor<128x128xf16, #htile.encoding<placement = shared>>, "
        "tensor<128x64xf16, #htile.encoding<placement = shared>>"
    ) in result.stdout


def test_triton_rejects_unfissioned_dot_transpose():
    require_translator_deps()
    with pytest.raises(NotImplementedError, match="htile-dot-transpose-to-load-order"):
        translate_file_with(str(MLIR_FILE), TritonTranslator, str(PLUGIN))
