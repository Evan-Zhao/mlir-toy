#!/usr/bin/env python3
"""Golden tests for HTile Python translators."""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "neptune"))

from translate_to_tilelang import translate_file as translate_tilelang  # noqa: E402
from translate_to_triton import translate_file as translate_triton  # noqa: E402

PLUGIN = ROOT / "build" / "libHTileDialectPlugin.dylib"
GOLDEN_DIR = ROOT / "tests" / "golden"
MLIR_FILE = ROOT / "tests" / "flash_attention_htile.mlir"


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
