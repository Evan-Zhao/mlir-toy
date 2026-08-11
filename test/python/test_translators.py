import ast
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import pytest

from neptune_mlir.translators.cutile import translate_mlir_text as translate_cutile
from neptune_mlir.translators.tilelang import translate_mlir_text as translate_tilelang
from neptune_mlir.translators.triton import translate_mlir_text as translate_triton

DATA_DIR = Path(__file__).resolve().parent / "data"
CAUSAL_HTILE_INPUT = DATA_DIR / "causal_attention_htile.mlir"
CAUSAL_TRITON_EXPECTED = DATA_DIR / "causal_attention_triton.py"
CAUSAL_CUTILE_EXPECTED = DATA_DIR / "causal_attention_cutile.py"
CAUSAL_TILELANG_EXPECTED = DATA_DIR / "causal_attention_tilelang.py"
CAUSAL_GRID = (4, 8)
CAUSAL_SHAPE = (1, 4, 1024, 64)
CAUSAL_BLOCK_ROWS = 128


def _assert_matches_python_source(actual: ast.Module, expected_path: Path) -> None:
    expected = ast.parse(expected_path.read_text())
    assert ast.unparse(actual) == ast.unparse(expected)


def test_triton_translator_matches_causal_attention():
    actual = translate_triton(CAUSAL_HTILE_INPUT.read_text())
    _assert_matches_python_source(actual, CAUSAL_TRITON_EXPECTED)


def test_cutile_translator_matches_causal_attention():
    actual = translate_cutile(CAUSAL_HTILE_INPUT.read_text())
    _assert_matches_python_source(actual, CAUSAL_CUTILE_EXPECTED)


def test_tilelang_translator_matches_causal_attention():
    actual = translate_tilelang(CAUSAL_HTILE_INPUT.read_text())
    _assert_matches_python_source(actual, CAUSAL_TILELANG_EXPECTED)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _require_cuda_torch():
    if not _module_available("torch"):
        pytest.skip("torch is required for functional translator tests")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("A working CUDA runtime is required for functional translator tests")
    try:
        torch.cuda.init()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA initialization failed: {exc}")
    return torch


def _exec_translated_module(module_ast: ast.Module, module_name: str):
    source = ast.unparse(module_ast) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"{module_name}_", delete=False
    ) as file:
        file.write(source)
        module_path = Path(file.name)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load translated module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_causal_attention_inputs(torch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    q = torch.randn(CAUSAL_SHAPE, dtype=torch.float16, device="cuda", generator=generator)
    k = torch.randn(CAUSAL_SHAPE, dtype=torch.float16, device="cuda", generator=generator)
    v = torch.randn(CAUSAL_SHAPE, dtype=torch.float16, device="cuda", generator=generator)
    out = torch.empty(CAUSAL_SHAPE, dtype=torch.float16, device="cuda")
    return q, k, v, out


def _reference_causal_attention(torch, q, k, v):
    reference = torch.empty(CAUSAL_SHAPE, dtype=torch.float32, device=q.device)
    k_t = k.transpose(-1, -2).float()
    v_f32 = v.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    key_positions = torch.arange(q.shape[2], device=q.device)

    for row_start in range(0, q.shape[2], CAUSAL_BLOCK_ROWS):
        row_end = min(row_start + CAUSAL_BLOCK_ROWS, q.shape[2])
        q_block = q[:, :, row_start:row_end, :].float()
        scores = torch.matmul(q_block, k_t) * scale
        row_positions = torch.arange(row_start, row_end, device=q.device)
        causal_mask = key_positions[None, :] <= row_positions[:, None]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        reference[:, :, row_start:row_end, :] = torch.matmul(probabilities, v_f32)

    return reference


def _assert_causal_attention_output(torch, out, q, k, v):
    assert tuple(out.shape) == CAUSAL_SHAPE
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all())
    reference = _reference_causal_attention(torch, q, k, v)
    torch.testing.assert_close(out.float(), reference, rtol=0, atol=2e-2)


def test_triton_translator_functional():
    torch = _require_cuda_torch()
    if not _module_available("triton"):
        pytest.skip("Triton is required for its functional translator test")
    module = _exec_translated_module(
        translate_triton(CAUSAL_HTILE_INPUT.read_text()), "translated_causal_attention_triton"
    )
    q, k, v, out = _make_causal_attention_inputs(torch)
    module.attention_kernel[CAUSAL_GRID](q, k, v, out)
    torch.cuda.synchronize()
    _assert_causal_attention_output(torch, out, q, k, v)


def test_cutile_translator_functional():
    torch = _require_cuda_torch()
    if not _module_available("cuda.tile"):
        pytest.skip("cuda.tile is required for its functional translator test")
    import cuda.tile as ct  # type: ignore

    module = _exec_translated_module(
        translate_cutile(CAUSAL_HTILE_INPUT.read_text()), "translated_causal_attention_cutile"
    )
    q, k, v, out = _make_causal_attention_inputs(torch)
    stream = torch.cuda.current_stream()
    grid = (*CAUSAL_GRID, 1)
    try:
        ct.launch(stream, grid, module.attention_kernel, (q, k, v, out))
    except TypeError:
        ct.launch(stream, grid, module.attention_kernel, q, k, v, out)
    torch.cuda.synchronize()
    _assert_causal_attention_output(torch, out, q, k, v)


def test_tilelang_translator_functional():
    torch = _require_cuda_torch()
    if not _module_available("tilelang"):
        pytest.skip("TileLang is required for its functional translator test")
    import tilelang

    module = _exec_translated_module(
        translate_tilelang(CAUSAL_HTILE_INPUT.read_text()),
        "translated_causal_attention_tilelang",
    )
    kernel = tilelang.compile(
        module.attention_kernel,
        out_idx=[3],
        execution_backend="tvm_ffi",
        target="cuda",
    )
    q, k, v, _ = _make_causal_attention_inputs(torch)
    out = kernel(q, k, v)
    if isinstance(out, (list, tuple)):
        out = out[0]
    torch.cuda.synchronize()
    _assert_causal_attention_output(torch, out, q, k, v)


def test_tilelang_scalar_memref_uses_one_element_buffer():
    source = """
    module {
      htile.kernel @scalar_load(%src: memref<f32>) attributes {program_bounds = array<i64: 1>} {
        %value = htile.load %src[] : memref<f32> -> tensor<f32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_tilelang(source))
    assert "buf_0: T.Tensor((1,), 'float32')" in translated
    assert "scalar_2 = buf_0[0]" in translated


def test_triton_translates_unsqueeze_and_squeeze():
    source = """
    module {
      htile.kernel @unit_dims() attributes {program_bounds = array<i64: 1>} {
        %cst = arith.constant 0.0 : f32
        %tile = htile.full %cst : f32 -> tensor<4x8xf32>
        %expanded = htile.unsqueeze %tile mask [false, true, false]
            : tensor<4x8xf32> -> tensor<4x1x8xf32>
        %collapsed = htile.squeeze %expanded mask [false, true, false]
            : tensor<4x1x8xf32> -> tensor<4x8xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_triton(source))
    assert "tile_2 = tile_1[:, None, :]" in translated
    assert translated.count("tl.reshape") == 1


def test_cutile_translates_unsqueeze_and_squeeze():
    source = """
    module {
      htile.kernel @unit_dims() {
        %cst = arith.constant 0.0 : f32
        %tile = htile.full %cst : f32 -> tensor<4x8xf32>
        %expanded = htile.unsqueeze %tile mask [false, true, false]
            : tensor<4x8xf32> -> tensor<4x1x8xf32>
        %collapsed = htile.squeeze %expanded mask [false, true, false]
            : tensor<4x1x8xf32> -> tensor<4x8xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_cutile(source))
    assert translated.count("ct.expand_dims") == 1
    assert translated.count("ct.reshape") == 1


def test_triton_masked_memory_uses_tensor_pointers():
    source = """
    module {
      htile.kernel @masked_memory(
          %src: memref<16x8xf32>, %dst: memref<16x8xf32>) {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %other = arith.constant 0.0 : f32
        %range = htile.arange %c0 to %c8 : tensor<8xindex>
        %indices = htile.broadcast %range dimensions = [0]
            : tensor<8xindex> -> tensor<4x8xindex>
        %zero = htile.full %c0 : index -> tensor<4x8xindex>
        %mask = arith.cmpi sge, %indices, %zero : tensor<4x8xindex>
        %value = htile.load %src[%c0, %c0]
            mask(%mask : tensor<4x8xi1>) other(%other : f32)
            : memref<16x8xf32> -> tensor<4x8xf32>
        htile.store %value, %dst[%c0, %c0] mask(%mask : tensor<4x8xi1>)
            : tensor<4x8xf32>, memref<16x8xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_triton(source))
    assert "tl.make_block_ptr" not in translated
    assert "tl.load(" in translated
    assert "other=" in translated
    assert translated.count("mask=") == 2
    assert "tl.store(" in translated


def test_cutile_masked_memory_uses_raw_offsets():
    source = """
    module {
      htile.kernel @masked_memory(
          %src: memref<16x8xf32>, %dst: memref<16x8xf32>) {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %other = arith.constant 0.0 : f32
        %range = htile.arange %c0 to %c8 : tensor<8xindex>
        %indices = htile.broadcast %range dimensions = [0]
            : tensor<8xindex> -> tensor<4x8xindex>
        %zero = htile.full %c0 : index -> tensor<4x8xindex>
        %mask = arith.cmpi sge, %indices, %zero : tensor<4x8xindex>
        %value = htile.load %src[%c0, %c0]
            mask(%mask : tensor<4x8xi1>) other(%other : f32)
            : memref<16x8xf32> -> tensor<4x8xf32>
        htile.store %value, %dst[%c0, %c0] mask(%mask : tensor<4x8xi1>)
            : tensor<4x8xf32>, memref<16x8xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_cutile(source))
    assert translated.count("get_raw_memory()") == 2
    assert ".load_offset(" in translated and "padding_value=" in translated
    assert ".store_offset(" in translated
    assert translated.count("mask=") == 2


def test_triton_translates_scalar_load_and_index_cast():
    source = """
    module {
      htile.kernel @scalar_metadata(%src: memref<4xi32>) {
        %c2 = arith.constant 2 : index
        %value = htile.load %src[%c2] : memref<4xi32> -> i32
        %index = arith.index_cast %value : i32 to index
        %next = arith.addi %index, %c2 : index
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_triton(source))
    assert translated.count("tl.load(") == 1
    assert "tl.cast" not in translated


def test_triton_scalar_memref_access_uses_pointer_arithmetic():
    source = """
    module {
      htile.kernel @scalar_access(%src: memref<4xf32>, %dst: memref<4xf32>) {
        %c2 = arith.constant 2 : index
        %value = htile.load %src[%c2] : memref<4xf32> -> tensor<f32>
        htile.store %value, %dst[%c2] : tensor<f32>, memref<4xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_triton(source))
    assert "tl.make_block_ptr" not in translated
    assert translated.count("tl.load(") == 1
    assert translated.count("tl.store(") == 1
    assert "shape=[]" not in translated
