import ast
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

from neptune_mlir.testing import make_attn_inputs, reference_attn
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
    batch, heads, seq_len, head_dim = CAUSAL_SHAPE
    inputs = make_attn_inputs(
        batch, heads, seq_len, head_dim, torch.float16, "cuda", seed=0, scale=1.0
    )
    return (*inputs, torch.empty_like(inputs[0]))


def _assert_causal_attention_output(torch, out, q, k, v):
    assert tuple(out.shape) == CAUSAL_SHAPE
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all())
    reference = reference_attn(q, k, v, block_rows=CAUSAL_BLOCK_ROWS)
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


def test_cutile_scalar_memref_uses_one_element_buffer():
    source = """
    module {
      htile.kernel @scalar_load(%src: memref<f32>) attributes {program_bounds = array<i64: 1>} {
        %value = htile.load %src[] : memref<f32> -> tensor<f32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_cutile(source))
    assert "ct.load(arr_0, (0,), (1,), order=(0,))" in translated
    assert ".item()" in translated


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


def test_tilelang_hoisted_load_and_converted_dot_rhs():
    source = """
    module {
      htile.kernel @converted_rhs(%q: memref<32x32xf16>, %k: memref<32x32xf8E4M3FN>)
          attributes {program_bounds = array<i64: 1>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %q_tile = htile.load %q[%c0, %c0] : memref<32x32xf16> -> tensor<32x32xf16>
        scf.for %i = %c0 to %c1 step %c1 {
          %k_tile = htile.load %k[%c0, %c0] : memref<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
          %k_f16 = arith.extf %k_tile : tensor<32x32xf8E4M3FN> to tensor<32x32xf16>
          %k_transposed = htile.permute %k_f16 permutation [1, 0]
              : tensor<32x32xf16> -> tensor<32x32xf16>
          %dot = htile.dot %q_tile, %k_transposed
              : tensor<32x32xf16>, tensor<32x32xf16> -> tensor<32x32xf32>
        }
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_tilelang(source))
    # Only the hoisted Q load must opt out of TMA; inner K loads still use it.
    copies = [line for line in translated.splitlines() if "T.copy(" in line]
    assert len(copies) == 2
    assert "buf_0[" in copies[0] and "disable_tma=True" in copies[0]
    assert "buf_1[" in copies[1] and "disable_tma" not in copies[1]
    assert translated.count("T.alloc_shared([32, 32], 'float16')") == 2
    assert "T.alloc_fragment([32, 32], 'float32')" in translated


def test_cutile_normalizes_dynamic_loop_bounds():
    source = """
    module {
      htile.kernel @offset_loop(%offsets: memref<2xi64>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %length = htile.load %offsets[%c1] : memref<2xi64> -> i64
        %bound = arith.index_cast %length : i64 to index
        scf.for %i = %c0 to %bound step %c1 {
        }
        htile.return
      }
    }
    """
    module = translate_cutile(source)
    loop = next(node for node in ast.walk(module) if isinstance(node, ast.For))
    assert all(
        ast.unparse(arg).startswith("ct.astype(") and ast.unparse(arg).endswith(", ct.int32)")
        for arg in loop.iter.args
    )


def test_tilelang_translates_varlen_primitives():
    source = """
    module {
      htile.kernel @varlen_primitives(
          %src: memref<16x8xf32>, %offsets: memref<2xi32>,
          %dst: memref<16x8xf32>) attributes {program_bounds = array<i64: 1>} {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %other = arith.constant 0.0 : f32
        %offset_i32 = htile.load %offsets[%c0] : memref<2xi32> -> i32
        %offset = arith.index_cast %offset_i32 : i32 to index
        %has_offset = arith.cmpi sgt, %offset, %c0 : index
        %start = arith.select %has_offset, %offset, %c0 : index
        %range = htile.arange %c0 to %c8 : tensor<8xindex>
        %indices = htile.broadcast %range dimensions = [0]
            : tensor<8xindex> -> tensor<4x8xindex>
        %zero = htile.full %c0 : index -> tensor<4x8xindex>
        %mask = arith.cmpi sge, %indices, %zero : tensor<4x8xindex>
        %expanded_mask = htile.unsqueeze %mask mask [false, true, false]
            : tensor<4x8xi1> -> tensor<4x1x8xi1>
        %value = htile.load %src[%start, %c0]
            mask(%mask : tensor<4x8xi1>) other(%other : f32)
            : memref<16x8xf32> -> tensor<4x8xf32>
        %expanded = htile.unsqueeze %value mask [false, true, false]
            : tensor<4x8xf32> -> tensor<4x1x8xf32>
        %collapsed = htile.squeeze %expanded mask [false, true, false]
            : tensor<4x1x8xf32> -> tensor<4x8xf32>
        htile.store %collapsed, %dst[%start, %c0] mask(%mask : tensor<4x8xi1>)
            : tensor<4x8xf32>, memref<16x8xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_tilelang(source))
    assert "scalar_" in translated and "buf_1[c_" in translated
    assert "T.cast(" in translated
    assert translated.count("T.reshape") == 3
    assert "= T.if_then_else(" in translated
    assert "if frag_" in translated


@pytest.mark.parametrize(
    ("translator", "expected"),
    [
        (
            translate_triton,
            (
                "from triton.language.extra import libdevice",
                "tl.abs(",
                "tl.exp(",
                "libdevice.log1p(",
                "tl.bfloat16",
                " != ",
            ),
        ),
        (
            translate_cutile,
            ("ct.abs(", "ct.exp(", "ct.log(", " + 1.0", "ct.bfloat16", " != "),
        ),
        (
            translate_tilelang,
            ("T.abs(", "T.exp(", "T.log1p(", "'bfloat16'", " != "),
        ),
    ],
    ids=("triton", "cutile", "tilelang"),
)
def test_translators_support_selective_scan_math(translator, expected):
    source = """
    module {
      htile.kernel @softplus(%src: memref<8xbf16>, %dst: memref<8xbf16>)
          attributes {program_bounds = array<i64: 1>} {
        %c0 = arith.constant 0 : index
        %input = htile.load %src[%c0] : memref<8xbf16> -> tensor<8xbf16>
        %x = arith.extf %input : tensor<8xbf16> to tensor<8xf32>
        %abs = math.absf %x : tensor<8xf32>
        %neg = arith.negf %abs : tensor<8xf32>
        %exp = math.exp %neg : tensor<8xf32>
        %log = math.log1p %exp : tensor<8xf32>
        %nan = arith.cmpf une, %x, %x : tensor<8xf32>
        %selected = arith.select %nan, %x, %log : tensor<8xi1>, tensor<8xf32>
        %output = arith.truncf %selected : tensor<8xf32> to tensor<8xbf16>
        htile.store %output, %dst[%c0] : tensor<8xbf16>, memref<8xbf16>
        htile.return
      }
    }
    """
    translated = ast.unparse(translator(source))
    for fragment in expected:
        assert fragment in translated
    assert " = -" in translated


def test_triton_translates_vector_dots_as_reductions():
    source = """
    module {
      htile.kernel @vector_dots() attributes {program_bounds = array<i64: 1>} {
        %zero = arith.constant 0.0 : f32
        %matrix_lhs = htile.full %zero : f32 -> tensor<4x8xf32>
        %vector_rhs = htile.full %zero : f32 -> tensor<8xf32>
        %matvec = htile.dot %matrix_lhs, %vector_rhs
            : tensor<4x8xf32>, tensor<8xf32> -> tensor<4xf32>
        %vector_lhs = htile.full %zero : f32 -> tensor<8xf32>
        %matrix_rhs = htile.full %zero : f32 -> tensor<8x4xf32>
        %acc = htile.full %zero : f32 -> tensor<4xf32>
        %vecmat = htile.dot %vector_lhs, %matrix_rhs, %acc
            : tensor<8xf32>, tensor<8x4xf32>, tensor<4xf32> -> tensor<4xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_triton(source))
    assert "tl.dot(" not in translated
    assert translated.count("tl.sum(") == 2
    assert "[None, :]" in translated
    assert "[:, None]" in translated
    assert "dtype=tl.float32" in translated
    assert ") + tile_" in translated


def test_cutile_translates_vector_dots_as_reductions():
    source = """
    module {
      htile.kernel @vector_dots() attributes {program_bounds = array<i64: 1>} {
        %zero = arith.constant 0.0 : f32
        %matrix_lhs = htile.full %zero : f32 -> tensor<4x8xf32>
        %vector_rhs = htile.full %zero : f32 -> tensor<8xf32>
        %matvec = htile.dot %matrix_lhs, %vector_rhs
            : tensor<4x8xf32>, tensor<8xf32> -> tensor<4xf32>
        %vector_lhs = htile.full %zero : f32 -> tensor<8xf32>
        %matrix_rhs = htile.full %zero : f32 -> tensor<8x4xf32>
        %acc = htile.full %zero : f32 -> tensor<4xf32>
        %vecmat = htile.dot %vector_lhs, %matrix_rhs, %acc
            : tensor<8xf32>, tensor<8x4xf32>, tensor<4xf32> -> tensor<4xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_cutile(source))
    assert "ct.mma(" not in translated
    assert translated.count("ct.sum(") == 2
    assert "ct.expand_dims(" in translated
    assert "ct.expand_dims(tile_" in translated
    assert ", 0)" in translated
    assert ", 1)" in translated
    assert ") + tile_" in translated


def test_tilelang_translates_vector_dots_as_reductions():
    source = """
    module {
      htile.kernel @vector_dots() attributes {program_bounds = array<i64: 1>} {
        %zero = arith.constant 0.0 : f32
        %matrix_lhs = htile.full %zero : f32 -> tensor<4x8xf32>
        %vector_rhs = htile.full %zero : f32 -> tensor<8xf32>
        %matvec = htile.dot %matrix_lhs, %vector_rhs
            : tensor<4x8xf32>, tensor<8xf32> -> tensor<4xf32>
        %vector_lhs = htile.full %zero : f32 -> tensor<8xf32>
        %matrix_rhs = htile.full %zero : f32 -> tensor<8x4xf32>
        %acc = htile.full %zero : f32 -> tensor<4xf32>
        %vecmat = htile.dot %vector_lhs, %matrix_rhs, %acc
            : tensor<8xf32>, tensor<8x4xf32>, tensor<4xf32> -> tensor<4xf32>
        htile.return
      }
    }
    """
    translated = ast.unparse(translate_tilelang(source))
    assert "T.gemm(" not in translated
    assert translated.count("in T.Parallel(4)") == 2
    assert translated.count("in T.serial(0, 8, 1)") == 2
    assert "] = 0.0" in translated
    assert "] = frag_" in translated
    assert translated.count("] += frag_") == 2


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
