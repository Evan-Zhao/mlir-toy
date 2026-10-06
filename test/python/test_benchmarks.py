"""Benchmark configuration/reporting and optional GPU smoke tests, not performance gates.

Use ``-k benchmark_gpu`` for GPU tests or ``-k 'not benchmark_gpu'`` for CPU-only tests.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neptune_mlir.benchmarks.cases import Case, suite
from neptune_mlir.benchmarks.runtime import BACKENDS, kernel_info, measure, prepare, summarize
from neptune_mlir.cli.bench import parse_args, profiler_command, run
from neptune_mlir.operator.variants import VARIANTS


def test_benchmark_presets():
    cases = suite()
    assert len(cases) == 8
    assert {case.operator for case in cases} == {"dense", "varlen", "mamba"}
    assert [case.options["sequence_length"] for case in cases if case.operator == "mamba"] == [
        128,
        2048,
    ]
    assert len({case.name for case in cases}) == len(cases)
    for case in cases:
        case.validate()
        assert Case(**case.config()).config() == case.config()


def test_benchmark_statistics():
    result = summarize([5.0, 1.0, 2.0, 3.0, 4.0])
    assert result["median_us"] == 3
    assert result["p20_us"] == pytest.approx(1.8)
    assert result["p80_us"] == pytest.approx(4.2)
    assert result["min_us"] == 1
    assert result["max_us"] == 5
    assert result["samples_us"] == [5.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("samples", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_benchmark_rejects_invalid_timings(samples):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize(samples)


def test_benchmark_cli_defaults():
    args = parse_args(["attn", "--profiler", "cudaevent", "--backend", "all"])
    assert args.profiler == "cudaevent"
    assert args.backends == ["triton", "cutile", "tilelang"]
    assert args.cases[0].options["variant"] == "causal"
    assert args.cases[0].options["seq_len"] == 128  # Same defaults as neptune-export.
    assert args.mode == "graph"
    assert args.cache == "warm"
    assert args.samples == 50


def test_benchmark_cli_overrides():
    args = parse_args(
        [
            "mamba",
            "--profiler",
            "cudaevent",
            "--batch",
            "2",
            "--model-dim",
            "256",
            "--expand",
            "1",
            "--seq-len",
            "8",
            "--dtype",
            "float32",
            "--mode",
            "events",
            "--cache",
            "cold",
        ]
    )
    assert args.cases[0].name.startswith("mamba-b2-s8-c256-")
    assert args.cases[0].options["activation_dtype"] == "float32"


@pytest.mark.parametrize(
    "argv",
    [
        ["suite", "--seq-len", "128"],
        ["attn", "--channels", "128"],
        ["mamba", "--heads", "8"],
        ["attn", "--seq-len", "127"],
        ["attn", "--dtype", "bfloat16"],
        ["mamba", "--model-dim", "3"],
        ["mamba", "--sequence-length", "1"],
        ["mamba", "--batch", "1"],
        ["mamba", "--model-dim", "64"],
        ["dense", "--q-heads", "3", "--kv-heads", "2"],
        ["dense", "--kv-seq-len", "127"],
        ["varlen", "--num-docs", "1", "--total-tokens", "1024", "--max-doc-tokens", "128"],
        ["varlen", "--doc-lengths", "-1", "1025"],
        ["mamba", "--state-dim", "3"],
        ["mamba", "--samples", "1"],
        ["mamba", "--sample-ms", "nan"],
        ["mamba", "--batch", "0"],
        ["mamba", "--device", "-1"],
        ["mamba", "--profiler-arg=--set=full"],
    ],
)
def test_benchmark_cli_rejects_invalid_config(argv):
    with pytest.raises(SystemExit):
        parse_args([*argv, "--profiler", "cudaevent"])


def test_benchmark_cli_requires_profiler(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["attn"])
    assert exc.value.code == 2
    assert "required: --profiler" in capsys.readouterr().err


def test_benchmark_cli_rejects_none_profiler():
    with pytest.raises(SystemExit):
        parse_args(["attn", "--profiler", "none"])


@pytest.mark.parametrize("profiler", ["cudaevent", "nsys", "ncu"])
@pytest.mark.parametrize("mode", ["graph", "events"])
@pytest.mark.parametrize("cache", ["warm", "cold"])
def test_benchmark_cli_composes_profiler_mode_cache(profiler, mode, cache):
    args = parse_args(["attn", "--profiler", profiler, "--mode", mode, "--cache", cache])
    assert (args.profiler, args.mode, args.cache) == (profiler, mode, cache)


def test_benchmark_grid_comes_from_htile():
    assert kernel_info("""module {
      htile.kernel @scan() attributes {program_bounds = array<i64: 8, 12>} {
        htile.return
      }
    }""") == ("scan", (8, 12))


def test_benchmark_profiler_commands(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: f"/tools/{name}")
    ncu = profiler_command("ncu", tmp_path / "profile", ["--set=full"])
    assert ncu[ncu.index("--profile-from-start") + 1] == "off"
    assert ncu[-1] == "--set=full"
    assert ncu[ncu.index("--replay-mode") + 1] == "application"
    assert ncu[ncu.index("--app-replay-mode") + 1] == "strict"
    assert ncu[ncu.index("--graph-profiling") + 1] == "node"
    nsys = profiler_command("nsys", tmp_path / "profile", [])
    assert "--capture-range=cudaProfilerApi" in nsys
    assert "--sample=none" in nsys
    assert "--cuda-graph-trace=node" in nsys
    assert profiler_command("cudaevent", tmp_path, []) == []


def test_benchmark_noise_is_informational(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("neptune_mlir.cli.bench.revision", dict)

    def worker(cmd, **kwargs):
        path = Path(cmd[-1]).parent
        result = {
            "environment": {"gpu": "fake GPU", "cuda": "test", "device": 0},
            "timing": summarize([1.0, 2.0, 3.0]),
        }
        (path / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", worker)
    output = tmp_path / "results"
    args = parse_args(["attn", "--profiler", "cudaevent", "--output", str(output)])
    assert run(args) == 0
    text = capsys.readouterr().out
    assert "noisy" in text
    assert "No performance thresholds" in text
    result = json.loads((output / "results.json").read_text())["results"][0]
    assert result["status"] == "ok"
    assert result["timing"]["noisy"] is True


def test_benchmark_worker_failure_keeps_diagnostics(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("neptune_mlir.cli.bench.revision", dict)

    def worker(cmd, *, stdout, **kwargs):
        stdout.write("backend compiler failed\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("subprocess.run", worker)
    output = tmp_path / "results"
    args = parse_args(["attn", "--profiler", "cudaevent", "--output", str(output)])
    assert run(args) == 1
    assert "backend compiler failed" in capsys.readouterr().out
    result = json.loads((output / "results.json").read_text())["results"][0]
    assert result["status"] == "error"


@pytest.mark.parametrize("bad_csv", [False, True])
def test_benchmark_nsight_reports_workload_and_timings(monkeypatch, tmp_path, capsys, bad_csv):
    monkeypatch.setattr("neptune_mlir.cli.bench.revision", dict)
    monkeypatch.setattr("neptune_mlir.cli.bench.profiler_command", lambda *args: ["fake-nsys"])
    monkeypatch.setattr("shutil.which", lambda tool: f"/tools/{tool}")

    def worker(cmd, **kwargs):
        if "stats" in cmd:
            text = "Duration (ns),Name\n900000,eviction\n"
            text += "".join(f"{n * 1000},attention_kernel\n" for n in range(10, 15))
            kwargs["stdout"].write("bad CSV" if bad_csv else text)
            return SimpleNamespace(returncode=0)
        path = Path(cmd[-1]).parent
        spec = json.loads(Path(cmd[-1]).read_text())
        assert spec["workload"] == {"mode": "graph", "cache": "cold", "warmup_ms": 200}
        assert "mode" not in spec["timing"]
        result = {
            "environment": {"gpu": "fake GPU", "cuda": "test", "device": 0},
            "kernel": "attention_kernel",
            "profile": {"tool": "nsys", "mode": "graph", "cache": "cold", "launches": 5},
        }
        (path / "result.json").write_text(json.dumps(result))
        (path / "profile.nsys-rep").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", worker)
    args = parse_args(
        [
            "attn",
            "--profiler",
            "nsys",
            "--mode",
            "graph",
            "--cache",
            "cold",
            "--output",
            str(tmp_path / "results"),
        ]
    )
    assert run(args) == (1 if bad_csv else 0)
    text = capsys.readouterr().out
    assert "graph / cold cache" in text
    assert "profile.nsys-rep" in text
    assert "MEDIAN" in text
    result = json.loads((tmp_path / "results" / "results.json").read_text())["results"][0]
    assert Path(result["profile"]["report"]).exists()
    if bad_csv:
        assert "PARSE ERROR" in text
        assert result["status"] == "parse_error"
        assert "CSV header" in result["profile"]["parse_error"]
    else:
        assert "profiled (nsys)" in text
        assert "12.000" in text
        timing = result["profile"]["timing"]
        assert timing["samples_us"] == [10, 11, 12, 13, 14]
        assert timing["excluded_activities"] == {"eviction": 1}
        saved = json.loads((Path(result["artifacts"]) / "result.json").read_text())
        assert saved["profile"]["timing"] == timing


# Real-GPU smoke coverage: correctness and timing execution, without speed gates.
GPU_CASES = [
    *[
        Case(
            "dense",
            variant=v,
            batch=2,
            q_heads=4,
            kv_heads=2,
            seq_len=128,
            kv_seq_len=256,
            window_size=73,
        )
        for v in VARIANTS
    ],
    Case("dense", variant="kv-fp8-causal", q_heads=4, kv_heads=1),
    *[
        Case(
            "varlen",
            num_docs=4,
            total_tokens=161,
            heads=2,
            max_doc_tokens=128,
            index_dtype=dtype,
            doc_lengths=(0, 33, 127, 1),
        )
        for dtype in ("int32", "int64")
    ],
    *[
        Case("mamba", batch=2, sequence_length=8, model_dim=128, expand=2, activation_dtype=dtype)
        for dtype in ("float16", "bfloat16", "float32")
    ],
]


@pytest.fixture(scope="module", params=GPU_CASES, ids=lambda c: c.name)
def lowered_gpu_case(request):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = request.param
    # Do not import torch_mlir in this process: it conflicts with standalone MLIR.
    dependency = "torch_mlir" if case.operator == "dense" else "jax"
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} required")
    return case, case.lower()


@pytest.mark.parametrize("backend", BACKENDS)
def test_benchmark_gpu(lowered_gpu_case, backend, tmp_path):
    import torch

    pytest.importorskip("cuda.tile" if backend == "cutile" else backend)
    case, htile = lowered_gpu_case
    if (
        backend == "cutile"
        and case.options.get("variant") == "kv-fp8-causal"
        and torch.cuda.get_device_capability()[0] < 10
    ):
        pytest.skip("cuTile FP8 requires sm100 or newer")
    args = case.make_args(torch, seed=0)
    with prepare(torch, htile, backend, args, tmp_path) as (launch, info):
        case.check(torch, args)
        for mode in ("graph", "events"):
            result = measure(
                torch, launch, mode=mode, cache="warm", warmup_ms=1, sample_ms=0.1, samples=3
            )
            assert result["median_us"] > 0
            assert len(result["samples_us"]) == 3
            case.check(torch, args)
    assert info["kernel"]
