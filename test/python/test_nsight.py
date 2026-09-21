"""Offline parser tests: no GPU or Nsight installation required."""

import csv
import json
from types import SimpleNamespace

import pytest

from neptune_mlir.benchmarks.nsight import (
    ProfileParseError,
    extract_profile,
    parse_ncu_csv,
    parse_nsys_csv,
)

KERNEL = "attention_kernel"
NSYS = """NOTICE: exporting a report
Processing report...
Start (ns),Duration (ns),CorrId,Name
1,900000,7,"void at::FillFunctor<int, float>()"
900001,10000,7,attention_kernel
910001,20000,7,attention_kernel
930001,700000,7,[CUDA memcpy DtoH]
"""
NCU = """==WARNING== Example diagnostic
"ID","Process ID","Kernel Name","gpu__time_duration.sum"
"","","","ns"
"0","123","void at::FillFunctor<int, float>()","900000"
"1","123","attention_kernel","10000"
"2","123","attention_kernel","20000"
"""
NCU_LONG = """"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"1","123","attention_kernel","launch__registers_per_thread","register/thread","64"
"1","123","attention_kernel","gpu__time_duration.sum","usecond","10"
"2","123","attention_kernel","gpu__time_duration.sum","usecond","20"
"""


@pytest.mark.parametrize(
    "parser,text", [(parse_nsys_csv, NSYS), (parse_ncu_csv, NCU), (parse_ncu_csv, NCU_LONG)]
)
def test_nsight_per_launch_statistics(parser, text):
    result = parser(text, KERNEL, expected_launches=2)
    assert result["samples_us"] == [10, 20]
    assert result["median_us"] == 15
    assert result["p20_us"] == 12
    assert result["p80_us"] == 18
    assert result["sample_count"] == 2
    assert result["warnings"] == []
    assert result["instrumented"] is True
    assert result["timing_scope"] == "kernel_duration"


def test_nsys_does_not_deduplicate_graph_nodes_by_correlation_id():
    result = parse_nsys_csv(NSYS, KERNEL)
    assert result["sample_count"] == 2  # Both nodes share CorrId=7.
    assert sum(result["excluded_activities"].values()) == 2


@pytest.mark.parametrize(
    "name",
    [
        "attention_kernel",
        "attention_kernel_kernel",
        "attention_kernel_Kt2_A3bf16_p16",
        "void attention_kernel_kernel(float *, float *)",
        "void ns::attention_kernel<float, int>(float *)",
    ],
)
def test_nsight_backend_kernel_names(name):
    text = f'Duration (ns),Name\n1000,"{name}"\n'
    assert parse_nsys_csv(text, KERNEL)["samples_us"] == [1]


@pytest.mark.parametrize(
    "name",
    [
        "not_attention_kernel",
        "attention_kernel_helper",
        "attention_kernel_kernel_helper",
        "other_kernel",
    ],
)
def test_nsight_does_not_guess_unmatched_kernels(name):
    with pytest.raises(ProfileParseError, match="no duration samples"):
        parse_nsys_csv(f"Duration (ns),Name\n1000,{name}\n", KERNEL)


@pytest.mark.parametrize(
    "unit,value,expected",
    [
        ("ns", "1000", 1),
        ("us", "1", 1),
        ("μs", "1", 1),
        ("µs", "1", 1),
        ("ms", ".001", 1),
        ("s", "1e-6", 1),
        ("usecond", "1,234.5", 1234.5),
    ],
)
def test_nsight_units(unit, value, expected):
    result = parse_nsys_csv(f'Duration ({unit}),Name\n"{value}",{KERNEL}\n', KERNEL)
    assert result["median_us"] == pytest.approx(expected)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "N/A", "", "1,5", "1e999"])
def test_nsight_rejects_invalid_matching_duration(value):
    with pytest.raises(ProfileParseError):
        parse_nsys_csv(f'Duration (ns),Name\n"{value}",{KERNEL}\n', KERNEL)


def test_nsight_rejects_unterminated_csv_quotes():
    with pytest.raises(csv.Error):
        parse_nsys_csv('Duration (ns),Name\n1000,"attention_kernel\n', KERNEL)


def test_nsight_unknown_units_fail():
    with pytest.raises(ProfileParseError, match="unsupported duration unit"):
        parse_nsys_csv(f"Duration (cycles),Name\n100,{KERNEL}\n", KERNEL)


@pytest.mark.parametrize(
    "parser,text",
    [
        (parse_nsys_csv, "No data"),
        (parse_ncu_csv, "No data"),
        (parse_ncu_csv, NCU.replace('"ns"', '"cycles"')),
        (parse_ncu_csv, NCU.replace("gpu__time_duration.sum", "other")),
        (parse_ncu_csv, NCU.replace('"","","","ns"\n', "")),
        (parse_ncu_csv, NCU.replace('"2","123"', '"1","123"')),
    ],
)
def test_nsight_rejects_missing_or_ambiguous_data(parser, text):
    with pytest.raises(ProfileParseError):
        parser(text, KERNEL)


def test_nsight_does_not_derive_percentiles_from_aggregate_stats():
    with pytest.raises(ProfileParseError, match="Duration"):
        parse_nsys_csv("Avg (ns),Instances,Name\n1000,20,attention_kernel\n", KERNEL)


def test_nsight_warns_about_partial_capture():
    result = parse_ncu_csv(NCU, KERNEL, expected_launches=5)
    assert result["sample_count"] == 2
    assert "observed 2" in result["warnings"][0]
    assert "requested 5" in result["warnings"][0]


@pytest.mark.parametrize("tool,text", [("nsys", NSYS), ("ncu", NCU)])
def test_export_and_parse_retains_diagnostics(monkeypatch, tmp_path, tool, text):
    monkeypatch.setattr("shutil.which", lambda tool: f"/tools/{tool}")

    def export(cmd, *, stdout, stderr, **kwargs):
        assert kwargs["env"]["LC_ALL"] == "C"
        assert kwargs["timeout"] == 120
        if tool == "ncu":
            assert cmd[-2:] == ["--metrics", "gpu__time_duration.sum"]
        else:
            assert "cuda_gpu_trace" in cmd
            assert "--force-export=true" in cmd
        stdout.write(text)
        stderr.write("export diagnostic\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", export)
    result = extract_profile(tool, tmp_path / f"profile.{tool}-rep", KERNEL, expected_launches=2)
    assert result["median_us"] == 15
    assert (tmp_path / "profile.csv").read_text() == text
    assert "export diagnostic" in (tmp_path / "stats.log").read_text()
    assert json.loads((tmp_path / "stats-command.json").read_text())[0] == f"/tools/{tool}"


def test_export_failure_is_not_reported_as_zero_latency(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda tool: f"/tools/{tool}")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ProfileParseError, match="export failed"):
        extract_profile("ncu", tmp_path / "profile.ncu-rep", KERNEL)
    assert "exit 1" in (tmp_path / "stats.log").read_text()
