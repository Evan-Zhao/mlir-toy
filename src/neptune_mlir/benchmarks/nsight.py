"""Extract operator timings from official Nsight CSV exports (no profiler SDK required)."""

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from .runtime import summarize

_DURATION = "gpu__time_duration.sum"


class ProfileParseError(ValueError):
    """A capture exists, but cannot yield a trustworthy operator timing summary."""


def _table(text, required):
    """Locate the CSV header after tool notices; preserve quoted commas in kernel names."""
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")), strict=True)
    for row in reader:
        header = [field.strip() for field in row]
        if required.issubset(header):
            break
    else:
        raise ProfileParseError(f"Nsight CSV header missing columns: {', '.join(sorted(required))}")
    for row in reader:
        if not row or not any(field.strip() for field in row):
            continue
        if row == header:
            continue
        if row[0].startswith(("==", "NOTICE:", "WARNING:", "Processing ", "Generating ")):
            continue
        if len(row) != len(header):
            raise ProfileParseError(
                f"malformed Nsight CSV row: expected {len(header)} columns, got {len(row)}"
            )
        yield dict(zip(header, (field.strip() for field in row)))


def _duration_us(value, unit):
    unit = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    scales = {
        "ns": 0.001,
        "nsecond": 0.001,
        "nanosecond": 0.001,
        "us": 1,
        "usecond": 1,
        "microsecond": 1,
        "ms": 1000,
        "msecond": 1000,
        "millisecond": 1000,
        "s": 1e6,
        "second": 1e6,
    }
    scale = scales.get(unit, scales.get(unit.removesuffix("s")))
    if scale is None:
        raise ProfileParseError(f"unsupported duration unit: {unit!r}")
    # Tool exports use the C locale; commas may separate thousands, never decimals.
    number = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if not re.fullmatch(number, value.strip()):
        raise ProfileParseError(f"invalid kernel duration: {value!r}")
    result = float(value.replace(",", "")) * scale
    if not math.isfinite(result) or result <= 0:
        raise ProfileParseError(f"kernel duration must be finite and positive, got {value!r}")
    return result


def _kernel_match(name, kernel):
    # Current translators preserve the HTile name. TileLang appends _kernel;
    # cuTile appends a _Kt... specialization signature. Reject arbitrary helper names.
    pattern = rf"(?<![\w]){re.escape(kernel)}(?:_kernel|_Kt\w*)?(?=[^\w]|$)"
    return re.search(pattern, name) is not None


def _summary(tool, samples, names, excluded, kernel, expected_launches):
    if not samples:
        available = ", ".join(excluded) or "no GPU activities"
        raise ProfileParseError(f"no duration samples for {kernel!r}; found: {available}")
    warnings = []
    if expected_launches is not None and len(samples) != expected_launches:
        warnings.append(
            f"observed {len(samples)} operator launches; requested {expected_launches}. "
            "Check capture filters, launch limits, and profiler range settings."
        )
    return {
        **summarize(samples),
        "profiler": tool,
        "instrumented": True,
        "timing_scope": "kernel_duration",
        "sample_count": len(samples),
        "launches_per_sample": 1,
        "expected_launches": expected_launches,
        "statistic": "per-profiled-launch GPU duration (not CUDA-event batch means)",
        "duration_metric": "Duration" if tool == "nsys" else _DURATION,
        "kernel_names": sorted(names),
        "excluded_activities": dict(excluded),
        "warnings": warnings,
    }


def parse_nsys_csv(text, kernel, *, expected_launches=None):
    """Parse individual GPU trace rows, not aggregate summaries or CPU API times."""
    samples, names, excluded = [], set(), Counter()
    for row in _table(text, {"Name"}):
        name = row["Name"]
        if not _kernel_match(name, kernel):
            excluded[name] += 1
            continue
        duration_cols = [(key, re.fullmatch(r"Duration \(([^)]+)\)", key)) for key in row]
        duration_cols = [(key, match.group(1)) for key, match in duration_cols if match]
        if len(duration_cols) != 1:
            raise ProfileParseError("nsys cuda_gpu_trace must contain one Duration (<unit>) column")
        column, unit = duration_cols[0]
        samples.append(_duration_us(row[column], unit))
        names.add(name)
    return _summary("nsys", samples, names, excluded, kernel, expected_launches)


def parse_ncu_csv(text, kernel, *, expected_launches=None):
    """Support raw wide CSV (with a units row) and metric-per-row CSV from older tools.

    One duration is consumed per action ID. Counter replay passes are NOT extra samples.
    """
    samples, names, excluded = [], set(), Counter()
    seen = set()
    unit = None
    for row in _table(text, {"ID", "Kernel Name"}):
        if "Metric Name" in row:
            if row["Metric Name"] != _DURATION:
                continue
            if not {"Metric Unit", "Metric Value"}.issubset(row):
                raise ProfileParseError("ncu metric CSV is missing Metric Unit/Metric Value")
            value, metric_unit = row["Metric Value"], row["Metric Unit"]
        elif _DURATION in row:
            if not row["ID"] and not row["Kernel Name"]:
                unit = row[_DURATION]
                continue
            if unit is None:
                raise ProfileParseError("ncu raw CSV is missing its duration units row")
            value, metric_unit = row[_DURATION], unit
        else:
            raise ProfileParseError(f"ncu report does not contain {_DURATION}; collect this metric")
        name = row["Kernel Name"]
        if not _kernel_match(name, kernel):
            excluded[name] += 1
            continue
        identity = (row.get("Process ID", ""), row["ID"])
        if not row["ID"] or identity in seen:
            raise ProfileParseError(f"missing or duplicate ncu duration action ID: {identity}")
        seen.add(identity)
        samples.append(_duration_us(value, metric_unit))
        names.add(name)
    return _summary("ncu", samples, names, excluded, kernel, expected_launches)


def extract_profile(tool, report, kernel, *, expected_launches=None):
    """Export a native report, retain diagnostics/CSV, and return operator statistics."""
    report = Path(report).resolve()
    if tool not in {"nsys", "ncu"}:
        raise ValueError(f"unsupported Nsight tool: {tool}")
    executable = shutil.which(tool)
    if executable is None:
        raise ProfileParseError(f"{tool} is not on PATH for report export")
    if tool == "nsys":
        cmd = [
            executable,
            "stats",
            "--report",
            "cuda_gpu_trace",
            "--format",
            "csv",
            "--force-export=true",
            str(report),
        ]
        parser = parse_nsys_csv
    else:
        cmd = [
            executable,
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
            "--metrics",
            _DURATION,
        ]
        parser = parse_ncu_csv
    directory = report.parent
    (directory / "stats-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    csv_path, log_path = directory / "profile.csv", directory / "stats.log"
    try:
        with csv_path.open("w") as out, log_path.open("w") as log:
            completed = subprocess.run(
                cmd,
                stdout=out,
                stderr=log,
                check=False,
                timeout=120,
                env={**os.environ, "LC_ALL": "C"},
            )
        if completed.returncode:
            raise ProfileParseError(f"{tool} report export failed (exit {completed.returncode})")
        return parser(csv_path.read_text(), kernel, expected_launches=expected_launches)
    except (OSError, subprocess.TimeoutExpired, ValueError, csv.Error) as exc:
        with log_path.open("a") as log:
            log.write(f"\n{exc}\n")
        raise ProfileParseError(f"{exc}; see {log_path}") from exc
