"""Benchmark generated GPU kernels; no regression thresholds or performance assertions."""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from neptune_mlir.benchmarks.cases import Case, suite
from neptune_mlir.benchmarks.nsight import ProfileParseError, extract_profile
from neptune_mlir.benchmarks.runtime import BACKENDS


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def positive_float(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return n


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="neptune-bench",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "operator",
        choices=("attn", "mamba", "suite"),
        help="causal FP16 attention, selective scan, or three preset cases",
    )
    parser.add_argument("--backend", nargs="+", choices=(*BACKENDS, "all"), default=["triton"])
    parser.add_argument(
        "--profiler",
        choices=("cudaevent", "nsys", "ncu"),
        required=True,
        default=argparse.SUPPRESS,
        help="cudaevent measures latency; nsys/ncu capture and summarize instrumented kernel durations",
    )
    parser.add_argument("--output", type=Path, help="new artifact directory (must not exist)")
    parser.add_argument("--device", type=int, default=0, help="CUDA logical device index")
    parser.add_argument("--seed", type=int, default=0)
    shape = parser.add_argument_group("case overrides (not applicable to suite)")
    for name in (
        "batch",
        "seq-len",
        "heads",
        "head-dim",
        "channels",
        "state-dim",
        "block-m",
        "block-n",
        "block-channels",
    ):
        shape.add_argument(f"--{name}", type=positive_int)
    shape.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    workload = parser.add_argument_group("workload (all profilers)")
    workload.add_argument(
        "--mode",
        choices=("graph", "events"),
        default="graph",
        help="graph replays captured launches; events uses ordinary launches with any profiler",
    )
    workload.add_argument(
        "--cache",
        choices=("warm", "cold"),
        default="warm",
        help="cold evicts L2 before every operator invocation, including inside graphs",
    )
    workload.add_argument(
        "--warmup-ms",
        type=positive_float,
        default=200,
        help="warmup budget before measurement or capture",
    )
    workload.add_argument("--skip-check", action="store_true", help="skip reference checking")
    timing = parser.add_argument_group("CUDA-event measurement (--profiler cudaevent)")
    timing.add_argument(
        "--sample-ms",
        type=positive_float,
        default=5,
        help="approximate work per sample including eviction, capped at 1024 launches",
    )
    timing.add_argument(
        "--samples",
        type=positive_int,
        default=50,
        help="number of separately timed batches (at least 3)",
    )
    profiling = parser.add_argument_group("Nsight (capture instead of timing)")
    profiling.add_argument(
        "--profile-launches",
        type=positive_int,
        default=5,
        help="operator invocations in the capture range (one graph replay in graph mode)",
    )
    profiling.add_argument(
        "--profiler-arg",
        action="append",
        default=[],
        help="extra tool argument; repeat, e.g. --profiler-arg=--set=full",
    )
    args = parser.parse_args(argv)
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.profiler_arg and args.profiler == "cudaevent":
        parser.error("--profiler-arg requires --profiler nsys or ncu")
    overrides = {
        name: getattr(args, name)
        for name in (
            "batch",
            "seq_len",
            "heads",
            "head_dim",
            "channels",
            "state_dim",
            "dtype",
            "block_m",
            "block_n",
            "block_channels",
        )
        if getattr(args, name) is not None
    }
    if args.operator == "suite":
        if overrides:
            parser.error("case overrides are not supported with suite; select attn or mamba")
        args.cases = suite()
    else:
        unused = (
            {"channels", "state_dim", "block_channels"}
            if args.operator == "attn"
            else {"heads", "head_dim", "block_m", "block_n"}
        ) & overrides.keys()
        if unused:
            parser.error(f"inapplicable case overrides: {', '.join(sorted(unused))}")
        defaults = {"dtype": "float16"} if args.operator == "attn" else {"batch": 8}
        args.cases = [Case(args.operator, **(defaults | overrides))]
    try:
        for case in args.cases:
            case.validate()
    except ValueError as exc:
        parser.error(str(exc))
    args.backends = list(BACKENDS) if "all" in args.backend else list(dict.fromkeys(args.backend))
    return args


def profiler_command(tool, report, extra):
    if tool == "cudaevent":
        return []
    executable = shutil.which(tool)
    if executable is None:
        raise ValueError(f"{tool} is not on PATH; install Nsight or use --profiler cudaevent")
    if tool == "ncu":
        return [
            executable,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--replay-mode",
            "application",
            "--app-replay-mode",
            "strict",
            "--graph-profiling",
            "node",
            "--set",
            "basic",
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            "--export",
            str(report),
            *extra,
        ]
    return [
        executable,
        "profile",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--output",
        str(report),
        *extra,
    ]


def revision():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        return {"commit": git("rev-parse", "HEAD"), "worktree_status": git("status", "--porcelain")}
    except OSError:
        return {"commit": None, "worktree_status": None}


def run(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or Path("benchmark-results") / stamp).resolve()
    # Validate profiler availability before creating an output directory.
    profiler_command(args.profiler, output / "unused", args.profiler_arg)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "created_at": stamp, "revision": revision(), "results": []}
    print(
        "Neptune GPU benchmark (cudaevent)"
        if args.profiler == "cudaevent"
        else f"Neptune GPU profile ({args.profiler})"
    )
    if args.profiler == "cudaevent":
        print(
            f"  {args.mode} / {args.cache} cache | {args.samples} batches | "
            f"{args.warmup_ms:g} ms warmup | median [p20, p80] in us"
        )
    else:
        print(
            f"  {args.mode} / {args.cache} cache | {args.profile_launches} operator invocations | "
            "instrumented kernel durations (not CUDA-event batch means)"
        )
    print(f"  Artifacts: {output}\n", flush=True)
    columns = f"{'MEDIAN us':>11} {'P20 us':>11} {'P80 us':>11} {'N':>5}  STATUS"
    print(f"{'CASE':<38} {'BACKEND':<10} {columns}", flush=True)
    failures = 0
    device_info = None
    for case in args.cases:
        for backend in args.backends:
            path = output / f"{case.name}-{backend}"
            path.mkdir()
            spec = {
                "case": case.config(),
                "backend": backend,
                "device": args.device,
                "seed": args.seed,
                "check": not args.skip_check,
                "profiler": args.profiler,
                "profile_launches": args.profile_launches,
                "artifact_dir": str(path),
                "workload": {
                    "mode": args.mode,
                    "cache": args.cache,
                    "warmup_ms": args.warmup_ms,
                },
                "timing": {"samples": args.samples, "sample_ms": args.sample_ms},
            }
            spec_path = path / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2) + "\n")
            cmd = profiler_command(args.profiler, path / "profile", args.profiler_arg)
            cmd += [sys.executable, "-m", "neptune_mlir.benchmarks.worker", str(spec_path)]
            (path / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            print(f"{case.name:<38} {backend:<10} ", end="", flush=True)
            start = time.perf_counter()
            with (path / "worker.log").open("w") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            elapsed = time.perf_counter() - start
            report = (path / "profile").with_suffix(
                ".ncu-rep" if args.profiler == "ncu" else ".nsys-rep"
            )
            capture_ok = args.profiler == "cudaevent" or report.is_file()
            if completed.returncode == 0 and (path / "result.json").exists() and capture_ok:
                result = json.loads((path / "result.json").read_text())
                result["status"] = "ok"
                device_info = result["environment"]
                timing = result.get("timing")
                if args.profiler != "cudaevent":
                    result["profile"]["report"] = str(report)
                    try:
                        timing = extract_profile(
                            args.profiler,
                            report,
                            result["kernel"],
                            expected_launches=result["profile"]["launches"],
                        )
                        result["profile"]["timing"] = timing
                    except (ProfileParseError, OSError) as exc:
                        failures += 1
                        result["status"] = "parse_error"
                        result["profile"]["parse_error"] = str(exc)
                        print(f"{'-':>11} {'-':>11} {'-':>11} {'-':>5}  PARSE ERROR", flush=True)
                        print(f"  {exc}", flush=True)
                if timing:
                    values = f"{timing['median_us']:11.3f} {timing['p20_us']:11.3f} {timing['p80_us']:11.3f}"
                    spread = (timing["p80_us"] - timing["p20_us"]) / timing["median_us"]
                    timing["noisy"] = spread > 0.1
                    status = "ok" if args.profiler == "cudaevent" else f"profiled ({args.profiler})"
                    if timing["noisy"]:
                        status += "; noisy (>10% spread)"
                    if args.skip_check:
                        status += "; unchecked"
                    if timing.get("warnings"):
                        status += "; count mismatch"
                    print(f"{values} {len(timing['samples_us']):5d}  {status}", flush=True)
                    for warning in timing.get("warnings", []):
                        print(f"  Warning: {warning}", flush=True)
                if args.profiler != "cudaevent":
                    print(f"  Report: {report.relative_to(output)}", flush=True)
                    # Persist the parsed summary alongside the worker's original metadata.
                    (path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            else:
                failures += 1
                result = {
                    "name": case.name,
                    "case": case.config(),
                    "backend": backend,
                    "status": "error",
                    "returncode": completed.returncode,
                }
                prefix = f"{'-':>11} {'-':>11} {'-':>11} {'-':>5}  "
                print(f"{prefix}ERROR", flush=True)
                lines = (path / "worker.log").read_text(errors="replace").strip().splitlines()
                detail = lines[-1] if lines else "worker did not produce a result"
                if completed.returncode == 0 and not capture_ok:
                    detail = (
                        f"profiler did not produce {report.name}; check tool permissions/support"
                    )
                print(f"  {detail[:240]}\n  See {path / 'worker.log'}", flush=True)
            result.update({"wall_seconds": elapsed, "artifacts": str(path)})
            summary["results"].append(result)
            (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if device_info:
        print(
            f"\nDevice: {device_info['gpu']} | CUDA {device_info['cuda']} | "
            f"logical GPU {device_info['device']}"
        )
    print(f"Saved {output / 'results.json'}", flush=True)
    print(
        "No performance thresholds applied. Compare only matching devices, versions, and timing policies."
    )
    return 1 if failures else 0


def main():
    args = parse_args()
    try:
        code = run(args)
    except (ValueError, OSError) as exc:
        print(f"neptune-bench: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
