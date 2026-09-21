"""Isolated benchmark worker; logs go to artifacts, not the CLI result table."""

import csv
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .cases import Case
from .runtime import measure, prepare, profile


def gpu_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,"
                    "clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used"
                ),
                "--format=csv,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(io.StringIO(result.stdout), skipinitialspace=True)
        ]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment(torch):
    versions = {}
    for package in (
        "neptune-mlir",
        "torch",
        "triton",
        "cuda-tile",
        "tilelang",
        "apache-tvm-ffi",
        "jax",
        "torch-mlir",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "device": torch.cuda.current_device(),
        "total_memory": props.total_memory,
        "gpu_uuid": str(getattr(props, "uuid", "unknown")),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_before": gpu_snapshot(),
    }


def run(spec):
    # Keep JAX export CPU-only and compiler verbosity in the worker log.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("benchmarking requires PyTorch with an available NVIDIA CUDA device")
    torch.cuda.set_device(spec["device"])
    case = Case(**spec["case"])
    case.validate()
    path = Path(spec["artifact_dir"])
    result = {
        "case": case.config(),
        "name": case.name,
        "backend": spec["backend"],
        "seed": spec["seed"],
        "environment": environment(torch),
    }
    start = time.perf_counter()
    htile = case.lower()
    result["lower_seconds"] = time.perf_counter() - start
    args = case.make_args(torch, spec["seed"])
    # Detect incomplete writes instead of accidentally validating uninitialized memory.
    args[-1].fill_(float("nan"))
    start = time.perf_counter()
    with prepare(torch, htile, spec["backend"], args, path) as (launch, kernel):
        result["prepare_seconds"] = time.perf_counter() - start
        result.update(kernel)
        result["check"] = case.check(torch, args) if spec["check"] else {"scope": "skipped"}
        if spec["profiler"] == "cudaevent":
            result["timing"] = measure(torch, launch, **spec["workload"], **spec["timing"])
        else:
            result["profile"] = profile(
                torch,
                launch,
                f"neptune/{case.name}/{spec['backend']}",
                **spec["workload"],
                launches=spec["profile_launches"],
            )
            result["profile"]["tool"] = spec["profiler"]
        # Also check graph replay and repeated state reuse, outside timing/capture.
        if spec["check"]:
            result["check_after"] = case.check(torch, args)
    result["environment"]["gpu_after"] = gpu_snapshot()
    (path / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
