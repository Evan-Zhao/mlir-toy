"""Common launch adapters and CUDA-event timing, independent of the code generator."""

import ast
import importlib
import math
import statistics
import time
from contextlib import contextmanager

from .workload import create_workload

BACKENDS = ("triton", "cutile", "tilelang")


def kernel_info(htile):
    from mlir import ir

    from neptune_mlir.translators.common import parse_mlir_module_from_text

    module = parse_mlir_module_from_text(htile)
    kernels = [op for op in module.body.operations if op.operation.name == "htile.kernel"]
    if len(kernels) != 1:
        raise ValueError(f"benchmark requires one kernel, got {len(kernels)}")
    kernel = kernels[0]
    name = ir.StringAttr(kernel.attributes["sym_name"]).value
    grid = tuple(ir.DenseI64ArrayAttr(kernel.attributes["program_bounds"]))
    if not 1 <= len(grid) <= 3 or any(dim <= 0 for dim in grid):
        raise ValueError(f"unsupported kernel grid: {grid}")
    return name, grid


@contextmanager
def prepare(torch, htile, backend, args, artifact_dir):
    """Yield a launch-only callable with persistent, caller-owned input/output buffers."""
    from neptune_mlir.pipeline import (
        _import_generated_source,
        _torch_dtype,
        get_htile_kernel_arguments,
    )

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    name, grid = kernel_info(htile)
    signature = get_htile_kernel_arguments(htile)
    if len(signature) != len(args):
        raise ValueError(f"ABI mismatch: kernel needs {len(signature)} arguments, got {len(args)}")
    for i, (arg, param) in enumerate(zip(args, signature)):
        if tuple(arg.shape) != (param.shape or (1,)) or arg.dtype != _torch_dtype(
            torch, param.dtype
        ):
            raise ValueError(f"ABI mismatch at argument {i}: {arg.shape}/{arg.dtype} vs {param}")
    translator = importlib.import_module(f"neptune_mlir.translators.{backend}")
    source = ast.unparse(translator.translate_mlir_text(htile)) + "\n"
    (artifact_dir / "kernel.py").write_text(source)
    (artifact_dir / "kernel.mlir").write_text(htile)
    with _import_generated_source(source, f"bench_{backend}") as module:
        kernel = getattr(module, name)
        if backend == "triton":

            def launch():
                return kernel[grid](*args)
        elif backend == "cutile":
            import cuda.tile as ct

            launch_grid = grid + (1,) * (3 - len(grid))

            def launch():
                ct.launch(torch.cuda.current_stream(), launch_grid, kernel, tuple(args))
        else:
            import tilelang

            # out_idx=[] is important: output allocation must not occur on each launch.
            compiled = tilelang.compile(
                kernel,
                out_idx=[],
                execution_backend="tvm_ffi",
                target="cuda",
            )
            (artifact_dir / "kernel.cu").write_text(compiled.kernel_source)

            def launch():
                compiled(*args)

        built = launch()  # Lazy JIT compilation and runtime initialization are outside timing.
        torch.cuda.synchronize()
        info = {"kernel": name, "grid": grid}
        if backend == "triton":
            (artifact_dir / "kernel.ptx").write_text(built.asm["ptx"])
            info["resources"] = {
                "registers": built.n_regs,
                "spills": built.n_spills,
                "shared_bytes": built.metadata.shared,
                "num_warps": built.metadata.num_warps,
                "num_stages": built.metadata.num_stages,
            }
        yield launch, info


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("GPU timing samples must be finite and positive")
    return {
        "median_us": statistics.median(samples),
        "p20_us": quantile(samples, 0.2),
        "p80_us": quantile(samples, 0.8),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def warmup(torch, launch, warmup_ms):
    # A time budget (not a fixed launch count) also warms long-running scan kernels.
    deadline = time.perf_counter() + warmup_ms / 1000
    while True:
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
        if time.perf_counter() >= deadline:
            return


class EventTimer:
    """Observe a workload without deciding its cache policy or launch mode."""

    def __init__(self, torch, workload, count):
        self.count = count
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # Warm graphs use outer events to avoid perturbing each kernel with event nodes.
        # All other cases bracket individual kernels, excluding cache-eviction work.
        self.per_launch = workload.mode != "graph" or workload.cache.policy == "cold"
        self.pairs = []
        if self.per_launch:
            event_args = {"enable_timing": True}
            if workload.mode == "graph":
                event_args["external"] = True
            self.pairs = [
                (torch.cuda.Event(**event_args), torch.cuda.Event(**event_args))
                for _ in range(count)
            ]
            # Capture the pairs, not self: avoid a timer -> batch -> timer reference cycle.
            pairs = self.pairs
            self.run_batch = workload.batch(
                count,
                before=lambda i: pairs[i][0].record(),
                after=lambda i: pairs[i][1].record(),
            )
        else:
            self.run_batch = workload.batch(count)

    def sample(self):
        """Return (operator latency, total batch duration), both in milliseconds."""
        self.start.record()
        self.run_batch()
        self.end.record()
        self.end.synchronize()
        total_ms = self.start.elapsed_time(self.end)
        latency_ms = (
            statistics.mean(a.elapsed_time(b) for a, b in self.pairs)
            if self.per_launch
            else total_ms / self.count
        )
        return latency_ms, total_ms


def measure(torch, launch, *, mode, cache, warmup_ms, sample_ms, samples):
    """Time the composed workload; eviction never contributes to kernel latency."""
    if samples < 3 or sample_ms <= 0 or warmup_ms <= 0:
        raise ValueError("need >= 3 samples and positive timing budgets")
    with create_workload(torch, launch, mode=mode, cache=cache) as workload:
        warmup(torch, workload.batch(1), warmup_ms)
        # Amortize graph submission in the pilot for microsecond kernels.
        pilot_count = 32 if mode == "graph" else 1
        pilot = EventTimer(torch, workload, pilot_count)
        pilot.sample()  # Instantiate/execute graph and event nodes before calibration.
        estimate_ms = statistics.median(pilot.sample()[1] / pilot_count for _ in range(5))
        # Include eviction in the calibration budget, but not the reported kernel latency.
        batch_size = max(1, min(1024, math.ceil(sample_ms / max(estimate_ms, 1e-6))))
        timer = EventTimer(torch, workload, batch_size)
        timer.sample()
        times = [timer.sample()[0] * 1000 for _ in range(samples)]
        metadata = workload.metadata()
    return {
        **summarize(times),
        **metadata,
        "launches_per_sample": batch_size,
        "warmup_ms": warmup_ms,
        "sample_ms": sample_ms,
        "sample_count": samples,
        "timing_scope": "per_launch_events" if timer.per_launch else "graph_batch",
        "statistic": "per-launch latency averaged within each batch",
    }


def profile(torch, launch, label, *, mode, cache, warmup_ms, launches):
    """Observe the same workload via Nsight, without inserting our timing events."""
    if launches <= 0 or warmup_ms <= 0:
        raise ValueError("need positive launch count and warmup budget")
    with create_workload(torch, launch, mode=mode, cache=cache) as workload:
        batch = workload.batch(launches)
        # Graph construction, instantiation and warmup all precede profiler activation.
        warmup(torch, batch, warmup_ms)
        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(label):
                batch()
                workload.stream.synchronize()
        finally:
            torch.cuda.profiler.stop()
        return {
            **workload.metadata(),
            "launches": launches,
            "graph_replays": 1 if mode == "graph" else 0,
        }
