"""CPU-only checks of workload composition and timer boundaries, not performance tests."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from neptune_mlir.benchmarks.runtime import EventTimer, profile
from neptune_mlir.benchmarks.workload import Workload


class FakeCuda:
    """Model deferred graph operations and a GPU clock with deterministic durations."""

    def __init__(self):
        self.capture = None
        self.clock = 0
        self.operations = []
        self.event_options = []

    def enqueue(self, name, duration=0, callback=None):
        def execute():
            self.operations.append(name)
            self.clock += duration
            if callback is not None:
                callback()

        if self.capture is None:
            execute()
        else:
            self.capture.commands.append(execute)

    def CUDAGraph(self):
        commands = []
        return SimpleNamespace(commands=commands, replay=lambda: [cmd() for cmd in commands])

    @contextmanager
    def graph(self, graph, **kwargs):
        self.capture = graph
        try:
            yield
        finally:
            self.capture = None

    def Event(self, **kwargs):
        self.event_options.append(kwargs)
        cuda = self

        class Event:
            def record(self):
                cuda.enqueue("event", callback=lambda: setattr(self, "timestamp", cuda.clock))

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        return Event()


def make_workload(mode, cache):
    cuda = FakeCuda()
    torch = SimpleNamespace(cuda=cuda)
    policy = SimpleNamespace(
        policy=cache,
        bytes=256 if cache == "cold" else 0,
        before_launch=lambda: cuda.enqueue("evict", 100) if cache == "cold" else None,
    )
    stream = SimpleNamespace(synchronize=lambda: None)
    return torch, Workload(torch, lambda: cuda.enqueue("kernel", 2), mode, policy, stream)


@pytest.mark.parametrize("mode", ["graph", "events"])
@pytest.mark.parametrize("cache", ["warm", "cold"])
def test_batch_composes_eviction_and_observation(mode, cache):
    torch, workload = make_workload(mode, cache)
    cuda = torch.cuda
    batch = workload.batch(
        3,
        before=lambda i: cuda.enqueue("begin"),
        after=lambda i: cuda.enqueue("end"),
    )
    # Constructing the workload (including graph capture) must not execute GPU work.
    assert cuda.operations == []
    batch()
    batch()
    expected = (["evict"] if cache == "cold" else []) + ["begin", "kernel", "end"]
    assert cuda.operations == expected * 6
    assert workload.metadata() == {
        "mode": mode,
        "cache": cache,
        "flush_bytes": 256 if cache == "cold" else 0,
    }


@pytest.mark.parametrize("mode", ["graph", "events"])
@pytest.mark.parametrize("cache", ["warm", "cold"])
def test_timer_excludes_eviction_on_every_replay(mode, cache):
    torch, workload = make_workload(mode, cache)
    timer = EventTimer(torch, workload, 3)
    for _ in range(2):
        kernel_ms, total_ms = timer.sample()
        assert kernel_ms == 2
        assert total_ms == 3 * (102 if cache == "cold" else 2)
    if mode == "graph" and cache == "cold":
        assert sum(options.get("external", False) for options in torch.cuda.event_options) == 6
    if mode == "graph" and cache == "warm":
        assert len(torch.cuda.event_options) == 2  # No per-node timing perturbation.


@pytest.mark.parametrize("mode", ["graph", "events"])
@pytest.mark.parametrize("cache", ["warm", "cold"])
def test_nsight_observes_same_workload_after_warmup(monkeypatch, mode, cache):
    torch, workload = make_workload(mode, cache)
    cuda = torch.cuda

    @contextmanager
    def create_workload(*args, **kwargs):
        yield workload

    @contextmanager
    def nvtx(label):
        cuda.enqueue("nvtx_start")
        yield
        cuda.enqueue("nvtx_end")

    cuda.profiler = SimpleNamespace(
        start=lambda: cuda.enqueue("start"), stop=lambda: cuda.enqueue("stop")
    )
    cuda.nvtx = SimpleNamespace(range=nvtx)
    monkeypatch.setattr("neptune_mlir.benchmarks.runtime.create_workload", create_workload)
    monkeypatch.setattr("neptune_mlir.benchmarks.runtime.warmup", lambda torch, batch, ms: batch())
    result = profile(
        torch, workload.launch, "case", mode=mode, cache=cache, warmup_ms=1, launches=2
    )
    kernels = (["evict"] if cache == "cold" else []) + ["kernel"]
    assert cuda.operations == kernels * 2 + ["start", "nvtx_start"] + kernels * 2 + [
        "nvtx_end",
        "stop",
    ]
    assert not cuda.event_options  # No harness timing events in Nsight captures.
    assert result["mode"] == mode
    assert result["cache"] == cache
    assert result["launches"] == 2
    assert result["graph_replays"] == (1 if mode == "graph" else 0)
