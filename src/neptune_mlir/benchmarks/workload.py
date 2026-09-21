"""Launch mode and cache policy, independent of timers and external profilers."""

from contextlib import contextmanager


class CachePolicy:
    """Apply best-effort L2 eviction before *each* operator invocation, or do nothing."""

    def __init__(self, torch, policy):
        if policy not in {"warm", "cold"}:
            raise ValueError(f"invalid cache policy: {policy}")
        self.policy = policy
        self.buffer = None
        if policy == "cold":
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            size = max(256 * 1024**2, 2 * getattr(props, "L2_cache_size", 0))
            self.buffer = torch.empty(size // 4, dtype=torch.int32, device="cuda")
            # Initialize the eviction operation before any graph capture.
            self.buffer.zero_()

    def before_launch(self):
        if self.buffer is not None:
            self.buffer.zero_()

    @property
    def bytes(self):
        return self.buffer.numel() * self.buffer.element_size() if self.buffer is not None else 0


class Workload:
    """Compose cache preparation, optional observation hooks, and launch execution.

    Hooks bracket only the operator, never eviction. Graph capture contains the same
    sequence as ordinary launches, including eviction before every invocation.
    """

    def __init__(self, torch, launch, mode, cache, stream):
        if mode not in {"graph", "events"}:
            raise ValueError(f"invalid launch mode: {mode}")
        self.torch = torch
        self.launch = launch
        self.mode = mode
        self.cache = cache
        self.stream = stream

    def batch(self, count, *, before=None, after=None):
        if count <= 0:
            raise ValueError("batch size must be positive")

        def sequence():
            for i in range(count):
                self.cache.before_launch()
                if before is not None:
                    before(i)
                self.launch()
                if after is not None:
                    after(i)

        if self.mode == "events":
            return sequence
        graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(graph, stream=self.stream):
            sequence()
        return graph.replay

    def metadata(self):
        return {"mode": self.mode, "cache": self.cache.policy, "flush_bytes": self.cache.bytes}


@contextmanager
def create_workload(torch, launch, *, mode, cache):
    # Capture/execute on a side stream with explicit input-initialization dependencies.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        workload = Workload(torch, launch, mode, CachePolicy(torch, cache), stream)
        try:
            yield workload
        finally:
            # Keep graphs, cache buffers, and caller-owned tensors alive through completion.
            stream.synchronize()
