# GPU Benchmarking And Profiling

`neptune-bench` benchmarks every operator supported by `neptune-export` on Triton, cuTile, and
TileLang: dense attention (all five variants, MHA/GQA/MQA, rectangular query/KV lengths), packed
variable-length attention, and Mamba selective scan. Operator flags, defaults, and export dispatch
are shared with `neptune-export`; input/ABI preparation and references are shared with pipeline tests.

## Quick Start

Activate `.venv` with the same MLIR/export/backend dependencies as the pipeline tests. Run
`neptune-bench -h` for subcommands and `neptune-bench dense -h` (or `varlen`/`mamba`) for options;
`python -m neptune_mlir.cli.bench` is also available.

```shell
# Eight preset cases, all backends (24 measurements).
neptune-bench suite --profiler cudaevent --backend all

# Same operator options as neptune-export.
neptune-bench dense --profiler cudaevent --backend triton --variant windowed-causal \
  --q-heads 8 --kv-heads 2 --seq-len 4096 --window-size 73 --block-m 128
neptune-bench dense --profiler cudaevent --backend all --variant kv-fp8-causal \
  --q-heads 4 --kv-heads 1 --seq-len 128 --kv-seq-len 256
neptune-bench varlen --profiler cudaevent --backend all --num-docs 4 \
  --total-tokens 161 --max-doc-tokens 128 --index-dtype int64 --doc-lengths 0 33 127 1
neptune-bench mamba --profiler cudaevent --backend cutile tilelang \
  --sequence-length 2048 --model-dim 768 --expand 2 --activation-dtype bfloat16

# Ordinary launches with best-effort L2 eviction.
neptune-bench attn --profiler cudaevent --backend all --mode events --cache cold

```

The suite contains:

| Case | Shape | Storage | Tile |
|---|---|---|---|
| Dense attention × 5 variants | batch 1, query/KV heads 8, seq_len 2048, head_dim 64 | FP16 Q/output; FP8 K/V for KV-FP8 | M=128, N=64 |
| Packed varlen | 8 documents, 1024 tokens, heads 4, head_dim 64, max_doc_tokens 512 | FP16, int32 offsets | M=128, N=64 |
| Mamba short | batch 8, seq_len 128, channels 1536, state_dim 16 | BF16 | channels=128 |
| Mamba long | batch 8, seq_len 2048, channels 1536, state_dim 16 | BF16 | channels=128 |

Dense variants are `global`, `causal`, `windowed-causal`, `alibi-causal`, and `kv-fp8-causal`.
Dense attention uses FP16 activations (with quantized K/V for KV-FP8), as in the exporter.
ALiBi slopes and FP8 scales are deterministic, non-unit per-head inputs from the shared test helpers.
`--q-heads`/`--kv-heads` select MHA/GQA/MQA; omitted KV heads/length default to query heads/length.

Varlen documents default to balanced lengths summing to `--total-tokens`. The benchmark-only
`--doc-lengths` option allows uneven and empty documents; it must match `--num-docs`, sum to
`--total-tokens`, and respect `--max-doc-tokens`. Exact lengths are saved in the case specification.
Mamba channels are `--model-dim * --expand`. Device selection respects `CUDA_VISIBLE_DEVICES`.
cuTile KV-FP8 execution requires sm100 or newer; unsupported configurations fail, rather than skip.

Migration from the initial benchmark CLI: `attn` remains an alias for `dense`, and dense `--heads`
remains an alias for `--q-heads`. Replace Mamba `--channels C` with `--model-dim C --expand 1`;
`--seq-len` and `--dtype` remain Mamba aliases for `--sequence-length` and `--activation-dtype`.
Single-case defaults now match `neptune-export` (dense: sequence 128, four query heads, default
causal variant); `suite` retains the original larger cases and adds the missing operators/variants.
Case JSON uses canonical export option names (results schema version 2); old worker specs are not
compatible. Use a subcommand, not `suite`, to override operator options.

The GPU schedules still impose shape constraints: dense query/KV lengths must divide by their
respective tiles; tile sizes and head/state dimensions have power-of-two requirements. Packed
varlen permits partial tiles. Mamba's known singleton batch/timestep/channel-block schedule
limitations are rejected before launching a worker.

## Measurement Method

With `--profiler cudaevent`, all backends use a shared PyTorch CUDA events/graphs implementation,
following the event-timed repeated graph approach used by Triton's `do_bench_cudagraph`.

1. Export, lower, translate, allocate deterministic inputs, and compile/launch once outside timing.
2. Validate against an FP32 Torch reference outside timing. Attention checks selected rows including
   tile boundaries across every batch/head; packed varlen checks every document independently and
   Mamba checks the full sequence. All check that the full output is finite. This is a benchmark sanity check, not exhaustive numerical coverage.
3. Warm up, estimate runtime, and size batches to the requested sample budget.
4. Collect timed batches and report median/p20/p80 of their per-launch average latencies, excluding
   eviction. The JSON retains every sample. These percentiles describe batch averages, not individual
   launch tail latencies or confidence intervals.
5. Check outputs again after measurement to catch state-reuse or graph-execution mistakes.

Setup and reference computation are outside timing; output buffers are reused. `lower_seconds` and
`prepare_seconds` are diagnostic wall times with backend caches enabled. Runs using `--skip-check`
are labeled unchecked. Input factories and Torch references are shared with correctness tests through
[`neptune_mlir.testing`](../src/neptune_mlir/testing).

### Graph Versus Ordinary Launches

With `--profiler cudaevent --mode graph --cache warm`, a single outer event pair times each batch.
This reduces measurement overhead and CPU/Python submission starvation for short kernels. It measures
steady-state GPU execution with graph scheduling and reused addresses, excluding Python call latency.
Graph capture failures are reported as errors.

`--mode events` uses ordinary launches. With `--profiler cudaevent`, an event pair brackets each
operator invocation. CPU submission gaps can inflate short-kernel timings; use graphs or inspect an
Nsight Systems trace in that situation.

Cold-cache eviction clears at least 256 MiB (or twice the reported L2 size, if larger). CUDA-event
timing brackets each operator after eviction, excluding flush time. Inside graphs this requires
`torch.cuda.Event(external=True)` support and adds instrumentation overhead. JSON records
`timing_scope` as `per_launch_events` rather than the warm-graph `graph_batch`.

Eviction is best effort, not a hardware cache reset. Warm means no eviction, not that the whole
working set fits in cache. Nsight capture uses the same workload without harness timing events;
eviction kernels appear separately in the trace/report and must not be counted as operator work.

### Controlling Noise

Use an idle GPU with stable clocks, adequate cooling, and a fixed software stack. Cases/backends run
sequentially in isolated processes. Increase warmup for cold GPUs. An administrator can lock supported
GPU clocks with `nvidia-smi`; the harness leaves power/clock settings unchanged.

The CLI flags p20-to-p80 spreads over 10% of the median as noisy. Inspect raw samples and GPU
snapshots, and repeat runs: narrow spreads can still hide clock drift or contention. Compare the same
case, dtype, tile sizes, backend/toolchain, device, cache policy, and timing mode. Backend order is
fixed; paired base-versus-candidate comparison is not implemented.

## Nsight Capture

The optional Nsight modes collect traces and profiles, then automatically print kernel-duration
summaries. These numbers include profiler effects; use `cudaevent` for uninstrumented benchmarking:

```shell
# Trace ordinary launches and see GPU duration versus CPU submission gaps.
neptune-bench attn --backend triton --profiler nsys --mode events --profile-launches 10

# Hardware counters for a warmed kernel; the default NCU section set is basic.
neptune-bench mamba --backend cutile --seq-len 128 --profiler ncu --profile-launches 1

```

The tools must be on `PATH`. `cudaProfilerStart/Stop` and the NVTX range `neptune/<case>/<backend>`
delimit capture, excluding compilation, graph construction, reference checking, and warmup. Launch
counts refer to one capture/application pass. Metadata records launch/cache policy, eviction buffer
size, operator count, and graph replay count.

Nsight Systems disables CPU sampling/context-switch tracing and requests graph node tracing so
individual operator and eviction kernels remain visible. Node tracing adds profiling overhead.

Nsight Compute uses application replay with strict kernel matching and graph-node profiling.
Each counter pass reruns the worker with the same launch/cache policy, requiring deterministic
execution. Automatic cache/clock control is disabled. Overriding these defaults with kernel replay
or automatic cache control can change cache state across passes.

Counter access may require administrator configuration and compatible tools, GPUs, and drivers.
Tool failure or a missing report is an error with a link to the log.

Open `profile.nsys-rep` or `profile.ncu-rep` in the corresponding GUI, or inspect from the command line:

```shell
nsys stats --report cuda_gpu_kern_sum path/to/profile.nsys-rep
ncu --import path/to/profile.ncu-rep --page details
```

### Automatic Result Parsing

After capture, the harness uses the tools' official CSV exports:

- `nsys stats --report cuda_gpu_trace --format csv --force-export=true <report>`:
  one GPU duration per matching kernel trace row, including graph nodes.
- `ncu --import <report> --page raw --csv --print-units base --metrics gpu__time_duration.sum`:
  one duration per matching action ID, excluding repeated counter passes.

The parser matches the generated HTile kernel name, including TileLang's `_kernel` suffix and
cuTile's specialization suffix. Eviction kernels, copies, and unrelated activities are excluded.
Durations are converted to microseconds. Nsight median/p20/p80 and `N` describe individual matched
launches; CUDA-event statistics describe timed batch averages. With one captured launch, all three
percentiles are identical.

Results live under `profile.timing` in per-case `result.json` and combined `results.json`. They include
raw samples, matched kernel names, excluded activity counts, and launch-count mismatch warnings.
The CLI shows the observed count.

Missing durations, unmatched kernels, unsupported units, malformed exports, and export failures
produce a `PARSE ERROR` with diagnostics while retaining the native report. Whole-graph-only traces
or NCU metric sets without `gpu__time_duration.sum` cannot provide these per-operator summaries.
Additional NCU counters remain in the native report.

## Output And Artifacts

Every profiler prints median/p20/p80 in microseconds and sample count `N`. Nsight rows are labeled
`profiled (nsys)` or `profiled (ncu)` and also show the report path. Compiler chatter and profiler
messages are captured in per-case logs. The default output root is the git-ignored
`benchmark-results/<UTC timestamp>/`.

- `results.json`: schema version, Neptune revision/dirty status, and all successes/errors. Case
  directories include a configuration hash to distinguish variants, layouts, dtypes, tiles and packing.
- `<case>-<backend>/spec.json`: exact input/configuration for the isolated worker, with shared
  `workload` options separated from CUDA-event `timing` options.
- `command.json`, `worker.log`: replay command and full diagnostics.
- `kernel.mlir`, `kernel.py`: generated Kernel HTile and backend source; TileLang also saves CUDA,
  and Triton saves PTX and records register/spill/shared-memory counts.
- `result.json`: case metadata, validation errors, launch grid, timing samples, and environment.
- `profile.nsys-rep` / `profile.ncu-rep`: optional Nsight captures.
- `profile.csv`, `stats-command.json`, `stats.log`: profiler timing export, exact export command,
  and export/parser diagnostics. Nsight Systems also creates a SQLite export beside its report.

Environment records include GPU details, software versions, visibility settings, and available
`nvidia-smi` telemetry before/after the worker run. These snapshots include reference checks and do
not establish clock stability during measurement. Nsight commands retain forwarded options.

Results are saved incrementally so a later backend failure does not discard earlier results. Missing
dependencies, compilation errors, failed reference checks, and profiler/export/parser failures produce
a nonzero exit code; a slow or noisy kernel does not. There are no stored performance baselines or CI gates.
The real-GPU smoke matrix in `test/python/test_benchmarks.py` exercises shared inputs, backend
preparation, graph/event timing and pre/post checks without performance thresholds. Select it with
`-k benchmark_gpu`, or use `-k 'not benchmark_gpu'` for that file's CPU-only tests. GPU tests skip when
CUDA is unavailable. CPU-only tests cover shared CLI option parity, serialized cases, workload
composition and profiler parsing.
See [backend test coverage](backend-test-coverage.md) for the separate automated correctness matrix.
