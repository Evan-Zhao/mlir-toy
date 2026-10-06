# Task: External Performance Baselines and Comparisons

## Goal and scope

Create a separate companion repository, provisionally `neptune-perf`, for collecting high-performance
operator implementations, comparing them with Neptune, profiling them, and plotting results.
Check it out alongside `neptune-mlir`; do not make it a submodule or vendor the baseline catalog here.
Keep this repository focused on the compiler, correctness tests, and Neptune benchmarking.
Small changes here to expose reusable measurement APIs are in scope.

## Starting points in this repository

Read [benchmarking.md](benchmarking.md), then inspect:

- [`cli/bench.py`](../src/neptune_mlir/cli/bench.py): CLI orchestration, isolated workers, profiler
  commands, and JSON/artifact conventions.
- [`benchmarks/runtime.py`](../src/neptune_mlir/benchmarks/runtime.py): Neptune-specific `prepare`,
  plus launch-callable-based `measure` and `profile` that external implementations can reuse.
- [`benchmarks/workload.py`](../src/neptune_mlir/benchmarks/workload.py): graph/ordinary launches,
  cache eviction, and stream handling.
- [`benchmarks/nsight.py`](../src/neptune_mlir/benchmarks/nsight.py): Nsight CSV parsing. Its current
  single generated-kernel matching is Neptune-specific, not a general external-operator solution.
- [`benchmarks/worker.py`](../src/neptune_mlir/benchmarks/worker.py): validation, setup boundaries,
  and environment/telemetry collection.
- [`testing/`](../src/neptune_mlir/testing): shared inputs and references, including variant-specific
  semantics and kernel ABI preparation. External adapters should share semantic inputs/references,
  not blindly reuse Neptune's ABI buffers.
- [`test_benchmarks.py`](../test/python/test_benchmarks.py),
  [`test_benchmark_workload.py`](../test/python/test_benchmark_workload.py), and
  [`test_nsight.py`](../test/python/test_nsight.py): harness tests and GPU smoke coverage.

## Suggested companion layout

```text
baselines/{attention,mamba}/   # Thin implementation adapters
workloads/                    # Named shapes, dtypes, semantics, measurement policies
experiments/                  # Reproducible comparison manifests
scripts/                      # Run, profile, compare, plot
environments/                 # Pinned dependencies; separate environments when necessary
results/                      # Small curated JSON results and provenance
docs/                         # Setup, sources/licenses, reproduction instructions
```

## Implementation requirements

1. Start with one external attention baseline and one Mamba selective-scan baseline, alongside
   Neptune. Prefer pinned upstream packages/source revisions over copied kernels; record source
   URLs, versions, licenses, and any local patches. Name implementations explicitly, not “SOTA.”
2. Define a small adapter contract: prepare persistent inputs/output/scratch, return a launch
   callable, provide correctness validation and implementation metadata. Keep compilation and
   allocation outside steady-state timing. Record or reject unsupported graph/cache policies.
3. Reuse Neptune's timing/workload implementation through a pinned Neptune dependency. Expose a
   small supported API if needed; do not fork the timing code or create a third package yet.
   Run conflicting dependency stacks in separate worker environments.
4. Use shared workload manifests and matching semantic inputs. Check attention mask alignment,
   GQA head mapping, precision, and scan/state semantics. Validate before and after measurement;
   explicitly report unsupported cases. Label native-tuned versus matched-configuration comparisons.
5. Produce versioned JSON with raw samples, workload and measurement policy, implementation/source
   revisions, tuning configuration, correctness status, hardware/software metadata, and artifact paths.
   Preserve failures and retain replay commands. Do not mix CUDA-event batch averages with
   profiler-instrumented per-kernel durations.
6. Support optional Nsight capture for both Neptune and external adapters. External operators may
   launch several kernels: retain native reports and define operator attribution explicitly rather
   than treating one matched kernel as the entire operator or blindly summing overlapping durations.
7. Plot latency and speedup from saved results without requiring a GPU. Reject or clearly flag
   incompatible comparisons and unchecked/failed cases. Commit compact curated results/manifests;
   keep large traces, binaries, and raw captures outside Git.

## Acceptance criteria

Document commands to run, profile, and plot a small attention and Mamba comparison from a clean
checkout. Both external baselines and Neptune must pass semantic correctness checks on the selected
cases. Demonstrate shared timing policy, reproducible provenance, retained failure diagnostics, and
plots regenerated solely from saved results. Add CPU tests for manifests/results/comparison logic
and optional GPU smoke tests; performance thresholds and a broad baseline catalog are follow-up work.
