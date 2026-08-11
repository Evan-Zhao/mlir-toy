# Transform Schedule Autotuning Plan

## Goal

Neptune should tune Transform dialect schedules without moving scheduling policy into the host
program. A schedule should declare both scalar choices and structurally different alternatives,
including a common schedule followed by backend-specific branches. The tuner should derive the
active choices from that program, compile and measure candidates, and preserve the winning choices
for deterministic replay.

The initial target is the global-attention schedule. Its tunable draft lives in
`test/Tune/draft_attention_schedule.mlir` while the tuning operations and workflow are under
development.

## Target Input and Outcome

A schedule declares finite scalar choices directly where their values are used:

```mlir
%block_m = transform.tune.sample_categorical "attention.block_m"
    candidates = [32, 64, 128] default = 128 : !transform.param<i64>
%block_n = transform.tune.sample_categorical "attention.block_n"
    candidates = [32, 64, 128] default = 64 : !transform.param<i64>

%tiled, %loop = transform.structured.tile_using_forall
    %matmul tile_sizes [1, 1, %block_m, %block_n, 0]
    : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
      -> (!transform.any_op, !transform.any_op)
```

Structural choices select named sequences with compatible signatures:

```mlir
transform.tune.choose_sequence "backend" %func, %loop
    default = "triton"
    cases = {cutile = @configure_cutile,
             tilelang = @configure_tilelang,
             triton = @configure_triton}
    : (!transform.any_op, !transform.any_op) -> ()
```

Only the selected sequence contributes active choices. For example, the Triton sequence may sample
`triton.num_warps`, while the cuTile sequence may sample load-latency hints. A concrete candidate is
a sparse decision trace:

```json
{
  "attention.block_m": 128,
  "attention.block_n": 64,
  "backend": "triton",
  "triton.num_warps": 8
}
```

Candidate materialization turns the tunable program into an ordinary Transform program. Scalar
choices become `transform.param.constant`, and structural choices become `transform.include` of the
selected sequence. The existing Transform interpreter then produces scheduled payload IR. The
backend choice and options are also retained as payload attributes or trace metadata for subsequent
translation and compilation.

The final tuning outcome consists of:

- a replayable decision trace keyed by schedule, workload, and target;
- measured correctness and latency results;
- optionally, a frozen Transform schedule or tuning-spec library containing the winning choices.

## Immediate Components

### 1. Tuning Transform Operations

**Status: initial implementation complete.**

`transform.tune.sample_categorical` declares a stable integer choice, its finite domain, and a
default. It currently evaluates to the default so that the operation can be tested before candidate
materialization exists.

`transform.tune.choose_sequence` declares labeled named-sequence alternatives and verifies that
their signatures match. It intentionally cannot execute before materialization.

The immediate syntax is deliberately small. Attribute-valued choices, payload-dependent domains,
and inline structural regions can be added after the first end-to-end workflow is working.

### 2. Candidate Materialization

**Status: next implementation step.**

Add a pass or standalone transformation over the Transform module:

```text
transform-tune-materialize{decisions-file=candidate.json}
```

It should:

1. Validate decision IDs and values.
2. Rewrite active `sample_categorical` operations to `transform.param.constant`.
3. Rewrite active `choose_sequence` operations to `transform.include`.
4. Use defaults when explicitly requested for normal, non-tuning compilation.
5. Diagnose missing active decisions and ignore decisions belonging to inactive branches.
6. Record the schedule hash and normalized active trace for reproducibility.

Keeping this as a preprocessing step lets Neptune continue using the upstream Transform interpreter
unchanged. A custom interpreter state becomes necessary only when a choice domain depends on the
partially transformed payload.

### 3. Conditional Space Discovery and Trace Replay

Walk the named-sequence call graph from `@__transform_main`. Report categorical choices and follow
the selected case at each `choose_sequence`. Repeating this with a partial trace discovers nested
backend choices lazily.

The complete space need not be counted or eagerly enumerated. The Transform program is the source
of truth; an optional index of observed choice sites and branch guards supports diagnostics and
search coverage. Traces use stable string IDs rather than SSA positions.

### 4. Initial Search and Measurement Loop

Start with random sampling and, where useful, exhaustive traversal of small finite branches. For
each normalized trace:

1. Materialize the schedule.
2. Run scheduling and lowering on a fresh payload.
3. Select the recorded backend and compile the generated kernel.
4. Check numerical correctness against a reference.
5. Warm up, synchronize, and measure repeated executions.
6. Store successes and failures in a simple JSONL or SQLite database.

Compilation failures, invalid transformations, unsupported backend configurations, and correctness
failures reject the candidate. Scheduled-IR or artifact hashes should deduplicate distinct traces
that produce the same kernel.

The builder, runner, and database interfaces should remain backend-neutral, but their exact shape
can be refined after Triton and one other backend work end to end.

## Extensions

### Richer and Payload-Dependent Choices

Add choices such as perfect-tile factorization, permutations, one-of arbitrary MLIR attributes, and
selection from a handle of legal fusion or compute locations. These require a decision provider
attached through `TransformState::Extension`, because their domains are known only while
interpreting a schedule against a payload.

Changing an early structural decision may expose a different set of later choices. Replay should
preserve still-active decisions, discard inactive ones, and sample newly encountered choices.

### Structural Schedule Rules

Beyond explicit backend branches, named sequences can represent different fusion frontiers,
reduction decompositions, materialized versus rolling softmax, and backend-specific lowering plans.
A later rule-based space generator may traverse matched operations and offer applicable scheduling
sequences automatically.

### Evolutionary and Cost-Model-Guided Search

Once traces and measurements are reliable, add typed mutations for categorical values, tile
factors, placement choices, and structural branches. Evolutionary search does not require a known
space size: mutation followed by replay normalizes a sparse conditional trace. Exploration should
cover every top-level backend and retain an epsilon-random fraction of measurements.

A learned cost model should eventually consume scheduled MLIR features, not only decision values.
Useful features include loop structure, reductions, fusion boundaries, memory traffic and reuse,
pipeline stages, target properties, and backend compilation statistics. Per-backend models may be
more effective than one model over heterogeneous code generators.

### Symbolic Constraints

SMT is optional for the initial workflow. Later it can express relationships among tile sizes,
thread counts, intrinsic layouts, pipeline stages, and target resource limits. The solver can seed
legal assignments or repair a mutated partial trace; Transform replay and backend verification
remain the authoritative legality checks. Upstream MLIR already provides the SMT dialect and an
SMT-LIB exporter.

### Deployment and Multi-Task Tuning

Winning traces can be frozen into matcher/action Transform libraries selected by workload shape,
dtype, target, and backend. Later work may share measurements across shape families, schedule trial
budgets among multiple kernels, and support remote or multi-device runners.

## Related Designs

### IREE Tuning

IREE identifies dispatch roots and represents many legal code-generation configurations as typed
knob templates plus compiler-generated SMT constraints. An external tuner enumerates satisfying
assignments, materializes each assignment as `compilation_info`, and emits a Transform dialect
tuning spec that matches and annotates the dispatch. AMDSHARK Tuner provides parallel compilation,
benchmarking, baseline checks, and tuning-spec selection.

Neptune can reuse the ideas of compiler-owned legality, typed one-of attributes, Transform tuning
specs, and robust builder/runner orchestration. IREE's current search is primarily constrained
configuration enumeration rather than arbitrary Transform-program mutation, so it does not replace
Neptune's trace and structural-choice work.

### TVM MetaSchedule

MetaSchedule represents deterministic scheduling operations and random decisions in replayable TIR
schedule traces. Schedule rules produce structural trace skeletons; sampling primitives fill tile,
placement, and categorical decisions. Its evolutionary search combines measured records, typed
trace mutation, a learned cost model, and limited random exploration.

Neptune follows the same separation between a generative schedule program, candidate traces, search,
building, measurement, and persistence. MLIR Transform programs already encode deterministic
scheduling, so Neptune initially needs to trace only tuning decisions rather than every transform
operation.
