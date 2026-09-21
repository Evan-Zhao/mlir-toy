# Mamba Selective-Scan Pipeline

This note complements the [attention pipeline](attention-pipeline.md) by describing only the parts
of Mamba lowering that differ from attention. The shared scheduled-L1, Semantic HTile, Kernel HTile,
and backend contracts are documented in the [HTile guide](htile-guide.md).

## Supported Operator

Neptune currently supports the real-valued Mamba-1 selective scan, not a complete Mamba block. For
each batch item, token, and expanded model channel, it computes

```text
dt_t = softplus(delta_t + delta_bias)
h_t  = exp(dt_t * A) * h_(t-1) + dt_t * b_t * u_t
y_t  = sum(c_t * h_t) + D * u_t
```

Activations use `[batch, sequence, channels]` storage. `A` and the recurrent state add a final
`state_dim` axis; `b` and `c` vary by batch and token but are shared across channels. Recurrence
arithmetic and state are `f32`, while activation storage may be `bf16`, `f16`, or `f32`.

Input projection, depthwise convolution, generation of `delta`/`b`/`c`, gating, and output
projection remain outside the supported operator.

## Scheduling Differences From Attention

The exported StableHLO contains a sequential `stablehlo.while`, rather than attention's pair of
contractions and softmax. The Mamba schedule:

1. normalizes the StableHLO loop to `scf.for`,
2. tiles the output over `(batch, channel-block)`,
3. fuses the per-token update and projection into that worker tile,
4. interchanges the worker loop with the time loop, producing one persistent worker per tile, and
5. carries a `[block_channels, state_dim]` state tile through the inner sequential loop.

The resulting kernel grid is `(batch, channels / block_channels)`. Unlike attention, each worker
owns its recurrent state for the full sequence and does not stream K/V tiles or maintain online
softmax state.

The output projection is a matrix-vector `htile.dot`. Triton and cuTile lower it to a broadcasted
multiply followed by a reduction; TileLang emits a parallel output loop with a serial reduction.
Logical scalar memrefs use one-element physical buffers where a backend cannot represent rank-zero
buffers.

## Export Stages And Constraints

For example:

```shell
neptune-export mamba --stage htile
neptune-export mamba --stage triton-ptx
neptune-export mamba --stage tilelang-cuda
neptune-export mamba --stage cutile
```

The source stages `triton` and `tilelang` are also available. Shapes are static. The expanded
channel count (`model_dim * expand`) must be divisible by `block_channels`, and the block size must
be a positive multiple of 16. The default block size is 128.

The outlined kernel contains buffers for the zero initial state and scalar zero in addition to the
seven public operator inputs and output. These are implementation details of the current outlining
path, not additions to the selective-scan operator ABI.

## Tests And Remaining Scope

[`test/Pipeline/jax_mamba_selective_scan.mlir`](../test/Pipeline/jax_mamba_selective_scan.mlir)
checks the scheduled recurrence and Kernel HTile structure. Python tests in
[`test/python/test_pipeline.py`](../test/python/test_pipeline.py) compile and run the generated
Triton, cuTile, and TileLang kernels against a deterministic reference implementation.

Current backend tests use one small `bf16` case. Additional activation dtypes, tile sizes, and shape
combinations remain useful coverage. See the [backend test coverage](backend-test-coverage.md) for
coverage across all exported operators.
