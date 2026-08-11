# HTile CPU+GPU Codegen Plan

This note tracks the path from Semantic HTile to a mixed CPU+GPU program with multiple device
kernels, explicit temporary buffers, and runtime workspace allocation.

SplitK decode attention now reaches the multi-kernel HTile IR milestone: its top-level loop nests are
outlined as separate device kernels, and the enclosing function contains ordered launches with
explicit memref temporaries. Runtime allocation and executable host lowering are still outstanding.

## Goal

Lower an HTile semantic program into:

1. device kernel bodies that can be lowered to Triton, GPU dialect, or another tile-code backend,
1. a host schedule that launches those kernels in order,
1. explicit device temporary buffers for values that flow between kernels,
1. a runtime ABI that supports kernel launch, device allocation, deallocation, and eventually
   workspace allocation and reuse.

The work should proceed in thrusts. The first thrust should be intentionally narrow: get the decode
attention cases working with minimal abstractions and a deliberately simple bufferization strategy.
Later thrusts can replace the narrow machinery with more principled bufferization, allocation, and
runtime support.

## Design Model

The current `func.func` should become a host schedule once kernel outlining begins. After that point:

- `htile.kernel` is a symbol-like device code operation with an isolated body.
- `htile.launch_func` is a host-side launch operation that references an `htile.kernel`.
- `htile.runtime.alloc` / `htile.runtime.free` represent device temporary allocation lifetime.
- Inline HTile ops inside `func.func` are allowed only as a temporary partially-outlined state.
- Before final runtime lowering, all executable HTile code must be inside `htile.kernel` bodies.

This mirrors the useful parts of MLIR GPU lowering without reusing GPU dialect as the HTile
abstraction. In stock MLIR, `gpu.launch` regions are outlined to `gpu.func` and called by
`gpu.launch_func`; allocations and copies become calls to a small runtime wrapper layer such as
`mgpuMemAlloc`, `mgpuMemFree`, `mgpuMemcpy`, and `mgpuLaunchKernel`.

For HTile, we should reuse that structure conceptually while keeping the dialect independent:

```mlir
%tmp0 = htile.runtime.alloc ... : memref<4x16xf32, #htile.device>
%tmp1 = htile.runtime.alloc ... : memref<4x64x16xf32, #htile.device>

htile.launch_func @decode_partial(%q, %k, %v, %tmp0, %tmp1)
htile.launch_func @decode_merge(%tmp0, %tmp1, %out)

htile.runtime.free %tmp1
htile.runtime.free %tmp0
```

## Thrust 1: Minimal Decode Path

The first thrust gets the current SplitK decode attention schedule through multi-kernel HTile
lowering with the least machinery that can be correct for the cases at hand. Kernel outlining and
targeted boundary bufferization are implemented; runtime allocation and host legality remain.

This is intentionally not a general bufferization/runtime solution. It is narrow enough to debug
quickly and broad enough to prove the CPU+GPU split.

### Checklist

- [x] Define `htile.kernel`, `htile.launch_func`, and `htile.return`.
- [x] Add targeted bufferization across kernel (loop nest) boundaries.
- [x] Add kernel outlining for selected top-level loop nests.
- [ ] Add per-temp `htile.runtime.alloc` / `htile.runtime.free`.
- [ ] Add a verifier or late legality check that rejects inline executable HTile code before final
      runtime lowering.

### Kernel Ops

The minimal kernel boundary operations are now:

1. `htile.kernel @name(%arg0: type, ...) { ... }`
   Symbol-like device-code container with one isolated single-block region. Kernel operands are
   explicit block arguments and are expected not to be tensors in the launchable form. The body
   terminates with `htile.return`.

1. `htile.launch_func @name(%arg0, ...) {launch_attrs} : type(%arg0), ...`
   Host-side launch operation that references an `htile.kernel` symbol and passes explicit operands.
   Launch configuration should live primarily on the launch op; kernel attributes can record codegen
   assumptions. It is designed to have no data results, so tensor values crossing this boundary must
   already have been bufferized into explicit operands.

1. `htile.return`
   Result-free terminator for `htile.kernel`.

### Transform-Driven Outlining

`transform.htile.outline_kernels` is an explicit transform op that takes an ordered set of selected
top-level `scf.forall` loop nests and turns them into result-free `htile.launch_func` operations plus
`htile.kernel` definitions.

Example shape:

```mlir
%launches, %kernels = transform.htile.outline_kernels %foralls
    {kernel_names = ["decode_partial", "decode_merge"]}
    : (!any) -> (!any, !any)
```

The transform first performs targeted kernel-boundary bufferization:

1. Validate that all selected loops are top-level `scf.forall` ops in the same host `func.func`;
   use handle order as launch order and `kernel_names` or deterministic generated names.
1. Allocate one explicit memref temporary for each ranked tensor result of each selected forall and
   map both the forall result and its tied output block argument to that buffer.
1. Rewrite tensor reads inside selected foralls to read from explicit memrefs. `tensor.extract_slice`
   users become `htile.load`; whole-tensor reads use an `htile.load` at zero offsets. External
   tensor operands that do not already have a mapped buffer get a `bufferization.to_buffer`
   materialization before the forall.
1. Rewrite each result publication `tensor.parallel_insert_slice` to `htile.store`.
1. Rewrite remaining host-side uses of selected forall tensor results, such as `func.return`, to
   read the corresponding memref through `bufferization.to_tensor`.
1. Rebuild selected foralls without tensor `shared_outs` / tensor results, leaving their bodies
   connected to explicit memrefs instead of result SSA.

This is intentionally a local boundary rewrite, not general MLIR bufferization. It does not attempt
alias analysis, copy minimization, workspace packing, complex control-flow handling, or async
lifetime reasoning; those belong to later bufferization and allocation work.

After boundary bufferization, the transform outlines kernels:

1. Create one `htile.kernel` per selected loop and set `program_bounds` from the static normalized
   forall trip counts.
1. Replace the forall induction variables with `htile.program_id` values and clone the forall body,
   not the enclosing forall op, into the kernel body with `IRMapping`.
1. Legalize remaining captures: memrefs become explicit kernel block arguments and launch operands;
   `arith.constant` values are cloned into the kernel; other captures fail.
1. Replace each original loop with result-free `htile.launch_func`, carrying the same
   `program_bounds` metadata, and return handles for the launches and kernels.

This should be schedule-driven rather than automatic discovery. The decode attention schedule already
knows which top-level loop nests should become kernels. Because `htile.launch_func` has no data
results, kernel-boundary buffer materialization is not optional: values that survive a selected loop
boundary must be passed through explicit temporary buffers before the loop is replaced by a launch.

### Simple Runtime Allocation

Use one allocation per inter-kernel temporary.

The first runtime surface can be:

1. `htile.runtime.alloc`
   Allocate one device buffer.

1. `htile.runtime.free`
   Free that buffer after the last launch that uses it.

No workspace allocation is required in this thrust. Allocation can be inefficient; the goal is to
make the dataflow explicit and executable.

### Host/Kernels After Outlining

The host `func.func` should eventually contain only:

- runtime allocations/frees,
- `htile.launch_func` ops,
- shape/index computations needed to form launch arguments,
- final result handling.

A late legality check should reject executable HTile ops that remain inline in the host function.

### Thrust 1 Milestone

The target output for decode attention should be:

```mlir
func.func @attention(...) -> ... {
  %tmp_m = htile.runtime.alloc ...
  %tmp_l = htile.runtime.alloc ...
  %tmp_acc = htile.runtime.alloc ...

  htile.launch_func @decode_partial(..., %tmp_m, %tmp_l, %tmp_acc)
  htile.launch_func @decode_merge(%tmp_m, %tmp_l, %tmp_acc, %out)

  htile.runtime.free %tmp_acc
  htile.runtime.free %tmp_l
  htile.runtime.free %tmp_m
  return ...
}

htile.kernel @decode_partial(...) { ... }
htile.kernel @decode_merge(...) { ... }
```

This milestone does not need workspace packing, async execution, full One-Shot Bufferize support, or
a general automatic kernel-region discovery pass.

## Thrust 2: Principled Bufferization

After the decode path works, replace the narrow cross-kernel rewrite with a bufferization model that
is closer to MLIR's tensor-to-buffer infrastructure.

### Checklist

- [ ] Decide whether inter-kernel buffers are device-space `memref` values or custom HTile buffer
      values.
- [ ] Implement principled bufferization support around `htile.launch_func` / `htile.kernel`.
- [ ] Decide where MLIR One-Shot Bufferize fits and keep the Thrust 1 path as a fallback until this
      path is proven.
- [ ] Add a pass boundary where tensor SSA crossing launches is illegal.

MLIR bufferization provides:

- `bufferization.to_buffer` / `bufferization.to_tensor` for explicit tensor-buffer boundaries,
- `bufferization.alloc_tensor` for tensor values that must become allocations,
- One-Shot Bufferize for destination-style in-place and out-of-place decisions,
- ownership-based deallocation for ordinary memref lifetime management.

The second thrust should decide how much of that infrastructure to reuse directly.

Useful steps:

1. Define whether inter-kernel buffers are ordinary `memref` values with a device memory space or a
   custom HTile buffer type.
1. Implement `BufferizableOpInterface` for `htile.launch_func` if we want One-Shot Bufferize to
   reason through launch boundaries.
1. Implement bufferization support for `htile.kernel` signatures if kernel operands should be
   transformed from tensor types to buffer types by stock bufferization.
1. Use MLIR's alias/equivalence model to avoid unnecessary temp allocation and copies.
1. Introduce a pass boundary where all tensor SSA crossing launches is illegal.

This thrust should keep the simple Thrust 1 path as a fallback until the general bufferization path
is proven on the decode schedule.

## Thrust 3: Runtime ABI

Lower the HTile host schedule to a concrete runtime ABI.

### Checklist

- [ ] Define the concrete runtime C ABI for device allocation, free, kernel launch, and memcpy.
- [ ] Lower `htile.launch_func` and runtime allocation ops to that ABI.
- [ ] Decide whether HTile-to-GPU dialect remains a supported backend path.
- [ ] Keep HTile-to-Triton and HTile-to-GPU backend choices independent of the host schedule model.

Initial lowering can target a small C ABI, modeled after MLIR's GPU runtime wrapper layer:

```text
neptuneDeviceAlloc(size, stream) -> void*
neptuneDeviceFree(ptr, stream)
neptuneLaunchKernel(kernel_handle, grid, block, args, stream)
neptuneDeviceMemcpy(dst, src, size, stream)
```

The compiler should not emit CUDA/HIP/Triton runtime calls directly in many places. It should lower
to a small stable wrapper ABI and let the runtime implementation choose CUDA, HIP, Triton, or another
backend.

For Triton integration, the host schedule can eventually lower to calls into a Triton runtime
launcher with already-allocated device pointers. Triton normally expects an external framework or
runtime to own tensor allocation and pass device pointers into compiled kernels, which matches the
`htile.runtime.alloc` / `htile.launch_func` split.

For TVM-like integration, the host schedule can lower to a graph/runtime module where allocation
planning and kernel launch sequencing are runtime responsibilities. This is the model to follow for
workspace planning and buffer reuse.

This thrust should also decide whether an intermediate lowering through GPU dialect is worth keeping
as a backend option:

1. HTile-to-Triton
   Lower each `htile.kernel` body to Triton-style code and lower `htile.launch_func` to the Triton
   launch ABI.

1. HTile-to-GPU dialect
   Lower `htile.kernel` bodies to `gpu.module` / `gpu.func` and lower launches to
   `gpu.launch_func`. This may reuse MLIR's existing GPU runtime lowering, but it requires a
   meaningful lowering from HTile tile ops to GPU-compatible IR.

The host schedule and runtime-buffer model should not assume either backend.

## Thrust 4: Workspace Allocation

After per-temp allocation works, replace it with workspace allocation.

### Checklist

- [ ] Compute temporary byte sizes, alignment, and launch-level liveness.
- [ ] Replace per-temp allocation with workspace allocation and view planning.
- [ ] Support runtime-provided workspace buffers when useful.
- [ ] Add async-safe lifetime management if launches become asynchronous.

The compiler should compute:

1. each temporary's byte size and alignment,
1. first launch that writes it,
1. last launch that reads it,
1. whether lifetimes overlap,
1. a packed offset assignment in one workspace buffer.

The resulting IR should look like:

```mlir
%workspace = htile.runtime.workspace_alloc ... : !htile.workspace
%tmp0 = htile.runtime.workspace_view %workspace offset 0
%tmp1 = htile.runtime.workspace_view %workspace offset 1024

htile.launch_func @k0(..., %tmp0, %tmp1)
htile.launch_func @k1(%tmp0, %tmp1, ...)

htile.runtime.workspace_free %workspace
```

This should be a separate pass from kernel outlining. Outlining defines kernel boundaries;
bufferization defines which values need storage; workspace planning optimizes the storage.

This thrust can later add:

- allocation hoisting,
- reuse across repeated calls,
- runtime-provided workspace buffers,
- dynamic-size workspace planning,
- async-safe lifetime management.

## Thrust 5: General Scheduling Support

Once decode attention is working, add more general scheduling conveniences.

### Checklist

- [ ] Add general kernel outlining for remaining top-level loop nests.
- [ ] Improve capture handling for constants, affine/index computations, and shape computations.
- [ ] Add host/device legality checks.
- [ ] Add async/dependency support if needed by runtime scheduling.
- [ ] Add host/device copy support if inputs or outputs are not already device buffers.

Possible additions:

1. A pass or transform op that outlines all remaining top-level `scf.forall` nests in a selected
   function.
1. Better capture handling for constants, affine/index computations, and shape computations.
1. Verification that host functions contain only host-legal HTile/runtime ops.
1. Async tokens and dependency tracking between launches.
1. Optional copies between host and device memory when inputs or outputs are not already device
   buffers.

These should not block the first decode-attention milestone.

## Open Questions

1. Should `htile.kernel` accept tensor block arguments during early development, or should it require
   memref/device-buffer arguments from the start?

1. Should launch grid/block configuration be mandatory on every `htile.launch_func`, or can it be
   inferred from the outlined `scf.forall` shape?

1. Should inter-kernel buffers use ordinary `memref` with a device memory space, or a custom HTile
   buffer type?

1. Should the first runtime lowering target `gpu.alloc` / `gpu.launch_func`, or a custom Neptune
   runtime ABI directly?

1. What is the minimum ABI that the existing HTile kernel translator needs once a kernel is no
   longer the whole `func.func`?
