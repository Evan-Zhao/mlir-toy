# HTile CPU+GPU Codegen Plan

This note sketches the path from the current single-kernel HTile lowering to a mixed CPU+GPU
program with multiple device kernels, explicit temporary buffers, and runtime workspace allocation.

The immediate target is decode-input attention after SplitK update. That schedule naturally creates
multiple top-level loop nests in one function: one loop nest computes split-local partial tensors,
and later loop nests merge or post-process those tensors. Each top-level loop nest should become a
separate device kernel, while the enclosing function becomes a host-side schedule.

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

The first thrust should get the current SplitK decode attention schedule through multi-kernel HTile
lowering with the least machinery that can be correct for the cases at hand.

This is intentionally not a general bufferization/runtime solution. It should be narrow enough to
debug quickly and broad enough to prove the CPU+GPU split.

### Checklist

- [ ] Define `htile.kernel`, `htile.launch_func`, and `htile.return`.
- [ ] Add transform-driven outlining for selected top-level `scf.forall` loop nests.
- [ ] Allow mixed host functions with outlined kernels and remaining inline HTile code.
- [ ] Add a verifier or late legality check that rejects inline executable HTile code before final
      runtime lowering.
- [ ] Add targeted, stupid kernel-boundary bufferization for decode SplitK partial tensors.
- [ ] Add per-temp `htile.runtime.alloc` / `htile.runtime.free`.
- [ ] Teach `semantic_to_kernel_abi` to operate per `htile.kernel`.

### Kernel Ops

Introduce the minimal kernel boundary operations:

1. `htile.kernel`
   A symbol op with one device-code region. The region is isolated from above. Kernel operands are
   explicit block arguments. The body terminates with `htile.return`.

1. `htile.launch_func`
   A host op that references an `htile.kernel` symbol and passes explicit operands. Launch
   configuration should live primarily on the launch op; kernel attributes can record ABI or codegen
   assumptions.

1. `htile.return`
   Terminator for `htile.kernel`.

Initial restrictions:

- support only single-block kernel bodies,
- support operands only, no kernel results,
- require inter-kernel data to be passed through explicit temp buffers,
- outline only selected top-level `scf.forall` loop nests,
- allow mixed host functions during partial lowering.

### Implementation Contract

Thrust 1 starts from the HTile semantic form produced by the decode attention schedule: a
`func.func` that contains two or more top-level `scf.forall` loop nests, with HTile ops inside those
loops and tensor SSA values connecting the loop nests. The implementation should not try to infer
arbitrary GPU regions; the transform schedule must explicitly select each loop nest to outline.

For decode attention, outlining and the first stupid bufferization pass should move toward this
shape:

```mlir
func.func @attention(%q: tensor<1x4x1x64xf16>,
                     %k: tensor<1x4x1024x64xf16>,
                     %v: tensor<1x4x1024x64xf16>) -> tensor<1x4x1x64xf16> {
  // Split-local partials produced by the first forall. These correspond to the
  // current HTile semantic tensors with shapes tensor<4x16xf32>,
  // tensor<4x64x16xf32>, and tensor<4x16xf32>.
  %m_rf = htile.runtime.alloc : memref<4x16xf32, #htile.device>
  %acc_rf = htile.runtime.alloc : memref<4x64x16xf32, #htile.device>
  %l_rf = htile.runtime.alloc : memref<4x16xf32, #htile.device>

  htile.launch_func @decode_partial(%q, %k, %v, %m_rf, %acc_rf, %l_rf)
      {grid = [4, 16, 1], block = [...]}

  %out = htile.runtime.alloc : memref<4x64xf16, #htile.device>
  htile.launch_func @decode_merge(%m_rf, %acc_rf, %l_rf, %out)
      {grid = [4, 1, 1], block = [...]}

  %result = htile.runtime.to_tensor %out
      : memref<4x64xf16, #htile.device> -> tensor<1x4x1x64xf16>
  htile.runtime.free %l_rf
  htile.runtime.free %acc_rf
  htile.runtime.free %m_rf
  return %result : tensor<1x4x1x64xf16>
}

htile.kernel @decode_partial(%q: tensor<1x4x1x64xf16>,
                             %k: tensor<1x4x1024x64xf16>,
                             %v: tensor<1x4x1024x64xf16>,
                             %m_rf: memref<4x16xf32, #htile.device>,
                             %acc_rf: memref<4x64x16xf32, #htile.device>,
                             %l_rf: memref<4x16xf32, #htile.device>) {
  scf.forall (%head, %split) in (4, 16) {
    %k_offset = affine.apply affine_map<(d0) -> (d0 * 64)>(%split)
    %q_tile = htile.load %q[0, %head, 0, 0] : tensor<64xf16>
    %k_tile = htile.load %k[0, %head, %k_offset, 0] : tensor<64x64xf16>
    %scores = htile.dot %q_tile, %k_tile {transpose_b}
        : tensor<64xf16>, tensor<64x64xf16> -> tensor<64xf32>
    %m_part = htile.reduce %scores axis 0 kind "max"
        : tensor<64xf32> -> tensor<f32>
    %p_tile = math.exp2(%scores - broadcast(%m_part))
        : tensor<64xf32>
    %v_tile = htile.load %v[0, %head, %k_offset, 0] : tensor<64x64xf16>
    %acc_part = htile.dot %p_tile, %v_tile
        : tensor<64xf32>, tensor<64x64xf16> -> tensor<64xf32>
    %l_part = htile.reduce %p_tile axis 0 kind "sum"
        : tensor<64xf32> -> tensor<f32>
    htile.store %m_part, %m_rf[%head, %split] : tensor<f32>
    htile.store %acc_part, %acc_rf[%head, 0, %split] : tensor<64xf32>
    htile.store %l_part, %l_rf[%head, %split] : tensor<f32>
  }
  htile.return
}

htile.kernel @decode_merge(%m_rf: memref<4x16xf32, #htile.device>,
                           %acc_rf: memref<4x64x16xf32, #htile.device>,
                           %l_rf: memref<4x16xf32, #htile.device>,
                           %out: memref<4x64xf16, #htile.device>) {
  scf.forall (%head) in (4) {
    %m_parts = htile.load %m_rf[%head, 0] [1, 16] : tensor<16xf32>
    %m = htile.reduce %m_parts axis 0 kind "max"
        : tensor<16xf32> -> tensor<f32>
    %l_parts = htile.load %l_rf[%head, 0] [1, 16] : tensor<16xf32>
    %l_scale = math.exp2(%m_parts - broadcast(%m)) : tensor<16xf32>
    %l = htile.reduce (%l_parts * %l_scale) axis 0 kind "sum"
        : tensor<16xf32> -> tensor<f32>

    %acc_parts = htile.load %acc_rf[%head, 0, 0] [1, 64, 16]
        : tensor<64x16xf32>
    %acc_scale = htile.broadcast %l_scale dimensions = [0]
        : tensor<16xf32> -> tensor<64x16xf32>
    %acc = htile.reduce (%acc_parts * %acc_scale) axis 1 kind "sum"
        : tensor<64x16xf32> -> tensor<64xf32>
    %out_vec = arith.truncf (%acc / broadcast(%l))
        : tensor<64xf32> to tensor<64xf16>
    htile.store %out_vec, %out[%head, 0] : tensor<64xf16>
  }
  htile.return
}
```

`htile.launch_func` should not return tensor results in this thrust. Values that must survive past a
kernel boundary should be explicit output buffer operands. If a selected loop currently produces a
tensor consumed by a later loop, the stupid bufferization pass should create a device temp and pass it
to both the producing and consuming kernels.

The first implementation may reject anything outside the decode pattern: nested unselected kernel
regions, complex control flow around selected loops, nontrivial tensor aliasing, dynamic workspace
packing, and async launch ordering. These are later-thrust problems.

### Transform-Driven Outlining

Add an explicit transform op that outlines a selected payload loop nest into `htile.kernel` +
`htile.launch_func`.

Example shape:

```mlir
%launch, %kernel = transform.htile.outline_kernel %forall
    : (!any) -> (!any, !any)
```

The transform should:

1. compute live-ins with MLIR region utilities such as `getUsedValuesDefinedAbove`,
1. reject unsupported captures initially, or clone simple constants and index computations,
1. create a unique `htile.kernel @name`,
1. clone or move the selected loop nest into the kernel body with `IRMapping`,
1. replace the original loop nest with `htile.launch_func @name(...)`,
1. return handles for both the launch and kernel.

This should be schedule-driven rather than automatic discovery. The decode attention schedule already
knows which top-level loop nest should become each kernel.

### Stupid Bufferization

Add a targeted, manually implemented bufferization pass for the exact inter-kernel tensors produced
by SplitK decode attention.

The pass should:

1. identify tensors produced by one top-level kernel and consumed by a later top-level kernel,
1. allocate one device temp for each such tensor,
1. rewrite the producing kernel to write into that temp,
1. rewrite consuming kernels to read from that temp,
1. remove the cross-kernel tensor SSA value.

This pass can be conservative:

- no alias analysis,
- no copy minimization,
- no in-place analysis beyond obvious destination-style outputs,
- no support for complex control flow,
- no workspace packing,
- no async lifetime reasoning.

It only needs to handle the decode attention pattern: split-local partial tensors produced by the
first forall and consumed by the merge/post-processing forall.

### Simple Runtime Allocation

Use one allocation per inter-kernel temporary.

The first runtime surface can be:

1. `htile.runtime.alloc`
   Allocate one device buffer.

1. `htile.runtime.free`
   Free that buffer after the last launch that uses it.

No workspace allocation is required in this thrust. Allocation can be inefficient; the goal is to
make the dataflow explicit and executable.

### Per-Kernel ABI Lowering

Teach `semantic_to_kernel_abi` to operate on each `htile.kernel` body rather than assuming the whole
`func.func` is one kernel.

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
