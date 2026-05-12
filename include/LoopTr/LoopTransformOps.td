include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LoopFuseIntoProducerOp : Op<Transform_Dialect, "loop.fuse_into_producer_op",
         [FunctionalStyleTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Fuse a consumer into a producer that is also a loop";

  let description = [{
    Fuse a single consumer operation into a producer that implements LoopLikeOpInterface.

    Delegates to upstream `scf::tileAndFuseConsumer`, which:
    - identifies operands of `consumer` defined by the containing loop,
    - locates the `tensor.insert_slice`/`tensor.parallel_insert_slice` that
      produces each such operand from within the loop body,
    - tiles the consumer to operate on those same slices in-place, and
    - adds the tiled result as an additional loop-carried value.

    The `producer_loop` handle is preserved and remapped to the rewritten loop.
    The `consumer_op` handle is consumed.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
                       TransformHandleTypeInterface:$producer_loop);
  let results = (outs TransformHandleTypeInterface:$fused_op);

  let assemblyFormat = "$consumer_op `into` $producer_loop attr-dict "
                       " `:` functional-type(operands, results)";

}

def LoopFuseReduceConsumerIntoForall : Op<Transform_Dialect, "loop.fuse_reduction_consumer_into_forall",
         [FunctionalStyleTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Fuse a reduction consumer into a forall producer, sequentializing one forall dimension";

  let description = [{
    Fuse a one-dimension reduction `linalg.generic` op into an `scf.forall`.

    Prerequisites:
    - The forall must have exactly one result (the producer tile).
    - The reduction consumer must take exactly one input, produced by the forall.

    The op removes that reduction-tiled dimension from the forall, adds the
    reduction result as a new shared_out, introduces an inner sequential
    `scf.for`, and computes both the producer tile and the reduction update
    inside that loop.  The outer forall issues only the final values.

    The `forall_loop` handle is preserved and remapped to the rewritten loop.
    The `consumer_op` handle is consumed. The new inner `scf.for` is returned
    explicitly because it did not exist before the transformation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
                       TransformHandleTypeInterface:$forall_loop);
  let results = (outs TransformHandleTypeInterface:$fused_op,
                      TransformHandleTypeInterface:$new_inner_loop);

  let assemblyFormat = "$consumer_op `into` $forall_loop attr-dict "
                       " `:` functional-type(operands, results)";
}

def LoopRURollingUpdateNextReduction
    : Op<Transform_Dialect, "match.loop_ru.rolling_update_next_reduction",
         [NavigationTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          MemoryEffectsOpInterface]> {
  let summary = "Find the nearest reduction consumer from a producer op via forward BFS";

  let description = [{
    Pure analysis op. Forward BFS from `producer_op` along def-use edges to find the nearest
    `linalg.generic` with at least one reduction iterator. Once found, walk
    backward from the reduction (bounded by the BFS visited set) to collect
    the elementwise ops on the path.

    Returns the reduction op and the elementwise chain (def-to-use order).
    Fails if no reduction is reachable.

    This analysis is a part of a larger "rolling-update" fusion transformation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_op);
  let results = (outs TransformHandleTypeInterface:$next_reduce_op,
                      TransformHandleTypeInterface:$elemwise_ops);

  let assemblyFormat = "$producer_op attr-dict `:` functional-type(operands, results)";
}

def LoopRUCloneFuseElemwise : Op<Transform_Dialect, "loop_ru.clone_fuse_elemwise",
         [FunctionalStyleTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Clone elementwise ops and force-fuse them into forall + inner for";

  let description = [{
    Create a duplicated version of each elementwise op (a "sidecar")
    and fuse it under an `scf.forall` outer loop, and then under an `scf.for` inner loop.
    Fusing into `scf.for` can change the semantics of the sidecar
    (when the elemwise reads from a reduction), while the original op remains unchanged.

    The transform also rebuilds both loops to carry and publish the full-tensor
    sidecar results as additional loop outputs. Those new loop results are not
    wired into the original users yet; they are only made available for a later
    repair step at the reduction frontier.

    All the three input handles are read-only (not consumed). The loops are changed (actually
    cloned and replaces the old ones) during fusion, and these handles are remapped to point
    to the new loops. The `elemwise_chain_ops` handle is preserved because the elemwise ops
    before fusion are just unchanged.
  }];

  let arguments = (ins TransformHandleTypeInterface:$elemwise_chain_ops,
                       TransformHandleTypeInterface:$outer_loop,
                       TransformHandleTypeInterface:$inner_loop);
  let results = (outs TransformHandleTypeInterface:$sidecar_chain_ops);

  let assemblyFormat = "$elemwise_chain_ops `into` $outer_loop `,` $inner_loop attr-dict "
                       " `:` functional-type(operands, results)";
}
