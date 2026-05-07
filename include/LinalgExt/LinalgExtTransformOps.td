include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgExtFuseElemwiseIntoProducerOp
    : Op<Transform_Dialect,
         "linalg_ext.fuse_elemwise_into_producer",
         [FunctionalStyleTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          MemoryEffectsOpInterface]> {
  let summary = "Fuse a unary elementwise consumer into an scf.for tile loop";

  let description = [{
    Fuses a single unary tensor `linalg.map` consumer into an `scf.for` or
    `scf.forall` loop that materializes tiles of the consumer input.

    This is a deliberately narrow reverse-fusion primitive for the
    FlashAttention score-scale epilogue:

    ```
    qk_tile = ...
    score_tile = map(qk_tile)
    insert_slice score_tile into loop result
    ```

    The op expects the containing loop result consumed by the map to be
    materialized through either a `tensor.insert_slice` (`scf.for`) or a
    `tensor.parallel_insert_slice` (`scf.forall`) whose source is the producer
    tile. It clones the `linalg.map` body on that tile, rewires the insert
    source to the mapped tile, replaces uses of the original full-tensor
    consumer result with the producer container result, and erases the original
    consumer when dead.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
                       TransformHandleTypeInterface:$containing_loop);
  let results = (outs TransformHandleTypeInterface:$fused_op,
                      TransformHandleTypeInterface:$new_containing_loop);

  let assemblyFormat = "$consumer_op `into` $containing_loop attr-dict "
                       " `:` functional-type(operands, results)";

}

def LinalgExtFuseReductionConsumerIntoForallOp
    : Op<Transform_Dialect,
         "linalg_ext.fuse_reduction_consumer_into_forall",
         [FunctionalStyleTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          MemoryEffectsOpInterface]> {
  let summary = "Fuse a unary reduction consumer by sequentializing one forall dimension";

  let description = [{
    Fuses a single unary tensor reduction `linalg.generic` consumer into an
    `scf.forall` loop that materializes tiles of the consumer input.

    This is a narrow reverse-fusion primitive for the FlashAttention row-max
    stage. It rewrites the target `scf.forall` by removing the tiled dimension
    that corresponds to the reduction axis, introduces an inner sequential
    `scf.for`, computes the producer tiles and reduction update inside that
    loop, and publishes the completed producer slice and reduced tile once per
    outer `scf.forall` instance.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
                       TransformHandleTypeInterface:$containing_loop);
  let results = (outs TransformHandleTypeInterface:$fused_op,
                      TransformHandleTypeInterface:$new_containing_loop,
                      TransformHandleTypeInterface:$new_inner_loop);

  let assemblyFormat = "$consumer_op `into` $containing_loop attr-dict "
                       " `:` functional-type(operands, results)";
}

def LinalgExtRollingUpdateNextReductionOp
    : Op<Transform_Dialect,
         "match.linalg_ext.rolling_update_next_reduction",
         [NavigationTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          MemoryEffectsOpInterface]> {
  let summary = "Find the nearest reduction consumer from a producer op via forward BFS";

  let description = [{
    Pure analysis op. Performs a forward BFS on the def-use graph starting from
    `producer_op` (each edge has unit length) and locates the first
    reduction-like op encountered — i.e., the reduction at minimum BFS distance.

    Returns the located reduction op; and the elementwise ops between it and the producer,
    sorted in def-to-use topological order.

    Fails if no reduction is reachable from `producer_op`.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_op);
  let results = (outs TransformHandleTypeInterface:$next_reduce_op,
                      TransformHandleTypeInterface:$elemwise_ops);

  let assemblyFormat = "$producer_op attr-dict `:` functional-type(operands, results)";
}

def LinalgExtRollingUpdateForceFuseElemwise
    : Op<Transform_Dialect,
         "linalg_ext.rolling_update_force_fuse_elemwise",
         [FunctionalStyleTransformOpTrait,
          DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Fuse elementwise into forall + inner for, returning new handles";

  let description = [{
    Fuse an elementwise consumer into a tiled loop nest.

    Takes an elementwise op, an `scf.forall` outer loop, and an `scf.for` inner
    loop. Performs:

      1. Fuse the elementwise op into the `scf.forall` via upstream
         `scf::tileAndFuseConsumer`.  The inner loop pointer remains valid
         because `mergeBlocks` moves (not clones) the body.
      2. Apply canonicalization to unify redundant `affine.apply` ops.
      3. Clone the fused elementwise op into the inner `scf.for` as a sidecar.

    Returns the sidecar ops, the new `scf.forall`, and the new inner `scf.for`.

    The outer and inner loop handles are read-only — they are not consumed by
    this op.  The caller is responsible for using the output handles instead.
  }];

  let arguments = (ins TransformHandleTypeInterface:$elemwise_chain_ops,
                       TransformHandleTypeInterface:$outer_loop,
                       TransformHandleTypeInterface:$inner_loop);
  let results = (outs TransformHandleTypeInterface:$sidecar_chain_ops,
                      TransformHandleTypeInterface:$new_outer_loop,
                      TransformHandleTypeInterface:$new_inner_loop);

  let assemblyFormat = "$elemwise_chain_ops `into` $outer_loop `,` $inner_loop attr-dict "
                       " `:` functional-type(operands, results)";
}
