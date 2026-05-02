include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgExtFuseMapConsumerIntoLoopOp
    : Op<Transform_Dialect,
         "linalg_ext.fuse_map_consumer_into_loop",
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
