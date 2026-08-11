#ifndef TUNE_TUNETRANSFORMOPS
#define TUNE_TUNETRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Produces the default today. A materialization pass will replace this op with
// transform.param.constant for each tuning candidate.
def TuneSampleCategoricalOp
    : Op<Transform_Dialect, "tune.sample_categorical",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ParamProducerTransformOpTrait]> {
  let summary = "Declare a categorical integer tuning choice";

  let description = [{
    Declares a stable, finite tuning choice and produces its selected value as a
    transform parameter. Before candidate materialization is implemented, the
    operation produces `default_value`.

    ```mlir
    %block_m = transform.tune.sample_categorical "attention.block_m"
        candidates = [32, 64, 128] default = 128
        : !transform.param<i64>
    ```
  }];

  let arguments = (ins StrAttr:$choice_id,
                       DenseI64ArrayAttr:$candidates,
                       I64Attr:$default_value);
  let results = (outs TransformParamTypeInterface:$result);

  let assemblyFormat = [{
    $choice_id `candidates` `=` $candidates `default` `=` $default_value
    attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

// Selects one named sequence. Candidate materialization will rewrite this op
// to transform.include before the transform interpreter runs.
def TuneChooseSequenceOp
    : Op<Transform_Dialect, "tune.choose_sequence",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Declare a structural choice among named transform sequences";

  let description = [{
    Declares a labeled choice among named transform sequences with compatible
    signatures. `cases` maps stable decision labels to symbol references.

    ```mlir
    transform.tune.choose_sequence "backend" %kernel
        default = "triton"
        cases = {triton = @configure_triton, cutile = @configure_cutile}
        : (!transform.any_op) -> ()
    ```

    This operation must be rewritten to `transform.include` by candidate
    materialization before it is executed.
  }];

  let arguments = (ins Variadic<TransformHandleTypeInterface>:$inputs,
                       StrAttr:$choice_id,
                       StrAttr:$default_case,
                       DictionaryAttr:$cases);
  let results = (outs Variadic<TransformHandleTypeInterface>:$results);

  let assemblyFormat = [{
    $choice_id $inputs `default` `=` $default_case `cases` `=` $cases
    attr-dict `:` functional-type($inputs, $results)
  }];

  let hasVerifier = 1;
}

#endif // TUNE_TUNETRANSFORMOPS
