#ifndef HTILE_HTILEPASSES
#define HTILE_HTILEPASSES

include "mlir/Pass/PassBase.td"

def DotTransposeToLoadOrderPass : Pass<"htile-dot-transpose-to-load-order", ""> {
  let summary = "Fission htile.dot transposes and fold load-fed permutes";
  let description = [{
    Fission htile.dot transpose attributes into htile.permute operations and
    fold load-fed permutes into htile.load dimension_order attributes.
  }];
  let dependentDialects = ["htile::HTileDialect"];
}

#endif // HTILE_HTILEPASSES
