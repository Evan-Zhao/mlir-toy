#ifndef NEPTUNE_STABLEHLO_STABLEHLOTILINGINTERFACEIMPL_H
#define NEPTUNE_STABLEHLO_STABLEHLOTILINGINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
}

namespace neptune {

/// Attaches TilingInterface external models to supported StableHLO operations.
void registerStableHLOTilingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace neptune

#endif // NEPTUNE_STABLEHLO_STABLEHLOTILINGINTERFACEIMPL_H
