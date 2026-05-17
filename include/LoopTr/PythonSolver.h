#ifndef LOOPTR_PYTHON_SOLVER_H
#define LOOPTR_PYTHON_SOLVER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace mlir {

/// Serializes the expression rooted at `output` to the solver JSON format.
///
/// Walks the value use-to-def chain starting from `output`. If the walk reaches
/// a value present in `variableNames`, emits a JSON variable node with the
/// corresponding name and does not recurse through that value's definition.
/// Fails if the walk would need to cross outside the body nested under `scope`
/// without first being stopped by `variableNames`, or if it encounters an
/// unsupported operation while building the JSON expression.
FailureOr<llvm::json::Value>
serializeMLIRExprToJSON(Value output, const llvm::DenseMap<Value, std::string> &variableNames,
                        Operation *scope);

struct DeserializedValueExpr {
  Value result;
  llvm::StringMap<Value> variablesByName;
};

/// Reconstructs an MLIR expression from the solver JSON format.
///
/// Materializes the expression using `rewriter` at `loc`, returns the final
/// result value of the reconstructed expression, and records the MLIR values
/// corresponding to any JSON variable names referenced by the expression.
/// The exact strategy for creating or looking up those variable values is left
/// to the implementation.
FailureOr<DeserializedValueExpr> deserializeMLIRExprFromJSON(const llvm::json::Value &expr,
                                                             RewriterBase &rewriter, Location loc);

/// Calls the Python rolling-update solver on the `g` expression (serialized) and returns the
/// result as a JSON value (that can be deserialized next).
llvm::Expected<llvm::json::Value> solveRollingUpdaterWithPython(const llvm::json::Value &gExpr);

} // namespace mlir

#endif // LOOPTR_PYTHON_SOLVER_H
