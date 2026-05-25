#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

using namespace mlir;

namespace mlir::transform {
namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message)

struct PredicateAtom {
  arith::CmpIOp cmp;
  bool liveWhenCmpIsTrue;
};

struct MatchedDeadSelect {
  SmallVector<PredicateAtom> atoms;
};

struct AffineScalarExpr {
  AffineExpr expr;
  SmallVector<Value> values;
};

class AffineScalarBuilder {
public:
  AffineScalarBuilder(MLIRContext *context, unsigned numIndexDims)
      : context(context), numIndexDims(numIndexDims) {}

  FailureOr<AffineExpr> getExpr(Value value) {
    if (auto index = value.getDefiningOp<linalg::IndexOp>())
      return getAffineDimExpr(index.getDim(), context);

    if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
      AffineMap map = apply.getAffineMap();
      SmallVector<AffineExpr> dimReplacements;
      SmallVector<AffineExpr> symbolReplacements;
      dimReplacements.reserve(map.getNumDims());
      symbolReplacements.reserve(map.getNumSymbols());

      for (Value operand : apply.getDimOperands()) {
        FailureOr<AffineExpr> replacement = getExpr(operand);
        if (failed(replacement))
          return failure();
        dimReplacements.push_back(*replacement);
      }
      for (Value operand : apply.getSymbolOperands()) {
        FailureOr<AffineExpr> replacement = getExpr(operand);
        if (failed(replacement))
          return failure();
        symbolReplacements.push_back(*replacement);
      }

      AffineExpr expr = map.getResult(0).replaceDimsAndSymbols(dimReplacements, symbolReplacements);
      return simplifyAffineMap(AffineMap::get(getNumDims(), getNumSymbols(), expr)).getResult(0);
    }

    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
      return getExpr(cast.getIn());

    if (auto add = value.getDefiningOp<arith::AddIOp>())
      return getBinaryExpr(add.getLhs(), add.getRhs(),
                           [](AffineExpr lhs, AffineExpr rhs) { return lhs + rhs; });

    if (auto sub = value.getDefiningOp<arith::SubIOp>())
      return getBinaryExpr(sub.getLhs(), sub.getRhs(),
                           [](AffineExpr lhs, AffineExpr rhs) { return lhs - rhs; });

    Attribute attr;
    if (matchPattern(value, m_Constant(&attr))) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return getAffineConstantExpr(intAttr.getInt(), context);
    }

    if (!value.getType().isIndex())
      return failure();

    return getAffineSymbolExpr(getOrAddValueSymbol(value), context);
  }

  AffineScalarExpr build(AffineExpr expr) const { return {expr, values}; }

  unsigned getNumIndexDims() const { return numIndexDims; }
  unsigned getNumDims() const { return numIndexDims; }
  unsigned getNumSymbols() const { return values.size(); }
  ArrayRef<Value> getValues() const { return values; }

private:
  template <typename Fn>
  FailureOr<AffineExpr> getBinaryExpr(Value lhsValue, Value rhsValue, Fn &&fn) {
    FailureOr<AffineExpr> lhs = getExpr(lhsValue);
    FailureOr<AffineExpr> rhs = getExpr(rhsValue);
    if (failed(lhs) || failed(rhs))
      return failure();
    return simplifyAffineMap(AffineMap::get(getNumDims(), getNumSymbols(), fn(*lhs, *rhs)))
        .getResult(0);
  }

  unsigned getOrAddValueSymbol(Value value) {
    auto it = valueSymbols.find(value);
    if (it != valueSymbols.end())
      return it->second;

    unsigned symbol = values.size();
    values.push_back(value);
    valueSymbols[value] = symbol;
    return symbol;
  }

  MLIRContext *context;
  unsigned numIndexDims;
  SmallVector<Value> values;
  DenseMap<Value, unsigned> valueSymbols;
};

struct AffineBound {
  AffineMap map;
  SmallVector<Value> operands;
};

std::string valueName(Value value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  value.printAsOperand(os, OpPrintingFlags().useLocalScope());
  return storage;
}

std::string formatAffineBound(AffineMap map, ValueRange operands) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  map.print(os);
  os << "(";
  llvm::interleaveComma(operands, os, [&](Value value) { os << valueName(value); });
  os << ")";
  return storage;
}

std::string formatLiveInterval(Value iv, std::optional<AffineBound> lower,
                               std::optional<AffineBound> upper) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "possible live interval for " << valueName(iv) << ": ";
  if (lower)
    os << valueName(iv) << " >= " << formatAffineBound(lower->map, lower->operands);
  else
    os << "unbounded below";
  os << ", ";
  if (upper)
    os << valueName(iv) << " <= " << formatAffineBound(upper->map, upper->operands);
  else
    os << "unbounded above";
  return storage;
}

LogicalResult addStaticIndexDomain(affine::FlatAffineValueConstraints &constraints,
                                   ArrayRef<int64_t> loopRanges) {
  for (auto [dim, range] : llvm::enumerate(loopRanges)) {
    if (ShapedType::isDynamic(range))
      return failure();
    constraints.addBound(presburger::BoundType::LB, dim, 0);
    constraints.addBound(presburger::BoundType::UB, dim, range - 1);
  }
  return success();
}

FailureOr<AffineBound> getProjectedBound(const AffineScalarExpr &expr, linalg::GenericOp generic,
                                         bool lowerBound) {
  MLIRContext *context = generic.getContext();
  unsigned numIndexDims = generic.getNumLoops();
  unsigned numSymbols = expr.values.size();

  SmallVector<std::optional<Value>> dimValues(numIndexDims, std::nullopt);
  for (Value value : expr.values)
    dimValues.push_back(value);

  affine::FlatAffineValueConstraints constraints(numIndexDims, numSymbols,
                                                 /*numLocals=*/0, dimValues);
  if (failed(addStaticIndexDomain(constraints, generic.getStaticLoopRanges())))
    return failure();

  AffineMap exprMap = simplifyAffineMap(AffineMap::get(numIndexDims, numSymbols, expr.expr));
  if (failed(constraints.composeMatchingMap(exprMap)))
    return failure();

  // `composeMatchingMap` adds the expression result as the leading dimension.
  // Projecting out the Linalg iteration dimensions computes a parametric bound
  // for the whole tile, expressed only in terms of surrounding SSA values.
  constraints.projectOut(/*pos=*/1, numIndexDims);

  AffineMap lbMap, ubMap;
  std::tie(lbMap, ubMap) =
      constraints.getLowerAndUpperBound(/*pos=*/0, /*offset=*/0, /*num=*/1,
                                        /*symStartPos=*/1, /*localExprs=*/{}, context,
                                        /*closedUB=*/true);
  if (!lbMap || !ubMap)
    return failure();

  SmallVector<Value> operands;
  constraints.getValues(/*start=*/1, constraints.getNumDimAndSymbolVars(), &operands);
  return AffineBound{lowerBound ? lbMap : ubMap, operands};
}

enum class RelationKind { LE, LT, GE, GT };

struct NecessaryLiveRelation {
  AffineBound lhs;
  AffineBound rhs;
  RelationKind kind;
};

LogicalResult collectPredicateAtoms(Value predicate, bool liveWhenPredicateIsTrue,
                                    SmallVectorImpl<PredicateAtom> &atoms) {
  if (auto andOp = predicate.getDefiningOp<arith::AndIOp>()) {
    if (!liveWhenPredicateIsTrue)
      return failure();
    if (failed(collectPredicateAtoms(andOp.getLhs(), liveWhenPredicateIsTrue, atoms)))
      return failure();
    return collectPredicateAtoms(andOp.getRhs(), liveWhenPredicateIsTrue, atoms);
  }

  auto cmp = predicate.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return failure();
  atoms.push_back(PredicateAtom{cmp, liveWhenPredicateIsTrue});
  return success();
}

FailureOr<NecessaryLiveRelation> getNecessaryLiveRelation(PredicateAtom atom,
                                                          linalg::GenericOp generic) {
  arith::CmpIPredicate predicate = atom.cmp.getPredicate();
  if (!atom.liveWhenCmpIsTrue)
    predicate = arith::invertPredicate(predicate);

  AffineScalarBuilder affineBuilder(generic.getContext(), generic.getNumLoops());
  FailureOr<AffineExpr> lhsExpr = affineBuilder.getExpr(atom.cmp.getLhs());
  FailureOr<AffineExpr> rhsExpr = affineBuilder.getExpr(atom.cmp.getRhs());
  if (failed(lhsExpr) || failed(rhsExpr))
    return failure();

  AffineScalarExpr lhs = {*lhsExpr, llvm::to_vector(affineBuilder.getValues())};
  AffineScalarExpr rhs = {*rhsExpr, llvm::to_vector(affineBuilder.getValues())};
  FailureOr<AffineBound> lhsLower = getProjectedBound(lhs, generic, /*lowerBound=*/true);
  FailureOr<AffineBound> lhsUpper = getProjectedBound(lhs, generic, /*lowerBound=*/false);
  FailureOr<AffineBound> rhsLower = getProjectedBound(rhs, generic, /*lowerBound=*/true);
  FailureOr<AffineBound> rhsUpper = getProjectedBound(rhs, generic, /*lowerBound=*/false);
  if (failed(lhsLower) || failed(lhsUpper) || failed(rhsLower) || failed(rhsUpper))
    return failure();

  switch (predicate) {
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return NecessaryLiveRelation{*lhsLower, *rhsUpper, RelationKind::LE};
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return NecessaryLiveRelation{*lhsLower, *rhsUpper, RelationKind::LT};
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return NecessaryLiveRelation{*lhsUpper, *rhsLower, RelationKind::GE};
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return NecessaryLiveRelation{*lhsUpper, *rhsLower, RelationKind::GT};
  default:
    return failure();
  }
}

LogicalResult addNecessaryLiveConstraint(const NecessaryLiveRelation &relation,
                                         affine::FlatAffineValueConstraints &constraints) {
  AffineMap alignedLhs = constraints.computeAlignedMap(relation.lhs.map, relation.lhs.operands);
  AffineMap alignedRhs = constraints.computeAlignedMap(relation.rhs.map, relation.rhs.operands);

  AffineExpr diff;
  int64_t lowerBound;
  switch (relation.kind) {
  case RelationKind::LE:
    diff = alignedRhs.getResult(0) - alignedLhs.getResult(0);
    lowerBound = 0;
    break;
  case RelationKind::LT:
    diff = alignedRhs.getResult(0) - alignedLhs.getResult(0);
    lowerBound = 1;
    break;
  case RelationKind::GE:
    diff = alignedLhs.getResult(0) - alignedRhs.getResult(0);
    lowerBound = 0;
    break;
  case RelationKind::GT:
    diff = alignedLhs.getResult(0) - alignedRhs.getResult(0);
    lowerBound = 1;
    break;
  }

  AffineMap diffMap = simplifyAffineMap(
      AffineMap::get(/*dimCount=*/1, /*symbolCount=*/constraints.getNumSymbolVars(), diff));
  if (failed(constraints.composeMatchingMap(diffMap)))
    return failure();
  constraints.addBound(presburger::BoundType::LB, /*pos=*/0, lowerBound);
  constraints.projectOut(/*pos=*/0);
  return success();
}

FailureOr<std::pair<std::optional<AffineBound>, std::optional<AffineBound>>>
getIvIntervalFromRelations(ArrayRef<NecessaryLiveRelation> relations, Value iv) {
  llvm::SetVector<Value> operandSet;
  for (const NecessaryLiveRelation &relation : relations) {
    for (Value value : relation.lhs.operands)
      operandSet.insert(value);
    for (Value value : relation.rhs.operands)
      operandSet.insert(value);
  }
  if (!operandSet.contains(iv))
    return failure();

  SmallVector<Value> operands;
  operands.push_back(iv);
  for (Value value : operandSet)
    if (value != iv)
      operands.push_back(value);

  SmallVector<std::optional<Value>> dimAndSymbolValues;
  dimAndSymbolValues.reserve(operands.size());
  for (Value value : operands)
    dimAndSymbolValues.push_back(value);

  affine::FlatAffineValueConstraints constraints(/*numDims=*/1,
                                                 /*numSymbols=*/operands.size() - 1,
                                                 /*numLocals=*/0, dimAndSymbolValues);
  for (const NecessaryLiveRelation &relation : relations) {
    if (failed(addNecessaryLiveConstraint(relation, constraints)))
      return failure();
  }

  unsigned ivPos;
  if (!constraints.findVar(iv, &ivPos))
    return failure();

  AffineMap lbMap, ubMap;
  std::tie(lbMap, ubMap) =
      constraints.getLowerAndUpperBound(/*pos=*/ivPos, /*offset=*/0, /*num=*/1,
                                        /*symStartPos=*/1, /*localExprs=*/{}, iv.getContext(),
                                        /*closedUB=*/true);

  SmallVector<Value> boundOperands;
  for (unsigned i = 0, e = constraints.getNumDimAndSymbolVars(); i < e; ++i) {
    if (i == ivPos)
      continue;
    if (!constraints.hasValue(i))
      return failure();
    boundOperands.push_back(constraints.getValue(i));
  }

  std::optional<AffineBound> lower;
  std::optional<AffineBound> upper;
  if (lbMap)
    lower = AffineBound{lbMap, boundOperands};
  if (ubMap)
    upper = AffineBound{ubMap, boundOperands};
  return std::make_pair(lower, upper);
}

bool isFalseConstant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return false;
  auto boolAttr = dyn_cast<BoolAttr>(attr);
  return boolAttr && !boolAttr.getValue();
}

bool floatAttrsBitwiseEqual(FloatAttr lhs, FloatAttr rhs) {
  return lhs.getValue().bitwiseIsEqual(rhs.getValue());
}

bool attrsEqualByValue(Attribute lhs, Attribute rhs) {
  if (lhs == rhs)
    return true;

  auto lhsFloat = dyn_cast<FloatAttr>(lhs);
  auto rhsFloat = dyn_cast<FloatAttr>(rhs);
  if (lhsFloat && rhsFloat)
    return floatAttrsBitwiseEqual(lhsFloat, rhsFloat);

  return false;
}

std::optional<Attribute> getSplatConstantAttr(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;

  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    if (!dense.isSplat())
      return std::nullopt;
    return dense.getSplatValue<Attribute>();
  }

  return attr;
}

std::optional<Attribute> getConstantAttrForScalarValue(linalg::GenericOp generic, Value value) {
  if (auto constant = getSplatConstantAttr(value))
    return constant;

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != generic.getBlock())
    return std::nullopt;

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber >= generic.getNumDpsInputs())
    return std::nullopt;

  SmallVector<OpOperand *> inputOperands = generic.getDpsInputOperands();
  OpOperand *inputOperand = inputOperands[argNumber];
  return getSplatConstantAttr(inputOperand->get());
}

bool scalarValueEqualsAttr(linalg::GenericOp generic, Value value, Attribute expected) {
  std::optional<Attribute> actual = getConstantAttrForScalarValue(generic, value);
  return actual && attrsEqualByValue(*actual, expected);
}

FailureOr<Value> getSufficientLivePredicate(Value predicate) {
  auto select = predicate.getDefiningOp<arith::SelectOp>();
  if (!select)
    return predicate;

  if (!isFalseConstant(select.getFalseValue()))
    return predicate;

  // If `select(%cmp, maybe_mask, false)` is false whenever `%cmp` is false,
  // then `%cmp` is a sufficient live predicate for proving all-dead tiles.
  return select.getCondition();
}

FailureOr<MatchedDeadSelect> matchDeadSelect(linalg::GenericOp generic, Attribute deadValue) {
  auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
  if (yield.getNumOperands() != 1)
    return failure();

  auto select = yield.getOperand(0).getDefiningOp<arith::SelectOp>();
  if (!select)
    return failure();

  bool trueIsDead = scalarValueEqualsAttr(generic, select.getTrueValue(), deadValue);
  bool falseIsDead = scalarValueEqualsAttr(generic, select.getFalseValue(), deadValue);
  if (trueIsDead == falseIsDead)
    return failure();

  FailureOr<Value> livePredicate = getSufficientLivePredicate(select.getCondition());
  if (failed(livePredicate))
    return failure();

  SmallVector<PredicateAtom> atoms;
  if (failed(collectPredicateAtoms(*livePredicate, /*liveWhenPredicateIsTrue=*/!trueIsDead, atoms)))
    return failure();
  return MatchedDeadSelect{atoms};
}

FailureOr<std::pair<std::optional<AffineBound>, std::optional<AffineBound>>>
derivePossibleLiveInterval(linalg::GenericOp generic, scf::ForOp loop,
                           const MatchedDeadSelect &match) {
  SmallVector<NecessaryLiveRelation> relations;
  relations.reserve(match.atoms.size());
  for (const PredicateAtom &atom : match.atoms) {
    FailureOr<NecessaryLiveRelation> relation = getNecessaryLiveRelation(atom, generic);
    if (failed(relation))
      return failure();
    relations.push_back(*relation);
  }

  if (relations.empty())
    return failure();
  return getIvIntervalFromRelations(relations, loop.getInductionVar());
}

} // namespace

void LoopSpecializeDeadTileOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerOpMutable(), effects);
  onlyReadsHandle(getLoopMutable(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure LoopSpecializeDeadTileOp::apply(TransformRewriter &rewriter,
                                                            TransformResults &transformResults,
                                                            TransformState &state) {
  (void)rewriter;
  (void)transformResults;
  auto transform = cast<TransformOpInterface>(getOperation());

  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getProducerOp, "producer", producer,
                               linalg::GenericOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getLoop, "loop", loop, scf::ForOp);

  FailureOr<MatchedDeadSelect> match = matchDeadSelect(producer, getDeadValue());
  if (failed(match))
    BAIL("expected producer to yield select(live_predicate, live_value, dead_value)");

  FailureOr<std::pair<std::optional<AffineBound>, std::optional<AffineBound>>> interval =
      derivePossibleLiveInterval(producer, loop, *match);
  if (failed(interval))
    BAIL("failed to derive an affine possible-live interval for the loop IV");

  loop.emitRemark() << formatLiveInterval(loop.getInductionVar(), interval->first,
                                          interval->second);
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
