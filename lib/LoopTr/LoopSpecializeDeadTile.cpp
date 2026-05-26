#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>
#include <variant>

using namespace mlir;

namespace mlir::transform {
namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message)

std::optional<Attribute> asSplatConstantAttr(Value value);
bool attrsEqualByValue(Attribute lhs, Attribute rhs);

template <typename T, typename Func>
FailureOr<SmallVector<T>> mapFAggFailure(ValueRange vec, Func &&func) {
  SmallVector<T> results;
  results.reserve(vec.size());
  for (Value value : vec) {
    auto result = func(value);
    if (failed(result))
      return failure();
    results.push_back(*result);
  }
  return results;
}

struct PredicateAtom {
  arith::CmpIOp cmp;
  bool liveWhenCmpIsTrue;
};

struct MatchedDeadSelect {
  SmallVector<PredicateAtom> atoms;
  Value liveValue;
  std::optional<unsigned> liveInputOperandNumber;
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
      auto self = [&](Value operand) { return getExpr(operand); };
      auto dimReplacements = mapFAggFailure<AffineExpr>(apply.getDimOperands(), self);
      auto symbolReplacements = mapFAggFailure<AffineExpr>(apply.getSymbolOperands(), self);
      if (failed(dimReplacements) || failed(symbolReplacements))
        return failure();

      AffineExpr expr =
          map.getResult(0).replaceDimsAndSymbols(*dimReplacements, *symbolReplacements);
      return simplifyAffineMap(AffineMap::get(getNumDims(), getNumSymbols(), expr)).getResult(0);
    }

#define RETURN_BINARY_EXPR(binOp, operator)                                                        \
  {                                                                                                \
    auto lhs = getExpr((binOp).getLhs()), rhs = getExpr((binOp).getRhs());                         \
    if (failed(lhs) || failed(rhs))                                                                \
      return failure();                                                                            \
    return simplifyAffineMap(AffineMap::get(getNumDims(), getNumSymbols(), (*lhs) operator(*rhs))) \
        .getResult(0);                                                                             \
  }

    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
      return getExpr(cast.getIn());
    if (auto add = value.getDefiningOp<arith::AddIOp>())
      RETURN_BINARY_EXPR(add, +);
    if (auto sub = value.getDefiningOp<arith::SubIOp>())
      RETURN_BINARY_EXPR(sub, -);

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

  AffineBound offset(int64_t delta) const {
    MLIRContext *context = map.getContext();
    AffineExpr expr = map.getResult(0) + getAffineConstantExpr(delta, context);
    return {simplifyAffineMap(AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr)),
            operands};
  }

  static FailureOr<AffineBound> project(const AffineScalarExpr &expr, linalg::GenericOp generic,
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

    auto [lbMap, ubMap] =
        constraints.getLowerAndUpperBound(/*pos=*/0, /*offset=*/0, /*num=*/1,
                                          /*symStartPos=*/1, /*localExprs=*/{}, context,
                                          /*closedUB=*/true);
    if (!lbMap || !ubMap)
      return failure();

    SmallVector<Value> operands;
    constraints.getValues(/*start=*/1, constraints.getNumDimAndSymbolVars(), &operands);
    return AffineBound{lowerBound ? lbMap : ubMap, operands};
  }

private:
  static LogicalResult addStaticIndexDomain(affine::FlatAffineValueConstraints &constraints,
                                            ArrayRef<int64_t> loopRanges) {
    for (auto [dim, range] : llvm::enumerate(loopRanges)) {
      if (ShapedType::isDynamic(range))
        return failure();
      constraints.addBound(presburger::BoundType::LB, dim, 0);
      constraints.addBound(presburger::BoundType::UB, dim, range - 1);
    }
    return success();
  }
};

template <typename T, std::enable_if_t<!std::is_same_v<T, Attribute>, int> = 0>
struct AbstractValueData {
  std::variant<Attribute, T, std::nullopt_t> value;

  static AbstractValueData getUnknown() { return {std::nullopt}; }
  static AbstractValueData getConstant(Attribute attr) { return {attr}; }
  static AbstractValueData getEquivalentTo(T value) { return {value}; }

  AbstractValueData getZeroLike() {
    auto attr = getConstantAttr();
    if (auto floatAttr = dyn_cast<FloatAttr>(*attr))
      return getConstant(FloatAttr::get(floatAttr.getType(), 0.0));
    if (auto intAttr = dyn_cast<IntegerAttr>(*attr))
      return getConstant(IntegerAttr::get(intAttr.getType(), 0));
    return getUnknown();
  }

  bool isKnown() const { return !std::holds_alternative<std::nullopt_t>(value); }

  std::optional<Attribute> getConstantAttr() const {
    return std::holds_alternative<Attribute>(value) ? std::make_optional(std::get<Attribute>(value))
                                                    : std::nullopt;
  }
  std::optional<T> getEquivalentValue() const {
    return std::holds_alternative<T>(value) ? std::make_optional(std::get<T>(value)) : std::nullopt;
  }

  bool isConstZero() const { return mapConstAttribute(isZeroAttr); }
  bool isConstOne() const { return mapConstAttribute(isOneAttr); }

  bool isConstBool(bool expectedValue) const {
    return mapConstAttribute([&](Attribute attr) {
      auto boolAttr = dyn_cast<BoolAttr>(attr);
      return boolAttr && boolAttr.getValue() == expectedValue;
    });
  }

  bool isNegativeInfinity() const {
    return mapConstAttribute([](Attribute attr) {
      auto floatAttr = dyn_cast<FloatAttr>(attr);
      return floatAttr && floatAttr.getValue().isInfinity() && floatAttr.getValue().isNegative();
    });
  }

  bool bitwiseEqualToAttr(Attribute rhs) const {
    return mapConstAttribute([&](Attribute attr) { return attrsEqualByValue(attr, rhs); });
  }

private:
  bool mapConstAttribute(std::function<bool(Attribute)> &&predicate) const {
    auto attr = getConstantAttr();
    return attr && predicate(*attr);
  }

  static bool isZeroAttr(Attribute attr) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return floatAttr.getValue().isZero();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isZero();
    return false;
  }

  static bool isOneAttr(Attribute attr) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return floatAttr.getValue().isExactlyValue(1.0);
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isOne();
    return false;
  }
};

using ScalarExprState = AbstractValueData<OpOperand *>;
using AbstractValue = AbstractValueData<Value>;

struct AffineInterval {
  // Semantics: [lower, upper), with absent endpoints meaning unbounded.
  std::optional<AffineBound> lower{}, upper{};
  // We don't have a good way to represent an empty interval with just lower and upper.
  // (A {std::nullopt, std::nullopt} interval would technically mean (-∞, +∞).)
  // Use the `empty` flag to represent an empty interval.
  bool empty{false};
};

FailureOr<Value> materializeAffineBound(RewriterBase &rewriter, Location loc,
                                        const AffineBound &bound) {
  if (!bound.map || bound.map.getNumResults() != 1)
    return failure();
  return affine::AffineApplyOp::create(rewriter, loc, bound.map, bound.operands).getResult();
}

Value selectMinOrMaxIndexValue(RewriterBase &rewriter, Location loc, Value lhs, Value rhs,
                               bool isMax) {
  arith::CmpIPredicate predicate = isMax ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::slt;
  Value lhsWins = arith::CmpIOp::create(rewriter, loc, predicate, lhs, rhs).getResult();
  return arith::SelectOp::create(rewriter, loc, lhsWins, lhs, rhs).getResult();
}

FailureOr<Value> materializeIntervalUpperBound(RewriterBase &rewriter, scf::ForOp loop,
                                               const AffineInterval &interval) {
  Location loc = loop.getLoc();
  Value value = loop.getUpperBound();
  if (interval.empty)
    value = loop.getLowerBound();
  else if (interval.upper)
    value = *materializeAffineBound(rewriter, loc, *interval.upper);
  value = selectMinOrMaxIndexValue(rewriter, loc, value, loop.getLowerBound(), /*isMax=*/true);
  value = selectMinOrMaxIndexValue(rewriter, loc, value, loop.getUpperBound(), /*isMax=*/false);
  return value;
}

using LoopCloneCustomizer =
    llvm::function_ref<FailureOr<bool>(RewriterBase &, IRMapping &, Operation &)>;

FailureOr<scf::ForOp> cloneForWithBody(RewriterBase &rewriter, scf::ForOp source, Value lowerBound,
                                       Value upperBound, ValueRange initArgs,
                                       LoopCloneCustomizer customizer = nullptr) {
  auto clonedLoop = scf::ForOp::create(rewriter, source.getLoc(), lowerBound, upperBound,
                                       source.getStep(), initArgs);
  Block &sourceBlock = source.getRegion().front();
  Block &targetBlock = clonedLoop.getRegion().front();

  IRMapping mapping;
  mapping.map(source.getInductionVar(), clonedLoop.getInductionVar());
  for (auto [oldArg, newArg] :
       llvm::zip_equal(source.getRegionIterArgs(), clonedLoop.getRegionIterArgs()))
    mapping.map(oldArg, newArg);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&targetBlock);
  for (Operation &op : sourceBlock.without_terminator()) {
    if (customizer) {
      FailureOr<bool> handled = customizer(rewriter, mapping, op);
      if (failed(handled))
        return failure();
      if (*handled)
        continue;
    }
    rewriter.clone(op, mapping);
  }

  auto sourceYield = cast<scf::YieldOp>(sourceBlock.getTerminator());
  SmallVector<Value> yieldedValues;
  yieldedValues.reserve(sourceYield.getNumOperands());
  for (Value operand : sourceYield.getOperands())
    yieldedValues.push_back(mapping.lookupOrDefault(operand));
  rewriter.setInsertionPointToEnd(&targetBlock);
  scf::YieldOp::create(rewriter, sourceYield.getLoc(), yieldedValues);
  return clonedLoop;
}

AbstractValue resolveAbstractValue(AbstractValue state,
                                   const DenseMap<Value, AbstractValue> &states) {
  SmallPtrSet<void *, 8> visited;
  while (auto equivalent = state.getEquivalentValue()) {
    if (!visited.insert(equivalent->getAsOpaquePointer()).second)
      break;

    if (auto attr = asSplatConstantAttr(*equivalent))
      return AbstractValue::getConstant(*attr);

    auto it = states.find(*equivalent);
    if (it == states.end() || !it->second.isKnown())
      break;
    state = it->second;
  }
  return state;
}

AbstractValue getKnownState(Value value, const DenseMap<Value, AbstractValue> &states) {
  auto it = states.find(value);
  if (it != states.end())
    return resolveAbstractValue(it->second, states);
  if (auto attr = asSplatConstantAttr(value))
    return AbstractValue::getConstant(*attr);
  if (isa<BlockArgument>(value))
    return AbstractValue::getEquivalentTo(value);
  return AbstractValue::getUnknown();
}

enum class RelationKind : uint8_t { LE, LT, GE, GT };
enum class LiveRelationStrength : uint8_t { Necessary, Sufficient };

struct NecessaryLiveRelation {
  AffineBound lhs;
  AffineBound rhs;
  RelationKind kind;
};

FailureOr<NecessaryLiveRelation> getLiveRelation(PredicateAtom atom, linalg::GenericOp generic,
                                                 LiveRelationStrength strength) {
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
  FailureOr<AffineBound> lhsLower = AffineBound::project(lhs, generic, /*lowerBound=*/true),
                         lhsUpper = AffineBound::project(lhs, generic, /*lowerBound=*/false),
                         rhsLower = AffineBound::project(rhs, generic, /*lowerBound=*/true),
                         rhsUpper = AffineBound::project(rhs, generic, /*lowerBound=*/false);
  if (failed(lhsLower) || failed(lhsUpper) || failed(rhsLower) || failed(rhsUpper))
    return failure();

  auto chooseBound = [&](AffineBound &lower, AffineBound &upper,
                         bool chooseLower) -> AffineBound & { return chooseLower ? lower : upper; };
  bool necessary = strength == LiveRelationStrength::Necessary;
  AffineBound &lhsLeLt = chooseBound(*lhsLower, *lhsUpper, necessary);
  AffineBound &rhsLeLt = chooseBound(*rhsLower, *rhsUpper, !necessary);
  AffineBound &lhsGeGt = chooseBound(*lhsLower, *lhsUpper, !necessary);
  AffineBound &rhsGeGt = chooseBound(*rhsLower, *rhsUpper, necessary);

  switch (predicate) {
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return NecessaryLiveRelation{lhsLeLt, rhsLeLt, RelationKind::LE};
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return NecessaryLiveRelation{lhsLeLt, rhsLeLt, RelationKind::LT};
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return NecessaryLiveRelation{lhsGeGt, rhsGeGt, RelationKind::GE};
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return NecessaryLiveRelation{lhsGeGt, rhsGeGt, RelationKind::GT};
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

FailureOr<AffineInterval> getIvIntervalFromRelations(ArrayRef<NecessaryLiveRelation> relations,
                                                     Value iv) {
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
  if (constraints.isEmpty())
    return AffineInterval{.empty = true};

  unsigned ivPos;
  if (!constraints.findVar(iv, &ivPos))
    return failure();

  auto [lbMap, ubMap] =
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
    upper = AffineBound{ubMap, boundOperands}.offset(1);
  return AffineInterval{lower, upper};
}

std::optional<unsigned> getGenericInputArgNumber(Value value, linalg::GenericOp generic) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != generic.getBlock())
    return std::nullopt;
  if (blockArg.getArgNumber() >= generic.getNumDpsInputs())
    return std::nullopt;
  return blockArg.getArgNumber();
}

OpOperand *getGenericInputOperand(Value value, linalg::GenericOp generic) {
  std::optional<unsigned> inputArgNumber = getGenericInputArgNumber(value, generic);
  if (!inputArgNumber)
    return nullptr;
  return generic.getDpsInputOperands()[*inputArgNumber];
}

ScalarExprState evaluateScalarValue(Value value, DenseMap<Value, ScalarExprState> &states,
                                    Attribute deadValue) {
  auto it = states.find(value);
  if (it != states.end())
    return it->second;

  if (auto attr = asSplatConstantAttr(value))
    return ScalarExprState::getConstant(*attr);

  Operation *def = value.getDefiningOp();
  if (!def || def->getNumResults() != 1)
    return ScalarExprState::getUnknown();

  auto foldAddLike = [](ScalarExprState lhs, ScalarExprState rhs) {
    if (lhs.isConstZero())
      return rhs;
    if (rhs.isConstZero())
      return lhs;
    return ScalarExprState::getUnknown();
  };
  auto foldSubLike = [](ScalarExprState lhs, ScalarExprState rhs) {
    return rhs.isConstZero() ? lhs : ScalarExprState::getUnknown();
  };
  auto foldMulLike = [](ScalarExprState lhs, ScalarExprState rhs) {
    if (lhs.isConstZero() || rhs.isConstOne())
      return lhs;
    if (rhs.isConstZero() || lhs.isConstOne())
      return rhs;
    return ScalarExprState::getUnknown();
  };
  auto foldDivLike = [](ScalarExprState lhs, ScalarExprState rhs) {
    return rhs.isConstOne() ? lhs : ScalarExprState::getUnknown();
  };
  auto foldAndLike = [](ScalarExprState lhs, ScalarExprState rhs) {
    if (lhs.isConstBool(false) || rhs.isConstBool(true))
      return lhs;
    if (lhs.isConstBool(true) || rhs.isConstBool(false))
      return rhs;
    return ScalarExprState::getUnknown();
  };
  auto foldSelectLike = [](ScalarExprState condition, ScalarExprState trueValue,
                           ScalarExprState falseValue) {
    if (condition.isConstBool(true))
      return trueValue;
    if (condition.isConstBool(false))
      return falseValue;
    // Can implement a trueValue == falseValue check here, but we don't have use for it.
    return ScalarExprState::getUnknown();
  };
  auto foldMaximumLike = [&](ScalarExprState lhs, ScalarExprState rhs) {
    if (lhs.bitwiseEqualToAttr(deadValue))
      return rhs;
    if (rhs.bitwiseEqualToAttr(deadValue))
      return lhs;
    return ScalarExprState::getUnknown();
  };
  auto foldExpLike = [](ScalarExprState operand) {
    if (operand.isNegativeInfinity())
      return operand.getZeroLike();
    return ScalarExprState::getUnknown();
  };
  auto evaluate = [&](Value operand) { return evaluateScalarValue(operand, states, deadValue); };

#define CASE_BIN_OP(foldLike)                                                                      \
  [&](auto op) { return foldLike(evaluate(op.getLhs()), evaluate(op.getRhs())); }

  ScalarExprState result =
      llvm::TypeSwitch<Operation *, ScalarExprState>(def)
          .Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
            return ScalarExprState::getConstant(constant.getValue());
          })
          .Case<math::ExpOp>(
              [&](math::ExpOp exp) { return foldExpLike(evaluate(exp.getOperand())); })
          .Case<arith::AddFOp, arith::AddIOp>(CASE_BIN_OP(foldAddLike))
          .Case<arith::SubFOp, arith::SubIOp>(CASE_BIN_OP(foldSubLike))
          .Case<arith::MulFOp, arith::MulIOp>(CASE_BIN_OP(foldMulLike))
          .Case<arith::DivFOp>(CASE_BIN_OP(foldDivLike))
          .Case<arith::AndIOp>(CASE_BIN_OP(foldAndLike))
          .Case<arith::MaximumFOp>(CASE_BIN_OP(foldMaximumLike))
          .Case<arith::SelectOp>([&](arith::SelectOp select) {
            return foldSelectLike(evaluate(select.getCondition()), evaluate(select.getTrueValue()),
                                  evaluate(select.getFalseValue()));
          })
          .Default([](Operation *) { return ScalarExprState::getUnknown(); });
  states[value] = result;
  return result;
}

DenseMap<Value, ScalarExprState>
seedGenericInputScalarStates(linalg::GenericOp generic,
                             const DenseMap<Value, AbstractValue> &states) {
  DenseMap<Value, ScalarExprState> scalarStates;
  for (auto [i, inputOperand] : llvm::enumerate(generic.getDpsInputOperands())) {
    AbstractValue inputState = getKnownState(inputOperand->get(), states);
    BlockArgument blockArg = generic.getBlock()->getArgument(i);
    if (auto attr = inputState.getConstantAttr())
      scalarStates[blockArg] = ScalarExprState::getConstant(*attr);
    else
      scalarStates[blockArg] = ScalarExprState::getEquivalentTo(inputOperand);
  }
  return scalarStates;
}

AbstractValue analyzeGeneric(linalg::GenericOp generic,
                             const DenseMap<Value, AbstractValue> &states, Attribute deadValue) {
  if (generic.getNumResults() != 1 || generic.getNumDpsInits() != 1)
    return AbstractValue::getUnknown();

  auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
  DenseMap<Value, ScalarExprState> scalarStates = seedGenericInputScalarStates(generic, states);
  SmallVector<OpOperand *> inputOperands = generic.getDpsInputOperands();
  OpOperand *initOperand = generic.getDpsInitOperand(0);
  BlockArgument accumulatorArg = generic.getBlock()->getArgument(generic.getNumDpsInputs());
  scalarStates[accumulatorArg] = ScalarExprState::getEquivalentTo(initOperand);
  bool isReduction = generic.getNumReductionLoops() != 0;
  ScalarExprState yielded = evaluateScalarValue(yield.getOperand(0), scalarStates, deadValue);

  if (isReduction && yielded.getEquivalentValue() == initOperand) {
    return AbstractValue::getEquivalentTo(initOperand->get());
  } else if (auto equivInput = yielded.getEquivalentValue()) {
    unsigned inputIndex = (*equivInput)->getOperandNumber();
    return AbstractValue::getEquivalentTo(inputOperands[inputIndex]->get());
  } else if (auto attr = yielded.getConstantAttr()) {
    return AbstractValue::getConstant(*attr);
  }
  return AbstractValue::getUnknown();
}

AbstractValue analyzeFill(linalg::FillOp fill) {
  if (auto attr = asSplatConstantAttr(fill.getInputs().front()))
    return AbstractValue::getConstant(*attr);
  return AbstractValue::getUnknown();
}

AbstractValue analyzeTensorLikeUnaryOp(Value source, const DenseMap<Value, AbstractValue> &states) {
  AbstractValue sourceState = getKnownState(source, states);
  if (sourceState.getConstantAttr())
    return sourceState;
  return AbstractValue::getUnknown();
}

void analyzeLoopDeadPropagation(scf::ForOp loop, linalg::GenericOp producer, Attribute deadValue,
                                DenseMap<Value, AbstractValue> &states) {
  if (producer.getNumResults() == 1)
    states[producer->getResult(0)] = AbstractValue::getConstant(deadValue);

  for (Operation &op : loop.getBody()->without_terminator()) {
    SmallVector<AbstractValue> resultStates(op.getNumResults(), AbstractValue::getUnknown());
    if (op.getNumResults() == 1) {
      if (&op == producer)
        resultStates[0] = AbstractValue::getConstant(deadValue);
      else if (auto fill = dyn_cast<linalg::FillOp>(&op))
        resultStates[0] = analyzeFill(fill);
      else if (auto generic = dyn_cast<linalg::GenericOp>(&op))
        resultStates[0] = analyzeGeneric(generic, states, deadValue);
      else if (auto extract = dyn_cast<tensor::ExtractSliceOp>(op))
        resultStates[0] = analyzeTensorLikeUnaryOp(extract.getSource(), states);
      else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op))
        resultStates[0] = analyzeTensorLikeUnaryOp(collapse.getSrc(), states);
      else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op))
        resultStates[0] = analyzeTensorLikeUnaryOp(expand.getSrc(), states);
    }

    for (auto [result, state] : llvm::zip(op.getResults(), resultStates))
      if (state.isKnown())
        states[result] = state;
  }
}

bool isFalseConstant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return false;
  auto boolAttr = dyn_cast<BoolAttr>(attr);
  return boolAttr && !boolAttr.getValue();
}

bool attrsEqualByValue(Attribute lhs, Attribute rhs) {
  if (lhs == rhs)
    return true;

  auto lhsFloat = dyn_cast<FloatAttr>(lhs);
  auto rhsFloat = dyn_cast<FloatAttr>(rhs);
  if (lhsFloat && rhsFloat)
    return lhsFloat.getValue().bitwiseIsEqual(rhsFloat.getValue());

  return false;
}

std::optional<Attribute> asSplatConstantAttr(Value value) {
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
  if (auto constant = asSplatConstantAttr(value))
    return constant;
  OpOperand *inputOperand = getGenericInputOperand(value, generic);
  if (!inputOperand)
    return std::nullopt;
  return asSplatConstantAttr(inputOperand->get());
}

FailureOr<MatchedDeadSelect> matchDeadSelect(linalg::GenericOp generic, Attribute deadValue) {
  auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
  if (yield.getNumOperands() != 1)
    return failure();

  auto select = yield.getOperand(0).getDefiningOp<arith::SelectOp>();
  if (!select)
    return failure();

  auto scalarValueEqualsAttr = [&](Value value) {
    auto actual = getConstantAttrForScalarValue(generic, value);
    return actual && attrsEqualByValue(*actual, deadValue);
  };
  bool trueIsDead = scalarValueEqualsAttr(select.getTrueValue());
  bool falseIsDead = scalarValueEqualsAttr(select.getFalseValue());
  if (trueIsDead == falseIsDead)
    return failure();
  bool liveWhenPredicateIsTrue = !trueIsDead;
  Value liveValue = trueIsDead ? select.getFalseValue() : select.getTrueValue();

  auto getSufficientLivePredicate = [](Value predicate) {
    auto select = predicate.getDefiningOp<arith::SelectOp>();
    if (!select)
      return predicate;
    if (!isFalseConstant(select.getFalseValue()))
      return predicate;
    // If `select(%cmp, maybe_mask, false)` is false whenever `%cmp` is false,
    // then `%cmp` is a sufficient live predicate for proving all-dead tiles.
    return select.getCondition();
  };
  FailureOr<Value> livePredicate = getSufficientLivePredicate(select.getCondition());
  if (failed(livePredicate))
    return failure();

  SmallVector<PredicateAtom> atoms;
  std::function<LogicalResult(Value)> collectPredicateAtoms = [&](Value predicate) {
    if (auto andOp = predicate.getDefiningOp<arith::AndIOp>()) {
      if (!liveWhenPredicateIsTrue)
        return failure();
      if (failed(collectPredicateAtoms(andOp.getLhs())))
        return failure();
      return collectPredicateAtoms(andOp.getRhs());
    }
    auto cmp = predicate.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return failure();
    atoms.push_back(PredicateAtom{cmp, liveWhenPredicateIsTrue});
    return success();
  };
  if (failed(collectPredicateAtoms(*livePredicate)))
    return failure();
  return MatchedDeadSelect{atoms, liveValue, getGenericInputArgNumber(liveValue, generic)};
}

FailureOr<AffineInterval> deriveLiveInterval(linalg::GenericOp generic, scf::ForOp loop,
                                             const MatchedDeadSelect &match,
                                             LiveRelationStrength strength) {
  SmallVector<NecessaryLiveRelation> relations;
  relations.reserve(match.atoms.size());
  for (const PredicateAtom &atom : match.atoms) {
    FailureOr<NecessaryLiveRelation> relation = getLiveRelation(atom, generic, strength);
    if (failed(relation))
      return failure();
    relations.push_back(*relation);
  }

  if (relations.empty())
    return failure();
  return getIvIntervalFromRelations(relations, loop.getInductionVar());
}

LogicalResult canSpecializeFullyLiveProducer(linalg::GenericOp producer,
                                             const MatchedDeadSelect &match) {
  if (!match.liveInputOperandNumber || producer->getNumResults() != 1 ||
      producer.getNumDpsInits() != 1)
    return failure();
  AffineMap liveInputMap =
      producer.getMatchingIndexingMap(producer.getDpsInputOperand(*match.liveInputOperandNumber));
  AffineMap outputMap = producer.getMatchingIndexingMap(producer.getDpsInitOperand(0));
  return success(liveInputMap == outputMap);
}

} // namespace

void LoopSpecializeDeadTileOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerOpMutable(), effects);
  consumesHandle(getLoopMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopSpecializeDeadTileOp::apply(TransformRewriter &rewriter,
                                                            TransformResults &transformResults,
                                                            TransformState &state) {
  (void)transformResults;
  auto transform = cast<TransformOpInterface>(getOperation());

  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getProducerOp, "producer", producer,
                               linalg::GenericOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getLoop, "loop", loop, scf::ForOp);

  FailureOr<MatchedDeadSelect> match = matchDeadSelect(producer, getDeadValue());
  if (failed(match))
    BAIL("expected producer to yield select(live_predicate, live_value, dead_value)");

  if (failed(deriveLiveInterval(producer, loop, *match, LiveRelationStrength::Necessary)))
    BAIL("failed to derive an affine possible-live interval for the loop IV");
  FailureOr<AffineInterval> fullyLiveInterval =
      deriveLiveInterval(producer, loop, *match, LiveRelationStrength::Sufficient);
  if (failed(fullyLiveInterval))
    BAIL("failed to derive an affine fully-live interval for the loop IV");

  DenseMap<Value, AbstractValue> states;
  analyzeLoopDeadPropagation(loop, producer, getDeadValue(), states);
  if (failed(canSpecializeFullyLiveProducer(producer, *match)))
    BAIL("expected producer live value to come from an input with the same indexing as the output");

  rewriter.setInsertionPoint(loop);
  FailureOr<Value> liveUpperBound =
      materializeIntervalUpperBound(rewriter, loop, *fullyLiveInterval);
  if (failed(liveUpperBound))
    BAIL("failed to materialize the fully-live prefix upper bound");
  FailureOr<scf::ForOp> liveLoop =
      cloneForWithBody(rewriter, loop, loop.getLowerBound(), *liveUpperBound, loop.getInitArgs(),
                       [&](RewriterBase &, IRMapping &mapping, Operation &op) -> FailureOr<bool> {
                         if (&op != producer.getOperation())
                           return false;
                         Value liveTensor =
                             producer.getDpsInputOperand(*match->liveInputOperandNumber)->get();
                         mapping.map(producer->getResult(0), mapping.lookupOrDefault(liveTensor));
                         return true;
                       });
  if (failed(liveLoop))
    BAIL("failed to clone the fully-live prefix loop");

  Operation *mixedProducer = nullptr;
  FailureOr<scf::ForOp> mixedLoop = cloneForWithBody(
      rewriter, loop, *liveUpperBound, loop.getUpperBound(), liveLoop->getResults(),
      [&](RewriterBase &rewriter, IRMapping &mapping, Operation &op) -> FailureOr<bool> {
        if (&op != producer.getOperation())
          return false;
        mixedProducer = rewriter.clone(op, mapping);
        return true;
      });
  if (failed(mixedLoop))
    BAIL("failed to clone the mixed loop");
  if (!mixedProducer)
    BAIL("failed to clone the producer into the mixed loop");
  if (failed(rewriter.notifyPayloadOperationReplaced(producer, mixedProducer)))
    BAIL("failed to preserve the producer handle");

  if (loop.getNumResults() == 0)
    rewriter.eraseOp(loop);
  else
    rewriter.replaceOp(loop, mixedLoop->getResults());
  transformResults.set(getOperation()->getResult(0), {liveLoop->getOperation()});
  transformResults.set(getOperation()->getResult(1), {mixedLoop->getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
