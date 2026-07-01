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
#include "llvm/Support/Debug.h"
#include <mlir/IR/Builders.h>
#include <optional>
#include <variant>

using namespace mlir;

namespace mlir::transform {
namespace {

#define DEBUG_TYPE "loop-specialize-dead-tile"
#define BAIL(message) return emitSilenceableFailure(transform, message)

std::optional<Attribute> asSplatConstantAttr(Value value);
bool attrsEqualByValue(Attribute lhs, Attribute rhs);

bool isZeroAttr(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

bool isOneAttr(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isExactlyValue(1.0);
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isOne();
  return false;
}

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
                                          /*closedUB=*/false);
    if (!lbMap || !ubMap)
      return failure();

    SmallVector<Value> operands;
    constraints.getValues(/*start=*/1, constraints.getNumDimAndSymbolVars(), &operands);
    if (lowerBound)
      return AffineBound{lbMap, operands};
    // The affine API represents upper bounds as open bounds. Convert to the
    // closed maximum value used when reasoning about "all elements in a tile".
    return AffineBound{ubMap, operands}.offset(-1);
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

  template <typename U> static AbstractValueData getConstantOfType(Type type, U constVal) {
    if (auto floatType = dyn_cast<FloatType>(type))
      return {FloatAttr::get(floatType, constVal)};
    if (auto intType = dyn_cast<IntegerType>(type))
      return {IntegerAttr::get(intType, constVal)};
    llvm::errs() << "Unsupported type for constant attribute: " << type << "\n";
    llvm_unreachable("unsupported type");
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

  bool equals(const AbstractValueData &rhs) const {
    auto rhsAttr = rhs.getConstantAttr();
    if (rhsAttr && bitwiseEqualToAttr(*rhsAttr))
      return true;
    return getEquivalentValue() == rhs.getEquivalentValue();
  };

private:
  bool mapConstAttribute(std::function<bool(Attribute)> &&predicate) const {
    auto attr = getConstantAttr();
    return attr && predicate(*attr);
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

Value minOrMaxIndexValue(RewriterBase &rewriter, Location loc, Value lhs, Value rhs, bool isMax) {
  if (isMax)
    return arith::MaxSIOp::create(rewriter, loc, lhs, rhs).getResult();
  return arith::MinSIOp::create(rewriter, loc, lhs, rhs).getResult();
}

FailureOr<std::pair<Value, Value>>
materializeIntervalBounds(RewriterBase &rewriter, scf::ForOp loop, const AffineInterval &interval) {
  Location loc = loop.getLoc();
  Value lower = loop.getLowerBound();
  Value upper = interval.empty ? loop.getLowerBound() : loop.getUpperBound();
  if (!interval.empty && interval.lower) {
    FailureOr<Value> bound = materializeAffineBound(rewriter, loc, *interval.lower);
    if (failed(bound))
      return failure();
    lower = *bound;
  }
  if (!interval.empty && interval.upper) {
    FailureOr<Value> bound = materializeAffineBound(rewriter, loc, *interval.upper);
    if (failed(bound))
      return failure();
    upper = *bound;
  }

  lower = minOrMaxIndexValue(rewriter, loc, lower, loop.getLowerBound(), /*isMax=*/true);
  lower = minOrMaxIndexValue(rewriter, loc, lower, loop.getUpperBound(), /*isMax=*/false);
  upper = minOrMaxIndexValue(rewriter, loc, upper, loop.getLowerBound(), /*isMax=*/true);
  upper = minOrMaxIndexValue(rewriter, loc, upper, loop.getUpperBound(), /*isMax=*/false);
  return {{lower, upper}};
}

using LoopCloneCustomizer =
    llvm::function_ref<FailureOr<bool>(RewriterBase &, IRMapping &, Operation &)>;

FailureOr<scf::ForOp> cloneForWithBody(RewriterBase &rewriter, scf::ForOp source, Value lowerBound,
                                       Value upperBound, ValueRange initArgs,
                                       LoopCloneCustomizer customizer) {
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
    FailureOr<bool> handled = customizer(rewriter, mapping, op);
    if (failed(handled))
      return failure();
    if (*handled)
      continue;
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
enum class LiveRelationStrength : uint8_t {
  // An iteration outside this interval is definitely dead. This is used to
  // truncate a suffix whose loop-carried state would not change.
  PossibleLive,
  // Every element in the producer tile is live. This is the mask-free interval
  // where the dead-select producer can be bypassed entirely.
  FullyLive
};

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
  bool possibleLive = strength == LiveRelationStrength::PossibleLive;
  AffineBound &lhsLeLt = chooseBound(*lhsLower, *lhsUpper, possibleLive);
  AffineBound &rhsLeLt = chooseBound(*rhsLower, *rhsUpper, !possibleLive);
  AffineBound &lhsGeGt = chooseBound(*lhsLower, *lhsUpper, !possibleLive);
  AffineBound &rhsGeGt = chooseBound(*rhsLower, *rhsUpper, possibleLive);

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
  default:
    llvm_unreachable("unexpected RelationKind");
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
  if (lbMap) {
    if (lbMap.getNumResults() > 1)
      return failure();
    if (lbMap.getNumResults() == 1)
      lower = AffineBound{lbMap, boundOperands};
  }
  if (ubMap) {
    if (ubMap.getNumResults() > 1)
      return failure();
    if (ubMap.getNumResults() == 1)
      upper = AffineBound{ubMap, boundOperands}.offset(1);
  }
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

  auto foldConstant = [](arith::ConstantOp constant) {
    return ScalarExprState::getConstant(constant.getValue());
  };
  auto foldAdd = [](ScalarExprState lhs, ScalarExprState rhs, Type _) {
    if (lhs.isConstZero())
      return rhs;
    if (rhs.isConstZero())
      return lhs;
    return ScalarExprState::getUnknown();
  };
  auto foldSub = [&](ScalarExprState lhs, ScalarExprState rhs, Type resultTy) {
    if (lhs.equals(rhs))
      return ScalarExprState::getConstantOfType(resultTy, 0);
    if (lhs.isNegativeInfinity() || rhs.isConstZero())
      return lhs;
    return ScalarExprState::getUnknown();
  };
  auto foldMul = [](ScalarExprState lhs, ScalarExprState rhs, Type _) {
    if (lhs.isConstZero() || rhs.isConstOne())
      return lhs;
    if (rhs.isConstZero() || lhs.isConstOne())
      return rhs;
    return ScalarExprState::getUnknown();
  };
  auto foldDiv = [&](ScalarExprState lhs, ScalarExprState rhs, Type resultTy) {
    if (lhs.equals(rhs))
      return ScalarExprState::getConstantOfType(resultTy, 1.0);
    if (lhs.isConstZero() || rhs.isConstOne())
      return lhs;
    return ScalarExprState::getUnknown();
  };
  auto foldAnd = [](ScalarExprState lhs, ScalarExprState rhs, Type _) {
    if (lhs.isConstBool(false) || rhs.isConstBool(true))
      return lhs;
    if (lhs.isConstBool(true) || rhs.isConstBool(false))
      return rhs;
    return ScalarExprState::getUnknown();
  };
  auto foldMaximum = [](ScalarExprState lhs, ScalarExprState rhs, Type _) {
    if (lhs.isNegativeInfinity())
      return rhs;
    if (rhs.isNegativeInfinity())
      return lhs;
    return ScalarExprState::getUnknown();
  };
  auto evaluate = [&](Value operand) { return evaluateScalarValue(operand, states, deadValue); };
  // Recognize the rolling-update idiom `(x * y) * (1 / y) -> x` (and swapped
  // operands). Plain recursive evaluation loses the reciprocal structure
  // because the current abstract domain does not represent `1 / y`.
  auto foldMulF = [&](arith::MulFOp mulf) -> ScalarExprState {
    Value lhsValue = mulf.getLhs();
    Value rhsValue = mulf.getRhs();
    auto tryFold = [&](Value mulValue, Value divValue) {
      auto mul = mulValue.getDefiningOp<arith::MulFOp>();
      auto div = divValue.getDefiningOp<arith::DivFOp>();
      if (!mul || !div || !evaluate(div.getLhs()).isConstOne())
        return ScalarExprState::getUnknown();

      ScalarExprState denominator = evaluate(div.getRhs());
      ScalarExprState lhs = evaluate(mul.getLhs());
      ScalarExprState rhs = evaluate(mul.getRhs());
      if (lhs.equals(denominator))
        return rhs;
      if (rhs.equals(denominator))
        return lhs;
      return ScalarExprState::getUnknown();
    };

    ScalarExprState folded = tryFold(lhsValue, rhsValue);
    if (folded.isKnown())
      return folded;
    folded = tryFold(rhsValue, lhsValue);
    if (folded.isKnown())
      return folded;
    return foldMul(evaluate(lhsValue), evaluate(rhsValue), mulf.getResult().getType());
  };
  auto foldExpOp = [&](auto exp) -> ScalarExprState {
    Type resultTy = exp.getResult().getType();
    auto operand = evaluate(exp.getOperand());
    if (operand.isNegativeInfinity())
      return ScalarExprState::getConstantOfType(resultTy, 0.0);
    if (operand.isConstZero())
      return ScalarExprState::getConstantOfType(resultTy, 1.0);
    return ScalarExprState::getUnknown();
  };
  auto foldSelectOp = [&](arith::SelectOp select) {
    auto condition = evaluate(select.getCondition()), trueValue = evaluate(select.getTrueValue()),
         falseValue = evaluate(select.getFalseValue());
    if (condition.isConstBool(true))
      return trueValue;
    if (condition.isConstBool(false))
      return falseValue;
    // Can implement a trueValue == falseValue check here, but we don't have use for it.
    return ScalarExprState::getUnknown();
  };

#define CASE_BIN_OP(foldLike)                                                                      \
  [&](auto op) {                                                                                   \
    return foldLike(evaluate(op.getLhs()), evaluate(op.getRhs()), op.getResult().getType());       \
  }

  ScalarExprState result = llvm::TypeSwitch<Operation *, ScalarExprState>(def)
                               .Case<arith::ConstantOp>(foldConstant)
                               .Case<math::ExpOp, math::Exp2Op>(foldExpOp)
                               .Case<arith::AddFOp, arith::AddIOp>(CASE_BIN_OP(foldAdd))
                               .Case<arith::SubFOp, arith::SubIOp>(CASE_BIN_OP(foldSub))
                               .Case<arith::MulFOp>(foldMulF)
                               .Case<arith::MulIOp>(CASE_BIN_OP(foldMul))
                               .Case<arith::DivFOp>(CASE_BIN_OP(foldDiv))
                               .Case<arith::AndIOp>(CASE_BIN_OP(foldAnd))
                               .Case<arith::MaximumFOp>(CASE_BIN_OP(foldMaximum))
                               .Case<arith::SelectOp>(foldSelectOp)
                               .Default([](Operation *) { return ScalarExprState::getUnknown(); });
  states[value] = result;
  return result;
}

DenseMap<Value, ScalarExprState>
seedGenericInputScalarStates(linalg::GenericOp generic,
                             const DenseMap<Value, AbstractValue> &states) {
  DenseMap<Value, ScalarExprState> scalarStates;
  SmallVector<OpOperand *> inputOperands = generic.getDpsInputOperands();
  DenseMap<Value, OpOperand *> equivalentValueToInputOperand;
  auto findInputOperandForValue = [&](Value value) -> OpOperand * {
    for (OpOperand *inputOperand : inputOperands)
      if (inputOperand->get() == value)
        return inputOperand;
    return nullptr;
  };

  for (auto [i, inputOperand] : llvm::enumerate(inputOperands)) {
    AbstractValue inputState = getKnownState(inputOperand->get(), states);
    BlockArgument blockArg = generic.getBlock()->getArgument(i);
    if (auto attr = inputState.getConstantAttr()) {
      scalarStates[blockArg] = ScalarExprState::getConstant(*attr);
      continue;
    }

    if (auto equivalent = inputState.getEquivalentValue()) {
      if (OpOperand *equivalentInput = findInputOperandForValue(*equivalent)) {
        scalarStates[blockArg] = ScalarExprState::getEquivalentTo(equivalentInput);
        continue;
      }
      if (auto it = equivalentValueToInputOperand.find(*equivalent);
          it != equivalentValueToInputOperand.end()) {
        scalarStates[blockArg] = ScalarExprState::getEquivalentTo(it->second);
        continue;
      }
      equivalentValueToInputOperand[*equivalent] = inputOperand;
    }

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

bool deadIterationPreservesLoopState(scf::ForOp loop,
                                     const DenseMap<Value, AbstractValue> &states) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [yieldedIdx, yieldedAndIterArg] :
       llvm::enumerate(llvm::zip_equal(yield.getOperands(), loop.getRegionIterArgs()))) {
    auto [yielded, iterArg] = yieldedAndIterArg;
    AbstractValue yieldedState = getKnownState(yielded, states);
    std::optional<Value> equivalent = yieldedState.getEquivalentValue();
    if (!equivalent || *equivalent != iterArg) {
      LLVM_DEBUG({
        auto printAbstractValue = [&](raw_ostream &os, AbstractValue value) {
          if (auto attr = value.getConstantAttr()) {
            os << "constant(" << *attr << ")";
            return;
          }
          if (auto equivalent = value.getEquivalentValue()) {
            os << "equivalent(" << *equivalent << ")";
            return;
          }
          os << "unknown";
        };

        llvm::dbgs() << "dead-tile suffix truncation failed for loop-carried value #" << yieldedIdx
                     << " in loop " << loop << "\n";
        llvm::dbgs() << "  yielded: " << yielded << "\n";
        llvm::dbgs() << "  iter_arg: " << iterArg << "\n";
        llvm::dbgs() << "  abstract value: ";
        printAbstractValue(llvm::dbgs(), yieldedState);
        llvm::dbgs() << "\n";
      });
      return false;
    }
  }
  return true;
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
  scf::ForOp loop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getLoop, "loop", loop, scf::ForOp);

  SmallVector<Operation *> producerOps = llvm::to_vector(state.getPayloadOps(getProducerOp()));
  linalg::GenericOp producer;
  bool producerWasTracked = llvm::hasSingleElement(producerOps);
  if (llvm::hasSingleElement(producerOps)) {
    producer = dyn_cast<linalg::GenericOp>(producerOps.front());
    if (!producer)
      BAIL("expected producer to be a linalg::GenericOp");
  } else if (producerOps.empty()) {
    SmallVector<linalg::GenericOp> candidates;
    loop.walk([&](linalg::GenericOp generic) {
      if (succeeded(matchDeadSelect(generic, getDeadValue())))
        candidates.push_back(generic);
    });
    if (!llvm::hasSingleElement(candidates))
      BAIL("expected exactly one dead-select producer in loop when producer handle is empty");
    producer = candidates.front();
  } else {
    return emitSilenceableFailure(transform, "expected exactly one producer payload op, got " +
                                                 std::to_string(producerOps.size()));
  }

  FailureOr<MatchedDeadSelect> match = matchDeadSelect(producer, getDeadValue());
  if (failed(match))
    BAIL("expected producer to yield select(live_predicate, live_value, dead_value)");

  FailureOr<AffineInterval> possibleLiveInterval =
      deriveLiveInterval(producer, loop, *match, LiveRelationStrength::PossibleLive);
  if (failed(possibleLiveInterval))
    BAIL("failed to derive an affine possible-live interval for the loop IV");
  FailureOr<AffineInterval> fullyLiveInterval =
      deriveLiveInterval(producer, loop, *match, LiveRelationStrength::FullyLive);
  if (failed(fullyLiveInterval))
    BAIL("failed to derive an affine fully-live interval for the loop IV");

  DenseMap<Value, AbstractValue> states;
  analyzeLoopDeadPropagation(loop, producer, getDeadValue(), states);
  bool canTruncateDeadSuffix = deadIterationPreservesLoopState(loop, states);
  if (failed(canSpecializeFullyLiveProducer(producer, *match)))
    BAIL("expected producer live value to come from an input with the same indexing as the output");

  rewriter.setInsertionPoint(loop);
  auto fullyLiveBounds = materializeIntervalBounds(rewriter, loop, *fullyLiveInterval);
  if (failed(fullyLiveBounds))
    BAIL("failed to materialize the fully-live prefix upper bound");
  auto [fullyLiveLower, fullyLiveUpper] = *fullyLiveBounds;
  FailureOr<scf::ForOp> liveLoop =
      cloneForWithBody(rewriter, loop, fullyLiveLower, fullyLiveUpper, loop.getInitArgs(),
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

  Value possiblyLiveUpper = loop.getUpperBound();
  if (canTruncateDeadSuffix) {
    auto possiblyLiveBounds = materializeIntervalBounds(rewriter, loop, *possibleLiveInterval);
    if (failed(possiblyLiveBounds))
      BAIL("failed to materialize the fully-dead suffix lower bound");
    possiblyLiveUpper = possiblyLiveBounds->second;
  }
  Operation *mixedProducer = nullptr;
  FailureOr<scf::ForOp> mixedLoop = cloneForWithBody(
      rewriter, loop, fullyLiveUpper, possiblyLiveUpper, liveLoop->getResults(),
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
  if (producerWasTracked &&
      failed(rewriter.notifyPayloadOperationReplaced(producer, mixedProducer)))
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
