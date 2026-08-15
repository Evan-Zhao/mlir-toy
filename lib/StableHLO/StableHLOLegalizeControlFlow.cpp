// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from IREE's StableHLO LegalizeControlFlow implementation at c9058ce882b5:
// compiler/plugins/input/StableHLO/Conversion/LegalizeControlFlow.cpp

#include "StableHLO/StableHLOLegalizeControlFlow.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace neptune::stablehlo {

using namespace mlir;

namespace {

// Move a StableHLO region into an SCF op and change only its terminator.
void inlineStableHLORegionIntoSCFRegion(PatternRewriter &rewriter, Region &hlo, Region &scf) {
  if (!scf.empty())
    rewriter.eraseBlock(&scf.back());
  rewriter.inlineRegionBefore(hlo, scf, scf.end());

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&scf.back());
  Operation *terminator = scf.back().getTerminator();
  rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator, terminator->getOperands());
}

// StableHLO predicates and loop bounds are tensors; SCF consumes scalars.
Value extractTensorValue(OpBuilder &builder, Value tensorValue) {
  Location loc = tensorValue.getLoc();
  if (auto type = dyn_cast<RankedTensorType>(tensorValue.getType()); type && type.getRank() != 0) {
    tensorValue = tensor::CollapseShapeOp::create(builder, loc, tensorValue,
                                                  SmallVector<ReassociationIndices>());
  }
  return tensor::ExtractOp::create(builder, loc, tensorValue, ValueRange());
}

struct ForBounds {
  Value lowerBound;
  Value upperBound;
  Value step;
  unsigned inductionArgument;
};

// Recognize the JAX fori_loop form:
//
//   cond(%i, ...) { %p = stablehlo.compare LT, %i, %ub }
//   body(%i, ...) { %next = stablehlo.add %i, %step }
std::optional<ForBounds> extractForBounds(mlir::stablehlo::WhileOp op) {
  Block &condition = op.getCond().front();
  Block &body = op.getBody().front();
  if (condition.getOperations().size() != 2)
    return std::nullopt;

  auto matchBlockArgument = [](Value value, Block &block) -> std::optional<unsigned> {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || value.getParentBlock() != &block)
      return std::nullopt;
    return argument.getArgNumber();
  };

  auto compare = dyn_cast<mlir::stablehlo::CompareOp>(condition.front());
  if (!compare || compare.getComparisonDirection() != mlir::stablehlo::ComparisonDirection::LT ||
      compare.getRhs().getParentBlock() == &condition ||
      !getElementTypeOrSelf(compare.getLhs().getType()).isSignlessIntOrIndex()) {
    return std::nullopt;
  }

  std::optional<unsigned> inductionArgument = matchBlockArgument(compare.getLhs(), condition);
  if (!inductionArgument)
    return std::nullopt;

  auto add = dyn_cast_if_present<mlir::stablehlo::AddOp>(
      body.getTerminator()->getOperand(*inductionArgument).getDefiningOp());
  if (!add || matchBlockArgument(add.getLhs(), body) != inductionArgument ||
      add.getRhs().getParentBlock() == &body) {
    return std::nullopt;
  }

  return ForBounds{op->getOperand(*inductionArgument), compare.getRhs(), add.getRhs(),
                   *inductionArgument};
}

struct WhileOpPattern final : OpConversionPattern<mlir::stablehlo::WhileOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::WhileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (std::optional<ForBounds> bounds = extractForBounds(op)) {
      Block &stableBody = op.getBody().front();
      auto matchIncrement = [&](unsigned argument) -> mlir::stablehlo::AddOp {
        auto add = dyn_cast_if_present<mlir::stablehlo::AddOp>(
            stableBody.getTerminator()->getOperand(argument).getDefiningOp());
        auto lhs = add ? dyn_cast<BlockArgument>(add.getLhs()) : BlockArgument();
        if (!add || !lhs || lhs.getOwner() != &stableBody || lhs.getArgNumber() != argument ||
            add.getRhs() != bounds->step || !add->hasOneUse())
          return {};
        return add;
      };

      mlir::stablehlo::AddOp inductionIncrement = matchIncrement(bounds->inductionArgument);
      bool dropInduction =
          op->getResult(bounds->inductionArgument).use_empty() && inductionIncrement;

      // Frontends may carry any number of counters synchronized with the
      // condition induction. Treat every unused counter with the same initial
      // value and update as an alias of the structural scf.for induction.
      SmallVector<unsigned> droppedArguments;
      SmallVector<Operation *> droppedIncrements;
      if (dropInduction) {
        droppedArguments.push_back(bounds->inductionArgument);
        droppedIncrements.push_back(inductionIncrement);
        for (unsigned argument = 0; argument < op->getNumOperands(); ++argument) {
          if (argument == bounds->inductionArgument ||
              op->getOperand(argument) != bounds->lowerBound ||
              !op->getResult(argument).use_empty())
            continue;
          if (mlir::stablehlo::AddOp increment = matchIncrement(argument)) {
            droppedArguments.push_back(argument);
            droppedIncrements.push_back(increment);
          }
        }
        llvm::sort(droppedArguments);
      }

      SmallVector<Value> initOperands(adaptor.getOperands());
      for (unsigned argument : llvm::reverse(droppedArguments))
        initOperands.erase(initOperands.begin() + argument);

      auto forOp =
          scf::ForOp::create(rewriter, loc, extractTensorValue(rewriter, bounds->lowerBound),
                             extractTensorValue(rewriter, bounds->upperBound),
                             extractTensorValue(rewriter, bounds->step), initOperands);
      inlineStableHLORegionIntoSCFRegion(rewriter, op.getBody(), forOp.getRegion());

      // SCF supplies a scalar induction variable. Rebuild the tensor form used
      // by the inlined StableHLO body and replace its old loop-carried index.
      Block &body = forOp.getRegion().front();
      BlockArgument induction =
          forOp.getRegion().insertArgument(unsigned{0}, forOp.getLowerBound().getType(), loc);
      BlockArgument oldInduction = body.getArgument(1 + bounds->inductionArgument);
      rewriter.setInsertionPointToStart(&body);
      Value tensorInduction =
          tensor::FromElementsOp::create(rewriter, loc, oldInduction.getType(), induction);
      oldInduction.replaceAllUsesWith(tensorInduction);

      if (!dropInduction) {
        rewriter.replaceOp(op, forOp.getResults());
        return success();
      }

      // Omit unused structural counters from the iter_args and yield. Current
      // iteration uses have already been redirected to the tensor induction.
      for (unsigned argument : droppedArguments)
        body.getArgument(1 + argument).replaceAllUsesWith(tensorInduction);
      auto yield = cast<scf::YieldOp>(body.getTerminator());
      rewriter.modifyOpInPlace(yield, [&]() {
        for (unsigned argument : llvm::reverse(droppedArguments))
          yield->eraseOperand(argument);
      });
      for (unsigned argument : llvm::reverse(droppedArguments))
        body.eraseArgument(1 + argument);
      for (Operation *increment : droppedIncrements)
        rewriter.eraseOp(increment);

      SmallVector<Value> replacements;
      replacements.reserve(op->getNumResults());
      unsigned forResult = 0;
      for (unsigned result = 0; result < op->getNumResults(); ++result) {
        // These values are unused, but keeping same-typed replacements lets
        // the conversion rewriter report one replacement per original result.
        if (llvm::is_contained(droppedArguments, result))
          replacements.push_back(bounds->lowerBound);
        else
          replacements.push_back(forOp.getResult(forResult++));
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }

    auto whileOp = scf::WhileOp::create(rewriter, loc, op.getResultTypes(), adaptor.getOperands());

    rewriter.inlineRegionBefore(op.getCond(), whileOp.getBefore(), whileOp.getBefore().end());
    auto conditionReturn =
        cast<mlir::stablehlo::ReturnOp>(whileOp.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&whileOp.getBefore().front());
    Value predicate = extractTensorValue(rewriter, conditionReturn->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(conditionReturn, predicate,
                                                  whileOp.getBeforeArguments());

    inlineStableHLORegionIntoSCFRegion(rewriter, op.getBody(), whileOp.getAfter());
    rewriter.replaceOp(op, whileOp.getResults());
    return success();
  }
};

struct IfOpPattern final : OpConversionPattern<mlir::stablehlo::IfOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::IfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                                  extractTensorValue(rewriter, adaptor.getPred()),
                                  /*withElseRegion=*/true);
    inlineStableHLORegionIntoSCFRegion(rewriter, op.getTrueBranch(), ifOp.getThenRegion());
    inlineStableHLORegionIntoSCFRegion(rewriter, op.getFalseBranch(), ifOp.getElseRegion());
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

struct CaseOpPattern final : OpConversionPattern<mlir::stablehlo::CaseOp> {
  using Base::Base;

  scf::IfOp createNestedCases(int currentIndex, mlir::stablehlo::CaseOp op, OpAdaptor adaptor,
                              PatternRewriter &outerBuilder) const {
    Location loc = op.getLoc();
    Value index = adaptor.getIndex();
    size_t finalIndex = op.getBranches().size() - 2;

    auto shapedType = cast<ShapedType>(index.getType());
    auto constantAttr = DenseElementsAttr::get(
        shapedType, {cast<Attribute>(outerBuilder.getI32IntegerAttr(currentIndex))});
    Value currentIndexValue =
        mlir::stablehlo::ConstantOp::create(outerBuilder, loc, index.getType(), constantAttr);
    Value predicate = mlir::stablehlo::CompareOp::create(
        outerBuilder, loc, index, currentIndexValue, mlir::stablehlo::ComparisonDirection::EQ);

    auto ifOp = scf::IfOp::create(outerBuilder, loc, op.getResultTypes(),
                                  extractTensorValue(outerBuilder, predicate),
                                  /*withElseRegion=*/true);
    inlineStableHLORegionIntoSCFRegion(outerBuilder, op.getBranches()[currentIndex],
                                       ifOp.getThenRegion());

    int nextIndex = currentIndex + 1;
    if (currentIndex == static_cast<int64_t>(finalIndex)) {
      inlineStableHLORegionIntoSCFRegion(outerBuilder, op.getBranches()[nextIndex],
                                         ifOp.getElseRegion());
      return ifOp;
    }

    PatternRewriter::InsertionGuard guard(outerBuilder);
    outerBuilder.setInsertionPointToEnd(&ifOp.getElseRegion().back());
    scf::IfOp nested = createNestedCases(nextIndex, op, adaptor, outerBuilder);
    scf::YieldOp::create(outerBuilder, loc, nested.getResults());
    return ifOp;
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::CaseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getBranches().size() == 1) {
      Block &block = op.getBranches().front().front();
      SmallVector<Value> results(block.getTerminator()->getOperands());
      rewriter.eraseOp(block.getTerminator());
      rewriter.inlineBlockBefore(&block, op.getOperation(), {});
      rewriter.replaceOp(op, results);
      return success();
    }

    rewriter.replaceOp(op, createNestedCases(0, op, adaptor, rewriter).getResults());
    return success();
  }
};

} // namespace

LogicalResult legalizeControlFlow(Operation *target, RewriterBase::Listener *listener) {
  MLIRContext *context = target->getContext();
  RewritePatternSet patterns(context);
  patterns.add<WhileOpPattern, IfOpPattern, CaseOpPattern>(context);

  ConversionTarget conversionTarget(*context);
  conversionTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  conversionTarget
      .addIllegalOp<mlir::stablehlo::WhileOp, mlir::stablehlo::IfOp, mlir::stablehlo::CaseOp>();
  ConversionConfig config;
  config.listener = listener;
  return applyPartialConversion(target, conversionTarget,
                                FrozenRewritePatternSet(std::move(patterns)), config);
}

} // namespace neptune::stablehlo
