// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <iter_visitor.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <optimization/combine_mul_sum.h>
#include <type.h>

namespace nvfuser::optimization {

namespace {

TensorView* get_tensorview_from_mul(TensorView* in) {
  if (in->getDataType() == DataType::BFloat16 ||
      in->getDataType() == DataType::Half) {
    return in;
  }
  TensorView* ret = nullptr;
  if (in->getDataType() == DataType::Float) {
    // get the producer.
    auto* cast = in->definition();
    auto* uCastOp = dynamic_cast<UnaryOp*>(cast);
    if (uCastOp && uCastOp->getUnaryOpType() == UnaryOpType::Cast) {
      auto* aTV = static_cast<TensorView*>(uCastOp->in());
      NVF_CHECK(
          (aTV->getDataType() == DataType::BFloat16 ||
           aTV->getDataType() == DataType::Half),
          "The output of the cast must be half or BF16");
      ret = aTV;
    }
  }
  return ret;
}

void add_mma_op(
    Fusion* fusion,
    std::vector<std::pair<ReductionOp*, BinaryOp*>>& sum_mul_pairs) {
  for (auto [sumOp, mulOp] : sum_mul_pairs) {
    // Create the output for the MMA Op.
    auto rootDomain = static_cast<TensorView*>(sumOp->out())->getRootDomain();
    std::vector<int> axes = {};
    int dimIdx = 0;
    for (const auto* id : rootDomain) {
      if (id->isReduction()) {
        axes.push_back(dimIdx);
      }
      ++dimIdx;
    }
    std::cout << "1 " << std::endl;
    auto x = get_tensorview_from_mul(static_cast<TensorView*>(mulOp->lhs()));
    auto y = get_tensorview_from_mul(static_cast<TensorView*>(mulOp->rhs()));

    auto* tvOut = fusedMultiplySum(x, y, axes);
    fusion->replaceOutput(sumOp->out(), tvOut);
  }
};
} // namespace

void CombineMulSum::dispatch(Expr* expr) {
  std::cout << "involing combine-mul-sum dispatch for exprs" << std::endl;
  IterVisitor::dispatch(expr);
};

void CombineMulSum::handle(BinaryOp* stmt) {
  if (stmt->getBinaryOpType() == BinaryOpType::Mul) {
    std::cout << "Handling a Mul Op type" << std::endl;
  }
};

void CombineMulSum::handle(ReductionOp* stmt) {
  if (stmt->getReductionOpType() == BinaryOpType::Add) {
    std::cout << "Handling a reduction of Add type" << std::endl;
    auto* inputOfSum = stmt->in();
    if (inputOfSum != nullptr) {
      auto* producer = inputOfSum->definition();
      if (producer && dynamic_cast<BinaryOp*>(producer)) {
        auto* bOp = dynamic_cast<BinaryOp*>(producer);
        if (bOp->getBinaryOpType() == BinaryOpType::Mul) {
          std::cout << "Reduce (Sum) Op with Mul Producer" << std::endl;
          sum_mul_pairs_.push_back({stmt, bOp});
        }
      }
    }
  }
};

bool CombineMulSum::run() {
  fusion_->printMath();
  traverse(fusion_);
  add_mma_op(fusion_, sum_mul_pairs_);
  fusion_->printMath();
  return true;
}
} // namespace nvfuser::optimization