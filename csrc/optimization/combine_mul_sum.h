// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exceptions.h>
#include <iter_visitor.h>

namespace nvfuser::optimization {

//! RemoveEmptyPass removes intermediate empty tensors (those with at least one
//! extent zero thar are neither a fusion output or input).
class CombineMulSum : public IterVisitor {
 public:
  CombineMulSum(Fusion* fusion) : IterVisitor(), fusion_(fusion){};

  //! Instead of traverseTo, run() is the entry point for this class, and we
  //! always traverse from outputs backward to their inputs.
  //!
  //! Returns a bool indicating whether the Fusion was modified or not.
  bool run();

  inline Fusion* fusion() const {
    return fusion_;
  }

 protected:
  std::vector<Statement*> stmts;
  using IterVisitor::handle;
  // void dispatch(Statement* stmt) override;
  virtual void dispatch(Expr* expr) override;

  virtual void handle(BinaryOp* stmt) override;
  virtual void handle(ReductionOp* stmt) override;

  // static void runPass(Fusion* fusion);

 private:
  //! The Fusion associated with live_statements_
  Fusion* fusion_;
  std::vector<std::pair<ReductionOp *, BinaryOp *>> sum_mul_pairs_ = {}; 
};

} // namespace nvfuser::optimization