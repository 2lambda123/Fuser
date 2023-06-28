// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <serde/expr_evaluator_serde.h>
#include <serde/utils.h>

namespace nvfuser::serde {

namespace {

template <typename VALTYPE>
std::vector<VALTYPE*> getImmediateProducers(VALTYPE* val) {
  return (val->definition()) ? val->definition()->inputs()
                             : std::vector<VALTYPE*>();
}

//! IR-Generic utility, collects all the producers required for the
//!  given list of IR values and returns them along with the original
//!  list in topological order.
template <typename VALTYPE>
std::vector<VALTYPE*> makeSortedEvaluationList(std::vector<VALTYPE*> input) {
  // Deduplicate
  std::vector<VALTYPE*> to_sort;
  std::unordered_set<VALTYPE*> visited;
  for (auto val : input) {
    if (!visited.count(val)) {
      to_sort.push_back(val);
      visited.insert(val);
    }
  }

  std::vector<VALTYPE*> sorted;
  visited.clear();

  // Topological Sort
  while (!to_sort.empty()) {
    auto top_val = to_sort.back();
    if (visited.count(top_val)) {
      to_sort.pop_back();
    } else {
      bool ready_to_pop = true;
      for (auto producer : getImmediateProducers(top_val)) {
        if (!visited.count(producer)) {
          ready_to_pop = false;
          to_sort.push_back(producer);
        }
      }
      if (ready_to_pop) {
        visited.insert(top_val);
        sorted.push_back(top_val);
        to_sort.pop_back();
      }
    }
  }

  return sorted;
}

//! Kernel IR utility, collects all the symbolic values used in allocation
//! nodes.
std::vector<kir::Allocate*> collectBufferSizes(
    const std::vector<Expr*>& exprs) {
  std::vector<kir::Allocate*> buffers;
  std::vector<Expr*> to_visit(exprs);
  while (!to_visit.empty()) {
    auto expr = to_visit.back();
    to_visit.pop_back();
    if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      buffers.push_back(allocate);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      auto for_loop_exprs = for_loop->body().exprs();
      to_visit.insert(
          to_visit.end(), for_loop_exprs.begin(), for_loop_exprs.end());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      auto ite_then_exprs = ite->thenBody().exprs();
      auto ite_else_exprs = ite->elseBody().exprs();
      to_visit.insert(
          to_visit.end(), ite_then_exprs.begin(), ite_then_exprs.end());
      to_visit.insert(
          to_visit.end(), ite_else_exprs.begin(), ite_else_exprs.end());
    }
  }
  return buffers;
}

void bind(std::vector<Val*>& all_values, Val* v) {
  all_values.push_back(v);
}

void bind(std::vector<Val*>& all_values, std::vector<IterDomain*> domain) {
  for (auto d : domain) {
    bind(all_values, d->extent());
  }
}

// 1. Generate extents for IterDomains that compose root domain
// 2. Create new extents using split, merge, reorder operations for rfactor,
// allocation, and leaf domains
void bind(std::vector<Val*>& all_values, nvfuser::TensorView* tv) {
  if (tv->getMemoryType() != MemoryType::Global) {
    return;
  }
  bind(all_values, tv->getRootDomain());
}

} // namespace

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeUnaryOp(
    flatbuffers::FlatBufferBuilder& builder,
    UnaryOp* uop) const {
  serde::DataType dtype = (uop->getUnaryOpType() == nvfuser::UnaryOpType::Cast)
      ? mapToSerdeDtype(uop->out()->getDataType().value())
      : serde::DataType_None;
  auto inst = serde::CreateInstructionDirect(
      builder,
      serde::InstructionType_Unary,
      mapToSerdeUnaryOp(uop->getUnaryOpType()),
      serde::BinaryOpType_None,
      dtype,
      operation_stack_.at(uop->inputs().front()),
      0,
      (int64_t)operation_stack_.size(),
      uop->toString().c_str());
  return inst;
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeBinaryOp(
    flatbuffers::FlatBufferBuilder& builder,
    BinaryOp* bop) const {
  auto inst = serde::CreateInstructionDirect(
      builder,
      serde::InstructionType_Binary,
      serde::UnaryOpType_None,
      mapToSerdeBinaryOp(bop->getBinaryOpType()),
      serde::DataType_None,
      operation_stack_.at(bop->inputs().front()),
      operation_stack_.at(bop->inputs().back()),
      (int64_t)operation_stack_.size(),
      bop->toString().c_str());
  return inst;
}

flatbuffers::Offset<serde::NaiveValueGenerator> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    kir::Kernel* kernel,
    const std::vector<const kir::Allocate*>& global_allocations) {
  // 1) Collect allocation sizes
  std::vector<Val*> all_values;
  for (auto allocate : collectBufferSizes(kernel->topLevelExprs())) {
    if (TensorView* tv = dynamic_cast<TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
    }
  }
  // A deserialized fusion may not contain global allocations in its
  // kir::Kernel. Add global allocations directly to handle this case.
  for (auto allocate : global_allocations) {
    if (TensorView* tv = dynamic_cast<TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
    }
  }

  // 2) Sort values by dependency order
  auto list = makeSortedEvaluationList(all_values);

  // 3) Divide values into NamedScalar, Int, Symbolic, and Derived values
  std::unordered_set<nvfuser::NamedScalar*> named_scalar_values;
  std::unordered_set<nvfuser::Int*> const_int_values;
  std::unordered_set<nvfuser::Val*> symbolic_values;
  std::vector<nvfuser::Val*> derived_values;
  for (auto v : list) {
    if (v->definition() == nullptr) {
      if (NamedScalar* ns = dynamic_cast<NamedScalar*>(v)) {
        named_scalar_values.insert(ns);
      } else if (v->isConstInt()) {
        const_int_values.insert(v->as<nvfuser::Int>());
      } else {
        symbolic_values.insert(v);
      }
    } else {
      derived_values.push_back(v);
    }
  }

  // Add TensorView RootDomain IterDomain Extents for all kernel inputs
  // TODO Get deterministic order
  for (auto input : kernel->inputs()) {
    if (TensorView* tv = dynamic_cast<TensorView*>(input)) {
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        if (!extent->isA<NamedScalar>() && !extent->isConstInt()) {
          symbolic_values.insert(extent);
        }
      }
    }
  }

  // 4) Serialize NaiveValueGenerator by converting each NvFuser value of into
  // an instruction.
  //
  // table NaiveValueGenerator {
  //   instructions : [Instruction];
  // }
  //
  // table Instruction {
  //  instruction : InstructionType;
  //  unary_type : UnaryOpType;
  //  binary_type : BinaryOpType;
  //  data_type : DataType;
  //  src0 : int;
  //  src1 : int;
  //  dest : int;
  //  name : string;
  // }

  using fb_instruction = flatbuffers::Offset<Instruction>;
  std::vector<fb_instruction> instructions_fb;

  for (auto& val : symbolic_values) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_Symbolic,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_Int,
        val->name(),
        0,
        0,
        val->toString().c_str());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(val, operation_stack_.size());
  }

  for (const auto& ns : named_scalar_values) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_NamedString,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_None,
        0,
        0,
        0,
        ns->name().c_str());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(ns, operation_stack_.size());
  }

  for (const auto& int_val : const_int_values) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_Scalar,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_Int,
        int_val->evaluateInt(),
        0,
        0,
        nullptr /* name */);
    instructions_fb.push_back(inst);
    operation_stack_.emplace(int_val, operation_stack_.size());
  }

  for (auto& val : derived_values) {
    auto def = val->definition();
    TORCH_INTERNAL_ASSERT(def, "Expected definition with derived value.");
    if (auto uop = dynamic_cast<UnaryOp*>(def)) {
      auto inst = serializeUnaryOp(builder, uop);
      instructions_fb.push_back(inst);
      operation_stack_.emplace(val, operation_stack_.size());
    } else if (auto bop = dynamic_cast<BinaryOp*>(def)) {
      auto inst = serializeBinaryOp(builder, bop);
      instructions_fb.push_back(inst);
      operation_stack_.emplace(val, operation_stack_.size());
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unknown Expression.");
    }
  }
  return serde::CreateNaiveValueGeneratorDirect(builder, &instructions_fb);
}

std::vector<flatbuffers::Offset<AllocateBuffer>> ExpressionSerializer::
    serialize(
        flatbuffers::FlatBufferBuilder& builder,
        const std::vector<const kir::Allocate*>& allocations) {
  using fb_allocate = flatbuffers::Offset<serde::AllocateBuffer>;
  std::vector<fb_allocate> fb_global_allocations;

  for (auto alloc : allocations) {
    auto alloc_buffer_tv = alloc->buffer()->as<nvfuser::TensorView>();
    TORCH_INTERNAL_ASSERT(alloc_buffer_tv);

    auto fb_alloc = serde::CreateAllocateBuffer(
        builder, serialize(builder, alloc_buffer_tv), alloc->zeroInit());
    fb_global_allocations.push_back(fb_alloc);
  }
  return fb_global_allocations;
}

// TODO create separate functions for TensorDomain and IterDomain
flatbuffers::Offset<serde::SymbolicTensor> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::TensorView* tv) {
  // Only serialize root domain because we do not support split, merge, reorder
  // operations to move between rfactor, allocate, and leaf domains.
  std::vector<flatbuffers::Offset<IterationDomain>> fb_root_domain;
  for (auto id : tv->getRootDomain()) {
    TORCH_INTERNAL_ASSERT(
        operation_stack_.count(id->extent()),
        "Missing value in NaiveValueGenerator stack.");
    auto extent_id = operation_stack_.at(id->extent());
    fb_root_domain.push_back(serde::CreateIterationDomain(builder, extent_id));
  }

  return serde::CreateSymbolicTensor(
      builder,
      mapToSerdeDtype(tv->getDataType().value()),
      serde::CreateDomainDirect(builder, &fb_root_domain));
}

ExpressionBuilder::ExpressionBuilder(kir::Kernel* kernel) : kernel_(kernel) {
  // Add TensorView RootDomain IterDomain Extents for all kernel inputs
  // TODO Get deterministic order
  std::unordered_set<nvfuser::Val*> symbolic_values;
  for (auto input : kernel->inputs()) {
    if (TensorView* tv = dynamic_cast<TensorView*>(input)) {
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        if (!extent->isA<NamedScalar>() && !extent->isConstInt()) {
          symbolic_values.insert(extent);
        }
      }
    }
  }
  operation_stack_.insert(
      operation_stack_.end(), symbolic_values.begin(), symbolic_values.end());
}

void ExpressionBuilder::deserialize(const NaiveValueGenerator* buffer) {
  // table NaiveValueGenerator {
  //   instructions : [Instruction];
  // }
  for (auto inst : *buffer->instructions()) {
    deserialize(inst);
  }
}

void ExpressionBuilder::deserialize(const Instruction* buffer) {
  // table Instruction {
  //  instruction : InstructionType;
  //  unary_type : UnaryOpType;
  //  binary_type : BinaryOpType;
  //  data_type : DataType;
  //  src0 : int;
  //  src1 : int;
  //  dest : int;
  //  name : string;
  // }
  FusionGuard fg(kernel_);
  switch (buffer->instruction()) {
    case serde::InstructionType_Symbolic:
      // Add check for symbolic extent
      break;
    case serde::InstructionType_NamedString: {
      auto ns = IrBuilder::create<NamedScalar>(
          buffer->name()->str(), nvfuser::DataType::Int);
      operation_stack_.push_back(ns);
      break;
    }
    case serde::InstructionType_Scalar: {
      auto int_val = IrBuilder::create<nvfuser::Int>(buffer->src0());
      operation_stack_.push_back(int_val);
      break;
    }
    case serde::InstructionType_Unary: {
      auto uop = buildUnaryOp(buffer);
      operation_stack_.push_back(uop);
      break;
    }
    case serde::InstructionType_Binary: {
      auto bop = buildBinaryOp(buffer);
      operation_stack_.push_back(bop);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported instruction.");
  }
}

Val* ExpressionBuilder::buildUnaryOp(const Instruction* buffer) {
  switch (buffer->unary_type()) {
    case serde::UnaryOpType_Cast:
      return castOp(
          mapToDtypeStruct(buffer->data_type()),
          operation_stack_.at(buffer->src0()));
    case serde::UnaryOpType_Neg:
      return neg(operation_stack_.at(buffer->src0()));
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported binary operation.\t");
      return nullptr;
  }
}

Val* ExpressionBuilder::buildBinaryOp(const Instruction* buffer) {
  switch (buffer->binary_type()) {
    case serde::BinaryOpType_Add:
      return add(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_CeilDiv:
      return ceilDiv(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Div:
      return div(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Mod:
      return mod(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Mul:
      return mul(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Sub:
      return sub(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported binary operation.\t");
      return nullptr;
  }
}

std::vector<const kir::Allocate*> ExpressionBuilder::deserialize(
    const ExpressionBuilder::Allocations* buffers) {
  // table IterationDomain {
  //  extent : long;
  // }
  //
  // table Domain {
  //  dims : [IterationDomain];
  // }
  //
  // table SymbolicTensor {
  //  dtype : DataType;
  //  root : Domain;
  //  rfactor : Domain;
  //  allocate : Domain;
  //  leaf : Domain;
  // }
  //
  // table AllocateBuffer {
  //  tv : SymbolicTensor;
  //  zero_init : bool;
  // }
  FusionGuard fg(kernel_);

  std::vector<const kir::Allocate*> results;
  for (auto buffer : *buffers) {
    std::vector<IterDomain*> new_buffer_ids;
    for (auto fb_id : *buffer->tv()->root()->dims()) {
      auto id = IrBuilder::create<IterDomain>(IterDomainBuilder(
          kernel_->zeroVal(), operation_stack_.at(fb_id->extent())));
      new_buffer_ids.push_back(id);
    }

    const auto buffer_domain = IrBuilder::create<TensorDomain>(new_buffer_ids);

    const auto buffer_tv = IrBuilder::create<TensorView>(
        buffer_domain,
        mapToNvfuserDtype(buffer->tv()->dtype()),
        MemoryType::Global);

    auto node = IrBuilder::create<kir::Allocate>(
        buffer_tv, buffer_tv->getMemoryType(), nullptr, buffer->zero_init());

    results.push_back(node);
  }
  return results;
}

} // namespace nvfuser::serde