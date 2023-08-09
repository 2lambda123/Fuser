
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <executor_kernel_arg.h>
#include <serde/factory.h>
#include <serde/fusion_cache_generated.h>
#include <functional>
#include <memory>

namespace nvfuser::serde {

class PolymorphicValueFactory
    : public Factory<serde::PolymorphicValue, nvfuser::PolymorphicValue> {
 public:
  PolymorphicValueFactory() : Factory((serde::PolymorphicValueData_MAX + 1)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();
};

nvfuser::PolymorphicValue deserializePolymorphicValue(const serde::Scalar* c);

flatbuffers::Offset<serde::PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    std::shared_ptr<nvfuser::PolymorphicValue> v);

flatbuffers::Offset<serde::Scalar> serializeScalarCpu(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    bool v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    int64_t v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    double v,
    nvfuser::DataType t);

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    c10::complex<double> v,
    nvfuser::DataType t);

} // namespace nvfuser::serde
