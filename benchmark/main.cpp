// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "utils.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "cupti_test.h"

namespace {

std::string getHostName() {
#if defined(__linux__)
  constexpr int len = 128;
  char name[len];
  gethostname(name, len);
  return std::string(name);
#else
  return "UNKNOWN";
#endif
}

std::string getDeviceName() {
  int dev_idx;
  CUDA_RT_SAFE_CALL(cudaGetDevice(&dev_idx));

  cudaDeviceProp prop;
  CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, dev_idx));

  return std::string(prop.name);
}

} // namespace

// Copied from BENCHMARK_MAIN with extra custom settings
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  ::benchmark::AddCustomContext("Host", getHostName());
  ::benchmark::AddCustomContext("GPU", getDeviceName());

  ProfilingData_t* prof = nullptr;
  if (getenv("CUPTI")) {
    prof = initCupti(argc, argv);
  }

  ::benchmark::RunSpecifiedBenchmarks();

  if (getenv("CUPTI")) {
    finishCupti(prof);
  }

  ::benchmark::Shutdown();
  return 0;
}
