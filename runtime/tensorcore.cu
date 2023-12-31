// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// Utility macro for this file

// MMA instruction wrappers:
//  The wrappers are subroutines that implement matrix of size
//    A(M,K) X B(K,N) = C(M,N)
//  The naming of the wrappers follow similar naming conventions
//    as the mma instructions.
//  All the mma macros follow the namespace and naming like
//    Arch::M (M-dim) N (N-dim) K(K-dim) (Layout), eg.
//    Volta::M16N16K4TT,
//  with the dimensions describing the size of the sub-matrices being
//   multiplied by this wrapper.
//  see [Operand Layout Convention] in mma_type.h for details on the layout
//   notation.

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

__device__ inline void M16N8K16TN(
    Array<float, 4, 1>& C,
    Array<unsigned, 4, 1>& A,
    Array<unsigned, 2, 1>& B) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[2]),
        "r"(A[3]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
}

__device__ inline void M16N16K16TN(
    Array<float, 8, 1>& C,
    Array<unsigned, 4, 1>& A,
    Array<unsigned, 4, 1>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 1>*>(&C);
  auto* _B = reinterpret_cast<Array<unsigned, 2, 1>*>(&B);
  M16N8K16TN(_C[0], A, _B[0]);
  M16N8K16TN(_C[1], A, _B[1]);
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace Ampere {

__device__ inline void M16N8K16TNF16(
    Array<float, 4, 1>& C,
    Array<unsigned, 4, 1>& A,
    Array<unsigned, 2, 1>& B) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(A[2]),
        "r"(A[3]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
}

__device__ inline void M16N8K16TNBF16(
    Array<float, 4, 1>& C,
    Array<unsigned, 4, 1>& A,
    Array<unsigned, 2, 1>& B) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(A[2]),
        "r"(A[3]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
}

__device__ inline void M16N16K16TNF16(
    Array<float, 8, 1>& C,
    Array<unsigned, 4, 1>& A,
    Array<unsigned, 4, 1>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 1>*>(&C);
  auto* _B = reinterpret_cast<Array<unsigned, 2, 1>*>(&B);
  M16N8K16TNF16(_C[0], A, _B[0]);
  M16N8K16TNF16(_C[1], A, _B[1]);
}

__device__ inline void M16N16K16TNBF16(
    Array<float, 8, 1>& C,
    Array<unsigned, 4, 1>& A,
    Array<unsigned, 4, 1>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 1>*>(&C);
  auto* _B = reinterpret_cast<Array<unsigned, 2, 1>*>(&B);
  M16N8K16TNBF16(_C[0], A, _B[0]);
  M16N8K16TNBF16(_C[1], A, _B[1]);
}

} // namespace Ampere

#endif // Arch 80
