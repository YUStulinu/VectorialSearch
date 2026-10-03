// distance.hpp — distance kernels and the runtime choice between them.
//
// Distance is where essentially all the time goes. A single search touches
// a few thousand vectors, and every one of them is a full pass over `dim`
// floats, so this is the one piece worth writing three times.
//
// The three versions are chosen at *runtime*, not compile time. That matters:
// a binary compiled with -mavx2 crashes with an illegal instruction on a CPU
// without AVX2, so the AVX2 code lives in its own translation unit, compiled
// with AVX2 enabled, and is only ever called after asking the CPU what it
// supports.
#pragma once

#include <cstddef>

#include "vecsearch/types.hpp"

namespace vecsearch {

/// Signature every kernel shares. `dim` is the padded dimension.
using DistanceFn = dist_t (*)(const float* a, const float* b, std::size_t dim);

namespace detail {

// Always available, used as the reference implementation in tests.
dist_t l2SquaredScalar(const float* a, const float* b, std::size_t dim);
dist_t innerProductScalar(const float* a, const float* b, std::size_t dim);

// Compiled in distance_avx2.cpp with AVX2+FMA enabled. These require
// dim % 8 == 0 and must not be called unless the CPU supports AVX2.
dist_t l2SquaredAvx2(const float* a, const float* b, std::size_t dim);
dist_t innerProductAvx2(const float* a, const float* b, std::size_t dim);

}  // namespace detail

/// What the CPU this process is running on can actually do.
struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
};

const CpuFeatures& cpuFeatures();

/// Human-readable name of the kernel that would be selected, for the banner
/// in the benchmark output — a benchmark that doesn't say which code path it
/// measured is not worth much.
const char* activeKernelName(Metric metric, std::size_t dim);

/// Picks the fastest correct kernel for this metric, dimension and CPU.
DistanceFn selectDistanceFn(Metric metric, std::size_t dim);

/// Forces the scalar path. Used by the tests to compare the two.
DistanceFn scalarDistanceFn(Metric metric);

/// Rounds a dimension up to a multiple of 8.
///
/// Vectors are stored padded with zeros to this width, which removes the
/// leftover-elements loop from every kernel. The padding cannot change the
/// result: zeros add nothing to a dot product, and (0-0)² adds nothing to a
/// squared distance.
inline std::size_t paddedDim(std::size_t dim) {
    return ((dim + 7) / 8) * 8;
}

}  // namespace vecsearch
