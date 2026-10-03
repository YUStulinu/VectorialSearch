// distance_avx2.cpp — the vectorised kernels.
//
// This file is compiled with AVX2 and FMA enabled (see CMakeLists / Makefile).
// Nothing here may be called unless cpuFeatures().avx2 is true: on an older
// CPU these instructions do not decode and the process dies instantly with
// an illegal-instruction fault.
//
// Both kernels assume dim is a multiple of 8, which the index guarantees by
// padding stored vectors with zeros.
#include <immintrin.h>

#include "vecsearch/distance.hpp"

namespace vecsearch::detail {

namespace {

/// Adds across the eight lanes of a register down to one float.
inline float horizontalSum(__m256 v) {
    // Fold the upper 128 bits onto the lower, then two shuffles.
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    low = _mm_add_ps(low, high);
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    low = _mm_add_ss(low, _mm_shuffle_ps(low, low, 0x55));
    return _mm_cvtss_f32(low);
}

}  // namespace

dist_t l2SquaredAvx2(const float* a, const float* b, std::size_t dim) {
    // Two independent accumulators, so two FMAs can be in flight at once;
    // one chain would stall on the ~4-cycle latency of each FMA.
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }

    return horizontalSum(_mm256_add_ps(acc0, acc1));
}

dist_t innerProductAvx2(const float* a, const float* b, std::size_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= dim; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }

    return 1.0f - horizontalSum(_mm256_add_ps(acc0, acc1));
}

}  // namespace vecsearch::detail
