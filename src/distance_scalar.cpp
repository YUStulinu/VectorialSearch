// distance_scalar.cpp — the reference implementations.
//
// Compiled without any special instruction flags so this file runs anywhere.
// The tests check the SIMD kernels against these, which is the only way to be
// confident the hand-vectorised versions are actually correct.
#include "vecsearch/distance.hpp"

namespace vecsearch::detail {

dist_t l2SquaredScalar(const float* a, const float* b, std::size_t dim) {
    // Four accumulators instead of one: floating-point addition is not
    // associative, so the compiler may not split the chain itself, and a
    // single accumulator serialises on the adder's latency.
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float d0 = a[i + 0] - b[i + 0];
        const float d1 = a[i + 1] - b[i + 1];
        const float d2 = a[i + 2] - b[i + 2];
        const float d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    float tail = 0.f;
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        tail += d * d;
    }
    return (s0 + s1) + (s2 + s3) + tail;
}

dist_t innerProductScalar(const float* a, const float* b, std::size_t dim) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float tail = 0.f;
    for (; i < dim; ++i) tail += a[i] * b[i];

    // Returned as a *distance*, so that smaller is always nearer and every
    // comparison in the graph code can stay a plain `<`.
    return 1.0f - ((s0 + s1) + (s2 + s3) + tail);
}

}  // namespace vecsearch::detail
