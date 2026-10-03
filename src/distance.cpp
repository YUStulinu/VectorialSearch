// distance.cpp — asks the CPU what it supports, then picks a kernel.
#include "vecsearch/distance.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cstring>

namespace vecsearch {

namespace {

void cpuidEx(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#else
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    regs[0] = static_cast<int>(eax);
    regs[1] = static_cast<int>(ebx);
    regs[2] = static_cast<int>(ecx);
    regs[3] = static_cast<int>(edx);
#endif
}

/// AVX requires the operating system to have enabled saving the wider
/// registers on a context switch. The CPU reporting the instructions is not
/// enough on its own, so XGETBV is checked too.
bool osSupportsAvx() {
    int regs[4];
    cpuidEx(1, 0, regs);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    const unsigned long long xcr0 = (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
    // Bit 1: SSE state saved. Bit 2: AVX state saved.
    return (xcr0 & 0x6) == 0x6;
}

CpuFeatures detectFeatures() {
    CpuFeatures f;
    int regs[4];

    cpuidEx(0, 0, regs);
    const int maxLeaf = regs[0];

    cpuidEx(1, 0, regs);
    f.fma = (regs[2] & (1 << 12)) != 0;

    if (maxLeaf >= 7 && osSupportsAvx()) {
        cpuidEx(7, 0, regs);
        f.avx2 = (regs[1] & (1 << 5)) != 0;
    }
    if (!osSupportsAvx()) f.fma = false;
    return f;
}

}  // namespace

const CpuFeatures& cpuFeatures() {
    // Function-local static: computed once, thread-safely, on first use.
    static const CpuFeatures features = detectFeatures();
    return features;
}

bool canUseAvx2(std::size_t dim) {
    const CpuFeatures& f = cpuFeatures();
    return f.avx2 && f.fma && (dim % 8 == 0);
}

DistanceFn selectDistanceFn(Metric metric, std::size_t dim) {
    if (canUseAvx2(dim)) {
        return metric == Metric::L2 ? &detail::l2SquaredAvx2 : &detail::innerProductAvx2;
    }
    return scalarDistanceFn(metric);
}

DistanceFn scalarDistanceFn(Metric metric) {
    return metric == Metric::L2 ? &detail::l2SquaredScalar : &detail::innerProductScalar;
}

const char* activeKernelName(Metric metric, std::size_t dim) {
    if (canUseAvx2(dim)) return metric == Metric::L2 ? "l2 (AVX2+FMA)" : "ip (AVX2+FMA)";
    return metric == Metric::L2 ? "l2 (scalar)" : "ip (scalar)";
}

}  // namespace vecsearch
