// types.hpp — basic types shared across the library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

namespace vecsearch {

using id_t = std::uint32_t;      // internal node id, dense from 0
using label_t = std::uint64_t;   // the caller's own id, carried through
using dist_t = float;

inline constexpr id_t kInvalidId = 0xFFFFFFFFu;

enum class Metric : std::uint8_t {
    L2 = 0,            // squared Euclidean distance (the square root is never needed for ranking)
    InnerProduct = 1   // 1 - dot(a,b); with unit vectors this is cosine distance
};

/**
 * A heap buffer with a guaranteed alignment.
 *
 * AVX2 loads are happiest on 32-byte boundaries, and the whole point of this
 * project is to let the CPU read eight floats at a time without penalty.
 * 64 bytes is used instead of 32 so a vector also starts on a cache line,
 * which keeps one vector from straddling two lines unnecessarily.
 */
class AlignedBuffer {
public:
    static constexpr std::size_t kAlignment = 64;

    AlignedBuffer() = default;

    explicit AlignedBuffer(std::size_t bytes) { allocate(bytes); }

    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.bytes_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            bytes_ = other.bytes_;
            other.data_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t bytes) {
        release();
        if (bytes == 0) return;
        // Both allocators require the size to be a multiple of the alignment.
        const std::size_t rounded = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
#if defined(_MSC_VER)
        data_ = static_cast<std::uint8_t*>(_aligned_malloc(rounded, kAlignment));
#else
        data_ = static_cast<std::uint8_t*>(std::aligned_alloc(kAlignment, rounded));
#endif
        if (!data_) throw std::bad_alloc();
        bytes_ = rounded;
    }

    void release() {
        if (!data_) return;
#if defined(_MSC_VER)
        _aligned_free(data_);
#else
        std::free(data_);
#endif
        data_ = nullptr;
        bytes_ = 0;
    }

    std::uint8_t* data() noexcept { return data_; }
    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }

    template <typename T> T* as() noexcept { return reinterpret_cast<T*>(data_); }
    template <typename T> const T* as() const noexcept { return reinterpret_cast<const T*>(data_); }

private:
    std::uint8_t* data_ = nullptr;
    std::size_t bytes_ = 0;
};

/** How the graph is built. The defaults are the ones the HNSW paper recommends. */
struct IndexConfig {
    std::size_t dim = 0;
    Metric metric = Metric::L2;

    /// Neighbours kept per node above layer 0. Higher means better recall,
    /// more memory, slower build. 16 is the usual starting point.
    std::size_t M = 16;

    /// How wide the search is while inserting. Affects build quality and
    /// build time, but costs nothing at query time.
    std::size_t ef_construction = 200;

    /// Reserve room up front so insertion never reallocates mid-build.
    std::size_t capacity = 0;

    /// Fixed seed by default, so two builds of the same data are identical —
    /// without that, benchmark numbers would wander between runs.
    std::uint64_t seed = 42;
};

/** Result of a search: one neighbour and how far away it was. */
struct SearchResult {
    label_t label;
    dist_t distance;
};

inline const char* metricName(Metric m) {
    return m == Metric::L2 ? "l2" : "ip";
}

}  // namespace vecsearch
