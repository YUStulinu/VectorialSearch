// hnsw.hpp — a Hierarchical Navigable Small World index.
//
// WHAT THE ALGORITHM DOES
//
// Finding the nearest vector by checking all of them is exact but linear: a
// million 384-dimension vectors is 1.5 GB to read per query. HNSW trades a
// little accuracy for a very large constant-factor win, and the trade is
// explicit — you choose it per query with `ef`.
//
// The structure is a stack of graphs:
//
//   layer 2      A . . . . . . . . E              sparse, long hops
//   layer 1      A . . C . . . . . E . . H        denser
//   layer 0      A B C D E F G H I J K L M        every vector
//
// Each vector gets a random level from an exponential distribution, so the
// upper layers hold exponentially fewer points. A search enters at the top,
// greedily walks to the closest node it can reach, drops a layer, and
// repeats. The upper layers cover ground quickly; the bottom one refines.
// The result is roughly logarithmic hops instead of linear scanning.
//
// Two details do most of the work for quality:
//
//  * `ef` — how many candidates stay in play during a layer search. Larger
//    means a wider beam, better recall, more distance computations. This is
//    the accuracy dial, and it costs nothing to change after the build.
//
//  * The neighbour selection heuristic. Keeping a node's M *closest*
//    neighbours sounds right and works badly: in a dense cluster every node
//    links only inside the cluster, and the graph falls into disconnected
//    islands a search can never escape. Instead a candidate is kept only if
//    it is closer to the node than to any neighbour already chosen, which
//    keeps links pointing in different directions and holds the graph
//    together.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/storage.hpp"
#include "vecsearch/types.hpp"

namespace vecsearch {

/// Counters worth reporting, mostly to make benchmark numbers explainable.
struct IndexStats {
    std::size_t count = 0;
    std::size_t deleted = 0;
    std::size_t dim = 0;
    std::size_t padded_dim = 0;
    std::size_t M = 0;
    std::size_t ef_construction = 0;
    int max_level = -1;
    std::size_t memory_bytes = 0;
    std::uint64_t distance_calls = 0;  // cumulative, for measuring search work
    const char* kernel = "";
};

class HnswIndex {
public:
    explicit HnswIndex(const IndexConfig& config);
    ~HnswIndex();

    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    HnswIndex(HnswIndex&&) noexcept;
    HnswIndex& operator=(HnswIndex&&) noexcept;

    /// Grows the backing storage. Called automatically, but doing it up front
    /// avoids reallocation during a large build.
    void reserve(std::size_t capacity);

    /// Inserts one vector. Thread-safe against other inserts.
    id_t add(const float* vector, label_t label);

    /// Inserts many, optionally across several threads.
    ///
    /// Insertion mutates shared graph structure, so each node carries its own
    /// lock. Threads contend only when they touch the same neighbourhood,
    /// which is rare once the graph has some size. The resulting graph differs
    /// slightly run to run — the interleaving changes which links survive
    /// pruning — so a single thread is used when `threads <= 1` and that path
    /// stays exactly reproducible.
    void addBatch(const float* vectors, const label_t* labels, std::size_t count, int threads = 1);

    /// Finds the `k` nearest. `ef` is the search beam; it must be >= k, and
    /// larger values mean better recall at a proportional cost.
    std::vector<SearchResult> search(const float* query, std::size_t k, std::size_t ef) const;

    /// Exact scan over every live vector. Slow by design: this is the ground
    /// truth that recall is measured against.
    std::vector<SearchResult> bruteForce(const float* query, std::size_t k) const;

    /// Soft delete. The node stays in the graph so paths through it still
    /// work; it is simply never returned. Removing nodes outright would tear
    /// holes in the connectivity the search depends on.
    bool markDeleted(label_t label);
    bool isDeleted(label_t label) const;

    void save(const std::string& path) const;

    /// Reads the whole index into memory. Use when it will be written to.
    static HnswIndex load(const std::string& path);

    /// Maps the file instead of reading it. Opening is instant whatever the
    /// size, pages arrive on demand, and several processes share one copy in
    /// RAM. The index is read-only in this mode.
    static HnswIndex openMapped(const std::string& path);

    std::size_t size() const noexcept { return count_; }
    std::size_t dim() const noexcept { return config_.dim; }
    Metric metric() const noexcept { return config_.metric; }
    bool readOnly() const noexcept { return read_only_; }

    IndexStats stats() const;
    void resetCounters() const { distance_calls_.store(0, std::memory_order_relaxed); }

private:
    struct Scratch;
    class ScratchPool;

    // --- layout helpers ------------------------------------------------
    const float* vectorAt(id_t id) const { return vectors_ + static_cast<std::size_t>(id) * padded_dim_; }
    float* mutableVectorAt(id_t id) { return vector_store_.as<float>() + static_cast<std::size_t>(id) * padded_dim_; }

    /// Links are stored as [count, n0, n1, ...] in a fixed-width slot.
    id_t* links(id_t id, int level);
    const id_t* readLinks(id_t node, int level, std::vector<id_t>& buffer) const;
    const id_t* links(id_t id, int level) const;
    std::size_t maxLinks(int level) const { return level == 0 ? M0_ : config_.M; }

    // --- algorithm ------------------------------------------------------
    int randomLevel();
    dist_t distance(const float* a, const float* b) const;

    /// Greedy descent through one upper layer: hop to a closer neighbour
    /// until none is closer. Used above the layers where the full beam
    /// search runs.
    id_t greedyDescend(const float* query, id_t entry, dist_t& entryDist, int level) const;

    /// The beam search, used for building (ef_construction) and querying (ef).
    /// Returns a max-heap: the furthest kept candidate sits on top, which is
    /// what makes "is this new candidate good enough" a single comparison.
    void searchLayer(const float* query, id_t entry, std::size_t ef, int level,
                     Scratch& scratch, bool collect_deleted) const;

    /// The diversity heuristic described at the top of this file.
    void selectNeighbours(const float* base, std::vector<std::pair<dist_t, id_t>>& candidates,
                          std::size_t M, std::vector<id_t>& out) const;

    void connect(id_t node, int level, const std::vector<id_t>& neighbours);

    IndexConfig config_;
    std::size_t padded_dim_ = 0;
    std::size_t M0_ = 0;          // layer-0 fan-out, twice M by convention
    double level_lambda_ = 0.0;   // 1 / ln(M)

    // Vector data: either owned (build/load) or a window into a mapping.
    AlignedBuffer vector_store_;
    const float* vectors_ = nullptr;

    std::vector<std::uint8_t> levels_;
    std::vector<label_t> labels_;
    std::vector<std::uint8_t> deleted_;      // uint8 not bool: vector<bool> is not safely addressable per element
    std::vector<id_t> links0_;
    std::vector<std::uint64_t> upper_offset_;
    std::vector<id_t> upper_blob_;

    // Views used when the index is memory-mapped.
    MappedFile mapping_;
    const std::uint8_t* mapped_levels_ = nullptr;
    const label_t* mapped_labels_ = nullptr;
    const std::uint8_t* mapped_deleted_ = nullptr;
    const id_t* mapped_links0_ = nullptr;
    const std::uint64_t* mapped_upper_offset_ = nullptr;
    const id_t* mapped_upper_blob_ = nullptr;

    std::size_t count_ = 0;
    std::size_t capacity_ = 0;
    std::size_t deleted_count_ = 0;
    id_t entry_point_ = kInvalidId;
    int max_level_ = -1;
    bool read_only_ = false;

    DistanceFn distance_fn_ = nullptr;
    mutable std::atomic<std::uint64_t> distance_calls_{0};

    /// True while addBatch is running threads, which forbids any reallocation
    /// of structures other threads hold raw pointers into.
    std::atomic<bool> concurrent_build_{false};

    // One lock per node for concurrent inserts, plus one guarding the entry
    // point and level counter.
    mutable std::vector<std::mutex> node_locks_;
    mutable std::mutex global_lock_;
    mutable std::mutex resize_lock_;

    std::unique_ptr<ScratchPool> scratch_pool_;
    std::uint64_t rng_state_ = 0;
};

}  // namespace vecsearch
