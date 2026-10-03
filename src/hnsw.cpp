#include "vecsearch/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>  // _mm_prefetch
#endif

namespace vecsearch {

namespace {

/// Candidate queues. `Nearest` puts the smallest distance on top (the next
/// node to explore); `Furthest` puts the largest on top (the first to evict).
using Candidate = std::pair<dist_t, id_t>;
struct NearestFirst {
    bool operator()(const Candidate& a, const Candidate& b) const { return a.first > b.first; }
};
using NearestQueue = std::priority_queue<Candidate, std::vector<Candidate>, NearestFirst>;
using FurthestQueue = std::priority_queue<Candidate, std::vector<Candidate>>;

/**
 * Asks the CPU to start loading an address now, so it has arrived by the time
 * the distance kernel wants it.
 *
 * Worth having on every compiler, not just GCC: the measurements showed this
 * code spends more time waiting on memory than doing arithmetic, so leaving
 * MSVC without a prefetch would hand Windows builds a slower search for no
 * reason.
 */
inline void prefetch(const void* address) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address, 0, 1);
#elif defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
#else
    (void)address;
#endif
}

/// splitmix64: small, fast, and good enough for choosing levels. Keeping our
/// own means the level sequence is identical on every platform and compiler,
/// which std::mt19937 plus a distribution does not guarantee.
inline std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline double nextUniform(std::uint64_t& state) {
    // 53 significant bits, in (0, 1).
    return ((splitmix64(state) >> 11) + 1) * (1.0 / 9007199254740994.0);
}

}  // namespace

/**
 * Per-search working memory.
 *
 * The visited set is the hot structure in the whole algorithm, consulted once
 * per edge. A hash set would dominate the profile, so instead there is one
 * slot per node holding the id of the search that last touched it: "visited"
 * means `stamp[node] == current`. Clearing is just incrementing a counter,
 * which turns an O(n) reset into O(1).
 */
struct HnswIndex::Scratch {
    std::vector<std::uint32_t> stamps;
    std::uint32_t current = 0;
    NearestQueue candidates;
    FurthestQueue results;
    /// Somewhere to copy a neighbour list into while a build is running, so
    /// the list is not read while another thread rewrites it.
    std::vector<id_t> link_buffer;

    void prepare(std::size_t capacity) {
        if (stamps.size() < capacity) stamps.resize(capacity + 1024, 0);
        if (++current == 0) {  // wrapped after 4 billion searches
            std::fill(stamps.begin(), stamps.end(), 0);
            current = 1;
        }
        while (!candidates.empty()) candidates.pop();
        while (!results.empty()) results.pop();
    }

    bool visit(id_t id) {
        // During a concurrent build another thread can link a brand-new node
        // into a neighbourhood this search is walking, so an id beyond the
        // size captured at prepare() time is possible. The branch is never
        // taken in the steady state and predicts perfectly.
        if (id >= stamps.size()) stamps.resize(static_cast<std::size_t>(id) + 1024, 0);
        if (stamps[id] == current) return false;
        stamps[id] = current;
        return true;
    }
};

/// Hands out Scratch objects so concurrent searches never share one, without
/// allocating a fresh visited array per query.
class HnswIndex::ScratchPool {
public:
    std::unique_ptr<Scratch> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) return std::make_unique<Scratch>();
        auto s = std::move(pool_.back());
        pool_.pop_back();
        return s;
    }
    void release(std::unique_ptr<Scratch> s) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.size() < 64) pool_.push_back(std::move(s));
    }

private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<Scratch>> pool_;
};

/* ==================================================================== */

HnswIndex::HnswIndex(const IndexConfig& config)
    : config_(config),
      padded_dim_(paddedDim(config.dim)),
      M0_(config.M * 2),
      level_lambda_(1.0 / std::log(static_cast<double>(std::max<std::size_t>(config.M, 2)))),
      scratch_pool_(std::make_unique<ScratchPool>()),
      rng_state_(config.seed) {
    if (config.dim == 0) throw std::invalid_argument("dim must be greater than 0");
    if (config.M < 2) throw std::invalid_argument("M must be at least 2");
    if (config.ef_construction < config.M) {
        throw std::invalid_argument("ef_construction should be at least M");
    }
    distance_fn_ = selectDistanceFn(config.metric, padded_dim_);
    if (config.capacity) reserve(config.capacity);
}

HnswIndex::~HnswIndex() = default;

// Mutexes and atomics are neither movable nor copyable, so the compiler
// cannot generate these. Locks are recreated at the right size rather than
// transferred — moving an index while another thread holds one of its locks
// would be a bug regardless.
HnswIndex::HnswIndex(HnswIndex&& other) noexcept
    : config_(other.config_),
      padded_dim_(other.padded_dim_),
      M0_(other.M0_),
      level_lambda_(other.level_lambda_),
      vector_store_(std::move(other.vector_store_)),
      vectors_(other.vectors_),
      levels_(std::move(other.levels_)),
      labels_(std::move(other.labels_)),
      deleted_(std::move(other.deleted_)),
      links0_(std::move(other.links0_)),
      upper_offset_(std::move(other.upper_offset_)),
      upper_blob_(std::move(other.upper_blob_)),
      mapping_(std::move(other.mapping_)),
      mapped_levels_(other.mapped_levels_),
      mapped_labels_(other.mapped_labels_),
      mapped_deleted_(other.mapped_deleted_),
      mapped_links0_(other.mapped_links0_),
      mapped_upper_offset_(other.mapped_upper_offset_),
      mapped_upper_blob_(other.mapped_upper_blob_),
      count_(other.count_),
      capacity_(other.capacity_),
      deleted_count_(other.deleted_count_),
      entry_point_(other.entry_point_),
      max_level_(other.max_level_),
      read_only_(other.read_only_),
      distance_fn_(other.distance_fn_),
      distance_calls_(other.distance_calls_.load(std::memory_order_relaxed)),
      concurrent_build_(false),
      node_locks_(other.capacity_),
      scratch_pool_(std::move(other.scratch_pool_)),
      rng_state_(other.rng_state_) {
    other.vectors_ = nullptr;
    other.count_ = 0;
    other.capacity_ = 0;
    other.entry_point_ = kInvalidId;
}

HnswIndex& HnswIndex::operator=(HnswIndex&& other) noexcept {
    if (this == &other) return *this;
    this->~HnswIndex();
    new (this) HnswIndex(std::move(other));
    return *this;
}

void HnswIndex::reserve(std::size_t capacity) {
    if (capacity <= capacity_ || read_only_) return;

    // Growing while other threads are inserting would move the link blob and
    // the lock array out from under them. addBatch reserves everything up
    // front, so this only fires if someone drives concurrent inserts by hand.
    if (concurrent_build_.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "Cannot grow the index while a multi-threaded build is running. "
            "Set IndexConfig::capacity, or call reserve() before addBatch().");
    }

    AlignedBuffer newStore(capacity * padded_dim_ * sizeof(float));
    std::memset(newStore.data(), 0, newStore.size());
    if (count_) {
        std::memcpy(newStore.data(), vector_store_.data(), count_ * padded_dim_ * sizeof(float));
    }
    vector_store_ = std::move(newStore);
    vectors_ = vector_store_.as<float>();

    levels_.resize(capacity, 0);
    labels_.resize(capacity, 0);
    deleted_.resize(capacity, 0);
    links0_.resize(capacity * (M0_ + 1), 0);
    upper_offset_.resize(capacity, UINT64_MAX);

    // Links above layer 0 go in one flat blob. A node's level comes from a
    // geometric distribution, so the expected number of slots per node is
    // (M+1)/(M-1) — about 1.13 at M=16. Reserving four times that makes a
    // reallocation during a build effectively impossible, which matters
    // because a reallocation would invalidate pointers held by other threads.
    const std::size_t expectedPerNode = (config_.M + 1) / (config_.M - 1) + 1;
    upper_blob_.reserve(capacity * expectedPerNode * 4 + 1024);

    // std::mutex is neither copyable nor movable, so the vector is rebuilt
    // rather than resized.
    std::vector<std::mutex> locks(capacity);
    node_locks_.swap(locks);

    capacity_ = capacity;
}

id_t* HnswIndex::links(id_t id, int level) {
    if (level == 0) return links0_.data() + static_cast<std::size_t>(id) * (M0_ + 1);
    const std::uint64_t base = upper_offset_[id];
    return upper_blob_.data() + base + static_cast<std::size_t>(level - 1) * (config_.M + 1);
}

const id_t* HnswIndex::links(id_t id, int level) const {
    if (level == 0) {
        const id_t* base = mapped_links0_ ? mapped_links0_ : links0_.data();
        return base + static_cast<std::size_t>(id) * (M0_ + 1);
    }
    const std::uint64_t* offsets = mapped_upper_offset_ ? mapped_upper_offset_ : upper_offset_.data();
    const id_t* blob = mapped_upper_blob_ ? mapped_upper_blob_ : upper_blob_.data();
    return blob + offsets[id] + static_cast<std::size_t>(level - 1) * (config_.M + 1);
}

int HnswIndex::randomLevel() {
    const double u = nextUniform(rng_state_);
    return static_cast<int>(-std::log(u) * level_lambda_);
}

dist_t HnswIndex::distance(const float* a, const float* b) const {
    distance_calls_.fetch_add(1, std::memory_order_relaxed);
    return distance_fn_(a, b, padded_dim_);
}

/* ---------------------------- search ------------------------------- */

/**
 * Reads one node's neighbour list.
 *
 * While a multi-threaded build is running, another thread may be rewriting
 * this very list, so it is copied out under the node's lock. That is a real
 * cost — but only during a build. Once the index is built the flag is false,
 * the list is read straight from memory with no lock at all, and queries pay
 * nothing for a safety measure they do not need.
 */
const id_t* HnswIndex::readLinks(id_t node, int level, std::vector<id_t>& buffer) const {
    if (!concurrent_build_.load(std::memory_order_acquire)) {
        return links(node, level);
    }
    const std::size_t slots = maxLinks(level) + 1;
    if (buffer.size() < slots) buffer.resize(slots);
    {
        std::lock_guard<std::mutex> lock(node_locks_[node]);
        const id_t* src = links(node, level);
        const std::size_t n = std::min<std::size_t>(src[0], maxLinks(level));
        buffer[0] = static_cast<id_t>(n);
        std::memcpy(buffer.data() + 1, src + 1, n * sizeof(id_t));
    }
    return buffer.data();
}

id_t HnswIndex::greedyDescend(const float* query, id_t entry, dist_t& entryDist, int level) const {
    std::vector<id_t> buffer;
    bool moved = true;
    while (moved) {
        moved = false;
        const id_t* lnk = readLinks(entry, level, buffer);
        const id_t n = lnk[0];
        for (id_t i = 1; i <= n; ++i) {
            const id_t candidate = lnk[i];
            const dist_t d = distance(query, vectorAt(candidate));
            if (d < entryDist) {
                entryDist = d;
                entry = candidate;
                moved = true;
            }
        }
    }
    return entry;
}

void HnswIndex::searchLayer(const float* query, id_t entry, std::size_t ef, int level,
                            Scratch& scratch, bool collect_deleted) const {
    const std::uint8_t* deletedFlags = mapped_deleted_ ? mapped_deleted_ : deleted_.data();

    const dist_t entryDist = distance(query, vectorAt(entry));
    scratch.visit(entry);
    scratch.candidates.emplace(entryDist, entry);
    if (collect_deleted || !deletedFlags[entry]) scratch.results.emplace(entryDist, entry);

    while (!scratch.candidates.empty()) {
        const Candidate current = scratch.candidates.top();

        // Everything still unexplored is further away than the worst result
        // already held, and the beam is full — nothing reachable from here
        // can improve the answer.
        if (scratch.results.size() >= ef && current.first > scratch.results.top().first) break;
        scratch.candidates.pop();

        const id_t* lnk = readLinks(current.second, level, scratch.link_buffer);
        const id_t n = lnk[0];

        // Pull the neighbours' vectors toward the cache before they are
        // needed. The graph walk is random access over a large array, so the
        // kernel usually waits on memory rather than arithmetic.
        for (id_t i = 1; i <= n; ++i) {
            prefetch(vectorAt(lnk[i]));
        }

        for (id_t i = 1; i <= n; ++i) {
            const id_t candidate = lnk[i];
            if (!scratch.visit(candidate)) continue;

            const dist_t d = distance(query, vectorAt(candidate));
            const bool worthKeeping = scratch.results.size() < ef || d < scratch.results.top().first;
            if (!worthKeeping) continue;

            scratch.candidates.emplace(d, candidate);
            if (collect_deleted || !deletedFlags[candidate]) {
                scratch.results.emplace(d, candidate);
                if (scratch.results.size() > ef) scratch.results.pop();
            }
        }
    }
}

std::vector<SearchResult> HnswIndex::search(const float* query, std::size_t k, std::size_t ef) const {
    if (count_ == 0 || entry_point_ == kInvalidId) return {};
    ef = std::max({ef, k, static_cast<std::size_t>(1)});

    // The query is padded the same way the stored vectors are.
    std::vector<float> padded(padded_dim_, 0.f);
    std::memcpy(padded.data(), query, config_.dim * sizeof(float));

    id_t entry = entry_point_;
    dist_t entryDist = distance(padded.data(), vectorAt(entry));
    for (int level = max_level_; level > 0; --level) {
        entry = greedyDescend(padded.data(), entry, entryDist, level);
    }

    auto scratch = scratch_pool_->acquire();
    scratch->prepare(capacity_);
    searchLayer(padded.data(), entry, ef, 0, *scratch, false);

    std::vector<SearchResult> out;
    out.reserve(std::min(k, scratch->results.size()));
    while (scratch->results.size() > k) scratch->results.pop();
    while (!scratch->results.empty()) {
        const Candidate c = scratch->results.top();
        scratch->results.pop();
        const label_t* lbl = mapped_labels_ ? mapped_labels_ : labels_.data();
        out.push_back({lbl[c.second], c.first});
    }
    std::reverse(out.begin(), out.end());  // the heap yields furthest first

    scratch_pool_->release(std::move(scratch));
    return out;
}

std::vector<SearchResult> HnswIndex::bruteForce(const float* query, std::size_t k) const {
    std::vector<float> padded(padded_dim_, 0.f);
    std::memcpy(padded.data(), query, config_.dim * sizeof(float));

    const std::uint8_t* deletedFlags = mapped_deleted_ ? mapped_deleted_ : deleted_.data();
    const label_t* lbl = mapped_labels_ ? mapped_labels_ : labels_.data();

    FurthestQueue best;
    for (std::size_t i = 0; i < count_; ++i) {
        if (deletedFlags[i]) continue;
        const dist_t d = distance_fn_(padded.data(), vectorAt(static_cast<id_t>(i)), padded_dim_);
        if (best.size() < k) {
            best.emplace(d, static_cast<id_t>(i));
        } else if (d < best.top().first) {
            best.pop();
            best.emplace(d, static_cast<id_t>(i));
        }
    }

    std::vector<SearchResult> out;
    out.reserve(best.size());
    while (!best.empty()) {
        out.push_back({lbl[best.top().second], best.top().first});
        best.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}

/* ---------------------------- insert ------------------------------- */

void HnswIndex::selectNeighbours(const float* /*base*/, std::vector<std::pair<dist_t, id_t>>& candidates,
                                 std::size_t M, std::vector<id_t>& out) const {
    out.clear();
    if (candidates.size() <= M) {
        std::sort(candidates.begin(), candidates.end());
        for (const auto& c : candidates) out.push_back(c.second);
        return;
    }

    std::sort(candidates.begin(), candidates.end());  // nearest first

    // Keep a candidate only if it is closer to the new node than to anything
    // already kept. Taking the M nearest instead would link every node inside
    // its own cluster and leave the graph in disconnected pieces.
    for (const auto& candidate : candidates) {
        if (out.size() >= M) break;
        bool keep = true;
        const float* candidateVec = vectorAt(candidate.second);
        for (const id_t chosen : out) {
            if (distance(candidateVec, vectorAt(chosen)) < candidate.first) {
                keep = false;
                break;
            }
        }
        if (keep) out.push_back(candidate.second);
    }
}

void HnswIndex::connect(id_t node, int level, const std::vector<id_t>& neighbours) {
    id_t* lnk = links(node, level);
    const std::size_t limit = maxLinks(level);
    const std::size_t n = std::min(neighbours.size(), limit);
    lnk[0] = static_cast<id_t>(n);
    for (std::size_t i = 0; i < n; ++i) lnk[i + 1] = neighbours[i];
}

id_t HnswIndex::add(const float* vector, label_t label) {
    if (read_only_) throw std::runtime_error("This index is memory-mapped and cannot be modified.");

    id_t id;
    int level;
    {
        std::lock_guard<std::mutex> lock(resize_lock_);
        if (count_ == capacity_) reserve(capacity_ ? capacity_ * 2 : 1024);
        id = static_cast<id_t>(count_);
        level = randomLevel();

        std::memset(mutableVectorAt(id), 0, padded_dim_ * sizeof(float));
        std::memcpy(mutableVectorAt(id), vector, config_.dim * sizeof(float));
        levels_[id] = static_cast<std::uint8_t>(level);
        labels_[id] = label;
        deleted_[id] = 0;
        std::memset(links0_.data() + static_cast<std::size_t>(id) * (M0_ + 1), 0, (M0_ + 1) * sizeof(id_t));

        if (level > 0) {
            const std::size_t need = static_cast<std::size_t>(level) * (config_.M + 1);
            if (concurrent_build_.load(std::memory_order_acquire) &&
                upper_blob_.size() + need > upper_blob_.capacity()) {
                // Reallocating here would dangle pointers other threads are
                // holding. Refusing loudly beats corrupting the heap quietly.
                throw std::runtime_error(
                    "The upper-link blob needs to grow during a multi-threaded build. "
                    "Reserve more capacity before building.");
            }
            upper_offset_[id] = upper_blob_.size();
            upper_blob_.resize(upper_blob_.size() + need, 0);
        }
        ++count_;
    }

    const float* self = vectorAt(id);

    // The first point becomes the entry point and there is nothing to link to.
    {
        std::lock_guard<std::mutex> lock(global_lock_);
        if (entry_point_ == kInvalidId) {
            entry_point_ = id;
            max_level_ = level;
            return id;
        }
    }

    id_t entry;
    int topLevel;
    {
        std::lock_guard<std::mutex> lock(global_lock_);
        entry = entry_point_;
        topLevel = max_level_;
    }

    dist_t entryDist = distance(self, vectorAt(entry));
    for (int l = topLevel; l > level; --l) {
        entry = greedyDescend(self, entry, entryDist, l);
    }

    auto scratch = scratch_pool_->acquire();
    std::vector<std::pair<dist_t, id_t>> candidates;
    std::vector<id_t> chosen;
    std::vector<id_t> pruneList;

    for (int l = std::min(level, topLevel); l >= 0; --l) {
        scratch->prepare(capacity_);
        // Deleted nodes still take part in building: they are part of the
        // graph's connectivity even though searches never return them.
        searchLayer(self, entry, config_.ef_construction, l, *scratch, true);

        candidates.clear();
        candidates.reserve(scratch->results.size());
        while (!scratch->results.empty()) {
            candidates.push_back(scratch->results.top());
            scratch->results.pop();
        }
        if (candidates.empty()) continue;

        // The nearest candidate at this layer is where the next one starts.
        entry = std::min_element(candidates.begin(), candidates.end())->second;

        selectNeighbours(self, candidates, maxLinks(l), chosen);
        {
            std::lock_guard<std::mutex> lock(node_locks_[id]);
            connect(id, l, chosen);
        }

        // Links are bidirectional: tell each chosen neighbour about this node.
        for (const id_t neighbour : chosen) {
            std::lock_guard<std::mutex> lock(node_locks_[neighbour]);
            id_t* lnk = links(neighbour, l);
            const std::size_t limit = maxLinks(l);

            if (lnk[0] < limit) {
                lnk[++lnk[0]] = id;
                continue;
            }

            // The neighbour is full. Re-run the heuristic over its existing
            // links plus this one, so the connection that gets dropped is the
            // least useful rather than simply the oldest.
            std::vector<std::pair<dist_t, id_t>> existing;
            existing.reserve(limit + 1);
            const float* neighbourVec = vectorAt(neighbour);
            for (id_t i = 1; i <= lnk[0]; ++i) {
                existing.emplace_back(distance(neighbourVec, vectorAt(lnk[i])), lnk[i]);
            }
            existing.emplace_back(distance(neighbourVec, self), id);

            selectNeighbours(neighbourVec, existing, limit, pruneList);
            lnk[0] = static_cast<id_t>(pruneList.size());
            for (std::size_t i = 0; i < pruneList.size(); ++i) lnk[i + 1] = pruneList[i];
        }
    }

    scratch_pool_->release(std::move(scratch));

    {
        std::lock_guard<std::mutex> lock(global_lock_);
        if (level > max_level_) {
            max_level_ = level;
            entry_point_ = id;
        }
    }
    return id;
}

void HnswIndex::addBatch(const float* vectors, const label_t* labels, std::size_t count, int threads) {
    reserve(count_ + count);

    if (threads <= 1) {
        for (std::size_t i = 0; i < count; ++i) {
            add(vectors + i * config_.dim, labels ? labels[i] : static_cast<label_t>(count_));
        }
        return;
    }

    concurrent_build_.store(true, std::memory_order_release);
    struct ClearFlag {
        std::atomic<bool>* flag;
        ~ClearFlag() { flag->store(false, std::memory_order_release); }
    } clearFlag{&concurrent_build_};

    std::atomic<std::size_t> next{0};
    std::vector<std::thread> workers;
    std::exception_ptr failure;
    std::mutex failureLock;

    const int n = std::min<int>(threads, static_cast<int>(std::thread::hardware_concurrency() * 2) + 1);
    for (int t = 0; t < n; ++t) {
        workers.emplace_back([&] {
            try {
                for (;;) {
                    const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) break;
                    add(vectors + i * config_.dim, labels ? labels[i] : static_cast<label_t>(i));
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failureLock);
                if (!failure) failure = std::current_exception();
            }
        });
    }
    for (auto& w : workers) w.join();
    if (failure) std::rethrow_exception(failure);
}

/* ---------------------------- deletes ------------------------------ */

bool HnswIndex::markDeleted(label_t label) {
    const label_t* lbl = mapped_labels_ ? mapped_labels_ : labels_.data();
    for (std::size_t i = 0; i < count_; ++i) {
        if (lbl[i] != label) continue;
        if (read_only_) throw std::runtime_error("A memory-mapped index is read-only.");
        if (deleted_[i]) return false;
        deleted_[i] = 1;
        ++deleted_count_;
        return true;
    }
    return false;
}

bool HnswIndex::isDeleted(label_t label) const {
    const label_t* lbl = mapped_labels_ ? mapped_labels_ : labels_.data();
    const std::uint8_t* flags = mapped_deleted_ ? mapped_deleted_ : deleted_.data();
    for (std::size_t i = 0; i < count_; ++i) {
        if (lbl[i] == label) return flags[i] != 0;
    }
    return false;
}

/* --------------------------- persistence --------------------------- */

void HnswIndex::save(const std::string& path) const {
    FileHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kFormatVersion;
    header.metric = static_cast<std::uint32_t>(config_.metric);
    header.dim = config_.dim;
    header.padded_dim = padded_dim_;
    header.count = count_;
    header.M = config_.M;
    header.M0 = M0_;
    header.ef_construction = config_.ef_construction;
    header.entry_point = entry_point_;
    header.max_level = max_level_;
    header.deleted_count = deleted_count_;

    const std::uint64_t upperBlobCount =
        mapped_upper_blob_ ? header.upper_blob_count : upper_blob_.size();

    std::uint64_t offset = alignUp(sizeof(FileHeader), kVectorAlignment);
    header.vectors_offset = offset;
    offset += count_ * padded_dim_ * sizeof(float);

    header.levels_offset = offset;             offset += count_ * sizeof(std::uint8_t);
    header.labels_offset = alignUp(offset, 8); offset = header.labels_offset + count_ * sizeof(label_t);
    header.deleted_offset = offset;            offset += count_ * sizeof(std::uint8_t);
    header.links0_offset = alignUp(offset, 4); offset = header.links0_offset + count_ * (M0_ + 1) * sizeof(id_t);
    header.upper_offset_offset = alignUp(offset, 8); offset = header.upper_offset_offset + count_ * sizeof(std::uint64_t);
    header.upper_blob_offset = alignUp(offset, 4);
    header.upper_blob_count = upperBlobCount;
    offset = header.upper_blob_offset + upperBlobCount * sizeof(id_t);
    header.file_size = offset;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not open \"" + path + "\" for writing");

    auto writeAt = [&](std::uint64_t at, const void* data, std::size_t bytes) {
        out.seekp(static_cast<std::streamoff>(at));
        if (bytes) out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };

    writeAt(0, &header, sizeof(header));
    writeAt(header.vectors_offset, vectors_, count_ * padded_dim_ * sizeof(float));
    writeAt(header.levels_offset, mapped_levels_ ? mapped_levels_ : levels_.data(), count_);
    writeAt(header.labels_offset, mapped_labels_ ? mapped_labels_ : labels_.data(), count_ * sizeof(label_t));
    writeAt(header.deleted_offset, mapped_deleted_ ? mapped_deleted_ : deleted_.data(), count_);
    writeAt(header.links0_offset, mapped_links0_ ? mapped_links0_ : links0_.data(), count_ * (M0_ + 1) * sizeof(id_t));
    writeAt(header.upper_offset_offset, mapped_upper_offset_ ? mapped_upper_offset_ : upper_offset_.data(),
            count_ * sizeof(std::uint64_t));
    writeAt(header.upper_blob_offset, mapped_upper_blob_ ? mapped_upper_blob_ : upper_blob_.data(),
            upperBlobCount * sizeof(id_t));

    // Make sure the file is exactly file_size bytes even if the last block is
    // empty, so a mapping of it has the expected length.
    out.seekp(static_cast<std::streamoff>(header.file_size) - 1);
    const char zero = 0;
    out.write(&zero, 1);

    out.flush();
    if (!out) throw std::runtime_error("Writing \"" + path + "\" failed");
}

namespace {

void validateHeader(const FileHeader& h, std::size_t fileSize) {
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("Not a vecsearch index file (bad magic bytes)");
    }
    if (h.version != kFormatVersion) {
        throw std::runtime_error("Index file version " + std::to_string(h.version) +
                                 " cannot be read by this build (expected " + std::to_string(kFormatVersion) + ")");
    }
    if (h.file_size > fileSize) {
        throw std::runtime_error("Index file is truncated: the header expects " + std::to_string(h.file_size) +
                                 " bytes but the file is " + std::to_string(fileSize));
    }
    if (h.dim == 0 || h.padded_dim < h.dim || h.M < 2) {
        throw std::runtime_error("Index file header is not self-consistent");
    }
}

IndexConfig configFromHeader(const FileHeader& h) {
    IndexConfig cfg;
    cfg.dim = static_cast<std::size_t>(h.dim);
    cfg.metric = static_cast<Metric>(h.metric);
    cfg.M = static_cast<std::size_t>(h.M);
    cfg.ef_construction = static_cast<std::size_t>(h.ef_construction);
    return cfg;
}

}  // namespace

HnswIndex HnswIndex::load(const std::string& path) {
    MappedFile file;
    file.open(path);
    const auto& header = *reinterpret_cast<const FileHeader*>(file.data());
    validateHeader(header, file.size());

    HnswIndex index(configFromHeader(header));
    const std::size_t count = static_cast<std::size_t>(header.count);
    index.reserve(std::max<std::size_t>(count, 1));

    const std::uint8_t* base = file.data();
    std::memcpy(index.vector_store_.data(), base + header.vectors_offset,
                count * index.padded_dim_ * sizeof(float));
    std::memcpy(index.levels_.data(), base + header.levels_offset, count);
    std::memcpy(index.labels_.data(), base + header.labels_offset, count * sizeof(label_t));
    std::memcpy(index.deleted_.data(), base + header.deleted_offset, count);
    std::memcpy(index.links0_.data(), base + header.links0_offset, count * (index.M0_ + 1) * sizeof(id_t));
    std::memcpy(index.upper_offset_.data(), base + header.upper_offset_offset, count * sizeof(std::uint64_t));

    index.upper_blob_.resize(static_cast<std::size_t>(header.upper_blob_count));
    if (header.upper_blob_count) {
        std::memcpy(index.upper_blob_.data(), base + header.upper_blob_offset,
                    static_cast<std::size_t>(header.upper_blob_count) * sizeof(id_t));
    }

    index.count_ = count;
    index.deleted_count_ = static_cast<std::size_t>(header.deleted_count);
    index.entry_point_ = static_cast<id_t>(header.entry_point);
    index.max_level_ = static_cast<int>(header.max_level);
    return index;
}

HnswIndex HnswIndex::openMapped(const std::string& path) {
    MappedFile file;
    file.open(path);
    file.adviseRandom();

    const auto& header = *reinterpret_cast<const FileHeader*>(file.data());
    validateHeader(header, file.size());

    HnswIndex index(configFromHeader(header));
    const std::uint8_t* base = file.data();

    index.vectors_ = reinterpret_cast<const float*>(base + header.vectors_offset);
    index.mapped_levels_ = base + header.levels_offset;
    index.mapped_labels_ = reinterpret_cast<const label_t*>(base + header.labels_offset);
    index.mapped_deleted_ = base + header.deleted_offset;
    index.mapped_links0_ = reinterpret_cast<const id_t*>(base + header.links0_offset);
    index.mapped_upper_offset_ = reinterpret_cast<const std::uint64_t*>(base + header.upper_offset_offset);
    index.mapped_upper_blob_ = reinterpret_cast<const id_t*>(base + header.upper_blob_offset);

    index.count_ = static_cast<std::size_t>(header.count);
    index.capacity_ = index.count_;
    index.deleted_count_ = static_cast<std::size_t>(header.deleted_count);
    index.entry_point_ = static_cast<id_t>(header.entry_point);
    index.max_level_ = static_cast<int>(header.max_level);
    index.read_only_ = true;
    index.mapping_ = std::move(file);
    return index;
}

IndexStats HnswIndex::stats() const {
    IndexStats s;
    s.count = count_;
    s.deleted = deleted_count_;
    s.dim = config_.dim;
    s.padded_dim = padded_dim_;
    s.M = config_.M;
    s.ef_construction = config_.ef_construction;
    s.max_level = max_level_;
    s.distance_calls = distance_calls_.load(std::memory_order_relaxed);
    s.kernel = activeKernelName(config_.metric, padded_dim_);
    s.memory_bytes = read_only_
        ? mapping_.size()
        : vector_store_.size() + links0_.size() * sizeof(id_t) + upper_blob_.size() * sizeof(id_t) +
              levels_.size() + labels_.size() * sizeof(label_t) + deleted_.size() +
              upper_offset_.size() * sizeof(std::uint64_t);
    return s;
}

}  // namespace vecsearch
