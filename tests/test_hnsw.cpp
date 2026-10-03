// Correctness tests.
//
// An approximate index is easy to get subtly wrong: a broken graph still
// returns *something*, ranked plausibly, and only a recall measurement
// against an exhaustive scan shows the difference. So most of these tests
// compare against brute force rather than against hard-coded expectations.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "harness.hpp"
#include "vecsearch/distance.hpp"
#include "vecsearch/hnsw.hpp"

using namespace vecsearch;

namespace {

std::vector<float> randomVectors(std::size_t n, std::size_t dim, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = dist(rng);
    return data;
}

/// Fraction of the true nearest neighbours that the approximate search found.
double recallAt(const HnswIndex& index, const std::vector<float>& queries, std::size_t dim,
                std::size_t queryCount, std::size_t k, std::size_t ef) {
    std::size_t hits = 0;
    for (std::size_t q = 0; q < queryCount; ++q) {
        const float* query = queries.data() + q * dim;
        const auto truth = index.bruteForce(query, k);
        const auto found = index.search(query, k, ef);

        std::unordered_set<label_t> truthSet;
        for (const auto& r : truth) truthSet.insert(r.label);
        for (const auto& r : found) {
            if (truthSet.count(r.label)) ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(queryCount * k);
}

HnswIndex buildIndex(const std::vector<float>& data, std::size_t n, std::size_t dim,
                     Metric metric = Metric::L2, std::size_t M = 16, int threads = 1) {
    IndexConfig cfg;
    cfg.dim = dim;
    cfg.metric = metric;
    cfg.M = M;
    cfg.ef_construction = 200;
    cfg.capacity = n;
    HnswIndex index(cfg);

    std::vector<label_t> labels(n);
    for (std::size_t i = 0; i < n; ++i) labels[i] = 1000 + i;
    index.addBatch(data.data(), labels.data(), n, threads);
    return index;
}

/// A writable path for test files.
///
/// "/tmp" does not exist on Windows, so hard-coding it makes the persistence
/// tests fail there for a reason that has nothing to do with the index.
std::string tempPath(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

}  // namespace

TEST(exact_match_is_always_found) {
    const std::size_t n = 2000, dim = 64;
    const auto data = randomVectors(n, dim, 1);
    auto index = buildIndex(data, n, dim);

    // Searching for a vector that is in the index must return that vector
    // first, at distance ~0. If this fails, the graph is disconnected.
    //
    // ef is deliberately generous here. At a small ef an occasional miss is
    // normal and says nothing about correctness, so testing at ef=50 over a
    // few dozen queries would be a coin flip rather than an assertion.
    std::size_t found = 0;
    std::size_t tried = 0;
    for (std::size_t i = 0; i < n; i += 3) {
        ++tried;
        const auto results = index.search(data.data() + i * dim, 1, 100);
        harness::check(!results.empty(), "search returned nothing for an indexed vector");
        if (results[0].label == 1000 + i) ++found;
    }
    harness::atLeast(static_cast<double>(found) / tried, 0.99, "self-retrieval rate at ef=100");
}

TEST(recall_rises_with_ef) {
    const std::size_t n = 5000, dim = 32, k = 10;
    const auto data = randomVectors(n, dim, 2);
    const auto queries = randomVectors(50, dim, 99);
    auto index = buildIndex(data, n, dim);

    const double r10 = recallAt(index, queries, dim, 50, k, 10);
    const double r50 = recallAt(index, queries, dim, 50, k, 50);
    const double r200 = recallAt(index, queries, dim, 50, k, 200);

    std::printf("       recall@10: ef=10 %.3f  ef=50 %.3f  ef=200 %.3f\n", r10, r50, r200);

    // The whole premise of the accuracy dial: a wider beam must not be worse.
    harness::check(r50 >= r10 - 0.02, "ef=50 should not be worse than ef=10");
    harness::check(r200 >= r50 - 0.02, "ef=200 should not be worse than ef=50");
    harness::atLeast(r200, 0.95, "recall at ef=200");
}

TEST(results_are_sorted_and_within_k) {
    const std::size_t n = 1500, dim = 48;
    const auto data = randomVectors(n, dim, 3);
    auto index = buildIndex(data, n, dim);
    const auto queries = randomVectors(5, dim, 7);

    for (std::size_t q = 0; q < 5; ++q) {
        const auto results = index.search(queries.data() + q * dim, 7, 64);
        harness::check(results.size() <= 7, "search returned more than k results");
        harness::check(!results.empty(), "search returned nothing");
        for (std::size_t i = 1; i < results.size(); ++i) {
            harness::check(results[i].distance >= results[i - 1].distance,
                           "results are not ordered by distance");
        }
    }
}

TEST(inner_product_metric_ranks_correctly) {
    const std::size_t n = 2000, dim = 32, k = 5;
    auto data = randomVectors(n, dim, 4);

    // Inner product only behaves like cosine on unit vectors, so normalise.
    for (std::size_t i = 0; i < n; ++i) {
        float norm = 0.f;
        for (std::size_t d = 0; d < dim; ++d) norm += data[i * dim + d] * data[i * dim + d];
        norm = std::sqrt(norm);
        for (std::size_t d = 0; d < dim; ++d) data[i * dim + d] /= norm;
    }

    auto index = buildIndex(data, n, dim, Metric::InnerProduct);
    auto queries = randomVectors(30, dim, 11);
    for (std::size_t q = 0; q < 30; ++q) {
        float norm = 0.f;
        for (std::size_t d = 0; d < dim; ++d) norm += queries[q * dim + d] * queries[q * dim + d];
        norm = std::sqrt(norm);
        for (std::size_t d = 0; d < dim; ++d) queries[q * dim + d] /= norm;
    }

    const double recall = recallAt(index, queries, dim, 30, k, 128);
    std::printf("       inner-product recall@%zu: %.3f\n", k, recall);
    harness::atLeast(recall, 0.95, "inner product recall");
}

TEST(build_is_reproducible_on_one_thread) {
    const std::size_t n = 1200, dim = 24;
    const auto data = randomVectors(n, dim, 5);

    auto a = buildIndex(data, n, dim);
    auto b = buildIndex(data, n, dim);

    const auto queries = randomVectors(20, dim, 13);
    for (std::size_t q = 0; q < 20; ++q) {
        const auto ra = a.search(queries.data() + q * dim, 10, 64);
        const auto rb = b.search(queries.data() + q * dim, 10, 64);
        harness::equal(ra.size(), rb.size(), "two identical builds disagree on result count");
        for (std::size_t i = 0; i < ra.size(); ++i) {
            harness::equal(ra[i].label, rb[i].label, "two identical builds returned different neighbours");
        }
    }
}

TEST(deleted_vectors_are_never_returned) {
    const std::size_t n = 2000, dim = 32;
    const auto data = randomVectors(n, dim, 6);
    auto index = buildIndex(data, n, dim);

    // Delete the true nearest neighbour of a query, then make sure it is gone
    // but its neighbours are still reachable — the node must stay in the graph
    // for connectivity even though it is filtered from results.
    const float* query = data.data();
    const auto before = index.search(query, 5, 100);
    harness::check(!before.empty(), "no results before deleting");

    const label_t victim = before[0].label;
    harness::check(index.markDeleted(victim), "markDeleted reported no change");
    harness::check(index.isDeleted(victim), "the vector is not marked deleted");

    const auto after = index.search(query, 5, 100);
    for (const auto& r : after) {
        harness::check(r.label != victim, "a deleted vector was returned");
    }
    harness::equal(after.size(), std::size_t{5}, "deleting one vector lost other results");
}

TEST(save_and_load_round_trip) {
    const std::size_t n = 1500, dim = 40;
    const auto data = randomVectors(n, dim, 8);
    auto index = buildIndex(data, n, dim);
    index.markDeleted(1000 + 7);

    const std::string path = tempPath("vecsearch_roundtrip.idx");
    index.save(path);

    auto loaded = HnswIndex::load(path);
    harness::equal(loaded.size(), index.size(), "size changed across save/load");
    harness::check(loaded.isDeleted(1000 + 7), "the deleted flag did not survive saving");

    const auto queries = randomVectors(25, dim, 17);
    for (std::size_t q = 0; q < 25; ++q) {
        const auto original = index.search(queries.data() + q * dim, 10, 80);
        const auto reopened = loaded.search(queries.data() + q * dim, 10, 80);
        harness::equal(reopened.size(), original.size(), "result count changed after loading");
        for (std::size_t i = 0; i < original.size(); ++i) {
            harness::equal(reopened[i].label, original[i].label, "a loaded index returned different neighbours");
        }
    }
    std::remove(path.c_str());
}

TEST(memory_mapped_index_matches_the_original) {
    const std::size_t n = 1500, dim = 40;
    const auto data = randomVectors(n, dim, 9);
    auto index = buildIndex(data, n, dim);

    const std::string path = tempPath("vecsearch_mmap.idx");
    index.save(path);

    {
        // Scoped: Windows cannot delete a file while it is mapped, so the
        // mapping must be released before std::remove below.
        auto mapped = HnswIndex::openMapped(path);
        harness::check(mapped.readOnly(), "a mapped index should report itself read-only");
        harness::equal(mapped.size(), index.size(), "mapped size differs");

        const auto queries = randomVectors(25, dim, 19);
        for (std::size_t q = 0; q < 25; ++q) {
            const auto direct = index.search(queries.data() + q * dim, 10, 80);
            const auto viaMap = mapped.search(queries.data() + q * dim, 10, 80);
            for (std::size_t i = 0; i < direct.size(); ++i) {
                harness::equal(viaMap[i].label, direct[i].label, "mapped index returned different neighbours");
                harness::near(viaMap[i].distance, direct[i].distance, 1e-6, "mapped distance differs");
            }
        }
    }
    harness::equal(std::remove(path.c_str()), 0, "the index file could not be deleted after unmapping");
}

TEST(a_corrupt_file_is_rejected_not_crashed_on) {
    const std::string path = tempPath("vecsearch_bad.idx");
    FILE* f = std::fopen(path.c_str(), "wb");
    const char junk[256] = {'N', 'O', 'T', 'A', 'N', 'I', 'D', 'X'};
    std::fwrite(junk, 1, sizeof(junk), f);
    std::fclose(f);

    bool threw = false;
    try {
        auto bad = HnswIndex::openMapped(path);
    } catch (const std::exception&) {
        threw = true;
    }
    harness::check(threw, "a file with bad magic bytes should be rejected");
    std::remove(path.c_str());
}

TEST(multithreaded_build_keeps_recall) {
    const std::size_t n = 4000, dim = 32, k = 10;
    const auto data = randomVectors(n, dim, 21);
    const auto queries = randomVectors(40, dim, 23);

    auto single = buildIndex(data, n, dim, Metric::L2, 16, 1);
    auto multi = buildIndex(data, n, dim, Metric::L2, 16, 4);

    const double r1 = recallAt(single, queries, dim, 40, k, 100);
    const double r4 = recallAt(multi, queries, dim, 40, k, 100);
    std::printf("       recall: 1 thread %.3f, 4 threads %.3f\n", r1, r4);

    harness::equal(multi.size(), single.size(), "the threaded build lost vectors");
    // The graphs differ because insert order interleaves, but quality must not.
    harness::check(r4 >= r1 - 0.03, "the threaded build lost recall");
}

TEST(high_dimensional_vectors_work) {
    // 1536 is OpenAI's embedding width, and 384 is the local model in Second
    // Brain. Both must pad cleanly to a multiple of 8.
    for (std::size_t dim : {384u, 768u, 1536u}) {
        const std::size_t n = 600;
        const auto data = randomVectors(n, dim, 31 + dim);
        auto index = buildIndex(data, n, dim, Metric::L2, 12);
        const auto results = index.search(data.data() + 5 * dim, 1, 64);
        harness::check(!results.empty(), "no result at dim " + std::to_string(dim));
        harness::equal(results[0].label, static_cast<label_t>(1005),
                       "wrong nearest neighbour at dim " + std::to_string(dim));
    }
}

TEST(dimension_not_divisible_by_eight_still_works) {
    // 100 pads to 104. The padding must not change any distance.
    const std::size_t n = 800, dim = 100;
    const auto data = randomVectors(n, dim, 41);
    auto index = buildIndex(data, n, dim);

    const auto results = index.search(data.data() + 11 * dim, 3, 64);
    harness::equal(results[0].label, static_cast<label_t>(1011), "padding broke the distance");
    harness::near(results[0].distance, 0.0, 1e-4, "an indexed vector should be at distance zero from itself");
}

TEST(single_and_empty_index_edge_cases) {
    IndexConfig cfg;
    cfg.dim = 16;
    HnswIndex empty(cfg);
    harness::check(empty.search(std::vector<float>(16, 0.f).data(), 5, 32).empty(),
                   "an empty index should return no results");

    std::vector<float> one(16, 0.5f);
    empty.add(one.data(), 7);
    const auto results = empty.search(one.data(), 5, 32);
    harness::equal(results.size(), std::size_t{1}, "a one-vector index should return exactly one result");
    harness::equal(results[0].label, static_cast<label_t>(7), "wrong label from a one-vector index");
}

int main() {
    std::printf("\nvecsearch tests  (kernel: %s)\n\n", activeKernelName(Metric::L2, 384));
    return harness::run();
}
