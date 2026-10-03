// bench.cpp — the point of the whole exercise.
//
// A vector index is only meaningful as a pair of numbers: how fast, and how
// often right. Quoting latency alone is meaningless, because any index can be
// made arbitrarily fast by returning worse answers. So every row here reports
// recall measured against an exhaustive scan of the same data.
//
//   ./bench --n 100000 --dim 384 --queries 200
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/hnsw.hpp"

using namespace vecsearch;
using Clock = std::chrono::steady_clock;

namespace {

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/**
 * Clustered vectors, not uniform noise.
 *
 * Uniformly random points in high dimensions are the pathological case for
 * every graph index: all distances converge to the same value and there is no
 * structure to navigate. Real embeddings sit in dense clusters, so the data
 * here is points scattered around a set of cluster centres.
 */
std::vector<float> makeCentres(std::size_t clusters, std::size_t dim, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> wide(0.f, 1.f);
    std::vector<float> centres(clusters * dim);
    for (auto& x : centres) x = wide(rng);
    return centres;
}

std::vector<float> sampleAround(const std::vector<float>& centres, std::size_t clusters,
                                std::size_t n, std::size_t dim, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> tight(0.f, 0.22f);
    std::vector<float> data(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = rng() % clusters;
        for (std::size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = centres[c * dim + d] + tight(rng);
        }
    }
    return data;
}

struct Percentiles {
    double p50 = 0, p95 = 0, p99 = 0, mean = 0;
};

Percentiles percentiles(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    Percentiles p;
    auto at = [&](double q) {
        if (samples.empty()) return 0.0;
        const std::size_t i = std::min(samples.size() - 1,
                                       static_cast<std::size_t>(q * samples.size()));
        return samples[i];
    };
    p.p50 = at(0.50);
    p.p95 = at(0.95);
    p.p99 = at(0.99);
    p.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    return p;
}

std::size_t argValue(int argc, char** argv, const char* flag, std::size_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return static_cast<std::size_t>(std::stoull(argv[i + 1]));
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argValue(argc, argv, "--n", 50000);
    const std::size_t dim = argValue(argc, argv, "--dim", 384);
    const std::size_t queryCount = argValue(argc, argv, "--queries", 200);
    const std::size_t k = argValue(argc, argv, "--k", 10);
    const std::size_t M = argValue(argc, argv, "--M", 16);
    const std::size_t efc = argValue(argc, argv, "--ef-construction", 200);
    const std::size_t threads = argValue(argc, argv, "--threads", std::thread::hardware_concurrency());

    // By default queries come from the same clusters as the data, the way a
    // question to a RAG system is about the documents it holds. --ood draws
    // them from clusters that do not exist in the collection: every query is
    // then nearly equidistant from hundreds of clusters, the hardest case a
    // graph index can face, and recall drops the larger the collection gets.
    bool ood = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ood") == 0) ood = true;
    }

    std::printf("\n");
    std::printf("vecsearch benchmark\n");
    std::printf("===================\n");
    std::printf("  vectors          %zu\n", n);
    std::printf("  dimensions       %zu\n", dim);
    std::printf("  queries          %zu  (k = %zu)\n", queryCount, k);
    std::printf("  graph            M = %zu, ef_construction = %zu\n", M, efc);
    std::printf("  distance kernel  %s\n", activeKernelName(Metric::L2, paddedDim(dim)));
    std::printf("  build threads    %zu\n", threads);
    std::printf("  queries drawn    %s\n\n", ood ? "from clusters NOT in the data (--ood, worst case)"
                                               : "from the data's own clusters (realistic)");

    std::printf("generating %.1f MB of clustered test data...\n",
                n * dim * 4.0 / 1048576.0);
    const std::size_t clusters = std::max<std::size_t>(n / 500, 20);
    const auto centres = makeCentres(clusters, dim, 1234);
    const auto data = sampleAround(centres, clusters, n, dim, 77);
    const auto queries = ood
        ? sampleAround(makeCentres(clusters, dim, 999), clusters, queryCount, dim, 555)
        : sampleAround(centres, clusters, queryCount, dim, 555);

    // ---- build ---------------------------------------------------------
    IndexConfig cfg;
    cfg.dim = dim;
    cfg.metric = Metric::L2;
    cfg.M = M;
    cfg.ef_construction = efc;
    cfg.capacity = n;

    HnswIndex index(cfg);
    std::vector<label_t> labels(n);
    std::iota(labels.begin(), labels.end(), 0);

    std::printf("building the index...\n");
    auto t0 = Clock::now();
    index.addBatch(data.data(), labels.data(), n, static_cast<int>(threads));
    const double buildMs = msSince(t0);

    const auto st = index.stats();
    std::printf("  built in %.2f s   (%.0f vectors/sec)\n", buildMs / 1000.0, n / (buildMs / 1000.0));
    std::printf("  graph levels     %d\n", st.max_level + 1);
    std::printf("  memory           %.1f MB  (%.0f bytes/vector, of which %zu are the vector itself)\n\n",
                st.memory_bytes / 1048576.0,
                static_cast<double>(st.memory_bytes) / n,
                dim * sizeof(float));

    // ---- exhaustive scan, for both ground truth and a baseline ---------
    std::printf("exhaustive scan (ground truth, and the thing we are trying to beat)\n");
    std::vector<std::vector<label_t>> truth(queryCount);
    std::vector<double> bruteTimes;
    bruteTimes.reserve(queryCount);

    for (std::size_t q = 0; q < queryCount; ++q) {
        auto qt = Clock::now();
        const auto exact = index.bruteForce(queries.data() + q * dim, k);
        bruteTimes.push_back(msSince(qt));
        truth[q].reserve(exact.size());
        for (const auto& r : exact) truth[q].push_back(r.label);
    }
    const auto brute = percentiles(bruteTimes);
    std::printf("  %.2f ms per query   (%.0f queries/sec)\n\n", brute.mean, 1000.0 / brute.mean);

    // ---- the actual measurement ----------------------------------------
    std::printf("HNSW: recall against that ground truth, and what it costs\n\n");
    std::printf("  %-6s %-9s %-11s %-11s %-11s %-12s %s\n",
                "ef", "recall", "mean", "p95", "p99", "queries/s", "speedup");
    std::printf("  %-6s %-9s %-11s %-11s %-11s %-12s %s\n",
                "----", "------", "----", "---", "---", "---------", "-------");

    for (std::size_t ef : {10u, 20u, 40u, 80u, 160u, 320u}) {
        if (ef < k) continue;

        std::vector<double> times;
        times.reserve(queryCount);
        std::size_t hits = 0;

        // One untimed pass so the caches are in the same state for every row.
        for (std::size_t q = 0; q < std::min<std::size_t>(queryCount, 20); ++q) {
            index.search(queries.data() + q * dim, k, ef);
        }

        index.resetCounters();
        for (std::size_t q = 0; q < queryCount; ++q) {
            auto qt = Clock::now();
            const auto found = index.search(queries.data() + q * dim, k, ef);
            times.push_back(msSince(qt));

            const std::unordered_set<label_t> expected(truth[q].begin(), truth[q].end());
            for (const auto& r : found) {
                if (expected.count(r.label)) ++hits;
            }
        }

        const double recall = static_cast<double>(hits) / (queryCount * k);
        const auto p = percentiles(times);
        const auto counters = index.stats();

        std::printf("  %-6zu %-9.4f %-11.3f %-11.3f %-11.3f %-12.0f %.0fx\n",
                    ef, recall, p.mean, p.p95, p.p99, 1000.0 / p.mean, brute.mean / p.mean);
        std::printf("         %zu distance computations per query, %.2f%% of the collection\n",
                    counters.distance_calls / queryCount,
                    100.0 * counters.distance_calls / queryCount / n);
    }

    // ---- persistence ----------------------------------------------------
    const std::string path = "bench_index.bin";
    std::printf("\npersistence\n");

    t0 = Clock::now();
    index.save(path);
    std::printf("  save              %.2f s\n", msSince(t0) / 1000.0);

    double loadMs;
    {
        t0 = Clock::now();
        auto loaded = HnswIndex::load(path);
        loadMs = msSince(t0);
    }
    std::printf("  load into memory  %.3f s\n", loadMs / 1000.0);

    // Scoped so the mapping is released before the file is deleted below —
    // Windows will not delete a file that is still mapped.
    {
    t0 = Clock::now();
    auto mapped = HnswIndex::openMapped(path);
    const double mapMs = msSince(t0);
    std::printf("  memory-map        %.4f s   (%.0fx faster to open: nothing is read until it is touched)\n",
                mapMs / 1000.0, loadMs / std::max(mapMs, 1e-4));

    // A mapped index is cold, so the first queries pay for page faults.
    std::vector<double> coldTimes;
    for (std::size_t q = 0; q < std::min<std::size_t>(queryCount, 50); ++q) {
        auto qt = Clock::now();
        mapped.search(queries.data() + q * dim, k, 80);
        coldTimes.push_back(msSince(qt));
    }
    std::vector<double> warmTimes;
    for (std::size_t q = 0; q < std::min<std::size_t>(queryCount, 50); ++q) {
        auto qt = Clock::now();
        mapped.search(queries.data() + q * dim, k, 80);
        warmTimes.push_back(msSince(qt));
    }
    std::printf("  mapped query      %.3f ms cold, %.3f ms warm (ef=80)\n",
                percentiles(coldTimes).mean, percentiles(warmTimes).mean);
    }

    if (std::remove(path.c_str()) != 0) {
        std::printf("  (could not delete %s - remove it by hand)\n", path.c_str());
    }
    std::printf("\n");
    return 0;
}
