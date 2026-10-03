# VectorialSearch

An HNSW vector index written from scratch in C++: hand-vectorised AVX2 distance kernels, a multi-threaded build, memory-mapped storage, and an N-API binding so Node can use it.

It was written to replace the exact vector scan in a personal RAG system, and the benchmark against that system's current engine is the point of the project.

> All headline numbers below were measured on an **Intel Core i7-10870H** (8 cores / 16 threads), Windows, MSVC 19.51, Node 26 — a laptop, not a server.

---

## Results

### One million vectors

1,000,000 vectors of 384 dimensions, k = 10, 100 queries drawn from the same clusters as the data, built on 16 threads:

| `ef`              | recall    | per query    | queries/sec | vs exhaustive scan | vectors touched |
| ----------------- | --------- | ------------ | ----------- | ------------------ | --------------- |
| 10                | 0.617     | 0.088 ms     | 11,354      | 1,109×             | 0.04%           |
| 20                | 0.759     | 0.139 ms     | 7,180       | 702×               | 0.05%           |
| 40                | 0.897     | 0.204 ms     | 4,899       | 479×               | 0.07%           |
| 80                | 0.969     | 0.240 ms     | 4,167       | 407×               | 0.08%           |
| **160**           | **1.000** | **0.308 ms** | **3,250**   | **318×**           | **0.09%**       |
| 320               | 1.000     | 0.368 ms     | 2,716       | 265×               | 0.09%           |
| _exhaustive scan_ | _1.000_   | _97.72 ms_   | _10_        | —                  | _100%_          |

**Perfect recall, 318× faster than reading every vector.** At `ef=160` a query touches 867 vectors out of a million.

|              |                                                                   |
| ------------ | ----------------------------------------------------------------- |
| build        | 119.5 s (8,366 vectors/sec, 16 threads)                           |
| memory       | 1.6 GB — 1,691 bytes/vector, of which 1,536 are the vector itself |
| graph levels | 5                                                                 |

### Against sqlite-vec

100,000 vectors of 384 dimensions, k = 10, queries from the data's own clusters. sqlite-vec is the engine the RAG system uses today:

|                              | recall    | per query  | queries/sec |
| ---------------------------- | --------- | ---------- | ----------- |
| sqlite-vec (exact scan)      | 1.000     | 66.28 ms   | 15          |
| VectorialSearch, `ef=16`     | 0.835     | 165 µs     | 6,074       |
| VectorialSearch, `ef=32`     | 0.947     | 177 µs     | 5,658       |
| VectorialSearch, **`ef=64`** | **0.996** | **203 µs** | **4,922**   |
| VectorialSearch, `ef=128`    | 0.999     | 284 µs     | 3,523       |

**About 325× faster than sqlite-vec at 0.996 recall.** Two separate wins are stacked in that number, and it is worth keeping them apart:

- **The kernel.** VectorialSearch's own _exhaustive_ scan runs at 11.42 ms on the same data — **5.8× faster than sqlite-vec's**, with no approximation at all. That is AVX2 and a cache-friendly layout, nothing more.
- **The graph.** On top of that, HNSW reads around 1% of the collection instead of all of it.

The cost is paid once, at build time: sqlite-vec inserts 100k vectors in 1.6 s with no graph to build; VectorialSearch takes 7.8 s to construct one. Build once, query cheaply forever.

### It scales logarithmically — measured, not claimed

The same benchmark at two sizes, `ef=160`:

|                                 | 50,000 vectors | 1,000,000 vectors | growth   |
| ------------------------------- | -------------- | ----------------- | -------- |
| collection size                 | 50k            | 1M                | **20×**  |
| exhaustive scan                 | 4.66 ms        | 97.72 ms          | 21×      |
| HNSW query                      | 0.144 ms       | 0.308 ms          | **2.1×** |
| distance computations per query | 618            | 867               | **1.4×** |

The collection grew twenty-fold; the work per query grew 1.4-fold. The exhaustive scan grew linearly, exactly as expected. That gap is the whole reason graph indexes exist.

---

## The benchmark bug I found at one million vectors

The first run at a million vectors looked like this:

| `ef` | recall | "speedup" |
| ---- | ------ | --------- |
| 10   | 0.057  | 1,485×    |
| 80   | 0.198  | 353×      |
| 320  | 0.365  | 134×      |

A speedup with 5.7% recall is meaningless — any index can be made arbitrarily fast by returning worse answers. Something was wrong, and the shape of the failure said where: recall had been fine at 20k, mediocre at 100k, and collapsed at 1M. It degraded _with collection size_.

The cause was in the benchmark, not the index. The data is generated as points around a set of cluster centres, and the queries were generated **with a different random seed — so they came from cluster centres that did not exist in the collection.** Such a query is almost equidistant from hundreds of clusters, which is the hardest case a graph index can face. With 2,000 clusters at a million vectors, it was nearly hopeless.

Confirmed side by side on the same index before changing anything:

|                         | queries from the data's clusters | queries from clusters _not_ in the data |
| ----------------------- | -------------------------------- | --------------------------------------- |
| 20,000 vectors, `ef=80` | **1.000**                        | 0.707                                   |
| 80,000 vectors, `ef=80` | **1.000**                        | 0.573                                   |

And the same million-vector run, before and after the fix:

| `ef` | before (queries from foreign clusters) | after (queries from the data's clusters) |
| ---- | -------------------------------------- | ---------------------------------------- |
| 80   | 0.198                                  | **0.969**                                |
| 160  | 0.290                                  | **1.000**                                |
| 320  | 0.365                                  | **1.000**                                |

A question to a RAG system is about the documents it holds, so queries now come from the data's own clusters, with fresh noise. The worst case is still available as `--ood`, because it is a genuinely useful thing to see: **an approximate index is only as good as the assumption that queries look like the data.**

The lesson I took from it: a benchmark is code, and it can be wrong like any other code. A number that looks too good — or too bad — is a reason to check the measurement before celebrating or debugging the thing being measured.

---

## Is it correct?

Matching a reference implementation is the only convincing answer, so the same data and parameters were run through **hnswlib**, written by the author of the HNSW paper. This comparison was made on the hard, out-of-distribution queries, single-threaded, on a separate Linux machine — both libraries faced exactly the same difficulty, which is what makes the match meaningful:

|                                | hnswlib     | VectorialSearch |
| ------------------------------ | ----------- | --------------- |
| recall @ M=16, efc=200, ef=320 | 0.9290      | **0.9290**      |
| recall @ M=32, efc=500, ef=320 | 0.9910      | **0.9910**      |
| build, 1 thread                | 6,613 vec/s | **6,762 vec/s** |
| query @ ef=80                  | 10,136 q/s  | **10,029 q/s**  |

Identical recall, and speed within a few percent either way.

---

## How it works

Finding the nearest vector exactly means reading all of them. HNSW trades a little accuracy for a very large constant factor, and makes the trade explicit: you choose it per query with `ef`.

The index is a stack of graphs:

```
layer 2     A . . . . . . . . E              sparse, long hops
layer 1     A . . C . . . . . E . . H        denser
layer 0     A B C D E F G H I J K L M        every vector
```

Each vector is assigned a random level from an exponential distribution, so upper layers hold exponentially fewer points. A search enters at the top, greedily walks to the closest node it can reach, drops a layer, and repeats. The upper layers cross the space quickly; the bottom one refines. Hops grow roughly logarithmically with the collection instead of linearly — which is exactly what the scaling table above measured.

Two details carry most of the quality:

**`ef` — the beam width.** How many candidates stay in play during a layer search. It is the accuracy dial, it costs nothing to change after the build, and the result tables above are really just this one number being turned.

**The neighbour selection heuristic.** Keeping a node's M _closest_ neighbours sounds obviously right and works badly: inside a dense cluster every node links only within the cluster, and the graph breaks into islands a search can never leave. Instead a candidate is kept only if it is closer to the node than to any neighbour already chosen. Links then point in different directions and the graph stays navigable.

---

## What the measurements taught me

### AVX2 helps four times as much when the data fits in cache

Same kernel, same work, different working-set size (measured on a Linux Xeon):

| working set       | AVX2 speedup over scalar |
| ----------------- | ------------------------ |
| 0.25 MB (fits L2) | **4.19×**                |
| 5 MB (fits L3)    | 2.24×                    |
| 293 MB (RAM)      | 1.94×                    |

Below L2 the kernel is compute-bound and the wider registers pay off fully. In RAM it is waiting on memory, and the arithmetic is nearly free by comparison. That is why `searchLayer` prefetches each neighbour's vector before it is needed — a graph walk is random access by definition, and the cache misses, not the FLOPs, are what a query actually spends its time on.

### "Scalar" code is not scalar any more

The first speedup measured was a disappointing 1.42×. The reason was in the compiler output: GCC had auto-vectorised the "scalar" reference loop _using 16-byte vectors_ — SSE. The real comparison is three-way:

|                                     | throughput        |
| ----------------------------------- | ----------------- |
| true scalar (`-fno-tree-vectorize`) | 3.4 M distances/s |
| compiler's automatic SSE            | 4.7 M             |
| hand-written AVX2 + FMA             | 6.7 M             |

Worth knowing before claiming a speedup against a baseline you did not inspect.

### ThreadSanitizer found real data races

The multi-threaded build reads a node's neighbour list while another thread may be rewriting it. Several well-known libraries accept this as a "benign race"; it is still undefined behaviour. The fix costs nothing where it matters: neighbour lists are copied out under a per-node lock **only while a build is running**, and a plain pointer read is used at query time, when the index is immutable.

```
before: 4 data races reported
after:  0, and query performance unchanged
```

### Porting to Windows found five more bugs

The code was first written and tested on Linux. Running it on Windows surfaced problems that Linux silently tolerates:

| problem                    | what happened                                                                                                                     | fix                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| hard-coded `/tmp` in tests | persistence tests failed: the path does not exist                                                                                 | `std::filesystem::temp_directory_path()`                                                     |
| deleting a mapped file     | `EBUSY`: Windows refuses to delete a file that is still memory-mapped, and JavaScript's GC releases the mapping whenever it likes | an explicit `close()`, plus scoped mappings in C++                                           |
| `close()` during a build   | would have freed the index under the worker thread                                                                                | `close()` refuses while `addBatch()` runs, and the JS object is kept alive until it finishes |
| prefetch only on GCC/Clang | MSVC builds silently lost the optimisation the measurements said mattered most                                                    | `_mm_prefetch` on MSVC                                                                       |
| `napi.h` not found         | node-addon-api's `include_dir` is a _relative_ path, and MSBuild compiles from inside `build/`                                    | the absolute `include` path                                                                  |

None of these were visible on Linux. Testing on the platform you ship to is not optional.

---

## Building

### Linux / macOS

```bash
make test      # builds and runs the correctness suite
make bench     # builds and runs the benchmark
make asan      # same tests under AddressSanitizer + UBSan
make tsan      # same tests under ThreadSanitizer
```

### Windows

With CMake (included in Visual Studio's "C++ CMake tools"):

```bat
cmake -B cmake-build
cmake --build cmake-build --config Release
cmake-build\Release\tests.exe
cmake-build\Release\bench.exe --n 1000000 --dim 384 --queries 100
```

**Use `cmake-build`, not `build`:** node-gyp owns `build/` and deletes it on every `npm install`, which would take your CMake binaries with it.

Without CMake, from a **Developer Command Prompt for VS**:

```bat
build.bat
out\tests.exe
out\bench.exe
```

### The Node addon

```bash
npm install          # runs node-gyp
npm test             # tests the binding
npm run bench -- --n 100000 --dim 384    # VectorialSearch vs sqlite-vec
```

`node-gyp` is the most reliable path on Windows, because it already knows where the Visual Studio toolchain is. npm ships its own copy, so the project does not depend on one — which keeps `npm audit` at **0 vulnerabilities** and the install at 4 packages.

`better-sqlite3` and `sqlite-vec` are optional dependencies, used only by the comparison benchmark. If they fail to install, the addon still builds and the benchmark simply skips the sqlite-vec column.

The `-flto=thin` warnings during the Windows build come from Node's own `common.gypi`, not from this project, and are harmless.

Only `src/distance_avx2.cpp` is compiled with AVX2 enabled — as a separate target in both CMake and `binding.gyp`. Compiling everything with `/arch:AVX2` would produce a binary that dies with an illegal-instruction fault on any CPU without it, which is exactly what the runtime dispatch exists to avoid.

### Benchmark options

| flag                | default   | meaning                                                       |
| ------------------- | --------- | ------------------------------------------------------------- |
| `--n`               | 50000     | number of vectors                                             |
| `--dim`             | 384       | dimensions                                                    |
| `--queries`         | 200       | queries to measure                                            |
| `--k`               | 10        | neighbours per query                                          |
| `--M`               | 16        | graph fan-out                                                 |
| `--ef-construction` | 200       | build-time beam                                               |
| `--threads`         | all cores | build threads                                                 |
| `--ood`             | off       | draw queries from clusters _not_ in the data — the worst case |

---

## Using it from Node

```js
const { VecIndex, cpuInfo } = require("VectorialSearch");

console.log(cpuInfo()); // { avx2: true, fma: true, kernel: 'l2 (AVX2+FMA)' }

const index = new VecIndex({
  dim: 384,
  metric: "l2", // or 'ip' — inner product, which is cosine on unit vectors
  M: 16,
  efConstruction: 200,
  capacity: 100000, // required before a threaded build
});

// vectors is one flat Float32Array of count * dim values — read without copying
await index.addBatch(vectors, labels, { threads: 8 }); // runs off the event loop

const hits = index.search(query, 10, 128);
// [ { label: 4210, distance: 0.0831 }, ... ]

index.save("index.bin");
const reopened = VecIndex.openMapped("index.bin"); // instant, read-only, shared

reopened.close(); // release it now, not whenever the GC runs
```

**Call `close()` when you are done with an index.** JavaScript frees native memory whenever the garbage collector gets round to it, which in a short script may be never. On Linux that is only wasteful; on Windows a memory-mapped file cannot be deleted or overwritten while the mapping exists, so deleting it fails with `EBUSY` until the GC happens to run. `close()` refuses while an `addBatch()` is still in progress, because the worker thread is using the index.

Vectors must be `Float32Array`. Anything else is rejected rather than quietly converted, because an element-by-element conversion in JavaScript would cost more than the search it feeds.

### Memory-mapped opening

| index size            | `load` (read into memory) | `openMapped` |                 |
| --------------------- | ------------------------- | ------------ | --------------- |
| 161 MB (100k vectors) | 130.7 ms                  | 3.2 ms       | 41× faster      |
| 1.6 GB (1M vectors)   | 2,514 ms                  | **0.1 ms**   | ~19,000× faster |

Nothing is read until it is touched. A search visits a few hundred vectors, so only those pages are faulted in, and several processes opening the same file share one copy in RAM.

The first queries on a freshly mapped index pay for those page faults: at a million vectors, 8.4 ms cold against 0.26 ms once the touched pages are resident.

---

## Tests

```bash
make test     # 13 C++ tests
npm test      # 10 Node tests
```

The C++ suite checks what an approximate index can get subtly wrong: a broken graph still returns _something_, ranked plausibly. So most tests measure against an exhaustive scan rather than hard-coded expectations — recall rising with `ef`, self-retrieval, deletes staying out of results, a memory-mapped index agreeing with the in-memory one, and a corrupt file being rejected instead of crashed on.

The Node suite covers the boundary: type checking, the event loop staying free during a build, `close()` refusing while a build runs, and results agreeing across save, load and memory-mapping.

Both suites pass on Windows (MSVC) and Linux (GCC), and the C++ suite passes clean under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer.

---

## Project layout

```
VectorialSearch/
├── include/VectorialSearch/
│   ├── types.hpp        aligned buffer, config, basic types
│   ├── distance.hpp     kernel interface and runtime dispatch
│   ├── hnsw.hpp         the index
│   └── storage.hpp      memory mapping, Windows + POSIX
├── src/
│   ├── distance_scalar.cpp   reference implementation
│   ├── distance_avx2.cpp     the only file built with AVX2
│   ├── distance.cpp          CPUID detection and selection
│   ├── hnsw.cpp              build, search, persistence
│   └── storage.cpp
├── binding/             N-API addon, JS wrapper, TypeScript types
├── bench/
│   ├── bench.cpp        recall vs latency, standalone
│   └── compare.js       VectorialSearch vs sqlite-vec
├── tests/               dependency-free C++ test harness
├── CMakeLists.txt       Windows / portable
├── Makefile             Linux / macOS
└── build.bat            MSVC without CMake
```

---

## Tuning

|                  | effect                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| `M`              | graph fan-out. 16 is a good default; 32–48 helps high-dimensional data. Memory grows with it.                       |
| `efConstruction` | build-time beam. Raising it improves recall at _no query cost_. On hard data this helped far more than raising `M`. |
| `ef`             | query-time beam. The accuracy dial; change it freely per query.                                                     |
| `metric`         | `l2`, or `ip` for cosine on unit-length vectors.                                                                    |

Start with M=16, efConstruction=200, and tune `ef` until recall is where you need it. On realistic queries that default already reaches perfect recall at `ef=160` with a million vectors.

A richer graph costs build time and helps most on hard queries. On the out-of-distribution query set at 100k vectors, raising M from 16 to 32 and efConstruction from 200 to 500 lifted recall at `ef=320` from 0.82 to 0.90 — and made the build take 26 s instead of 7.

---

## Platform notes

Everything is portable C++17 with no third-party dependencies (and also compiles cleanly as C++20, which node-gyp now uses by default), but a few things differ per platform and are handled explicitly rather than assumed:

|                        | handled how                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| aligned allocation     | `_aligned_malloc` on MSVC, `std::aligned_alloc` elsewhere                 |
| memory mapping         | `CreateFileMapping` / `MapViewOfFile` on Windows, `mmap` on POSIX         |
| prefetch               | `_mm_prefetch` on MSVC, `__builtin_prefetch` on GCC and Clang             |
| AVX2 flags             | `/arch:AVX2` on MSVC, `-mavx2 -mfma` elsewhere — applied to one file only |
| CPU detection          | `__cpuidex` / `_xgetbv` on MSVC, `cpuid.h` and inline `xgetbv` elsewhere  |
| temp paths in tests    | `std::filesystem::temp_directory_path()`, never a hard-coded `/tmp`       |
| deleting a mapped file | mappings are released before deletion, because Windows refuses otherwise  |

---

## Honest limits

- **It is approximate.** Recall is a setting, not a guarantee. Where an exact answer is required, `bruteForce()` is still there and still several times faster than sqlite-vec.
- **At small scale, this is not worth it.** Below ~10,000 vectors an exact scan answers in a couple of milliseconds and needs no index to build, no parameters to tune and no recall to worry about. The crossover is where the collection stops fitting comfortably in cache.
- **Recall depends on queries resembling the data.** That is true of every graph index, and the benchmark bug above is a vivid demonstration. Queries from a different distribution than the indexed data will see lower recall at the same `ef`.
- **Build scaling is sub-linear.** During a multi-threaded build every neighbour-list read takes a per-node lock — the price of the ThreadSanitizer fix. 16 threads give a solid speedup, but not 16×. Read-mostly locking or lock-free link updates are the obvious next optimisation.
- **Deletes are soft.** A deleted node stays in the graph so paths through it keep working, and is filtered from results. Deleting a large fraction of the collection should be followed by a rebuild.
- **No quantisation.** Vectors are stored as raw float32: 1,536 bytes each at 384 dimensions, which is why a million vectors need 1.6 GB. Product quantisation would cut that several-fold at some cost in recall, and is the obvious next feature.
- **One process writes.** A memory-mapped index is read-only, by design.

---

## Publishing to GitHub

The repository's `.gitignore` already excludes everything that is generated locally — `build/`, `cmake-build/`, `out/`, `node_modules/`, compiled `.node` addons and benchmark index files — so only source code is committed.

Create an **empty** repository on GitHub first (no README, no `.gitignore`, no licence — this project already has its own), then, from the project folder:

```bash
git init
git add .
git status                 # check: no build/, cmake-build/ or node_modules/ in the list
git commit -m "VectorialSearch: HNSW vector index in C++ with AVX2, mmap storage and a Node binding"
git branch -M main
git remote add origin https://github.com/<your-username>/VectorialSearch.git
git push -u origin main
```

Later changes:

```bash
git add .
git commit -m "Describe what changed"
git push
```

If the GitHub repository was created _with_ a README or licence, the first push is rejected because the two histories are unrelated. Either recreate the repository empty, or merge GitHub's commit in first:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

Anyone cloning the repository rebuilds everything from source:

```bash
git clone https://github.com/<your-username>/VectorialSearch.git
cd VectorialSearch
npm install                         # Node addon
cmake -B cmake-build                # C++ tests and benchmark
cmake --build cmake-build --config Release
```
