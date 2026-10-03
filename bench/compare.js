'use strict';
/**
 * vecsearch versus sqlite-vec — the comparison this project exists for.
 *
 * These are not the same kind of thing, and the point of measuring is to
 * show where the difference starts to matter:
 *
 *   sqlite-vec scans every vector. The answer is exactly right, and the time
 *   grows linearly with the collection. At ten thousand vectors that is
 *   perfectly fine; the honest conclusion at that size is that Second Brain
 *   does not need this project.
 *
 *   vecsearch walks a graph. It reads a few thousand vectors instead of all
 *   of them, so the time grows roughly logarithmically — but it can miss a
 *   true neighbour, which is why every row reports recall measured against
 *   the exhaustive answer.
 *
 * Run:  node bench/compare.js --n 100000 --dim 384
 */
const { performance } = require('node:perf_hooks');
const os = require('node:os');
const path = require('node:path');
const fs = require('node:fs');

const { VecIndex, cpuInfo } = require('../binding/index.js');

/* ---------------- options ---------------- */

function arg(flag, fallback) {
  const i = process.argv.indexOf(flag);
  return i > 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
}

const N = arg('--n', 50000);
const DIM = arg('--dim', 384);
const QUERIES = arg('--queries', 100);
const K = arg('--k', 10);

/* ---------------- data ---------------- */

/**
 * Clustered rather than uniform: uniform noise in high dimensions is the
 * worst case for every graph index and nothing like a real embedding.
 *
 * Queries come from the same cluster centres as the data, the way a question
 * to a RAG system is about the documents it holds. With --ood they come from
 * centres that do not exist in the collection, which is the hardest case
 * possible and gets harder the more clusters there are.
 */
function rng(seed) {
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const normal = () => {
    const u = Math.max(rand(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  return { rand, normal };
}

function makeCentres(clusters, dim, seed) {
  const { normal } = rng(seed);
  const centres = new Float32Array(clusters * dim);
  for (let i = 0; i < centres.length; i++) centres[i] = normal();
  return centres;
}

function sampleAround(centres, clusters, n, dim, seed) {
  const { rand, normal } = rng(seed);
  const data = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rand() * clusters) * dim;
    for (let d = 0; d < dim; d++) data[i * dim + d] = centres[c + d] + normal() * 0.22;
  }
  return data;
}

const OOD = process.argv.includes('--ood');

const fmt = (ms) => (ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : `${ms.toFixed(2)} ms`);

/* ---------------- sqlite-vec side ---------------- */

function trySqliteVec() {
  try {
    const Database = require('better-sqlite3');
    const sqliteVec = require('sqlite-vec');
    return { Database, sqliteVec };
  } catch {
    return null;
  }
}

async function main() {
  console.log('\nvecsearch vs sqlite-vec');
  console.log('='.repeat(60));
  console.log(`  vectors     ${N.toLocaleString()}`);
  console.log(`  dimensions  ${DIM}`);
  console.log(`  queries     ${QUERIES}  (k = ${K})`);
  console.log(`  kernel      ${cpuInfo().kernel}`);
  console.log(`  cpu         ${os.cpus()[0].model.trim()} (${os.cpus().length} cores)`);
  console.log(`  queries     ${OOD ? "from clusters NOT in the data (--ood, worst case)" : "from the data's own clusters (realistic)"}\n`);

  console.log(`generating ${((N * DIM * 4) / 1048576).toFixed(0)} MB of clustered data...`);
  const clusters = Math.max(20, Math.floor(N / 500));
  const centres = makeCentres(clusters, DIM, 1234);
  const data = sampleAround(centres, clusters, N, DIM, 77);
  const queries = OOD
    ? sampleAround(makeCentres(clusters, DIM, 999), clusters, QUERIES, DIM, 555)
    : sampleAround(centres, clusters, QUERIES, DIM, 555);

  /* ---- build ---- */
  console.log('\nbuilding');
  const index = new VecIndex({ dim: DIM, metric: 'l2', M: 16, efConstruction: 200, capacity: N });

  let t0 = performance.now();
  await index.addBatch(data, null, { threads: Math.max(1, os.cpus().length) });
  const buildMs = performance.now() - t0;
  const stats = index.stats();
  console.log(`  vecsearch   ${(buildMs / 1000).toFixed(1)} s  (${Math.round(N / (buildMs / 1000)).toLocaleString()} vectors/sec, ` +
              `${(stats.memoryBytes / 1048576).toFixed(0)} MB, ${Math.round(stats.memoryBytes / N)} bytes/vector)`);

  const deps = trySqliteVec();
  let db = null;
  if (deps) {
    const { Database, sqliteVec } = deps;
    db = new Database(':memory:');
    sqliteVec.load(db);
    db.exec(`CREATE VIRTUAL TABLE v USING vec0(embedding float[${DIM}])`);
    const insert = db.prepare('INSERT INTO v(rowid, embedding) VALUES (?, ?)');

    t0 = performance.now();
    const tx = db.transaction(() => {
      for (let i = 0; i < N; i++) {
        // sqlite-vec needs a Buffer view, and better-sqlite3 needs a BigInt rowid.
        const view = Buffer.from(data.buffer, i * DIM * 4, DIM * 4);
        insert.run(BigInt(i), view);
      }
    });
    tx();
    const sqliteBuildMs = performance.now() - t0;
    console.log(`  sqlite-vec  ${(sqliteBuildMs / 1000).toFixed(1)} s  (${Math.round(N / (sqliteBuildMs / 1000)).toLocaleString()} vectors/sec, no graph to build)`);
  } else {
    console.log('  sqlite-vec  not installed — run `npm i better-sqlite3 sqlite-vec` to include it');
  }

  /* ---- ground truth ---- */
  console.log('\nground truth (exhaustive scan inside vecsearch)');
  const truth = [];
  t0 = performance.now();
  for (let q = 0; q < QUERIES; q++) {
    truth.push(new Set(index.bruteForce(queries.subarray(q * DIM, (q + 1) * DIM), K).map((r) => r.label)));
  }
  const bruteMs = (performance.now() - t0) / QUERIES;
  console.log(`  ${fmt(bruteMs)} per query`);

  /* ---- sqlite-vec queries ---- */
  if (db) {
    const stmt = db.prepare(
      `SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = ${K} ORDER BY distance`
    );
    for (let q = 0; q < 5; q++) stmt.all(Buffer.from(queries.buffer, q * DIM * 4, DIM * 4));

    const times = [];
    let hits = 0;
    for (let q = 0; q < QUERIES; q++) {
      const buf = Buffer.from(queries.buffer, q * DIM * 4, DIM * 4);
      const s = performance.now();
      const rows = stmt.all(buf);
      times.push(performance.now() - s);
      for (const row of rows) if (truth[q].has(Number(row.rowid))) hits++;
    }
    times.sort((a, b) => a - b);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    console.log('\nsqlite-vec (exact, scans everything)');
    console.log(`  recall ${(hits / (QUERIES * K)).toFixed(4)}   mean ${fmt(mean)}   p95 ${fmt(times[Math.floor(times.length * 0.95)])}   ${Math.round(1000 / mean).toLocaleString()} q/s`);
  }

  /* ---- vecsearch queries ---- */
  console.log('\nvecsearch (approximate, walks a graph)');
  console.log(`  ${'ef'.padEnd(6)}${'recall'.padEnd(10)}${'mean'.padEnd(12)}${'p95'.padEnd(12)}${'queries/s'.padEnd(12)}vs exhaustive`);
  console.log(`  ${'-'.repeat(4).padEnd(6)}${'-'.repeat(6).padEnd(10)}${'-'.repeat(4).padEnd(12)}${'-'.repeat(3).padEnd(12)}${'-'.repeat(9).padEnd(12)}-------------`);

  for (const ef of [16, 32, 64, 128, 256]) {
    if (ef < K) continue;
    for (let q = 0; q < 10; q++) index.search(queries.subarray(q * DIM, (q + 1) * DIM), K, ef);

    const times = [];
    let hits = 0;
    for (let q = 0; q < QUERIES; q++) {
      const query = queries.subarray(q * DIM, (q + 1) * DIM);
      const s = performance.now();
      const found = index.search(query, K, ef);
      times.push(performance.now() - s);
      for (const r of found) if (truth[q].has(r.label)) hits++;
    }
    times.sort((a, b) => a - b);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const recall = hits / (QUERIES * K);
    console.log(
      `  ${String(ef).padEnd(6)}${recall.toFixed(4).padEnd(10)}${fmt(mean).padEnd(12)}` +
      `${fmt(times[Math.floor(times.length * 0.95)]).padEnd(12)}` +
      `${Math.round(1000 / mean).toLocaleString().padEnd(12)}${(bruteMs / mean).toFixed(0)}x`
    );
  }

  /* ---- persistence ---- */
  const file = path.join(os.tmpdir(), `vecsearch-bench-${process.pid}.idx`);
  t0 = performance.now();
  index.save(file);
  const saveMs = performance.now() - t0;

  t0 = performance.now();
  const loaded = VecIndex.load(file);
  const loadMs = performance.now() - t0;
  loaded.close();

  t0 = performance.now();
  const mapped = VecIndex.openMapped(file);
  const mapMs = performance.now() - t0;

  console.log('\npersistence');
  console.log(`  file          ${(fs.statSync(file).size / 1048576).toFixed(0)} MB`);
  console.log(`  save          ${fmt(saveMs)}`);
  console.log(`  load          ${fmt(loadMs)}`);
  console.log(`  memory-map    ${fmt(mapMs)}   (${Math.round(loadMs / Math.max(mapMs, 0.001))}x faster to open — nothing is read until it is touched)`);

  const warm = [];
  for (let q = 0; q < Math.min(QUERIES, 50); q++) {
    const s = performance.now();
    mapped.search(queries.subarray(q * DIM, (q + 1) * DIM), K, 64);
    warm.push(performance.now() - s);
  }
  console.log(`  mapped query  ${fmt(warm.reduce((a, b) => a + b, 0) / warm.length)} (ef=64)`);

  // Windows refuses to delete a file that is still mapped.
  mapped.close();
  index.close();
  fs.unlinkSync(file);
  if (db) db.close();
  console.log('');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
