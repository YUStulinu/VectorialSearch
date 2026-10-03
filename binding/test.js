'use strict';
/**
 * Tests for the Node binding.
 *
 * The C++ tests already cover the algorithm; these cover the boundary — type
 * checking, promises, zero-copy views, and whether a result that crossed from
 * C++ into JavaScript still says the same thing.
 */
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { VecIndex, cpuInfo } = require('./index.js');

const DIM = 64;

function randomVectors(n, dim, seed = 1) {
  // A small deterministic generator, so a failure is reproducible.
  let s = seed >>> 0;
  const next = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const data = new Float32Array(n * dim);
  for (let i = 0; i < data.length; i++) {
    // Box-Muller, for something roughly normal rather than uniform.
    const u = Math.max(next(), 1e-9);
    data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  }
  return data;
}

test('reports which distance kernel it selected', () => {
  const info = cpuInfo();
  assert.equal(typeof info.avx2, 'boolean');
  assert.match(info.kernel, /AVX2|scalar/);
  console.log(`      kernel in use: ${info.kernel}`);
});

test('finds a vector that is in the index', async () => {
  const n = 3000;
  const data = randomVectors(n, DIM, 7);
  const index = new VecIndex({ dim: DIM, M: 16, efConstruction: 200, capacity: n });

  const labels = Array.from({ length: n }, (_, i) => i * 10);
  const size = await index.addBatch(data, labels);
  assert.equal(size, n);
  assert.equal(index.size, n);

  const query = data.subarray(42 * DIM, 43 * DIM);
  const hits = index.search(query, 5, 100);
  assert.equal(hits.length, 5);
  assert.equal(hits[0].label, 420, 'the exact vector should rank first');
  assert.ok(hits[0].distance < 1e-4, 'distance to itself should be ~0');

  for (let i = 1; i < hits.length; i++) {
    assert.ok(hits[i].distance >= hits[i - 1].distance, 'results must be sorted');
  }
});

test('recall against the exhaustive scan is high', async () => {
  const n = 4000;
  const k = 10;
  const data = randomVectors(n, DIM, 11);
  const index = new VecIndex({ dim: DIM, M: 16, efConstruction: 200, capacity: n });
  await index.addBatch(data);

  const queries = randomVectors(50, DIM, 999);
  let hits = 0;
  for (let q = 0; q < 50; q++) {
    const query = queries.subarray(q * DIM, (q + 1) * DIM);
    const truth = new Set(index.bruteForce(query, k).map((r) => r.label));
    for (const r of index.search(query, k, 200)) {
      if (truth.has(r.label)) hits++;
    }
  }
  const recall = hits / (50 * k);
  console.log(`      recall@${k} at ef=200: ${recall.toFixed(4)}`);
  assert.ok(recall > 0.95, `recall too low: ${recall}`);
});

test('a batch build does not block the event loop', async () => {
  const n = 20000;
  const data = randomVectors(n, DIM, 13);
  const index = new VecIndex({ dim: DIM, capacity: n });

  // If addBatch ran on the main thread this interval would never fire until
  // the build finished.
  let ticks = 0;
  const timer = setInterval(() => { ticks++; }, 5);
  await index.addBatch(data, null, { threads: 4 });
  clearInterval(timer);

  console.log(`      the event loop ran ${ticks} times while ${n} vectors were indexed`);
  assert.ok(ticks > 0, 'the build blocked the event loop');
  assert.equal(index.size, n);
});

test('close() refuses while a build is running, then works after it', async () => {
  const n = 8000;
  const index = new VecIndex({ dim: DIM, capacity: n });
  const building = index.addBatch(randomVectors(n, DIM, 37));
  assert.throws(() => index.close(), /still running/);
  await building;
  index.close();
  assert.equal(index.size, 0);
  assert.throws(() => index.search(new Float32Array(DIM), 1), /closed/);
});

test('rejects the wrong input types instead of reading garbage', () => {
  const index = new VecIndex({ dim: DIM });
  assert.throws(() => index.add([1, 2, 3], 0), /Float32Array/);
  assert.throws(() => index.add(new Float32Array(DIM + 1), 0), /expects/);
  assert.throws(() => index.search(new Float64Array(DIM), 5), /Float32Array/);
  assert.throws(() => new VecIndex({}), /dim/);
  assert.throws(() => new VecIndex({ dim: 8, metric: 'manhattan' }), /l2|ip/);
});

test('save, load and memory-map all agree', async () => {
  const n = 2000;
  const data = randomVectors(n, DIM, 17);
  const index = new VecIndex({ dim: DIM, capacity: n });
  await index.addBatch(data);

  const file = path.join(os.tmpdir(), `vecsearch-node-${process.pid}.idx`);
  index.save(file);

  const loaded = VecIndex.load(file);
  const mapped = VecIndex.openMapped(file);
  assert.equal(loaded.size, n);
  assert.equal(mapped.size, n);
  assert.equal(mapped.stats().readOnly, true);

  const query = randomVectors(1, DIM, 31);
  const a = index.search(query, 10, 100);
  const b = loaded.search(query, 10, 100);
  const c = mapped.search(query, 10, 100);
  assert.deepEqual(a.map((r) => r.label), b.map((r) => r.label));
  assert.deepEqual(a.map((r) => r.label), c.map((r) => r.label));

  assert.throws(() => mapped.add(new Float32Array(DIM), 1), /read-only|memory-mapped/i);

  // On Windows a mapped file cannot be deleted until the mapping is released.
  mapped.close();
  loaded.close();
  assert.equal(mapped.size, 0);
  assert.throws(() => mapped.search(query, 1), /closed/);
  fs.unlinkSync(file);
});

test('deleted vectors stop being returned', async () => {
  const n = 1500;
  const data = randomVectors(n, DIM, 19);
  const index = new VecIndex({ dim: DIM, capacity: n });
  await index.addBatch(data);

  const query = data.subarray(0, DIM);
  const before = index.search(query, 3, 100);
  assert.equal(index.markDeleted(before[0].label), true);

  const after = index.search(query, 3, 100);
  assert.ok(!after.some((r) => r.label === before[0].label), 'a deleted vector came back');
  assert.equal(after.length, 3, 'deleting one result should not shrink the rest');
});

test('inner product works for normalised vectors', async () => {
  const n = 1500;
  const data = randomVectors(n, DIM, 23);
  for (let i = 0; i < n; i++) {
    let norm = 0;
    for (let d = 0; d < DIM; d++) norm += data[i * DIM + d] ** 2;
    norm = Math.sqrt(norm);
    for (let d = 0; d < DIM; d++) data[i * DIM + d] /= norm;
  }

  const index = new VecIndex({ dim: DIM, metric: 'ip', capacity: n });
  await index.addBatch(data);

  const hits = index.search(data.subarray(0, DIM), 1, 100);
  assert.equal(hits[0].label, 0);
  // A unit vector against itself: dot = 1, so the distance 1 - dot is ~0.
  assert.ok(Math.abs(hits[0].distance) < 1e-4, `expected ~0, got ${hits[0].distance}`);
});

test('stats describe the index honestly', async () => {
  const n = 1000;
  const index = new VecIndex({ dim: DIM, M: 12, efConstruction: 150, capacity: n });
  await index.addBatch(randomVectors(n, DIM, 29));

  const s = index.stats();
  assert.equal(s.count, n);
  assert.equal(s.dim, DIM);
  assert.equal(s.M, 12);
  assert.equal(s.efConstruction, 150);
  assert.ok(s.levels >= 1);
  assert.ok(s.memoryBytes > n * DIM * 4, 'memory should exceed the raw vectors (there is a graph too)');
  console.log(`      ${(s.memoryBytes / n).toFixed(0)} bytes per vector, of which ${DIM * 4} are the vector`);
});
