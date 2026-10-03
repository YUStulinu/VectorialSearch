export interface IndexOptions {
  /** Vector width. Required. */
  dim: number;
  /** "l2" for squared Euclidean, "ip" for inner product (cosine on unit vectors). */
  metric?: 'l2' | 'ip';
  /** Graph fan-out above layer 0. Higher = better recall, more memory. Default 16. */
  M?: number;
  /** Beam width while building. Default 200. Costs build time only. */
  efConstruction?: number;
  /** Pre-allocate for this many vectors. Required before a threaded build. */
  capacity?: number;
  seed?: number;
}

export interface SearchHit { label: number; distance: number; }

export interface Stats {
  count: number; deleted: number; dim: number; paddedDim: number;
  M: number; efConstruction: number; levels: number;
  memoryBytes: number; distanceCalls: number; kernel: string; readOnly: boolean;
}

export class VecIndex {
  constructor(options: IndexOptions);
  readonly size: number;
  add(vector: Float32Array, label: number): number;
  /** Runs on a worker thread; the returned promise resolves with the new size. */
  addBatch(vectors: Float32Array, labels?: number[], options?: { threads?: number }): Promise<number>;
  search(query: Float32Array, k: number, ef?: number): SearchHit[];
  /** Exhaustive scan, for measuring recall. */
  bruteForce(query: Float32Array, k: number): SearchHit[];
  markDeleted(label: number): boolean;
  save(path: string): void;
  stats(): Stats;
  /**
   * Frees the index now instead of whenever the GC runs. Required on Windows
   * before deleting or overwriting a file opened with openMapped().
   */
  close(): void;
  static load(path: string): VecIndex;
  /** Opens without reading: instant, read-only, shared between processes. */
  static openMapped(path: string): VecIndex;
}

export function cpuInfo(): { avx2: boolean; fma: boolean; kernel: string };
