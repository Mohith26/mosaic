import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, fuseEmbeddings, layout, personalizedPageRank, score, similarityMatrix } from '../src/graph.mjs';
import { normalize, seededRandom } from '../src/math.mjs';
import { averagePrecision, ndcgAtK, precisionAtK, recallAtK } from '../src/metrics.mjs';

// Two synthetic "boards" in 16-d: cluster A near e0, cluster B near e1.
function fixture(n = 12, seed = 5) {
  const rand = seededRandom(seed);
  const items = [];
  for (let i = 0; i < n; i += 1) {
    const v = Array.from({ length: 16 }, () => (rand() - 0.5) * 0.3);
    v[i < n / 2 ? 0 : 1] += 1;
    items.push(normalize(v));
  }
  const boards = items.map((_, i) => (i < n / 2 ? 'a' : 'b'));
  return { E: items, boards };
}

test('similarity matrix is symmetric with unit diagonal', () => {
  const { E } = fixture();
  const S = similarityMatrix(E);
  for (let i = 0; i < E.length; i += 1) {
    assert.ok(Math.abs(S[i][i] - 1) < 1e-6);
    for (let j = 0; j < E.length; j += 1) assert.ok(Math.abs(S[i][j] - S[j][i]) < 1e-6);
  }
});

test('graph has no self loops, valid transition probabilities, and both edge types', () => {
  const { E, boards } = fixture();
  const { edges, adjacency } = buildGraph(similarityMatrix(E), boards, { visualK: 3, boardK: 2 });
  assert.ok(edges.every((e) => e.src !== e.dst));
  assert.ok(edges.some((e) => e.type === 'visual') && edges.some((e) => e.type === 'board'));
  for (const list of adjacency) {
    const total = list.reduce((s, x) => s + x.probability, 0);
    assert.ok(Math.abs(total - 1) < 1e-9);
  }
});

test('board edges only connect items on the same board', () => {
  const { E, boards } = fixture();
  const { edges } = buildGraph(similarityMatrix(E), boards);
  for (const e of edges.filter((x) => x.type === 'board')) assert.equal(boards[e.src], boards[e.dst]);
});

test('hidden items lose every board edge but keep visual edges (no label leakage)', () => {
  const { E, boards } = fixture();
  const { adjacency } = buildGraph(similarityMatrix(E), boards, { hidden: new Set([2]) });
  assert.ok(adjacency[2].length > 0);
  assert.ok(adjacency[2].every((x) => x.type === 'visual'));
});

test('personalized PageRank converges and conserves probability mass', () => {
  const { E, boards } = fixture();
  const { adjacency } = buildGraph(similarityMatrix(E), boards);
  const { rank, residual } = personalizedPageRank(adjacency, [{ index: 0, weight: 1 }], { iterations: 60, returnResidual: true });
  assert.ok(residual < 1e-6, `residual ${residual}`);
  assert.ok(Math.abs(rank.reduce((s, x) => s + x, 0) - 1) < 1e-6);
  const inA = rank.slice(0, 6).reduce((s, x) => s + x, 0);
  assert.ok(inA > 0.8, 'walk should stay mostly inside the seed board');
});

test('fused embeddings stay unit length', () => {
  const { E, boards } = fixture();
  const { adjacency } = buildGraph(similarityMatrix(E), boards);
  for (const v of fuseEmbeddings(E, adjacency)) assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-6);
});

test('graph mode re-ranks differently from baseline and excludes the seed', () => {
  const { E, boards } = fixture();
  const { adjacency } = buildGraph(similarityMatrix(E), boards);
  const ctx = { embeddings: E, fusion: fuseEmbeddings(E, adjacency), adjacency };
  const g = score(ctx, E[0], { mode: 'graph', seedIndex: 0, exclude: new Set([0]) });
  const b = score(ctx, E[0], { mode: 'baseline', exclude: new Set([0]) });
  assert.equal(g.length, E.length - 1);
  assert.ok(!g.some((r) => r.index === 0));
  assert.notDeepEqual(g.map((r) => r.score), b.map((r) => r.score));
  assert.ok(g.slice(0, 5).every((r) => boards[r.index] === 'a'));
});

test('layout returns finite coordinates in [0,1] with no stacked points', () => {
  const { E, boards } = fixture(20);
  const { edges } = buildGraph(similarityMatrix(E), boards);
  const pos = layout(E, edges, { iterations: 80 });
  assert.equal(pos.length, 20);
  for (const p of pos) assert.ok(Number.isFinite(p.x) && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
  for (let i = 0; i < pos.length; i += 1) for (let j = i + 1; j < pos.length; j += 1) assert.ok(Math.hypot(pos[i].x - pos[j].x, pos[i].y - pos[j].y) > 0.01);
});

test('IR metrics match hand-computed values', () => {
  const rel = new Set([1, 3]);
  const ranked = [1, 2, 3, 4];
  assert.equal(precisionAtK(ranked, rel, 2), 0.5);
  assert.equal(recallAtK(ranked, rel, 3), 1);
  assert.ok(Math.abs(averagePrecision(ranked, rel) - (1 + 2 / 3) / 2) < 1e-9);
  assert.ok(Math.abs(ndcgAtK(ranked, rel, 4) - (1 + 1 / Math.log2(4)) / (1 + 1 / Math.log2(3))) < 1e-9);
});
