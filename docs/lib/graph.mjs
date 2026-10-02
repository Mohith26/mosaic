import { addWeighted, cosine, dot, minMax, seededRandom } from './math.mjs';

export const DEFAULTS = Object.freeze({ visualK: 6, boardK: 4, damping: 0.85, iterations: 20, graphWeight: 0.3, fusion: 0.3 });

// Pairwise cosine matrix (embeddings are L2-normalized, so dot == cosine).
export function similarityMatrix(embeddings) {
  const n = embeddings.length;
  const sim = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i += 1) {
    sim[i][i] = 1;
    for (let j = i + 1; j < n; j += 1) {
      const value = dot(embeddings[i], embeddings[j]);
      sim[i][j] = value; sim[j][i] = value;
    }
  }
  return sim;
}

// Two edge types, mirroring PinCLIP/PinSage's split between content similarity and
// the pin-board graph: "visual" = mutual-ish kNN in CLIP space, "board" = items saved
// to the same board. `hidden` items keep visual edges but lose board edges, which is
// how a brand-new, unboarded pin enters the graph (and how evaluation avoids leakage).
export function buildGraph(sim, boards, { visualK = DEFAULTS.visualK, boardK = DEFAULTS.boardK, hidden = new Set() } = {}) {
  const n = sim.length;
  const edgeMap = new Map();
  const add = (a, b, weight, type) => {
    if (a === b) return;
    const [src, dst] = a < b ? [a, b] : [b, a];
    const key = `${src}:${dst}:${type}`;
    const prev = edgeMap.get(key);
    if (!prev || weight > prev.weight) edgeMap.set(key, { src, dst, weight, type });
  };
  for (let i = 0; i < n; i += 1) {
    const order = [...Array(n).keys()].filter((j) => j !== i).sort((a, b) => sim[i][b] - sim[i][a]);
    for (const j of order.slice(0, visualK)) add(i, j, Math.max(0.01, sim[i][j]), 'visual');
    if (hidden.has(i)) continue;
    const same = order.filter((j) => !hidden.has(j) && boards[j] === boards[i]).slice(0, boardK);
    for (const j of same) add(i, j, Math.max(0.01, sim[i][j]) * 1.15, 'board');
  }
  const edges = [...edgeMap.values()];
  const adjacency = Array.from({ length: n }, () => []);
  for (const e of edges) {
    adjacency[e.src].push({ index: e.dst, weight: e.weight, type: e.type });
    adjacency[e.dst].push({ index: e.src, weight: e.weight, type: e.type });
  }
  for (const list of adjacency) {
    const total = list.reduce((s, x) => s + x.weight, 0) || 1;
    for (const x of list) x.probability = x.weight / total;
  }
  return { edges, adjacency };
}

// One step of neighbor aggregation (a parameter-free GraphSAGE-style mean), giving
// each item a "fusion" embedding that blends its own CLIP vector with its neighborhood.
export function fuseEmbeddings(embeddings, adjacency, strength = DEFAULTS.fusion) {
  return embeddings.map((vector, i) => {
    const nbrs = adjacency[i];
    if (!nbrs.length) return vector;
    const hood = addWeighted(nbrs.map((x) => embeddings[x.index]), nbrs.map((x) => x.weight));
    return addWeighted([vector, hood], [1 - strength, strength]);
  });
}

export function personalizedPageRank(adjacency, seeds, { damping = DEFAULTS.damping, iterations = DEFAULTS.iterations, returnResidual = false } = {}) {
  const n = adjacency.length;
  const restart = new Float64Array(n);
  const total = seeds.reduce((s, x) => s + Math.max(0, x.weight), 0) || 1;
  for (const s of seeds) restart[s.index] += Math.max(0, s.weight) / total;
  let rank = Float64Array.from(restart);
  let residual = 0;
  for (let step = 0; step < iterations; step += 1) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i += 1) next[i] = (1 - damping) * restart[i];
    for (let i = 0; i < n; i += 1) {
      if (!adjacency[i].length) { next[i] += damping * rank[i]; continue; }
      for (const x of adjacency[i]) next[x.index] += damping * rank[i] * x.probability;
    }
    residual = 0;
    for (let i = 0; i < n; i += 1) residual += Math.abs(next[i] - rank[i]);
    rank = next;
  }
  return returnResidual ? { rank: Array.from(rank), residual } : Array.from(rank);
}

// Score every item for a query vector. baseline = plain CLIP cosine.
// graph = (1-w) * cosine(query, fusion embedding) + w * normalized PPR seeded at the query's top CLIP hits.
export function score({ embeddings, fusion, adjacency }, query, { mode = 'graph', graphWeight = DEFAULTS.graphWeight, seedIndex = null, seeds: seedList = null, exclude = new Set() } = {}) {
  const n = embeddings.length;
  const base = embeddings.map((v) => dot(query, v));
  if (mode === 'baseline') {
    return base.map((s, i) => ({ index: i, score: s, clip: s, graph: 0 })).filter((r) => !exclude.has(r.index)).sort((a, b) => b.score - a.score);
  }
  const seeds = seedList?.length ? seedList : seedIndex !== null
    ? [{ index: seedIndex, weight: 1 }]
    : base.map((w, index) => ({ index, w })).sort((a, b) => b.w - a.w).slice(0, 6).map(({ index, w }) => ({ index, weight: Math.max(0.01, w) }));
  const ppr = personalizedPageRank(adjacency, seeds);
  if (seedIndex !== null) ppr[seedIndex] = 0;
  if (seedList?.length) for (const s of seedList) ppr[s.index] = 0;
  const pprNorm = minMax(ppr);
  const fused = fusion.map((v) => dot(query, v));
  const rows = [];
  for (let i = 0; i < n; i += 1) {
    if (exclude.has(i)) continue;
    rows.push({ index: i, score: (1 - graphWeight) * fused[i] + graphWeight * pprNorm[i], clip: base[i], graph: pprNorm[i] });
  }
  return rows.sort((a, b) => b.score - a.score);
}

// 2D layout for the constellation view: classical MDS of the CLIP Gram matrix as the
// starting point, then force-directed refinement over the graph and a collision pass.
export function layout(embeddings, edges, { iterations = 300, seed = 3, repulsion = 0.006, length = 0.06, spring = 0.1, center = 0.03, minGap = 0.05 } = {}) {
  const n = embeddings.length;
  const rand = seededRandom(seed);
  const G = embeddings.map((a) => embeddings.map((b) => dot(a, b)));
  const rowMean = G.map((r) => r.reduce((s, x) => s + x, 0) / n);
  const grand = rowMean.reduce((s, x) => s + x, 0) / n;
  const B = G.map((r, i) => r.map((x, j) => x - rowMean[i] - rowMean[j] + grand));
  const axes = [];
  for (let k = 0; k < 2; k += 1) {
    let v = Array.from({ length: n }, () => rand() - 0.5);
    for (let t = 0; t < 150; t += 1) {
      let w = B.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      for (const u of axes) { const proj = w.reduce((s, x, j) => s + x * u[j], 0); w = w.map((x, j) => x - proj * u[j]); }
      const m = Math.hypot(...w) || 1; v = w.map((x) => x / m);
    }
    axes.push(v);
  }
  let pos = axes[0].map((x, i) => ({ x, y: axes[1][i] }));
  const s0 = Math.max(...pos.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)])) || 1;
  pos = pos.map((p) => ({ x: p.x / s0, y: p.y / s0 }));
  for (let it = 0; it < iterations; it += 1) {
    const f = pos.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) {
      const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
      const k = repulsion / (dx * dx + dy * dy + 0.002);
      f[i].x += dx * k; f[i].y += dy * k; f[j].x -= dx * k; f[j].y -= dy * k;
    }
    for (const e of edges) {
      const dx = pos[e.dst].x - pos[e.src].x, dy = pos[e.dst].y - pos[e.src].y;
      const dist = Math.hypot(dx, dy) || 1e-6;
      const k = (dist - length) * spring * e.weight * (e.type === 'board' ? 1.4 : 1);
      f[e.src].x += dx / dist * k; f[e.src].y += dy / dist * k;
      f[e.dst].x -= dx / dist * k; f[e.dst].y -= dy / dist * k;
    }
    for (let i = 0; i < n; i += 1) { f[i].x -= pos[i].x * center; f[i].y -= pos[i].y * center; }
    const cool = 1 - it / iterations;
    pos = pos.map((p, i) => ({ x: p.x + Math.max(-0.05, Math.min(0.05, f[i].x)) * cool, y: p.y + Math.max(-0.05, Math.min(0.05, f[i].y)) * cool }));
  }
  const norm = () => {
    const xs = pos.map((p) => p.x), ys = pos.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    pos = pos.map((p) => ({ x: (p.x - x0) / ((x1 - x0) || 1), y: (p.y - y0) / ((y1 - y0) || 1) }));
  };
  norm();
  for (let pass = 0; pass < 60; pass += 1) {
    let moved = false;
    for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) {
      const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y; const d = Math.hypot(dx, dy) || 1e-6;
      if (d < minGap) { const push = (minGap - d) / 2; pos[i].x -= dx / d * push; pos[i].y -= dy / d * push; pos[j].x += dx / d * push; pos[j].y += dy / d * push; moved = true; }
    }
    if (!moved) break;
  }
  norm();
  return pos;
}

export { cosine };
