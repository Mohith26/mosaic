import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { buildGraph, fuseEmbeddings, score, similarityMatrix, DEFAULTS } from '../src/graph.mjs';
import { loadIndex, ROOT } from '../src/index-store.mjs';
import { embedTexts } from '../src/clip.mjs';
import { averagePrecision, mean, ndcgAtK, pairedBootstrap, percentile, precisionAtK, recallAtK } from '../src/metrics.mjs';

const index = await loadIndex();
const { embeddings, itemBoards: boards } = index;
const n = embeddings.length;
const sim = similarityMatrix(embeddings);
const r4 = (x) => Math.round(x * 10000) / 10000;

// Task A: Related Pins for a cold pin. Each item is hidden from the board graph in turn
// (it keeps only its visual edges), then we retrieve its board-mates.
function relatedPins(graphWeight) {
  const rows = { baseline: [], graph: [] };
  for (let q = 0; q < n; q += 1) {
    const relevant = new Set([...Array(n).keys()].filter((j) => j !== q && boards[j] === boards[q]));
    const { adjacency } = buildGraph(sim, boards, { ...DEFAULTS, hidden: new Set([q]) });
    const fusion = fuseEmbeddings(embeddings, adjacency, DEFAULTS.fusion);
    const ctx = { embeddings, fusion, adjacency };
    for (const mode of ['baseline', 'graph']) {
      const ranked = score(ctx, embeddings[q], { mode, graphWeight, seedIndex: q, exclude: new Set([q]) }).map((r) => r.index);
      rows[mode].push({ r10: recallAtK(ranked, relevant, 10), ap: averagePrecision(ranked, relevant), ndcg: ndcgAtK(ranked, relevant, 10) });
    }
  }
  return rows;
}

// Task B: text-to-image search with held-out hand-written queries (full graph).
const heldOut = JSON.parse(await fs.readFile(path.join(ROOT, 'eval/queries.json'), 'utf8')).queries;
const textVectors = await embedTexts(heldOut.map((q) => q.text));
const full = { embeddings, fusion: index.fusion, adjacency: index.adjacency };
function textSearch(graphWeight) {
  const rows = { baseline: [], graph: [] };
  heldOut.forEach((q, qi) => {
    const relevant = new Set([...Array(n).keys()].filter((j) => boards[j] === q.board));
    for (const mode of ['baseline', 'graph']) {
      const ranked = score(full, textVectors[qi], { mode, graphWeight }).map((r) => r.index);
      rows[mode].push({ p10: precisionAtK(ranked, relevant, 10), ap: averagePrecision(ranked, relevant), ndcg: ndcgAtK(ranked, relevant, 10) });
    }
  });
  return rows;
}

const summarize = (rows, keys) => Object.fromEntries(Object.entries(rows).map(([mode, list]) => [mode, Object.fromEntries(keys.map((k) => [k, r4(mean(list.map((x) => x[k])))]))]));

const related = relatedPins(DEFAULTS.graphWeight);
const search = textSearch(DEFAULTS.graphWeight);
const relatedSummary = summarize(related, ['r10', 'ap', 'ndcg']);
const searchSummary = summarize(search, ['p10', 'ap', 'ndcg']);
const relatedBoot = pairedBootstrap(related.baseline.map((x) => x.ap), related.graph.map((x) => x.ap));
const searchBoot = pairedBootstrap(search.baseline.map((x) => x.ap), search.graph.map((x) => x.ap));

// Sensitivity sweep, reported for transparency; the shipped weight (0.3) was fixed before this sweep.
const sweep = [];
for (const w of [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) {
  const a = relatedPins(w); const b = textSearch(w);
  sweep.push({ graphWeight: w, relatedR10: r4(mean(a.graph.map((x) => x.r10))), relatedMAP: r4(mean(a.graph.map((x) => x.ap))), searchP10: r4(mean(b.graph.map((x) => x.p10))), searchMAP: r4(mean(b.graph.map((x) => x.ap))) });
}

// Latency: ranking only (no model), and text encoding (warm model).
const rankMs = [];
for (let i = 0; i < 500; i += 1) {
  const v = textVectors[i % textVectors.length];
  const t = performance.now(); score(full, v, { mode: 'graph' }); rankMs.push(performance.now() - t);
}
const encodeMs = [];
for (let i = 0; i < 40; i += 1) {
  const t = performance.now(); await embedTexts([heldOut[i % heldOut.length].text]); encodeMs.push(performance.now() - t);
}

const metrics = {
  generatedAt: new Date().toISOString(),
  corpus: { images: n, boards: index.boards.length, edges: index.edges.length, visualEdges: index.edges.filter((e) => e.type === 'visual').length, boardEdges: index.edges.filter((e) => e.type === 'board').length, model: index.model },
  params: DEFAULTS,
  relatedPins: { queries: n, ...relatedSummary, bootstrapMAP: { pGraphBetter: r4(relatedBoot.pGraphBetter), ci95: relatedBoot.ci95.map(r4) } },
  textSearch: { queries: heldOut.length, ...searchSummary, bootstrapMAP: { pGraphBetter: r4(searchBoot.pGraphBetter), ci95: searchBoot.ci95.map(r4) } },
  sweep,
  latencyMs: { rankP50: r4(percentile(rankMs, 50)), rankP95: r4(percentile(rankMs, 95)), textEncodeP50: r4(percentile(encodeMs, 50)), textEncodeP95: r4(percentile(encodeMs, 95)) },
};
await fs.mkdir(path.join(ROOT, 'results'), { recursive: true });
await fs.writeFile(path.join(ROOT, 'results/metrics.json'), `${JSON.stringify(metrics, null, 2)}\n`);
console.log(JSON.stringify(metrics, null, 2));
