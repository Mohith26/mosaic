import fs from 'node:fs/promises';
import path from 'node:path';
import { buildGraph, fuseEmbeddings, similarityMatrix, DEFAULTS } from './graph.mjs';

export const ROOT = path.resolve(new URL('..', import.meta.url).pathname);
export const INDEX_PATH = path.join(ROOT, 'data/index.json');

export async function loadIndex(file = INDEX_PATH) {
  const raw = JSON.parse(await fs.readFile(file, 'utf8'));
  return hydrate(raw);
}

export function hydrate(raw) {
  const embeddings = raw.items.map((item) => item.embedding);
  const itemBoards = raw.items.map((item) => item.board);
  const sim = similarityMatrix(embeddings);
  const { edges, adjacency } = buildGraph(sim, itemBoards, raw.graphParams || DEFAULTS);
  const fusion = fuseEmbeddings(embeddings, adjacency, (raw.graphParams || DEFAULTS).fusion);
  const idToIndex = new Map(raw.items.map((item, i) => [item.id, i]));
  return { ...raw, embeddings, itemBoards, sim, edges, adjacency, fusion, idToIndex };
}
