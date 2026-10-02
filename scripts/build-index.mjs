import fs from 'node:fs/promises';
import path from 'node:path';
import { embedImage, embedTexts, warmVisionModel, MODEL_ID } from '../src/clip.mjs';
import { buildGraph, layout, similarityMatrix, DEFAULTS } from '../src/graph.mjs';
import { ROOT, INDEX_PATH } from '../src/index-store.mjs';

const manifest = JSON.parse(await fs.readFile(path.join(ROOT, 'data/manifest.json'), 'utf8'));
const round = (v) => v.map((x) => Math.round(x * 1e5) / 1e5);

// Remove stray downloads that are not in the manifest.
const keep = new Set(manifest.items.map((item) => path.basename(item.image)));
for (const name of await fs.readdir(path.join(ROOT, 'data/images'))) {
  if (!keep.has(name)) await fs.rm(path.join(ROOT, 'data/images', name), { force: true });
}

console.log(`Loading ${MODEL_ID}...`);
await warmVisionModel();
console.log(`Embedding ${manifest.items.length} images...`);
const started = Date.now();
const items = [];
for (const [i, item] of manifest.items.entries()) {
  const embedding = await embedImage(path.join(ROOT, 'data', item.image));
  items.push({ ...item, embedding: round(embedding) });
  if ((i + 1) % 10 === 0) console.log(`  ${i + 1}/${manifest.items.length} (${((Date.now() - started) / 1000).toFixed(1)}s)`);
}
const imageSeconds = (Date.now() - started) / 1000;

const suggestions = [
  'warm minimal living room', 'brutalist staircase', 'pressed flower print', 'speckled glaze pottery',
  'koi pond', 'street style denim', 'chocolate dessert', 'cafe racer motorcycle', 'cliffs at golden hour', 'bauhaus colors',
];
const suggestionVectors = await embedTexts(suggestions);
const boardVectors = await embedTexts(manifest.boards.map((b) => b.title + ': ' + b.query));

const embeddings = items.map((item) => item.embedding);
const sim = similarityMatrix(embeddings);
const { edges } = buildGraph(sim, items.map((x) => x.board), DEFAULTS);
const positions = layout(embeddings, edges);
items.forEach((item, i) => { item.x = Math.round(positions[i].x * 1e4) / 1e4; item.y = Math.round(positions[i].y * 1e4) / 1e4; });

const index = {
  model: MODEL_ID,
  builtAt: new Date().toISOString(),
  embedSecondsPerImage: Math.round((imageSeconds / items.length) * 1000) / 1000,
  graphParams: DEFAULTS,
  source: { name: manifest.source, url: manifest.sourceUrl, licensePolicy: manifest.licensePolicy },
  boards: manifest.boards.map((b, i) => ({ ...b, embedding: round(boardVectors[i]) })),
  suggestions: suggestions.map((text, i) => ({ text, embedding: round(suggestionVectors[i]) })),
  items,
};
await fs.writeFile(INDEX_PATH, JSON.stringify(index));
console.log(`Wrote ${INDEX_PATH}: ${items.length} items, ${edges.length} edges, ${imageSeconds.toFixed(1)}s of image embedding.`);
