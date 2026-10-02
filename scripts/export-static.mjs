// Builds a self-contained static copy of the UI in docs/ (GitHub Pages) that runs ranking
// and, on demand, CLIP itself in the browser. Nothing here needs the Node server.
import fs from 'node:fs/promises';
import path from 'node:path';
import { ROOT } from '../src/index-store.mjs';

const OUT = path.join(ROOT, process.argv[2] || 'docs');
await fs.rm(OUT, { recursive: true, force: true });
await fs.mkdir(path.join(OUT, 'lib'), { recursive: true });
await fs.mkdir(path.join(OUT, 'images'), { recursive: true });
let html = await fs.readFile(path.join(ROOT, 'public/index.html'), 'utf8');
html = html.replace('<script type="module" src="app.js"></script>', '<script>window.MOSAIC_STATIC = true;</script>\n<script type="module" src="app.js"></script>');
await fs.writeFile(path.join(OUT, 'index.html'), html);
for (const f of ['app.js', 'styles.css']) await fs.copyFile(path.join(ROOT, 'public', f), path.join(OUT, f));
for (const f of ['graph.mjs', 'math.mjs']) await fs.copyFile(path.join(ROOT, 'src', f), path.join(OUT, 'lib', f));
await fs.copyFile(path.join(ROOT, 'data/index.json'), path.join(OUT, 'index.json'));
await fs.copyFile(path.join(ROOT, 'results/metrics.json'), path.join(OUT, 'metrics.json'));
for (const f of await fs.readdir(path.join(ROOT, 'data/images'))) await fs.copyFile(path.join(ROOT, 'data/images', f), path.join(OUT, 'images', f));
await fs.writeFile(path.join(OUT, '.nojekyll'), '');
console.log(`Static demo written to ${OUT}`);
