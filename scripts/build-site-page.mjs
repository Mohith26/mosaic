// Builds one self-contained HTML file for a personal-site page: CSS, ranking code,
// index, metrics and (recompressed) images inlined. Only CLIP (for free-text and Lens
// searches) is fetched from a CDN, on demand.
// usage: node scripts/build-site-page.mjs <imagesDir(webp)> <out.html>
import fs from 'node:fs/promises';
import path from 'node:path';
import { ROOT } from '../src/index-store.mjs';

const [imgDir, out] = process.argv.slice(2);
if (!imgDir || !out) { console.error('usage: build-site-page.mjs <imagesDir> <out.html>'); process.exit(1); }
const strip = (code) => code.split('\n').filter((l) => !/^import .* from '.*';$/.test(l.trim())).join('\n').replace(/^export \{[^}]*\};?$/m, '').replace(/^export (?=(async )?function|const|class|let)/gm, '');
const js = [await fs.readFile(path.join(ROOT, 'src/math.mjs'), 'utf8'), await fs.readFile(path.join(ROOT, 'src/graph.mjs'), 'utf8'), await fs.readFile(path.join(ROOT, 'public/app.js'), 'utf8')].map(strip).join('\n;\n');
const index = JSON.parse(await fs.readFile(path.join(ROOT, 'data/index.json'), 'utf8'));
for (const item of index.items) {
  const file = path.join(imgDir, path.basename(item.image).replace(/\.[^.]+$/, '.webp'));
  item.image = `data:image/webp;base64,${(await fs.readFile(file)).toString('base64')}`;
}
const metrics = JSON.parse(await fs.readFile(path.join(ROOT, 'results/metrics.json'), 'utf8'));
const safe = (o) => JSON.stringify(o).replace(/</g, '\\u003c');
let html = await fs.readFile(path.join(ROOT, 'public/index.html'), 'utf8');
html = html.replace('<title>Mosaic · Graph-aware visual discovery</title>', '<title>System 05 / Mosaic · Mohith Gajjela</title>');
const css = await fs.readFile(path.join(ROOT, 'public/styles.css'), 'utf8');
// Function replacements: a string replacement would interpret `$$`, `$'` etc. inside the inlined code.
html = html.replace('<link rel="stylesheet" href="styles.css">', () => `<style>${css}</style>`);
html = html.replace('<script type="module" src="app.js"></script>', () => `<script>window.MOSAIC_STATIC=true;window.MOSAIC_INDEX=${safe(index)};window.MOSAIC_METRICS=${safe(metrics)};</script>\n<script type="module">\n${js}\n</script>`);
if (html.includes('src="app.js"') || html.includes('href="styles.css"')) throw new Error('unresolved asset reference');
await fs.writeFile(out, html);
console.log(`${out}: ${(Buffer.byteLength(html) / 1048576).toFixed(2)} MB`);
