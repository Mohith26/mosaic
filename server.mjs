import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { embedDataUrl, embedTexts, warmTextModel, MODEL_ID } from './src/clip.mjs';
import { ROOT } from './src/index-store.mjs';

const PORT = Number(process.env.PORT || 8790);
const HOST = process.env.HOST || '127.0.0.1';
const MAX_BODY = 6 * 1024 * 1024;
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.svg': 'image/svg+xml' };

// Static routes: the UI, the shared ranking modules (pure JS, same code the browser runs), the index, images.
const ROUTES = [
  ['/lib/', path.join(ROOT, 'src')],
  ['/images/', path.join(ROOT, 'data/images')],
  ['/', path.join(ROOT, 'public')],
];
const FILES = { '/index.json': path.join(ROOT, 'data/index.json'), '/metrics.json': path.join(ROOT, 'results/metrics.json') };
const BROWSER_SAFE_LIB = new Set(['graph.mjs', 'math.mjs', 'metrics.mjs']);

function send(res, status, body, type = 'application/json; charset=utf-8') {
  res.writeHead(status, { 'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
}

async function readBody(req) {
  const chunks = []; let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw Object.assign(new Error('Request body too large (6 MB max).'), { status: 413 });
    chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}'); }
  catch { throw Object.assign(new Error('Body must be JSON.'), { status: 400 }); }
}

async function serveStatic(res, urlPath) {
  if (FILES[urlPath]) return send(res, 200, await fs.readFile(FILES[urlPath]), TYPES['.json']);
  for (const [prefix, dir] of ROUTES) {
    if (!urlPath.startsWith(prefix)) continue;
    let rel = decodeURIComponent(urlPath.slice(prefix.length)) || 'index.html';
    if (prefix === '/lib/' && !BROWSER_SAFE_LIB.has(rel)) return send(res, 404, { error: 'not found' });
    const file = path.resolve(dir, rel);
    if (!file.startsWith(dir + path.sep) && file !== dir) return send(res, 403, { error: 'forbidden' });
    try {
      const body = await fs.readFile(file);
      return send(res, 200, body, TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream');
    } catch { return send(res, 404, { error: 'not found' }); }
  }
  return send(res, 404, { error: 'not found' });
}

export function createServer() {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    try {
      if (req.method === 'GET' && url.pathname === '/api/health') return send(res, 200, { ok: true, model: MODEL_ID, mode: 'live' });
      if (req.method === 'POST' && url.pathname === '/api/embed') {
        const body = await readBody(req);
        const started = performance.now();
        let vector;
        if (typeof body.text === 'string' && body.text.trim()) {
          if (body.text.length > 300) return send(res, 400, { error: 'text must be 300 characters or fewer' });
          [vector] = await embedTexts([body.text.trim()]);
        } else if (typeof body.image === 'string') {
          vector = await embedDataUrl(body.image);
        } else {
          return send(res, 400, { error: 'Provide {text} or {image: dataURL}.' });
        }
        return send(res, 200, { vector: vector.map((x) => Math.round(x * 1e5) / 1e5), ms: Math.round(performance.now() - started) });
      }
      if (req.method === 'GET' || req.method === 'HEAD') return await serveStatic(res, url.pathname === '/' ? '/index.html' : url.pathname);
      return send(res, 405, { error: 'method not allowed' });
    } catch (error) {
      return send(res, error.status || 500, { error: error.message });
    }
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try { await fs.access(path.join(ROOT, 'data/index.json')); }
  catch { console.error('No index found. Run `npm run setup` first.'); process.exit(1); }
  createServer().listen(PORT, HOST, () => console.log(`Mosaic running at http://${HOST}:${PORT}`));
  warmTextModel().then(() => console.log('CLIP text encoder warm.')).catch((e) => console.error('Warmup failed:', e.message));
}
