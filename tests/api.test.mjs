import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createServer } from '../server.mjs';

const hasIndex = fs.existsSync(new URL('../data/index.json', import.meta.url));

test('server: health, static index, library allow-list, path traversal, embed validation', { skip: !hasIndex && 'run npm run setup first' }, async () => {
  const server = createServer();
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const health = await (await fetch(`${base}/api/health`)).json();
    assert.equal(health.ok, true);
    const index = await (await fetch(`${base}/index.json`)).json();
    assert.ok(index.items.length >= 80 && index.items[0].embedding.length === 512);
    assert.equal((await fetch(`${base}/lib/graph.mjs`)).status, 200);
    assert.equal((await fetch(`${base}/lib/clip.mjs`)).status, 404, 'server-only modules are not exposed');
    assert.notEqual((await fetch(`${base}/images/..%2f..%2fpackage.json`)).status, 200);
    const bad = await fetch(`${base}/api/embed`, { method: 'POST', body: '{}', headers: { 'content-type': 'application/json' } });
    assert.equal(bad.status, 400);
    const long = await fetch(`${base}/api/embed`, { method: 'POST', body: JSON.stringify({ text: 'x'.repeat(301) }) });
    assert.equal(long.status, 400);
  } finally { server.close(); }
});

test('server: real CLIP text embedding is unit length and lands near the right board', { skip: (!hasIndex || process.env.MOSAIC_SKIP_MODEL) && 'model test skipped' }, async () => {
  const server = createServer();
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const res = await fetch(`${base}/api/embed`, { method: 'POST', body: JSON.stringify({ text: 'koi fish swimming in a pond' }) });
    const { vector } = await res.json();
    assert.equal(vector.length, 512);
    assert.ok(Math.abs(Math.hypot(...vector) - 1) < 1e-3);
    const index = JSON.parse(fs.readFileSync(new URL('../data/index.json', import.meta.url)));
    const best = index.items.map((it) => ({ b: it.board, s: it.embedding.reduce((a, x, k) => a + x * vector[k], 0) })).sort((a, b) => b.s - a.s)[0];
    assert.equal(best.b, 'still-water');
  } finally { server.close(); }
});
