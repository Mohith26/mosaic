import { buildGraph, fuseEmbeddings, score, similarityMatrix, DEFAULTS } from './lib/graph.mjs';
import { addWeighted, dot } from './lib/math.mjs';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;
const REPO = 'https://github.com/Mohith26/mosaic';

const state = { mode: 'graph', tab: 'discover', query: null, board: [], selected: null, rows: { graph: [], baseline: [] }, stats: null };
let INDEX, E, FUSION, ADJ, EDGES, ITEMS, BOARD_CENTROIDS, METRICS = null;
let STATIC = window.MOSAIC_STATIC === true;

// ---------- boot ----------
async function boot() {
  if (!STATIC) {
    try { const r = await fetch('api/health'); STATIC = !r.ok; } catch { STATIC = true; }
  }
  INDEX = window.MOSAIC_INDEX || await (await fetch('index.json')).json();
  try { METRICS = window.MOSAIC_METRICS || await (await fetch('metrics.json')).json(); } catch { METRICS = null; }
  ITEMS = INDEX.items;
  E = ITEMS.map((x) => x.embedding);
  const sim = similarityMatrix(E);
  const g = buildGraph(sim, ITEMS.map((x) => x.board), INDEX.graphParams || DEFAULTS);
  ADJ = g.adjacency; EDGES = g.edges;
  FUSION = fuseEmbeddings(E, ADJ, (INDEX.graphParams || DEFAULTS).fusion);
  BOARD_CENTROIDS = INDEX.boards.map((b) => ({ ...b, centroid: addWeighted(ITEMS.filter((x) => x.board === b.slug).map((x) => x.embedding)) }));
  if (STATIC) document.body.insertAdjacentHTML('afterbegin', `<div class="demo-banner"><b>This page is a demo.</b> It runs Mosaic entirely in your browser over a fixed 100-image index; free-text and Lens searches download CLIP (~150 MB) once. The real program is <a href="${REPO}">on GitHub</a>: the image pipeline, indexer, evaluator, tests and a live server, which run locally on Node 20+.</div>`);
  renderChips(); renderFoot(); renderHow(); bind(); render();
}

// ---------- encoders: server in live mode, transformers.js in the browser in static mode ----------
let hfPromise, textEnc, visionEnc;
const hf = () => (hfPromise ??= import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0'));
async function browserText(text) {
  const T = await hf();
  textEnc ??= Promise.all([T.AutoTokenizer.from_pretrained(INDEX.model), T.CLIPTextModelWithProjection.from_pretrained(INDEX.model, { dtype: 'q8' })]);
  const [tok, model] = await textEnc;
  const { text_embeds } = await model(tok([text], { padding: true, truncation: true }));
  return addWeighted([Array.from(text_embeds.data)]);
}
async function browserImage(dataUrl) {
  const T = await hf();
  visionEnc ??= Promise.all([T.AutoProcessor.from_pretrained(INDEX.model), T.CLIPVisionModelWithProjection.from_pretrained(INDEX.model, { dtype: 'q8' })]);
  const [proc, model] = await visionEnc;
  const img = await T.RawImage.fromURL(dataUrl);
  const { image_embeds } = await model(await proc(img));
  return addWeighted([Array.from(image_embeds.data)]);
}
async function encode(payload) {
  const t0 = performance.now();
  const preset = payload.text && INDEX.suggestions.find((s) => s.text.toLowerCase() === payload.text.toLowerCase());
  if (preset) return { vector: preset.embedding, ms: 0, cached: true };
  if (STATIC) {
    toast(textEnc || visionEnc ? 'Encoding…' : 'Loading CLIP in your browser (first time only)…', 0);
    const vector = payload.text ? await browserText(payload.text) : await browserImage(payload.image);
    hideToast();
    return { vector, ms: Math.round(performance.now() - t0) };
  }
  const r = await fetch('api/embed', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || 'Encoding failed');
  return { vector: body.vector, ms: Math.round(performance.now() - t0) };
}

// ---------- ranking ----------
function context() {
  const q = state.query; const b = state.board;
  if (!q && !b.length) return null;
  const boardVec = b.length ? addWeighted(b.map((i) => E[i])) : null;
  if (!q) return { vector: boardVec, seeds: b.map((index) => ({ index, weight: 1 })), exclude: new Set(b) };
  const vector = boardVec ? addWeighted([q.vector, boardVec], [1, 0.55]) : q.vector;
  const exclude = new Set(q.seedIndex != null ? [q.seedIndex] : []);
  return { vector, seedIndex: q.seedIndex ?? null, exclude };
}

function rank() {
  const ctx = context();
  if (!ctx) { state.rows = { graph: [], baseline: [] }; state.stats = null; return; }
  const t0 = performance.now();
  const ranker = { embeddings: E, fusion: FUSION, adjacency: ADJ };
  const opts = { seedIndex: ctx.seedIndex, seeds: ctx.seeds, exclude: ctx.exclude };
  const graph = score(ranker, ctx.vector, { ...opts, mode: 'graph' });
  const t1 = performance.now();
  const baseline = score(ranker, ctx.vector, { ...opts, mode: 'baseline' });
  state.rows = { graph, baseline };
  state.stats = { rankMs: t1 - t0, vector: ctx.vector };
}

function defaultFeed() {
  const byBoard = new Map();
  ITEMS.forEach((it, i) => { if (!byBoard.has(it.board)) byBoard.set(it.board, []); byBoard.get(it.board).push(i); });
  const lists = [...byBoard.values()]; const out = [];
  for (let k = 0; out.length < ITEMS.length; k += 1) for (const l of lists) if (l[k] !== undefined) out.push(l[k]);
  return out.map((index) => ({ index }));
}

// ---------- render ----------
function render() {
  rank();
  renderQuery();
  if (state.tab === 'discover') renderResults();
  if (state.tab === 'constellation') drawSky();
  renderBoard();
  renderWhy();
}

function rankMaps() {
  const g = new Map(state.rows.graph.map((r, i) => [r.index, i + 1]));
  const b = new Map(state.rows.baseline.map((r, i) => [r.index, i + 1]));
  return { g, b };
}

function cardHTML(row, pos, mode, maps) {
  const it = ITEMS[row.index]; const saved = state.board.includes(row.index);
  let tag = '';
  if (maps && mode === 'graph') {
    const before = maps.b.get(row.index); const delta = before - pos;
    if (before > 24) tag = `<span class="tag new">found by graph · was #${before}</span>`;
    else if (delta >= 3) tag = `<span class="tag up">↑${delta}</span>`;
    else if (delta <= -3) tag = `<span class="tag down">↓${-delta}</span>`;
  }
  if (maps && mode === 'baseline') {
    const after = maps.g.get(row.index); const delta = pos - after;
    if (after > 24) tag = `<span class="tag down">graph drops to #${after}</span>`;
    else if (Math.abs(delta) >= 3) tag = `<span class="tag ${delta > 0 ? 'up' : 'down'}">graph: #${after}</span>`;
  }
  const num = maps ? `<span class="tag">#${pos}</span>` : '';
  return `<div class="metawrap"><div class="card ${saved ? 'saved' : ''} ${state.selected === row.index ? 'sel' : ''}" data-i="${row.index}" tabindex="0">
    <img src="${esc(it.image)}" alt="${esc(it.title)}" width="${it.width}" height="${it.height}" loading="lazy" onload="this.classList.add('ld')">
    <div class="badge">${num}${tag}</div>
    <div class="ov">
      <div class="ovt"><button class="btn" data-act="lens" title="Lens: search one region">⌖ Lens</button><button class="btn ${saved ? '' : 'red'}" data-act="save" title="${saved ? 'Remove from board' : 'Save to board'}">${saved ? '✓ Saved' : '+ Save'}</button></div>
      <div class="ovb"><button class="btn" data-act="more">More like this</button></div>
    </div></div>
    <div class="meta"><b><span class="dotb" style="background:${it.boardColor}"></span>${esc(it.boardTitle)}</b><span>${row.score != null ? row.score.toFixed(3) : ''}</span></div></div>`;
}

// Row-major masonry: item k goes into the currently shortest column, so rank order reads left to right.
function masonry(rows, html, width, minCol, gap = 16) {
  const n = Math.max(1, Math.floor((width + gap) / (minCol + gap)));
  const cols = Array.from({ length: n }, () => ({ h: 0, parts: [] }));
  rows.forEach((r, i) => {
    const it = ITEMS[r.index]; const c = cols.reduce((a, b) => (b.h < a.h - 1e-6 ? b : a));
    c.parts.push(html(r, i)); c.h += it.height / it.width + 0.12;
  });
  return cols.map((c) => `<div class="mcol">${c.parts.join('')}</div>`).join('');
}

function renderResults() {
  const el = $('#results');
  const before = new Map($$('.card', el).map((c) => [c.dataset.i + '|' + (c.closest('.col')?.dataset.col || ''), c.getBoundingClientRect()]));
  const hasQuery = state.rows.graph.length > 0;
  const maps = hasQuery ? rankMaps() : null;
  if (state.mode === 'compare' && hasQuery) {
    el.className = 'results compare';
    const half = (el.clientWidth - 22) / 2;
    const col = (mode, title, sub) => `<div class="col" data-col="${mode}"><h3>${title}<small>${sub}</small></h3><div class="stack">${masonry(state.rows[mode].slice(0, 12), (r, i) => cardHTML(r, i + 1, mode, maps), half, 170)}</div></div>`;
    const overlap = state.rows.graph.slice(0, 12).filter((r) => (maps.b.get(r.index) || 99) <= 12).length;
    el.innerHTML = col('baseline', 'CLIP only', 'cosine similarity') + col('graph', 'Graph-aware', `${12 - overlap} of top 12 differ`);
  } else {
    el.className = 'results';
    const rows = hasQuery ? state.rows[state.mode === 'baseline' ? 'baseline' : 'graph'].slice(0, 32) : defaultFeed();
    el.innerHTML = rows.length ? masonry(rows, (r, i) => cardHTML(r, i + 1, state.mode === 'compare' ? 'graph' : state.mode, maps), el.clientWidth, 210) : '<div class="empty">No results.</div>';
  }
  // FLIP: animate cards from their previous positions so re-ranking is visible.
  for (const c of $$('.card', el)) {
    const prev = before.get(c.dataset.i + '|' + (c.closest('.col')?.dataset.col || ''));
    if (!prev) { c.animate([{ opacity: 0, transform: 'scale(.96)' }, { opacity: 1, transform: 'none' }], { duration: 380, easing: 'ease-out' }); continue; }
    const now = c.getBoundingClientRect();
    const dx = prev.left - now.left, dy = prev.top - now.top;
    if (Math.abs(dx) + Math.abs(dy) > 1) c.animate([{ transform: `translate(${dx}px,${dy}px)` }, { transform: 'none' }], { duration: 520, easing: 'cubic-bezier(.2,.8,.2,1)' });
    $('img', c).classList.add('ld');
  }
}

function renderQuery() {
  const el = $('#querybar'); const q = state.query;
  if (!q && !state.board.length) { el.hidden = true; return; }
  el.hidden = false;
  const thumb = q?.thumb ? `<img src="${esc(q.thumb)}" alt="">` : `<span class="qt">${q ? 'Aa' : '★'}</span>`;
  const label = q ? esc(q.label) : `Recommendations for your board (${state.board.length} pins)`;
  const boardNote = q && state.board.length ? ` <span class="muted">+ your board</span>` : '';
  const s = state.stats;
  const stat = s ? `${q?.encodeMs != null ? (q.encodeMs ? `encoded in ${q.encodeMs} ms · ` : 'precomputed query · ') : ''}ranked ${ITEMS.length} pins with a graph walk in ${s.rankMs.toFixed(2)} ms` : '';
  el.innerHTML = `<div class="qpill">${thumb}<span>${label}${boardNote}</span><button id="clearq" aria-label="Clear">×</button></div><span class="qstat">${stat}</span>`;
  $('#clearq').onclick = () => { state.query = null; if (!q) state.board = []; state.selected = null; render(); };
}

function renderBoard() {
  const b = state.board;
  $('#board').innerHTML = b.map((i) => `<div><img src="${esc(ITEMS[i].image)}" alt=""><button data-rm="${i}" aria-label="Remove">×</button></div>`).join('');
  $('#boardhint').hidden = b.length > 0;
  $('#clearboard').hidden = !b.length;
  $('#forboard').hidden = !b.length;
  if (!b.length) { $('#taste').innerHTML = ''; return; }
  const centroid = addWeighted(b.map((i) => E[i]));
  const sims = BOARD_CENTROIDS.map((bc) => ({ bc, s: dot(centroid, bc.centroid) }));
  const exps = sims.map((x) => Math.exp(x.s * 18)); const Z = exps.reduce((a, c) => a + c, 0);
  const top = sims.map((x, k) => ({ ...x, p: exps[k] / Z })).sort((a, c) => c.p - a.p).slice(0, 4);
  $('#taste').innerHTML = `<p class="hint" style="margin:6px 0 2px">Your board leans toward</p>` + top.map((x) => `<div class="trow"><span>${esc(x.bc.title)}</span><span class="bar"><i style="width:${pct(x.p)};background:${x.bc.color}"></i></span><em>${pct(x.p)}</em></div>`).join('');
}

function explain(i) {
  if (!state.stats || state.mode === 'baseline') return null;
  const v = state.stats.vector; let best = null;
  for (const nb of ADJ[i]) {
    const s = dot(v, E[nb.index]) * nb.probability;
    if (!best || s > best.s) best = { ...nb, s, cos: dot(v, E[nb.index]) };
  }
  return best;
}

function renderWhy() {
  const el = $('#why'); const i = state.selected;
  if (i == null) { el.hidden = true; return; }
  el.hidden = false;
  const it = ITEMS[i]; const maps = state.rows.graph.length ? rankMaps() : null;
  const row = state.rows.graph.find((r) => r.index === i);
  const lic = `${(it.license || '').toUpperCase()} ${it.licenseVersion || ''}`.trim();
  let body = '';
  if (row) {
    const path = explain(i);
    const pathHTML = path ? `<div class="path"><img src="${esc(ITEMS[path.index].image)}" alt=""><span>${path.type === 'board' ? `Shares the <b>${esc(ITEMS[path.index].boardTitle)}</b> board with` : 'Is a visual neighbor of'} this pin, which matches your query at cosine ${path.cos.toFixed(2)}. The graph walk flows through that edge.</span></div>` : '';
    body = `<div class="scores">
      <div class="srow"><span>CLIP match</span><span class="bar"><i style="width:${pct(Math.max(0, row.clip) / 0.4)};background:var(--blue)"></i></span><em>${row.clip.toFixed(3)}</em></div>
      <div class="srow"><span>Graph walk</span><span class="bar"><i style="width:${pct(row.graph)};background:var(--gold)"></i></span><em>${row.graph.toFixed(2)}</em></div>
      <div class="srow"><span>Final score</span><span class="bar"><i style="width:${pct(Math.max(0, row.score) / 0.6)};background:var(--red)"></i></span><em>${row.score.toFixed(3)}</em></div></div>
      <p class="rankline">Graph-aware #${maps.g.get(i)} · CLIP only #${maps.b.get(i)}</p>${pathHTML}`;
  }
  el.innerHTML = `<img class="hero" src="${esc(it.image)}" alt=""><h3>${esc(it.title.slice(0, 90))}</h3>
    <p class="credit">by ${it.creatorUrl ? `<a href="${esc(it.creatorUrl)}" target="_blank" rel="noopener">${esc(it.creator)}</a>` : esc(it.creator)} · ${it.licenseUrl ? `<a href="${esc(it.licenseUrl)}" target="_blank" rel="noopener">${esc(lic)}</a>` : esc(lic)} · <a href="${esc(it.landingUrl)}" target="_blank" rel="noopener">source ↗</a></p>
    ${body || '<p class="hint">Search for something, or press “More like this”, to see why this pin ranks where it does.</p>'}
    <div style="display:flex;gap:6px;margin-top:10px"><button class="ghost" style="margin:0" data-why="more">More like this</button><button class="ghost" style="margin:0" data-why="lens">Lens</button></div>`;
  $('[data-why=more]', el).onclick = () => searchPin(i);
  $('[data-why=lens]', el).onclick = () => openLens(it.image, it.title);
}

function renderChips() {
  $('#chips').innerHTML = INDEX.suggestions.map((s) => `<button class="chip" data-q="${esc(s.text)}">${esc(s.text)}</button>`).join('');
}

function renderFoot() {
  const c = METRICS?.corpus;
  $('#foot').innerHTML = `<span>${ITEMS.length} openly licensed images from <a href="https://openverse.org" target="_blank" rel="noopener">Openverse</a> (CC0, public domain, CC BY, CC BY-SA); credit and license on every pin.</span>
  <span>Model: ${esc(INDEX.model)} (OpenAI CLIP ViT-B/16, 8-bit).</span><span>${c ? `${c.visualEdges} visual + ${c.boardEdges} board edges.` : ''}</span><span><a href="${REPO}" target="_blank" rel="noopener">Source on GitHub</a></span><span>${STATIC ? 'Static demo build.' : 'Live mode: local server.'}</span>`;
}

function renderHow() {
  const m = METRICS;
  const row = (label, a, b, digits = 3) => `<tr><td>${label}</td><td class="n">${a.toFixed(digits)}</td><td class="n ${b > a ? 'win' : ''}">${b.toFixed(digits)}</td><td class="n">${b > a ? '+' : ''}${(((b - a) / a) * 100).toFixed(0)}%</td></tr>`;
  const metricsHTML = m ? `
    <h3>Does the graph actually help? Measured, not asserted.</h3>
    <p>Two tasks over the same ${m.corpus.images} images, same CLIP vectors. The only difference between columns is the graph.</p>
    <table><thead><tr><th>Related pins, cold start (${m.relatedPins.queries} queries)</th><th>CLIP only</th><th>Graph-aware</th><th>Change</th></tr></thead><tbody>
      ${row('Recall@10', m.relatedPins.baseline.r10, m.relatedPins.graph.r10)}${row('Mean average precision', m.relatedPins.baseline.ap, m.relatedPins.graph.ap)}${row('NDCG@10', m.relatedPins.baseline.ndcg, m.relatedPins.graph.ndcg)}</tbody></table>
    <p class="muted" style="font-size:13px">Each image is queried in turn with <b>its own board label hidden from the graph</b>, so it enters like a brand-new pin with only visual edges. Relevant = its 9 board-mates. Paired bootstrap on MAP: graph better in ${pct(m.relatedPins.bootstrapMAP.pGraphBetter)} of 5,000 resamples, 95% CI of the gain [${m.relatedPins.bootstrapMAP.ci95.join(', ')}].</p>
    <table><thead><tr><th>Text search, held-out queries (${m.textSearch.queries})</th><th>CLIP only</th><th>Graph-aware</th><th>Change</th></tr></thead><tbody>
      ${row('Precision@10', m.textSearch.baseline.p10, m.textSearch.graph.p10)}${row('Mean average precision', m.textSearch.baseline.ap, m.textSearch.graph.ap)}${row('NDCG@10', m.textSearch.baseline.ndcg, m.textSearch.graph.ndcg)}</tbody></table>
    <p class="muted" style="font-size:13px">20 hand-written queries, none of which were used to collect the images. 95% CI of the MAP gain [${m.textSearch.bootstrapMAP.ci95.join(', ')}].</p>
    <h3>How much graph?</h3>
    <p>Related-pins MAP as the graph weight <code>w</code> goes from 0 to 0.6. The shipped weight (0.3, red) was fixed before this sweep; the curve is flat past 0.2, so the result is not a tuned spike.</p>
    <div class="sweep">${m.sweep.map((s) => { const lo = 0.8; const h = Math.max(4, ((s.relatedMAP - lo) / (0.9 - lo)) * 100); return `<div class="${s.graphWeight === 0.3 ? 'on' : ''}" style="height:${h}%"><em>${s.relatedMAP.toFixed(3)}</em><span>w=${s.graphWeight}</span></div>`; }).join('')}</div>
    <p class="muted" style="font-size:13px;margin-top:28px">At w=0 the walk is off but neighbor-fused embeddings remain, which already lifts MAP from ${m.relatedPins.baseline.ap.toFixed(3)} to ${m.sweep[0].relatedMAP.toFixed(3)}.</p>
    <h3>Speed</h3>
    <p>Ranking all pins with fusion + personalized PageRank: <b>${m.latencyMs.rankP50.toFixed(2)} ms</b> p50, ${m.latencyMs.rankP95.toFixed(2)} ms p95. Encoding a text query with the warm CLIP text tower: ${m.latencyMs.textEncodeP50.toFixed(1)} ms p50 on an Apple Silicon laptop CPU.</p>` : '<p>Run <code>npm run evaluate</code> to generate metrics.</p>';
  $('#how').innerHTML = `<h2>Images understood through the company they keep.</h2>
    <p class="lede">CLIP knows what an image looks like. A board graph knows what people think belongs together. Mosaic combines the two, the idea behind Pinterest's PinCLIP and PinSage, at a scale you can inspect pin by pin.</p>
    <div class="pipe"><div><b>1 · Encode</b>Every image and every query goes through the same CLIP ViT-B/16 model into one 512-d space.</div><div><b>2 · Connect</b>Two edge types: each pin's 6 nearest visual neighbors, and its 4 closest board-mates.</div><div><b>3 · Fuse</b>Each pin's vector is blended 70/30 with its weighted neighborhood, one GraphSAGE-style hop.</div><div><b>4 · Walk</b>Personalized PageRank from the query's best matches; final score = 0.7 fused cosine + 0.3 walk.</div></div>
    ${metricsHTML}
    <h3>What is real and what is not</h3>
    <div class="note"><p style="margin:0 0 6px"><b>Real:</b> the images and licenses, the CLIP model, every embedding, the graph, the ranking, the metrics. Nothing here is precomputed per query except the ten suggestion chips' text vectors.</p>
    <p style="margin:0 0 6px"><b>Approximated:</b> “boards” are ten curated collections assembled by search, not real users' boards, and labels are weak: a few images sit on a board their content does not fit. Pinterest's graph has billions of human saves. This is a 100-pin model of the idea, not of the system.</p>
    <p style="margin:0"><b>Not done:</b> no trained projection head (fusion is parameter-free), no ANN index (at 100 pins, exact search is faster), no user personalization beyond the board you build here.</p></div>
    <h3>Run it</h3><p><code>git clone ${REPO}</code>, then <code>npm install &amp;&amp; npm run setup &amp;&amp; npm start</code>. Setup downloads the images, embeds them, builds the graph and re-runs the evaluation; <code>npm test</code> runs the unit and API tests.</p>`;
}

// ---------- constellation ----------
const sky = { imgs: new Map(), hover: -1, pts: [] };
function drawSky() {
  const cv = $('#sky'); const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight; if (!w) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#14130f'; g.fillRect(0, 0, w, h);
  const pad = 60;
  sky.pts = ITEMS.map((it) => ({ x: pad + it.x * (w - pad * 2), y: pad + it.y * (h - pad * 2 - 30) }));
  const top = new Map((state.rows[state.mode === 'baseline' ? 'baseline' : 'graph'] || []).slice(0, 12).map((r, k) => [r.index, k + 1]));
  const seeds = new Set([...(state.query?.seedIndex != null ? [state.query.seedIndex] : []), ...state.board]);
  const active = top.size > 0;
  for (const e of EDGES) {
    const a = sky.pts[e.src], b = sky.pts[e.dst];
    const lit = active && (top.has(e.src) || seeds.has(e.src)) && (top.has(e.dst) || seeds.has(e.dst));
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
    g.setLineDash(e.type === 'visual' ? [3, 4] : []);
    g.strokeStyle = e.type === 'board' ? `rgba(228,178,106,${lit ? 0.85 : active ? 0.07 : 0.22})` : `rgba(154,166,180,${lit ? 0.75 : active ? 0.05 : 0.16})`;
    g.lineWidth = lit ? 1.8 : 1; g.stroke();
  }
  g.setLineDash([]);
  ITEMS.forEach((it, i) => {
    const p = sky.pts[i]; const hit = top.get(i); const seed = seeds.has(i);
    const r = hit ? 16 : seed ? 18 : i === sky.hover ? 20 : 12;
    let img = sky.imgs.get(i);
    if (!img) { img = new Image(); img.src = it.image; img.onload = () => drawSky(); sky.imgs.set(i, img); }
    g.save(); g.globalAlpha = active && !hit && !seed && i !== sky.hover ? 0.32 : 1;
    g.beginPath(); g.arc(p.x, p.y, r, 0, Math.PI * 2); g.closePath();
    g.fillStyle = it.boardColor; g.fill();
    if (img.complete && img.naturalWidth) {
      g.clip(); const s = Math.max((2 * r) / img.naturalWidth, (2 * r) / img.naturalHeight);
      g.drawImage(img, p.x - (img.naturalWidth * s) / 2, p.y - (img.naturalHeight * s) / 2, img.naturalWidth * s, img.naturalHeight * s);
    }
    g.restore();
    g.beginPath(); g.arc(p.x, p.y, r + 1.5, 0, Math.PI * 2);
    g.strokeStyle = seed ? '#ffffff' : hit ? '#ff6b5e' : active ? 'rgba(255,255,255,.08)' : it.boardColor; g.lineWidth = seed || hit ? 3 : 1.5; g.stroke();
    if (hit) { g.fillStyle = '#ff6b5e'; g.beginPath(); g.arc(p.x + r * 0.75, p.y - r * 0.75, 9, 0, Math.PI * 2); g.fill(); g.fillStyle = '#fff'; g.font = '600 10px ui-sans-serif,system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(hit), p.x + r * 0.75, p.y - r * 0.75); }
  });
  // board labels at centroids
  g.font = '600 12px "Iowan Old Style",Georgia,serif'; g.textAlign = 'center';
  for (const b of INDEX.boards) {
    const idx = ITEMS.map((x, i) => (x.board === b.slug ? i : -1)).filter((i) => i >= 0);
    const cx = idx.reduce((s, i) => s + sky.pts[i].x, 0) / idx.length; const cy = Math.min(...idx.map((i) => sky.pts[i].y)) - 22;
    const ly = Math.max(16, cy - 6); const tw = g.measureText(b.title).width + 14;
    g.fillStyle = 'rgba(20,19,15,.78)'; g.beginPath(); g.roundRect(cx - tw / 2, ly - 10, tw, 19, 9); g.fill();
    g.fillStyle = active ? 'rgba(243,237,226,.45)' : 'rgba(243,237,226,.9)'; g.textBaseline = 'middle'; g.fillText(b.title, cx, ly);
  }
}
function skyHit(ev) {
  const r = $('#sky').getBoundingClientRect(); const x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = -1, bd = 26 * 26;
  sky.pts.forEach((p, i) => { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = i; } });
  return { best, x, y };
}

// ---------- actions ----------
async function searchText(text) {
  text = text.trim(); if (!text) return;
  try {
    const { vector, ms, cached } = await encode({ text });
    state.query = { kind: 'text', label: `“${text}”`, vector, encodeMs: cached ? 0 : ms };
    state.selected = null; switchTab(state.tab === 'how' ? 'discover' : state.tab); render();
  } catch (e) { hideToast(); toast(e.message); }
}
function searchPin(i) {
  const it = ITEMS[i];
  state.query = { kind: 'pin', label: `More like: ${it.title.slice(0, 48)}`, vector: E[i], seedIndex: i, thumb: it.image, encodeMs: null };
  state.selected = null; if (state.tab === 'how') switchTab('discover'); render(); window.scrollTo({ top: 0, behavior: 'smooth' });
}
async function searchImage(dataUrl, label) {
  try {
    const { vector, ms } = await encode({ image: dataUrl });
    state.query = { kind: 'image', label, vector, thumb: dataUrl, encodeMs: ms };
    state.selected = null; switchTab('discover'); render(); window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (e) { hideToast(); toast(e.message); }
}
function toggleSave(i) {
  const k = state.board.indexOf(i);
  if (k >= 0) state.board.splice(k, 1); else { state.board.push(i); toast(`Saved to your board · rankings now blend ${state.board.length} pin${state.board.length > 1 ? 's' : ''}`); }
  render();
}
function switchTab(tab) {
  state.tab = tab;
  $$('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  $$('.tab').forEach((t) => t.classList.toggle('on', t.id === `tab-${tab}`));
  if (tab === 'constellation') requestAnimationFrame(drawSky);
  if (tab === 'discover') renderResults();
}

// ---------- lens ----------
const lens = { sel: null, src: null, label: '' };
function openLens(src, label) {
  lens.src = src; lens.label = label; lens.sel = null;
  $('#lensimg').src = src; $('#lensrect').style.display = 'none';
  $('#lensinfo').textContent = 'No region selected: the whole image will be used.';
  $('#lens').hidden = false;
}
function cropDataUrl() {
  const img = $('#lensimg'); const W = img.naturalWidth, H = img.naturalHeight;
  const s = lens.sel ? { x: lens.sel.x * W, y: lens.sel.y * H, w: lens.sel.w * W, h: lens.sel.h * H } : { x: 0, y: 0, w: W, h: H };
  const scale = Math.min(1, 448 / Math.max(s.w, s.h));
  const c = document.createElement('canvas'); c.width = Math.max(8, Math.round(s.w * scale)); c.height = Math.max(8, Math.round(s.h * scale));
  c.getContext('2d').drawImage(img, s.x, s.y, s.w, s.h, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.9);
}

// ---------- toast ----------
let toastTimer;
function toast(msg, ms = 2400) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastTimer); if (ms) toastTimer = setTimeout(hideToast, ms); }
function hideToast() { $('#toast').hidden = true; }

// ---------- events ----------
function bind() {
  $('#search').addEventListener('submit', (e) => { e.preventDefault(); searchText($('#q').value); });
  $('#chips').addEventListener('click', (e) => { const b = e.target.closest('[data-q]'); if (b) { $('#q').value = b.dataset.q; searchText(b.dataset.q); } });
  $$('.tabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
  $$('.modes button').forEach((b) => b.addEventListener('click', () => {
    state.mode = b.dataset.mode; $$('.modes button').forEach((x) => x.classList.toggle('on', x === b));
    if (state.mode === 'compare' && !state.query && !state.board.length) toast('Search for something first, then compare the two rankings.');
    render();
  }));
  $('#results').addEventListener('click', (e) => {
    const card = e.target.closest('.card'); if (!card) return; const i = Number(card.dataset.i);
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'save') return toggleSave(i);
    if (act === 'more') return searchPin(i);
    if (act === 'lens') return openLens(ITEMS[i].image, ITEMS[i].title);
    state.selected = i; $$('.card').forEach((c) => c.classList.toggle('sel', Number(c.dataset.i) === i)); renderWhy();
  });
  $('#board').addEventListener('click', (e) => { const b = e.target.closest('[data-rm]'); if (b) toggleSave(Number(b.dataset.rm)); });
  $('#clearboard').onclick = () => { state.board = []; render(); };
  $('#forboard').onclick = () => { state.query = null; state.selected = null; switchTab('discover'); render(); };
  $('#upload').addEventListener('change', async (e) => {
    const f = e.target.files[0]; if (!f) return; e.target.value = '';
    const url = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(f); });
    const img = new Image(); img.src = url; await img.decode();
    const s = Math.min(1, 640 / Math.max(img.width, img.height)); const c = document.createElement('canvas');
    c.width = Math.round(img.width * s); c.height = Math.round(img.height * s); c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    openLens(c.toDataURL('image/jpeg', 0.92), f.name);
  });
  // lens drag
  const stage = $('.lensstage'); let start = null;
  stage.addEventListener('dragstart', (e) => e.preventDefault());
  stage.addEventListener('pointerdown', (e) => { e.preventDefault(); const r = $('#lensimg').getBoundingClientRect(); start = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height }; stage.setPointerCapture(e.pointerId); });
  stage.addEventListener('pointermove', (e) => {
    if (!start) return; const r = $('#lensimg').getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    lens.sel = { x: Math.min(x, start.x), y: Math.min(y, start.y), w: Math.abs(x - start.x), h: Math.abs(y - start.y) };
    Object.assign($('#lensrect').style, { display: 'block', left: `${lens.sel.x * r.width}px`, top: `${lens.sel.y * r.height}px`, width: `${lens.sel.w * r.width}px`, height: `${lens.sel.h * r.height}px` });
  });
  stage.addEventListener('pointerup', () => { start = null; if (lens.sel && (lens.sel.w < 0.04 || lens.sel.h < 0.04)) { lens.sel = null; $('#lensrect').style.display = 'none'; } $('#lensinfo').textContent = lens.sel ? `Region: ${pct(lens.sel.w)} × ${pct(lens.sel.h)} of the image.` : 'No region selected: the whole image will be used.'; });
  $('#lensclose').onclick = () => { $('#lens').hidden = true; };
  $('#lens').addEventListener('click', (e) => { if (e.target.id === 'lens') $('#lens').hidden = true; });
  $('#lensgo').onclick = () => { const url = cropDataUrl(); $('#lens').hidden = true; searchImage(url, lens.sel ? `Lens crop of “${lens.label.slice(0, 40)}”` : `Image: ${lens.label.slice(0, 48)}`); };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lens').hidden = true; if (e.key === '/' && document.activeElement !== $('#q')) { e.preventDefault(); $('#q').focus(); } });
  // constellation
  const cv = $('#sky');
  cv.addEventListener('mousemove', (e) => {
    const { best, x, y } = skyHit(e); const tip = $('#tip');
    if (best !== sky.hover) { sky.hover = best; drawSky(); }
    if (best < 0) { tip.hidden = true; return; }
    const it = ITEMS[best]; tip.hidden = false;
    tip.innerHTML = `<img src="${esc(it.image)}" alt=""><b>${esc(it.title.slice(0, 60))}</b><br><span class="muted">${esc(it.boardTitle)} · ${ADJ[best].length} edges</span>`;
    tip.style.left = `${Math.min(x + 18, cv.clientWidth - 215)}px`; tip.style.top = `${Math.min(y + 12, cv.clientHeight - 200)}px`;
  });
  cv.addEventListener('mouseleave', () => { sky.hover = -1; $('#tip').hidden = true; drawSky(); });
  cv.addEventListener('click', (e) => { const { best } = skyHit(e); if (best >= 0) { searchPin(best); switchTab('constellation'); } });
  let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (state.tab === 'constellation') drawSky(); else renderResults(); }, 120); });
}

boot().catch((e) => { document.body.insertAdjacentHTML('beforeend', `<p class="empty">Failed to load Mosaic: ${esc(e.message)}. Run <code>npm run setup</code> first.</p>`); console.error(e); });
