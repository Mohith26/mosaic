export function recallAtK(ranked, relevant, k) {
  if (!relevant.size) return 0;
  let hits = 0;
  for (const r of ranked.slice(0, k)) if (relevant.has(r)) hits += 1;
  return hits / Math.min(k, relevant.size);
}
export function precisionAtK(ranked, relevant, k) {
  let hits = 0;
  for (const r of ranked.slice(0, k)) if (relevant.has(r)) hits += 1;
  return hits / k;
}
export function averagePrecision(ranked, relevant) {
  let hits = 0; let sum = 0;
  ranked.forEach((r, i) => { if (relevant.has(r)) { hits += 1; sum += hits / (i + 1); } });
  return relevant.size ? sum / relevant.size : 0;
}
export function ndcgAtK(ranked, relevant, k) {
  let dcg = 0;
  ranked.slice(0, k).forEach((r, i) => { if (relevant.has(r)) dcg += 1 / Math.log2(i + 2); });
  let ideal = 0;
  for (let i = 0; i < Math.min(k, relevant.size); i += 1) ideal += 1 / Math.log2(i + 2);
  return ideal ? dcg / ideal : 0;
}
export const mean = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
export function percentile(xs, p) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}
// Paired bootstrap: fraction of resamples where graph beats baseline on the mean.
export function pairedBootstrap(a, b, rounds = 5000, seed = 11) {
  let state = seed; const rand = () => ((state = (1664525 * state + 1013904223) >>> 0) / 4294967296);
  let wins = 0; const deltas = [];
  for (let r = 0; r < rounds; r += 1) {
    let d = 0;
    for (let i = 0; i < a.length; i += 1) { const j = Math.floor(rand() * a.length); d += b[j] - a[j]; }
    d /= a.length; deltas.push(d); if (d > 0) wins += 1;
  }
  deltas.sort((x, y) => x - y);
  return { pGraphBetter: wins / rounds, ci95: [deltas[Math.floor(rounds * 0.025)], deltas[Math.floor(rounds * 0.975)]] };
}
