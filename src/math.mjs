export function dot(a, b) {
  let value = 0;
  for (let i = 0; i < a.length; i += 1) value += a[i] * b[i];
  return value;
}

export function magnitude(a) {
  return Math.sqrt(Math.max(0, dot(a, a)));
}

export function normalize(a) {
  const m = magnitude(a) || 1;
  return Array.from(a, (x) => x / m);
}

export function addWeighted(vectors, weights = []) {
  if (!vectors.length) return [];
  const out = new Array(vectors[0].length).fill(0);
  for (let i = 0; i < vectors.length; i += 1) {
    const weight = weights[i] ?? 1;
    for (let j = 0; j < out.length; j += 1) out[j] += vectors[i][j] * weight;
  }
  return normalize(out);
}

export function cosine(a, b) {
  return dot(a, b) / ((magnitude(a) || 1) * (magnitude(b) || 1));
}

export function minMax(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const span = max - min || 1;
  return values.map((value) => (value - min) / span);
}

export function seededRandom(seed = 42) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}
