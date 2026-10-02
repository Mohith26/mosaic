import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const ROOT = path.resolve(new URL('..', import.meta.url).pathname);
const IMAGE_DIR = path.join(ROOT, 'data/images');
const MANIFEST_PATH = path.join(ROOT, 'data/manifest.json');
const FORCE = process.argv.includes('--force');

const boards = [
  { slug: 'quiet-rooms', title: 'Quiet rooms', query: 'minimalist interior natural light', alternates: ['minimalist interior', 'scandinavian living room'], color: '#d7c2a3' },
  { slug: 'concrete-poetry', title: 'Concrete poetry', query: 'brutalist architecture concrete', alternates: ['concrete building', 'modernist architecture'], color: '#8f9998' },
  { slug: 'living-forms', title: 'Living forms', query: 'botanical illustration vintage', alternates: ['botanical illustration', 'vintage flower illustration', 'herbarium plant'], color: '#87a687' },
  { slug: 'earth-and-fire', title: 'Earth & fire', query: 'handmade ceramic pottery', alternates: ['ceramic vase', 'pottery bowl'], color: '#c77c55' },
  { slug: 'still-water', title: 'Still water', query: 'japanese garden pond', alternates: ['japanese garden', 'koi pond'], color: '#6a8b78' },
  { slug: 'street-language', title: 'Street language', query: 'street style fashion editorial', alternates: ['street fashion', 'fashion portrait'], color: '#b95f69' },
  { slug: 'small-rituals', title: 'Small rituals', query: 'artisan dessert food photography', alternates: ['pastry dessert', 'cake slice', 'latte art coffee', 'fruit tart'], color: '#d39c68' },
  { slug: 'machines-with-soul', title: 'Machines with soul', query: 'vintage motorcycle photography', alternates: ['vintage motorcycle', 'classic motorbike'], color: '#786f67' },
  { slug: 'edge-of-land', title: 'Edge of land', query: 'coastal landscape cliffs', alternates: ['sea cliffs', 'rocky coastline'], color: '#5f8fa7' },
  { slug: 'shape-and-color', title: 'Shape & color', query: 'abstract geometric art', alternates: ['geometric abstract painting', 'bauhaus poster'], color: '#9b69bf' },
];

async function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

async function fetchJson(url, attempts = 4) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { headers: { 'User-Agent': 'MosaicVisualDiscovery/1.0 (portfolio research project)' } });
      if (response.status === 429) {
        await sleep((attempt + 1) * 3500);
        continue;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      lastError = error;
      await sleep((attempt + 1) * 1200);
    }
  }
  throw lastError;
}

async function download(url, destination, attempts = 3) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { redirect: 'follow', headers: { 'User-Agent': 'MosaicVisualDiscovery/1.0' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.startsWith('image/')) throw new Error(`Unexpected content type ${contentType}`);
      const bytes = Buffer.from(await response.arrayBuffer());
      if (bytes.length < 7000) throw new Error(`Image too small (${bytes.length} bytes)`);
      await fs.writeFile(destination, bytes);
      return { bytes: bytes.length, contentType };
    } catch (error) {
      lastError = error;
      await sleep((attempt + 1) * 700);
    }
  }
  throw lastError;
}

function extensionFor(contentType, sourceUrl) {
  if (contentType.includes('png')) return '.png';
  if (contentType.includes('webp')) return '.webp';
  if (contentType.includes('gif')) return '.gif';
  const ext = path.extname(new URL(sourceUrl).pathname).toLowerCase();
  return ['.jpg', '.jpeg', '.png', '.webp'].includes(ext) ? ext : '.jpg';
}

await fs.mkdir(IMAGE_DIR, { recursive: true });
if (!FORCE) {
  try {
    const existing = JSON.parse(await fs.readFile(MANIFEST_PATH, 'utf8'));
    if (existing.items?.length >= 80) {
      console.log(`Using existing manifest with ${existing.items.length} images. Pass --force to rebuild.`);
      process.exit(0);
    }
  } catch {}
}

if (FORCE) {
  const old = await fs.readdir(IMAGE_DIR).catch(() => []);
  await Promise.all(old.map((name) => fs.rm(path.join(IMAGE_DIR, name), { force: true })));
}

const PARTIAL_PATH = path.join(ROOT, 'data/manifest.partial.json');
let items = [];
if (!FORCE) {
  try { items = JSON.parse(await fs.readFile(PARTIAL_PATH, 'utf8')).items || []; } catch {}
}
const seenIds = new Set(items.map((item) => item.sourceId).filter(Boolean));
const seenUrls = new Set(items.map((item) => item.sourceUrl).filter(Boolean));
for (let boardIndex = 0; boardIndex < boards.length; boardIndex += 1) {
  const board = boards[boardIndex];
  const existingCount = items.filter((item) => item.board === board.slug).length;
  if (existingCount >= 8) {
    console.log(`[${boardIndex + 1}/${boards.length}] ${board.title}: reusing ${existingCount} images`);
    continue;
  }
  items = items.filter((item) => item.board !== board.slug);
  console.log(`[${boardIndex + 1}/${boards.length}] Searching ${board.title}...`);
  const creatorCounts = new Map();
  let accepted = 0;
  const searches = [board.query, ...(board.alternates || [])].flatMap((q) => [1, 2].map((page) => ({ q, page })));
  for (const { q, page } of searches) {
    if (accepted >= 10) break;
    const params = new URLSearchParams({
      q,
      page_size: '20',
      page: String(page),
      license: 'cc0,pdm,by,by-sa',
      mature: 'false',
    });
    const payload = await fetchJson(`https://api.openverse.org/v1/images/?${params}`);
    const candidates = (payload.results || []).filter((item) => {
      if (!item.thumbnail || !item.foreign_landing_url || !item.id || seenIds.has(item.id) || seenUrls.has(item.url)) return false;
      if ((item.width || 0) < 320 || (item.height || 0) < 260) return false;
      const ratio = item.width / item.height;
      return ratio >= 0.48 && ratio <= 2.2;
    });

    for (const candidate of candidates) {
      if (accepted >= 10) break;
      const creator = candidate.creator || 'Unknown creator';
      if ((creatorCounts.get(creator) || 0) >= 4) continue;
      const temporary = path.join(IMAGE_DIR, `${candidate.id}.download`);
      try {
        const info = await download(candidate.thumbnail, temporary);
        const extension = extensionFor(info.contentType, candidate.url || candidate.thumbnail);
        const filename = `${board.slug}-${String(accepted + 1).padStart(2, '0')}${extension}`;
        const destination = path.join(IMAGE_DIR, filename);
        await fs.rename(temporary, destination);
        const hash = crypto.createHash('sha256').update(await fs.readFile(destination)).digest('hex').slice(0, 16);
        if (items.some((item) => item.sha256 === hash)) {
          await fs.rm(destination, { force: true });
          continue;
        }
        creatorCounts.set(creator, (creatorCounts.get(creator) || 0) + 1);
        seenIds.add(candidate.id);
        seenUrls.add(candidate.url);
        items.push({
          id: `${board.slug}-${String(accepted + 1).padStart(2, '0')}`,
          board: board.slug,
          boardTitle: board.title,
          boardColor: board.color,
          query: board.query,
          title: candidate.title || board.title,
          creator,
          creatorUrl: candidate.creator_url || null,
          landingUrl: candidate.foreign_landing_url,
          license: candidate.license || 'unknown',
          licenseVersion: candidate.license_version || null,
          licenseUrl: candidate.license_url || null,
          attribution: candidate.attribution || null,
          provider: candidate.provider || candidate.source || 'Openverse',
          tags: (candidate.tags || []).slice(0, 12).map((tag) => tag.name),
          width: candidate.width,
          height: candidate.height,
          image: `images/${filename}`,
          bytes: info.bytes,
          sha256: hash,
          sourceId: candidate.id,
          sourceUrl: candidate.url,
        });
        accepted += 1;
        process.stdout.write(`  ${accepted}/10 ${filename}\n`);
      } catch (error) {
        await fs.rm(temporary, { force: true });
        console.warn(`  skipped ${candidate.id}: ${error.message}`);
      }
    }
    if (accepted < 10) await sleep(1100);
  }
  if (accepted < 8) throw new Error(`Only downloaded ${accepted} images for ${board.title}`);
  await fs.writeFile(PARTIAL_PATH, `${JSON.stringify({ items }, null, 2)}\n`);
  await sleep(900);
}

const manifest = {
  generatedAt: new Date().toISOString(),
  source: 'Openverse API',
  sourceUrl: 'https://openverse.org/',
  licensePolicy: 'Only CC0, public-domain, CC BY, and CC BY-SA results were requested. Per-image attribution is preserved.',
  boards,
  items,
};
await fs.writeFile(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`);
await fs.rm(PARTIAL_PATH, { force: true });
console.log(`Wrote ${items.length} images to ${MANIFEST_PATH}`);
