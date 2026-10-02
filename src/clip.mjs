import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs/promises';
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from '@huggingface/transformers';
import { normalize } from './math.mjs';

export const MODEL_ID = process.env.MOSAIC_MODEL || 'Xenova/clip-vit-base-patch16';
env.cacheDir = process.env.MOSAIC_CACHE || path.resolve('.cache/huggingface');
env.allowLocalModels = true;

let processorPromise;
let tokenizerPromise;
let visionPromise;
let textPromise;

export function getProcessor() {
  processorPromise ??= AutoProcessor.from_pretrained(MODEL_ID);
  return processorPromise;
}

export function getTokenizer() {
  tokenizerPromise ??= AutoTokenizer.from_pretrained(MODEL_ID);
  return tokenizerPromise;
}

export function getVisionModel() {
  visionPromise ??= CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8' });
  return visionPromise;
}

export function getTextModel() {
  textPromise ??= CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8' });
  return textPromise;
}

export async function warmTextModel() {
  await Promise.all([getTokenizer(), getTextModel()]);
}

export async function warmVisionModel() {
  await Promise.all([getProcessor(), getVisionModel()]);
}

export async function embedTexts(texts) {
  const [tokenizer, model] = await Promise.all([getTokenizer(), getTextModel()]);
  const inputs = tokenizer(texts, { padding: true, truncation: true });
  const { text_embeds: tensor } = await model(inputs);
  const dimension = tensor.dims.at(-1);
  const data = Array.from(tensor.data);
  return texts.map((_, index) => normalize(data.slice(index * dimension, (index + 1) * dimension)));
}

export async function embedImage(input) {
  const [processor, model] = await Promise.all([getProcessor(), getVisionModel()]);
  const image = input instanceof RawImage ? input : await RawImage.read(input);
  const inputs = await processor(image);
  const { image_embeds: tensor } = await model(inputs);
  return normalize(Array.from(tensor.data));
}

export async function embedDataUrl(dataUrl) {
  const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/.exec(dataUrl || '');
  if (!match) throw new Error('Expected a base64 image data URL.');
  const extension = match[1].includes('png') ? 'png' : match[1].includes('webp') ? 'webp' : 'jpg';
  const filename = path.join(os.tmpdir(), `mosaic-upload-${process.pid}-${Date.now()}.${extension}`);
  await fs.writeFile(filename, Buffer.from(match[2], 'base64'));
  try {
    return await embedImage(filename);
  } finally {
    await fs.rm(filename, { force: true });
  }
}
