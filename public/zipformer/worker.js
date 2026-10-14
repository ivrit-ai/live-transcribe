// On-device Hebrew streaming ASR worker. Model: zipformer2-66m-streaming-he by
// Yakov Kolani (https://huggingface.co/thewh1teagle/zipformer2-66m-streaming-he),
// loaded from the ivrit-ai/zipformer2-66m-streaming-he mirror.
// Runs the int8 sherpa-onnx export on onnxruntime-web's WASM backend
// (multi-threaded when the page is cross-origin isolated). In testing this was
// as fast as WebGPU fp16 on integrated GPUs at half the download. Model files
// come from Hugging Face at a pinned revision and are kept in Cache Storage.

// Both modules are imported dynamically inside init(): a failed static import
// kills the worker with an empty error event (WebKit), while a failed dynamic
// import is an ordinary exception we can report to the page.
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
let ort = null;
let mod = null; // ./asr.js

const REPO = 'ivrit-ai/zipformer2-66m-streaming-he';
const REVISION = '3d01a65d00e45cbb238996795be87e8e800a3226';
const BASE = `https://huggingface.co/${REPO}/resolve/${REVISION}/`;
const CACHE_NAME = `zipformer2-he-${REVISION}`;

const FILES = {
  encoder: 'sherpa-onnx/encoder.int8.onnx',
  decoder: 'sherpa-onnx/decoder.int8.onnx',
  joiner: 'sherpa-onnx/joiner.int8.onnx',
};
const CONFIG_FILE = 'onnx/config.json';
const TOKENS_FILE = 'onnx/tokens.txt';

let asr = null;
let resampler = null;
let receivedSamples = 0;
let queue = Promise.resolve();

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

// Anything that escapes the message queue (e.g. a failure inside the WASM
// runtime's own threads) still reaches the page with a reason attached.
self.addEventListener('error', (e) => {
  e.preventDefault();
  post({ type: 'error', message: e.message || (e.error && e.error.message) || 'uncaught error in worker' });
});
self.addEventListener('unhandledrejection', (e) => {
  e.preventDefault();
  post({ type: 'error', message: (e.reason && e.reason.message) || String(e.reason) });
});

async function cachedResponses(paths, onlyIfCached) {
  const cache = self.caches ? await caches.open(CACHE_NAME).catch(() => null) : null;
  const out = [];
  for (const p of paths) {
    const url = BASE + p;
    const hit = cache && await cache.match(url);
    if (hit) { out.push({ path: p, cached: true, res: hit }); continue; }
    if (onlyIfCached) return null;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Model download failed (${res.status}) for ${p}`);
    out.push({ path: p, cached: false, res });
  }
  return { cache, out };
}

// Download (or read from cache) all files, reporting combined progress.
async function loadFiles(paths, onlyIfCached) {
  const got = await cachedResponses(paths, onlyIfCached);
  if (!got) return null;
  const { cache, out } = got;
  const total = out.reduce((s, f) => s + Number(f.res.headers.get('content-length') || 0), 0);
  let loaded = 0;
  let lastPost = 0;
  const bytes = await Promise.all(out.map(async (f) => {
    // Write straight into one buffer when the size is known, so the model is
    // never held twice in memory while downloading (matters on phones).
    const size = Number(f.res.headers.get('content-length') || 0);
    const reader = f.res.body.getReader();
    let buf = new Uint8Array(size);
    let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (n + value.length > buf.length) {
        const grown = new Uint8Array(Math.max(n + value.length, buf.length * 2));
        grown.set(buf.subarray(0, n));
        buf = grown;
      }
      buf.set(value, n);
      n += value.length;
      loaded += value.length;
      const now = Date.now();
      if (now - lastPost > 150) { lastPost = now; post({ type: 'progress', loaded, total, cached: f.cached }); }
    }
    if (n !== buf.length) buf = buf.slice(0, n);
    if (!f.cached && cache) {
      const headers = { 'content-type': 'application/octet-stream', 'content-length': String(n) };
      cache.put(BASE + f.path, new Response(buf, { headers })).catch(() => {});
    }
    return buf;
  }));
  post({ type: 'progress', loaded: total, total });
  return Object.fromEntries(paths.map((p, i) => [p, bytes[i]]));
}

// threads: 0 = auto (multi-threaded when cross-origin isolated), 1 = single.
async function loadRuntime(threads) {
  if (ort) return;
  [ort, mod] = await Promise.all([import(ORT_URL), import('./asr.js')]);
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  ort.env.wasm.numThreads = threads || (self.crossOriginIsolated
    ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1);
}

async function init({ onlyIfCached, threads }) {
  if (asr) { post({ type: 'ready', threads: ort.env.wasm.numThreads }); return; }
  await loadRuntime(threads);
  const paths = [CONFIG_FILE, TOKENS_FILE, FILES.encoder, FILES.decoder, FILES.joiner];
  const data = await loadFiles(paths, onlyIfCached);
  if (!data) { post({ type: 'not-cached' }); return; }

  post({ type: 'status', text: 'Loading model…' });
  const dec = new TextDecoder();
  const config = JSON.parse(dec.decode(data[CONFIG_FILE]));
  const symbols = mod.parseTokens(dec.decode(data[TOKENS_FILE]));
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const sessions = {
    encoder: await ort.InferenceSession.create(data[FILES.encoder], opts),
    decoder: await ort.InferenceSession.create(data[FILES.decoder], opts),
    joiner: await ort.InferenceSession.create(data[FILES.joiner], opts),
  };
  asr = new mod.StreamingTransducer(ort, sessions, config, symbols);

  // Warm-up: the first runs allocate buffers and spin up WASM threads.
  post({ type: 'status', text: 'Warming up…' });
  await asr.accept(new Float32Array(16000 * 1.5));
  asr.reset();
  post({ type: 'ready', threads: ort.env.wasm.numThreads });
}

function lagSeconds() {
  // Audio received but not yet through the encoder, minus the chunk the
  // encoder always needs in hand (T frames).
  const processedSamples = asr.processed * 160;
  return Math.max(0, (receivedSamples - processedSamples) / 16000 - (asr.cfg.T * 0.01));
}

async function handle(msg) {
  switch (msg.type) {
    case 'init':
      await init(msg);
      break;
    case 'start':
      asr.reset();
      resampler = new mod.Resampler(msg.sampleRate, 16000);
      receivedSamples = 0;
      break;
    case 'audio': {
      if (!asr || !resampler) break;
      const pcm = resampler.process(msg.data);
      receivedSamples += pcm.length;
      const r = await asr.accept(pcm);
      post({ type: 'result', finals: r.finals, partial: r.partial, lag: lagSeconds() });
      break;
    }
    case 'stop': {
      if (!asr) break;
      const r = await asr.finish();
      resampler = null;
      post({ type: 'stopped', finals: r.finals, timing: asr.timing });
      break;
    }
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  queue = queue.then(() => handle(msg)).catch((err) => {
    post({ type: 'error', message: (err && err.message) || String(err), during: msg.type });
  });
};
