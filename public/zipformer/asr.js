// Streaming Zipformer2 transducer for the browser, on top of onnxruntime-web.
// The model, zipformer2-66m-streaming-he, was developed by Yakov Kolani:
// https://huggingface.co/thewh1teagle/zipformer2-66m-streaming-he
// Mirrors sherpa-onnx's online pipeline: Kaldi-compatible fbank, chunked encoder
// with cached states, greedy transducer search, and its default endpoint rules.
// No DOM access; runs inside a worker. `ort` is injected by the caller.

const FLT_EPSILON = 1.1920929e-7;

// ---------------------------------------------------------------------------
// Fbank: 80-bin log-mel, 25 ms povey window, 10 ms shift, preemph 0.97,
// DC removal, no dither, snip_edges=false, 20..7600 Hz. Same as
// kaldi-native-fbank with sherpa-onnx's defaults.
// ---------------------------------------------------------------------------
const FRAME_LEN = 400;
const FRAME_SHIFT = 160;
const FFT_SIZE = 512;

function melScale(f) { return 1127 * Math.log(1 + f / 700); }

function buildMelBanks(numBins, sampleRate, lowFreq, highFreq) {
  const numFftBins = FFT_SIZE / 2;
  const binWidth = sampleRate / FFT_SIZE;
  const hi = highFreq <= 0 ? sampleRate / 2 + highFreq : highFreq;
  const melLow = melScale(lowFreq), melHigh = melScale(hi);
  const delta = (melHigh - melLow) / (numBins + 1);
  const banks = [];
  for (let b = 0; b < numBins; b++) {
    const left = melLow + b * delta, center = left + delta, right = center + delta;
    let first = -1;
    const w = [];
    for (let i = 0; i < numFftBins; i++) {
      const mel = melScale(binWidth * i);
      if (mel > left && mel < right) {
        if (first < 0) first = i;
        w.push(mel <= center ? (mel - left) / (center - left) : (right - mel) / (right - center));
      }
    }
    banks.push({ offset: first, weights: Float64Array.from(w) });
  }
  return banks;
}

// In-place iterative radix-2 complex FFT.
function makeFft(n) {
  const levels = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let j = 0; j < levels; j++) r |= ((i >>> j) & 1) << (levels - 1 - j);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
  return (re, im) => {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >>> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half;
          const tr = re[b] * cos[k] + im[b] * sin[k];
          const ti = -re[b] * sin[k] + im[b] * cos[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  };
}

export class OnlineFbank {
  constructor({ sampleRate = 16000, numBins = 80, lowFreq = 20, highFreq = -400 } = {}) {
    this.numBins = numBins;
    this.banks = buildMelBanks(numBins, sampleRate, lowFreq, highFreq);
    this.window = new Float64Array(FRAME_LEN);
    for (let i = 0; i < FRAME_LEN; i++) {
      this.window[i] = Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FRAME_LEN - 1)), 0.85);
    }
    this.fft = makeFft(FFT_SIZE);
    this.re = new Float64Array(FFT_SIZE);
    this.im = new Float64Array(FFT_SIZE);
    this.wave = new Float32Array(0);
    this.waveOffset = 0;   // absolute index of wave[0]
    this.numSamples = 0;   // absolute samples received
    this.frames = [];      // Float32Array(numBins) per frame
    this.framesOffset = 0; // absolute index of frames[0]
  }

  get numFramesReady() { return this.framesOffset + this.frames.length; }

  static firstSample(frame) { return frame * FRAME_SHIFT + FRAME_SHIFT / 2 - FRAME_LEN / 2; }

  acceptWaveform(samples) {
    const merged = new Float32Array(this.wave.length + samples.length);
    merged.set(this.wave);
    merged.set(samples, this.wave.length);
    this.wave = merged;
    this.numSamples += samples.length;

    let f = this.numFramesReady;
    while (OnlineFbank.firstSample(f) + FRAME_LEN <= this.numSamples) {
      this.frames.push(this.computeFrame(OnlineFbank.firstSample(f)));
      f++;
    }
    const keepFrom = Math.max(0, OnlineFbank.firstSample(f));
    if (keepFrom > this.waveOffset) {
      this.wave = this.wave.slice(keepFrom - this.waveOffset);
      this.waveOffset = keepFrom;
    }
  }

  computeFrame(start) {
    const { re, im, wave, waveOffset } = this;
    const total = this.numSamples;
    let mean = 0;
    for (let i = 0; i < FRAME_LEN; i++) {
      let s = start + i;
      // Reflect at the signal start (only frame 0 reaches before sample 0).
      while (s < 0 || s >= total) s = s < 0 ? -s - 1 : 2 * total - 1 - s;
      re[i] = wave[s - waveOffset];
      mean += re[i];
    }
    mean /= FRAME_LEN;
    for (let i = 0; i < FRAME_LEN; i++) re[i] -= mean;
    for (let i = FRAME_LEN - 1; i > 0; i--) re[i] -= 0.97 * re[i - 1];
    re[0] -= 0.97 * re[0];
    for (let i = 0; i < FRAME_LEN; i++) re[i] *= this.window[i];
    re.fill(0, FRAME_LEN);
    im.fill(0);
    this.fft(re, im);

    const out = new Float32Array(this.numBins);
    for (let b = 0; b < this.numBins; b++) {
      const { offset, weights } = this.banks[b];
      let e = 0;
      for (let k = 0; k < weights.length; k++) {
        const i = offset + k;
        e += weights[k] * (re[i] * re[i] + im[i] * im[i]);
      }
      out[b] = Math.log(Math.max(e, FLT_EPSILON));
    }
    return out;
  }

  getFrames(start, n) {
    const out = new Float32Array(n * this.numBins);
    for (let i = 0; i < n; i++) out.set(this.frames[start + i - this.framesOffset], i * this.numBins);
    return out;
  }

  discardBefore(frame) {
    const drop = frame - this.framesOffset;
    if (drop > 0) { this.frames.splice(0, drop); this.framesOffset = frame; }
  }
}

// ---------------------------------------------------------------------------
// Streaming windowed-sinc resampler (mic rate -> 16 kHz) with anti-aliasing.
// ---------------------------------------------------------------------------
export class Resampler {
  constructor(inRate, outRate, zeroCrossings = 16) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.step = inRate / outRate;
    this.cutoff = Math.min(1, outRate / inRate) * 0.95;
    this.half = Math.ceil(zeroCrossings / this.cutoff);
    this.bufStart = -this.half; // absolute input index of buf[0] (left-padded with zeros)
    this.buf = new Float32Array(this.half);
    this.next = 0; // absolute (fractional) input position of the next output sample
  }

  process(input) {
    if (this.inRate === this.outRate) return input;
    const merged = new Float32Array(this.buf.length + input.length);
    merged.set(this.buf);
    merged.set(input, this.buf.length);
    this.buf = merged;
    const end = this.bufStart + this.buf.length; // absolute index past last input
    const out = [];
    const { half, cutoff } = this;
    while (Math.floor(this.next) + half < end) {
      const t = this.next;
      const c = Math.floor(t);
      let acc = 0;
      for (let k = c - half + 1; k <= c + half; k++) {
        const d = t - k;
        const x = cutoff * d;
        const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
        const win = 0.5 + 0.5 * Math.cos(Math.PI * d / half);
        acc += this.buf[k - this.bufStart] * sinc * win;
      }
      out.push(acc * cutoff);
      this.next += this.step;
    }
    const keepFrom = Math.floor(this.next) - half + 1;
    if (keepFrom > this.bufStart) {
      this.buf = this.buf.slice(keepFrom - this.bufStart);
      this.bufStart = keepFrom;
    }
    return Float32Array.from(out);
  }
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------
export function parseTokens(text) {
  const syms = [];
  for (const line of text.split('\n')) {
    const m = line.match(/^(.*) (\d+)$/);
    if (m) syms[Number(m[2])] = m[1];
  }
  return syms;
}

function detokenize(syms, ids) {
  return ids.map((i) => syms[i]).join('').replace(/▁/g, ' ').trim();
}

// ---------------------------------------------------------------------------
// Streaming recognizer
// ---------------------------------------------------------------------------
// Endpoint rules, as sherpa-onnx's defaults (seconds).
const RULE1_SILENCE = 2.4;  // nothing decoded yet
const RULE2_SILENCE = 1.2;  // after some text
const RULE3_MAX_UTTERANCE = 20;

export class StreamingTransducer {
  constructor(ort, sessions, config, symbols) {
    this.ort = ort;
    this.encoder = sessions.encoder;
    this.decoder = sessions.decoder;
    this.joiner = sessions.joiner;
    this.cfg = config;
    this.syms = symbols;
    this.frameSec = (config.decode_chunk_len * 0.01) / config.encoder_frames_per_chunk;
    this.reset();
  }

  reset() {
    const { ort, cfg } = this;
    this.fbank = new OnlineFbank({ sampleRate: cfg.sample_rate, numBins: cfg.feature_dim });
    this.processed = 0;
    this.states = cfg.states.map((s) => {
      const size = s.shape.reduce((a, b) => a * b, 1);
      const data = s.type === 'int64' ? new BigInt64Array(size) : new Float32Array(size);
      return new ort.Tensor(s.type, data, s.shape);
    });
    this.context = [-1, cfg.blank_id];
    this.decoderOut = null;
    this.uttTokens = [];
    this.pendingTokens = []; // endpointed, not yet emitted (see emit())
    this.uttFrames = 0;
    this.trailingBlanks = 0;
    this.timing = { encoder: 0, search: 0, chunks: 0 };
  }

  async runDecoder() {
    const y = new this.ort.Tensor('int64', BigInt64Array.from(this.context.map(BigInt)), [1, this.cfg.context_size]);
    const out = await this.decoder.run({ y });
    this.decoderOut = out.decoder_out;
  }

  get chunkReady() { return this.processed + this.cfg.T < this.fbank.numFramesReady; }

  // Feed 16 kHz samples. Returns { finals: [text...], partial: text }.
  async accept(samples) {
    this.fbank.acceptWaveform(samples);
    const finals = [];
    while (this.chunkReady) {
      await this.runChunk(finals);
    }
    return { finals, partial: this.partialText() };
  }

  partialText() { return detokenize(this.syms, this.pendingTokens.concat(this.uttTokens)); }

  // An endpointed segment is held back until the next token arrives: if that
  // token continues a word (no leading ▁, e.g. "." or a suffix), the pause
  // was mid-word and the segment is reopened instead of split.
  emit(token, finals) {
    if (this.pendingTokens.length) {
      if (this.syms[token].startsWith('▁')) {
        finals.push(detokenize(this.syms, this.pendingTokens));
      } else {
        this.uttTokens = this.pendingTokens.concat(this.uttTokens);
      }
      this.pendingTokens = [];
    }
    this.uttTokens.push(token);
  }

  // Flush the tail with silence so the last partial chunk is decoded, then
  // close out the current utterance.
  async finish() {
    const padFrames = this.cfg.T + 1 - (this.fbank.numFramesReady - this.processed);
    const finals = [];
    if (padFrames > 0) this.fbank.acceptWaveform(new Float32Array((padFrames + 2) * FRAME_SHIFT));
    while (this.chunkReady) await this.runChunk(finals);
    const last = this.partialText();
    if (last) finals.push(last);
    this.pendingTokens = [];
    this.uttTokens = [];
    return { finals, partial: '' };
  }

  async runChunk(finals) {
    const { ort, cfg } = this;
    if (!this.decoderOut) await this.runDecoder();
    const x = new ort.Tensor('float32', this.fbank.getFrames(this.processed, cfg.T), [1, cfg.T, cfg.feature_dim]);
    const feeds = { x };
    cfg.states.forEach((s, i) => { feeds[s.name] = this.states[i]; });
    const t0 = performance.now();
    const out = await this.encoder.run(feeds);
    const t1 = performance.now();
    this.timing.encoder += t1 - t0;
    this.states = cfg.states.map((s) => out['new_' + s.name]);
    this.processed += cfg.decode_chunk_len;
    this.fbank.discardBefore(this.processed);

    const enc = out.encoder_out;
    const [, frames, dim] = enc.dims;
    const encData = enc.data;
    // Greedy search, one symbol per frame. The decoder output only changes when
    // a token is emitted, so score all remaining frames against it in a single
    // batched joiner call, and rescore from the frame after each emission.
    let t = 0;
    while (t < frames) {
      const n = frames - t;
      const e = new ort.Tensor('float32', encData.slice(t * dim, frames * dim), [n, dim]);
      const d = this.decoderOut.data;
      const dRep = new Float32Array(n * d.length);
      for (let i = 0; i < n; i++) dRep.set(d, i * d.length);
      const { logit } = await this.joiner.run({
        encoder_out: e,
        decoder_out: new ort.Tensor('float32', dRep, [n, d.length]),
      });
      const V = logit.dims[1], l = logit.data;
      let emitted = false;
      for (let i = 0; i < n && !emitted; i++, t++) {
        let best = 0;
        for (let k = 1, base = i * V; k < V; k++) if (l[base + k] > l[base + best]) best = k;
        this.uttFrames++;
        if (best !== cfg.blank_id && best !== cfg.unk_id) {
          this.emit(best, finals);
          this.context = [...this.context.slice(1), best];
          this.trailingBlanks = 0;
          emitted = true;
        } else {
          this.trailingBlanks++;
        }
        this.checkEndpoint(finals);
      }
      if (emitted) await this.runDecoder();
    }
    this.timing.search += performance.now() - t1;
    this.timing.chunks++;
  }

  checkEndpoint(finals) {
    const silence = this.trailingBlanks * this.frameSec;
    const length = this.uttFrames * this.frameSec;
    const hasText = this.uttTokens.length > 0;
    if ((hasText && silence >= RULE2_SILENCE) || (!hasText && silence >= RULE1_SILENCE) || length >= RULE3_MAX_UTTERANCE) {
      if (hasText) {
        if (this.pendingTokens.length) finals.push(detokenize(this.syms, this.pendingTokens));
        this.pendingTokens = this.uttTokens;
      }
      // Like sherpa-onnx, keep encoder states and decoder context across endpoints.
      this.uttTokens = [];
      this.uttFrames = 0;
      this.trailingBlanks = 0;
    }
  }
}
