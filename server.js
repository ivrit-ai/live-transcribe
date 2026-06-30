'use strict';

const path = require('path');
const express = require('express');

const config = require('./config');
const db = require('./db');

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// ---------------------------------------------------------------------------
// Rate limiting: in-memory token buckets keyed by (scope). Both the per-IP and
// per-anon bucket must have a token; the tighter one wins.
// ---------------------------------------------------------------------------
const buckets = new Map(); // key -> { tokens, last }

function peek(key, ratePerMin) {
  if (ratePerMin <= 0) return { bucket: null, tokens: 0, retryAfter: 60 };
  const now = Date.now();
  const refillPerMs = ratePerMin / 60000;
  let b = buckets.get(key);
  if (!b) {
    b = { tokens: ratePerMin, last: now };
    buckets.set(key, b);
  }
  b.tokens = Math.min(ratePerMin, b.tokens + (now - b.last) * refillPerMs);
  b.last = now;
  const retryAfter = b.tokens >= 1 ? 0 : Math.ceil((1 - b.tokens) / refillPerMs / 1000);
  return { bucket: b, tokens: b.tokens, retryAfter };
}

function rateCheck(ip, anon) {
  const a = peek(`ip:${ip}`, config.rate.ipPerMin);
  const b = peek(`a:${anon}`, config.rate.anonPerMin);
  if (a.tokens >= 1 && b.tokens >= 1) {
    a.bucket.tokens -= 1;
    b.bucket.tokens -= 1;
    return { ok: true };
  }
  const retryAfter = Math.max(
    a.tokens >= 1 ? 0 : a.retryAfter,
    b.tokens >= 1 ? 0 : b.retryAfter,
    1
  );
  return { ok: false, retryAfter };
}

function sweepBuckets() {
  const cutoff = Date.now() - 5 * 60 * 1000;
  for (const [key, b] of buckets) {
    if (b.last < cutoff) buckets.delete(key);
  }
}

// ---------------------------------------------------------------------------
// Usage accumulation: in-memory, flushed to rollup tables periodically so the
// request hot path never blocks on the DB.
// ---------------------------------------------------------------------------
let accumBuckets = new Map(); // minuteISO -> seconds
let accumUsers = new Map();   // anonId -> { seconds, sessions, lastSeen }
let accumTotal = { seconds: 0, sessions: 0 };

function recordUsage(anonId, seconds) {
  const minute = new Date(Math.floor(Date.now() / 60000) * 60000).toISOString();
  accumBuckets.set(minute, (accumBuckets.get(minute) || 0) + seconds);
  const now = new Date().toISOString();
  const u = accumUsers.get(anonId);
  if (u) {
    u.seconds += seconds;
    u.sessions += 1;
    u.lastSeen = now;
  } else {
    accumUsers.set(anonId, { seconds, sessions: 1, lastSeen: now });
  }
  accumTotal.seconds += seconds;
  accumTotal.sessions += 1;
}

function mergeBack(bucketDeltas, userDeltas, totalDelta) {
  for (const d of bucketDeltas) {
    accumBuckets.set(d.bucket, (accumBuckets.get(d.bucket) || 0) + d.seconds);
  }
  for (const d of userDeltas) {
    const u = accumUsers.get(d.anonId);
    if (u) {
      u.seconds += d.seconds;
      u.sessions += d.sessions;
      if (d.lastSeen > u.lastSeen) u.lastSeen = d.lastSeen;
    } else {
      accumUsers.set(d.anonId, { seconds: d.seconds, sessions: d.sessions, lastSeen: d.lastSeen });
    }
  }
  accumTotal.seconds += totalDelta.seconds;
  accumTotal.sessions += totalDelta.sessions;
}

async function flushUsage() {
  if (accumBuckets.size === 0 && accumUsers.size === 0) return;
  const bucketDeltas = [];
  for (const [bucket, seconds] of accumBuckets) bucketDeltas.push({ bucket, seconds });
  const userDeltas = [];
  for (const [anonId, v] of accumUsers) {
    userDeltas.push({ anonId, seconds: v.seconds, sessions: v.sessions, lastSeen: v.lastSeen });
  }
  const totalDelta = accumTotal;
  accumBuckets = new Map();
  accumUsers = new Map();
  accumTotal = { seconds: 0, sessions: 0 };
  try {
    await db.flushUsage(bucketDeltas, userDeltas, totalDelta);
  } catch (err) {
    console.error('usage flush failed; will retry next cycle', err);
    mergeBack(bucketDeltas, userDeltas, totalDelta);
  }
}

// ---------------------------------------------------------------------------
// Stats read + short-TTL cache.
// ---------------------------------------------------------------------------
let statsCache = { at: 0, data: null };

async function getStats() {
  const now = Date.now();
  if (statsCache.data && now - statsCache.at < config.statsCacheTtlSeconds * 1000) {
    return statsCache.data;
  }
  const data = await db.readStats(config.windowDays);
  statsCache = { at: now, data };
  return data;
}

// ---------------------------------------------------------------------------
// HTTP app
// ---------------------------------------------------------------------------
const app = express();
app.set('trust proxy', true);
app.disable('x-powered-by');
app.use(express.json({ limit: '4kb' }));

app.get('/healthz', (req, res) => res.json({ ok: true }));

app.post('/api/usage', (req, res) => {
  const body = req.body || {};
  const anonId = String(body.anonId || '').toLowerCase();
  if (!UUID_RE.test(anonId)) {
    return res.status(400).json({ error: 'invalid anonId (expected RFC-4122 uuid)' });
  }
  let seconds = Number(body.seconds);
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return res.status(400).json({ error: 'seconds must be a positive number' });
  }
  seconds = Math.min(Math.round(seconds), config.maxSessionSeconds);

  const rc = rateCheck(req.ip, anonId);
  if (!rc.ok) {
    res.set('Retry-After', String(rc.retryAfter));
    return res.status(429).json({ error: 'rate limit exceeded', retryAfter: rc.retryAfter });
  }

  recordUsage(anonId, seconds);
  res.json({ ok: true });
});

app.get('/api/stats', async (req, res, next) => {
  try {
    res.json(await getStats());
  } catch (err) {
    next(err);
  }
});

app.use(express.static(path.join(__dirname, 'public')));

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (err && (err.type === 'entity.too.large' || err.status === 413)) {
    return res.status(413).json({ error: 'body too large' });
  }
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'invalid json' });
  }
  console.error('unhandled error', err);
  res.status(500).json({ error: 'internal error' });
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function main() {
  await db.init();

  const runSweep = () =>
    db.sweep(config.bucketRetentionDays, config.windowDays)
      .catch((e) => console.error('sweep failed', e));

  setInterval(() => { flushUsage(); }, config.statsFlushSeconds * 1000);
  setInterval(() => { sweepBuckets(); }, 60 * 1000);
  setInterval(runSweep, 60 * 60 * 1000);
  runSweep();

  const server = app.listen(config.port, '0.0.0.0', () => {
    console.log(`live-transcribe listening on :${config.port}`);
  });

  const shutdown = async () => {
    try { await flushUsage(); } catch (_) { /* best effort */ }
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 5000).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

main().catch((err) => {
  console.error('fatal: startup failed', err);
  process.exit(1);
});
