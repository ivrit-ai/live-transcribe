'use strict';

function intEnv(name, def) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return def;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) {
    throw new Error(`Invalid value for ${name}: ${raw}`);
  }
  return n;
}

const config = {
  port: intEnv('PORT', 3000),

  // Largest single session we'll credit (seconds). Guards against a client
  // reporting an absurd duration; generously above the 2-min default max.
  maxSessionSeconds: intEnv('MAX_SESSION_SECONDS', 6 * 60 * 60),

  // Window for the "unique users" and "transcribed" rolling metrics.
  windowDays: intEnv('STATS_WINDOW_DAYS', 7),

  // Per-minute rate limits for the single write endpoint, by scope.
  rate: {
    ipPerMin: intEnv('RL_IP_PER_MIN', 30),
    anonPerMin: intEnv('RL_ANON_PER_MIN', 20),
  },

  // Keep per-minute buckets a little longer than the window so the rolling
  // sum is always complete; user_activity rows expire at the window edge.
  bucketRetentionDays: intEnv('BUCKET_RETENTION_DAYS', 8),
  statsFlushSeconds: intEnv('STATS_FLUSH_SECONDS', 5),
  statsCacheTtlSeconds: intEnv('STATS_CACHE_TTL_SECONDS', 5),
};

module.exports = config;
