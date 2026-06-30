'use strict';

const { Pool } = require('pg');

// DATABASE_URL is injected by xhost; search_path is pinned to the channel schema.
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 8,
  idleTimeoutMillis: 30000,
});

// Schema is owned by node-pg-migrate (run at build time from install.sh).
// At boot we only confirm the DB is reachable.
async function init() {
  await pool.query('SELECT 1');
}

// Flush accumulated usage in three batched upserts. `bucketDeltas` is
// [{ bucket: ISO-minute, seconds }], `userDeltas` is
// [{ anonId, seconds, sessions, lastSeen }], and `totalDelta` is { seconds, sessions }.
async function flushUsage(bucketDeltas, userDeltas, totalDelta) {
  if (bucketDeltas.length) {
    await pool.query(
      `INSERT INTO usage_buckets (bucket, seconds)
         SELECT * FROM unnest($1::timestamptz[], $2::bigint[])
       ON CONFLICT (bucket) DO UPDATE
         SET seconds = usage_buckets.seconds + EXCLUDED.seconds`,
      [bucketDeltas.map((d) => d.bucket), bucketDeltas.map((d) => d.seconds)]
    );
  }
  if (userDeltas.length) {
    await pool.query(
      `INSERT INTO user_activity (anon_id, total_seconds, sessions, last_seen)
         SELECT * FROM unnest($1::uuid[], $2::bigint[], $3::bigint[], $4::timestamptz[])
       ON CONFLICT (anon_id) DO UPDATE
         SET total_seconds = user_activity.total_seconds + EXCLUDED.total_seconds,
             sessions      = user_activity.sessions + EXCLUDED.sessions,
             last_seen     = GREATEST(user_activity.last_seen, EXCLUDED.last_seen)`,
      [
        userDeltas.map((d) => d.anonId),
        userDeltas.map((d) => d.seconds),
        userDeltas.map((d) => d.sessions),
        userDeltas.map((d) => d.lastSeen),
      ]
    );
  }
  if (totalDelta && (totalDelta.seconds || totalDelta.sessions)) {
    await pool.query(
      `UPDATE totals
         SET total_seconds = total_seconds + $1,
             total_sessions = total_sessions + $2
       WHERE id = 1`,
      [totalDelta.seconds, totalDelta.sessions]
    );
  }
}

async function readStats(windowDays) {
  const [users, week, ever] = await Promise.all([
    pool.query(
      `SELECT count(*)::bigint AS n FROM user_activity
         WHERE last_seen > now() - make_interval(days => $1)`,
      [windowDays]
    ),
    pool.query(
      `SELECT COALESCE(sum(seconds), 0)::bigint AS n FROM usage_buckets
         WHERE bucket > now() - make_interval(days => $1)`,
      [windowDays]
    ),
    pool.query('SELECT total_seconds FROM totals WHERE id = 1'),
  ]);
  return {
    uniqueUsers7d: Number(users.rows[0].n),
    seconds7d: Number(week.rows[0].n),
    secondsEver: Number(ever.rows[0] ? ever.rows[0].total_seconds : 0),
  };
}

// Trim per-minute buckets past their retention and drop user rows that have
// aged out of the rolling window. Totals are never touched.
async function sweep(bucketRetentionDays, windowDays) {
  await pool.query(
    'DELETE FROM usage_buckets WHERE bucket < now() - make_interval(days => $1)',
    [bucketRetentionDays]
  );
  await pool.query(
    'DELETE FROM user_activity WHERE last_seen < now() - make_interval(days => $1)',
    [windowDays]
  );
}

module.exports = { pool, init, flushUsage, readStats, sweep };
