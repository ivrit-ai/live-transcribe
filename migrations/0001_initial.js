'use strict';

// Initial schema for anonymous usage stats: usage_buckets, user_activity, totals.
// Idempotent DDL so the migration cleanly adopts an already-deployed database.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    CREATE TABLE IF NOT EXISTS usage_buckets (
      bucket  timestamptz NOT NULL PRIMARY KEY,
      seconds bigint      NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS user_activity (
      anon_id       uuid        PRIMARY KEY,
      total_seconds bigint      NOT NULL DEFAULT 0,
      sessions      bigint      NOT NULL DEFAULT 0,
      last_seen     timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS user_activity_last_seen_idx ON user_activity (last_seen);

    -- Single monotonic row of all-time counters; never pruned so "ever" totals
    -- survive bucket/activity retention.
    CREATE TABLE IF NOT EXISTS totals (
      id             int    PRIMARY KEY DEFAULT 1 CHECK (id = 1),
      total_seconds  bigint NOT NULL DEFAULT 0,
      total_sessions bigint NOT NULL DEFAULT 0
    );
    INSERT INTO totals (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    DROP TABLE IF EXISTS totals;
    DROP TABLE IF EXISTS user_activity;
    DROP TABLE IF EXISTS usage_buckets;
  `);
};
