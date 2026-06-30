'use strict';

// Idempotent schema setup, run from install.sh at build time. We use plain
// CREATE TABLE IF NOT EXISTS instead of a migration framework: there's a single
// schema and it must adopt an already-provisioned database cleanly, even if the
// build step runs more than once concurrently. Ignorable "already exists" races
// (codes below) are swallowed so a concurrent run can't fail the deploy.
const { Client } = require('pg');

const DDL = `
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

  CREATE TABLE IF NOT EXISTS totals (
    id             int    PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    total_seconds  bigint NOT NULL DEFAULT 0,
    total_sessions bigint NOT NULL DEFAULT 0
  );
  INSERT INTO totals (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
`;

// 42P07 duplicate_table, 42710 duplicate_object, 23505 unique_violation:
// all mean another run already created the object — safe to ignore.
const IGNORABLE = new Set(['42P07', '42710', '23505']);

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  try {
    await client.query(DDL);
    console.log('schema ready');
  } catch (err) {
    if (IGNORABLE.has(err.code)) {
      console.log(`schema already present (${err.code}); continuing`);
    } else {
      throw err;
    }
  } finally {
    await client.end();
  }
}

main().catch((err) => {
  console.error('migrate failed', err);
  process.exit(1);
});
