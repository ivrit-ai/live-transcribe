#!/bin/sh
set -eu

npm install --omit=dev --no-audit --no-fund

# Schema is managed by node-pg-migrate. Run the migration at build time, before
# the server launches; DATABASE_URL is set and the DB is reachable here. The
# migration is idempotent, so it cleanly adopts an already-deployed database.
npm run migrate:up
