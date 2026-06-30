#!/bin/sh
set -eu

npm install --omit=dev --no-audit --no-fund

# Set up the schema at build time, before the server launches; DATABASE_URL is
# set and the DB is reachable here. migrate.js is idempotent, so it cleanly
# adopts an already-deployed database even across concurrent build runs.
node migrate.js
