#!/bin/sh
set -eu

# Set up the schema before the server launches; DATABASE_URL is set and the DB
# is reachable here. migrate.js is idempotent, so it cleanly adopts an
# already-deployed database.
node migrate.js

exec node server.js
