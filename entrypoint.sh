#!/usr/bin/env bash
set -euo pipefail

PG_HOST="${PG_HOST:-postgres}"
PG_PORT="${PG_PORT:-5432}"

echo "entrypoint: waiting for database ${PG_HOST}:${PG_PORT}…"
python - "$PG_HOST" "$PG_PORT" <<'PY'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
for _ in range(60):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"entrypoint: {host}:{port} reachable")
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"entrypoint: database {host}:{port} not reachable after 60s")
PY

echo "entrypoint: applying migrations…"
alembic upgrade head

echo "entrypoint: starting app…"
exec "$@"
