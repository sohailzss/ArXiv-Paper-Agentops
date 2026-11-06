#!/usr/bin/env bash
# Runs ON the EC2 server, called by the deploy job in .github/workflows/ci-cd.yml after the code and a
# freshly rendered .env have been synced to ~/arxiv-rag.
#
# Reads the UI basic-auth password from stdin (so it never appears in a process list or log), then:
#   1. installs the new .env and nginx password file
#   2. rebuilds and restarts the stack (Docker layer cache keeps unchanged images fast)
#   3. waits until /api/v1/health reports "ok" - fails the deploy if it doesn't
#   4. prunes dangling images so the 40 GB disk doesn't fill up over many deploys
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/arxiv-rag}"
UI_USER="${UI_USER:-arxiv}"
COMPOSE=(docker compose -f compose.yml -f compose.prod.yml)
cd "$APP_DIR"

echo "==> Installing .env"
test -s .env.incoming || { echo "missing .env.incoming"; exit 1; }
mv .env.incoming .env
chmod 600 .env

echo "==> Writing nginx password file for user '$UI_USER'"
IFS= read -r UI_PASSWORD
test -n "$UI_PASSWORD" || { echo "empty UI password on stdin"; exit 1; }
printf '%s\n' "$UI_PASSWORD" | htpasswd -icB deploy/.htpasswd "$UI_USER"
chmod 644 deploy/.htpasswd  # nginx (another user in its container) must read it; it only holds a bcrypt hash
unset UI_PASSWORD

echo "==> Building and starting containers"
"${COMPOSE[@]}" up -d --build --remove-orphans
# nginx reads the password file at startup; restart it so a changed password takes effect.
"${COMPOSE[@]}" restart nginx >/dev/null

echo "==> Waiting for the API to be healthy"
for i in $(seq 1 60); do
  body=$(curl -s -m 5 http://localhost:8000/api/v1/health || true)
  if printf '%s' "$body" | grep -q '"status":"ok"'; then
    echo "healthy after ~$((i * 5))s: $body"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "API not healthy after 5 minutes. Last response: ${body:-none}"
    "${COMPOSE[@]}" ps
    "${COMPOSE[@]}" logs --tail 80 api
    exit 1
  fi
  sleep 5
done

echo "==> Cleaning up old images"
docker image prune -f >/dev/null
"${COMPOSE[@]}" ps --format 'table {{.Name}}\t{{.Status}}'
df -h / | tail -1 | awk '{print "disk: " $3 " used, " $4 " free"}'
