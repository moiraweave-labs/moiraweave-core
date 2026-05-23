#!/usr/bin/env bash
# =============================================================================
# MoiraWeave AI Workload Platform — local control-plane demo
#
# Prerequisites:
#   • Docker Compose stack running:  docker compose up -d
#   • jq installed:                  brew install jq  /  apt install jq
#
# Usage:
#   bash scripts/demo.sh [--base-url http://localhost:8000]
# =============================================================================

set -euo pipefail

BASE_URL="http://localhost:8000"
USERNAME="admin"
PASSWORD="demo-password"
MAX_WAIT=120
POLL_INTERVAL=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

log() { printf "${BOLD}==> %s${RESET}\n" "$*"; }
ok() { printf "${GREEN}    ✓ %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}    ⚠ %s${RESET}\n" "$*"; }
die() { printf "${RED}    ✗ %s${RESET}\n" "$*" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not found."; }

require_cmd curl
require_cmd jq

log "Checking liveness — GET $BASE_URL/health"
HEALTH=$(curl -sf "$BASE_URL/health") || die "Service not reachable. Is the stack running?"
[[ "$(echo "$HEALTH" | jq -r '.status')" == "ok" ]] || die "Unexpected health response: $HEALTH"
ok "API process is alive"

log "Checking readiness — GET $BASE_URL/ready"
READY=$(curl -sf "$BASE_URL/ready") || die "Readiness endpoint unreachable"
READY_STATUS=$(echo "$READY" | jq -r '.status')
[[ "$READY_STATUS" == "ready" ]] || warn "Readiness: $READY_STATUS — continuing anyway"
REDIS_STATUS=$(echo "$READY" | jq -r '.checks.redis.status')
POSTGRES_STATUS=$(echo "$READY" | jq -r '.checks.postgres.status')
QDRANT_STATUS=$(echo "$READY" | jq -r '.checks.qdrant.status')
ok "Redis: $REDIS_STATUS | Postgres: $POSTGRES_STATUS | Qdrant: $QDRANT_STATUS"

log "Authenticating — POST $BASE_URL/auth/token"
AUTH_RESP=$(curl -sf -X POST "$BASE_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}") || die "Login failed"
TOKEN=$(echo "$AUTH_RESP" | jq -r '.access_token')
[[ -n "$TOKEN" && "$TOKEN" != "null" ]] || die "No token in response"
ok "Token acquired (${#TOKEN} chars)"

log "Registering mock workload"
WORKLOAD_RESP=$(curl -sf -X POST "$BASE_URL/v1/workloads" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "apiVersion": "moiraweave.io/v1alpha1",
    "kind": "Workload",
    "metadata": {"name": "demo-echo"},
    "spec": {
      "type": "model-service",
      "image": "ghcr.io/example/demo-echo:latest",
      "execution": {"mode": "async", "timeoutSeconds": 120}
    }
  }') || die "Failed to register workload"
ok "Workload registered: $(echo "$WORKLOAD_RESP" | jq -r '.name')"

log "Submitting workload run"
RUN_RESP=$(curl -sf -X POST "$BASE_URL/v1/workloads/demo-echo/runs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"prompt": "hello moiraweave", "value": 42}}') || die "Run submit failed"
RUN_ID=$(echo "$RUN_RESP" | jq -r '.run_id')
[[ -n "$RUN_ID" && "$RUN_ID" != "null" ]] || die "No run_id in response"
ok "Run queued: $RUN_ID"

log "Watching run completion"
ELAPSED=0
RUN_STATUS="queued"
while [[ "$RUN_STATUS" == "queued" || "$RUN_STATUS" == "starting" || "$RUN_STATUS" == "running" || "$RUN_STATUS" == "cancel_requested" || "$RUN_STATUS" == "cancelling" ]]; do
  sleep "$POLL_INTERVAL"
  ELAPSED=$((ELAPSED + POLL_INTERVAL))
  RUN_DATA=$(curl -sf "$BASE_URL/v1/runs/$RUN_ID" \
    -H "Authorization: Bearer $TOKEN") || die "Failed to poll run status"
  RUN_STATUS=$(echo "$RUN_DATA" | jq -r '.status')
  printf "    [%3ds] status: %s\n" "$ELAPSED" "$RUN_STATUS"
  if [[ "$ELAPSED" -ge "$MAX_WAIT" ]]; then
    die "Run did not complete within ${MAX_WAIT}s. Last status: $RUN_STATUS"
  fi
done

[[ "$RUN_STATUS" == "succeeded" ]] || die "Run ended with status: $RUN_STATUS"
ok "Run completed"
echo "$RUN_DATA" | jq '.result'

printf "\n${GREEN}${BOLD}Demo completed successfully!${RESET}\n"
