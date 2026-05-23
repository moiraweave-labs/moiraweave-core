#!/usr/bin/env bash
# promote-model.sh — Promote a registered MLflow model to production and
# trigger an Argo Rollouts canary deployment.
#
# Usage:
#   ./scripts/promote-model.sh \
#       --model-name demo-model \
#       --model-version 3 \
#       --image ghcr.io/jgallego9/moiraweave/demo-model:sha-abc1234
#
# Required tools:
#   - curl (for MLflow REST API)
#   - kubectl (for Argo Rollouts)
#   - jq (for JSON parsing)
#
# Environment variables (override defaults via .env or shell export):
#   MLFLOW_TRACKING_URI  – MLflow server base URL
#   KUBECONFIG           – Path to kubeconfig (default: ~/.kube/config)
#   ROLLOUT_NAMESPACE    – Kubernetes namespace of the Rollout
#   ROLLOUT_NAME         – Argo Rollout resource name
#
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
ROLLOUT_NAMESPACE="${ROLLOUT_NAMESPACE:-moiraweave}"
ROLLOUT_NAME="${ROLLOUT_NAME:-moiraweave-model}"
CONTAINER_NAME="${CONTAINER_NAME:-workload}"

# ── Argument parsing ─────────────────────────────────────────────────────────
MODEL_NAME=""
MODEL_VERSION=""
NEW_IMAGE=""

usage() {
    cat >&2 <<EOF
Usage: $0 --model-name NAME --model-version VERSION --image IMAGE

Options:
  --model-name      Registered model name in MLflow  (required)
  --model-version   Model version number             (required)
  --image           Full container image reference   (required)
  --help            Show this message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)    MODEL_NAME="$2";    shift 2 ;;
        --model-version) MODEL_VERSION="$2"; shift 2 ;;
        --image)         NEW_IMAGE="$2";     shift 2 ;;
        --help)          usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

[[ -z "$MODEL_NAME" ]]    && { echo "ERROR: --model-name is required" >&2; usage; }
[[ -z "$MODEL_VERSION" ]] && { echo "ERROR: --model-version is required" >&2; usage; }
[[ -z "$NEW_IMAGE" ]]     && { echo "ERROR: --image is required" >&2; usage; }

# ── Dependency checks ────────────────────────────────────────────────────────
for cmd in curl jq kubectl; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "ERROR: required command '$cmd' not found in PATH" >&2
        exit 1
    }
done

# ── Step 1: Set MLflow alias 'champion' on the requested model version ────────
# MLflow 3.x deprecates lifecycle stages in favour of registered-model aliases.
# API: POST /api/2.0/mlflow/registered-models/alias
# Ref: https://mlflow.org/docs/latest/api_reference/rest-api.html#set-registered-model-alias
echo "==> [1/3] Setting MLflow alias 'champion' on '${MODEL_NAME}' v${MODEL_VERSION} …"

MLFLOW_API="${MLFLOW_TRACKING_URI}/api/2.0/mlflow"

# Confirm the version exists and is accessible.
curl --fail --silent --output /dev/null \
    "${MLFLOW_API}/model-versions/get?name=${MODEL_NAME}&version=${MODEL_VERSION}"

# Set (or update) the 'champion' alias to point at the requested version.
ALIAS_PAYLOAD=$(jq -n \
    --arg name    "$MODEL_NAME" \
    --arg alias   "champion" \
    --arg version "$MODEL_VERSION" \
    '{name: $name, alias: $alias, version: $version}')

curl --fail --silent --output /dev/null \
    --request POST \
    --header "Content-Type: application/json" \
    --data "$ALIAS_PAYLOAD" \
    "${MLFLOW_API}/registered-models/alias"

echo "    Alias 'champion' set to version ${MODEL_VERSION}."

# ── Step 2: Trigger Argo Rollouts canary ────────────────────────────────────
echo "==> [2/3] Updating Argo Rollout '${ROLLOUT_NAME}' image to '${NEW_IMAGE}' …"

kubectl argo rollouts set image \
    "${ROLLOUT_NAME}" \
    "${CONTAINER_NAME}=${NEW_IMAGE}" \
    --namespace "${ROLLOUT_NAMESPACE}"

echo "    Rollout image updated. Canary deployment started."

# ── Step 3: Print rollout status ────────────────────────────────────────────
echo "==> [3/3] Current rollout status:"
kubectl argo rollouts status \
    "${ROLLOUT_NAME}" \
    --namespace "${ROLLOUT_NAMESPACE}" \
    --timeout 10s \
    || true  # status may return non-zero while rollout is in progress

cat <<EOF

Promotion initiated.

  Model    : ${MODEL_NAME} v${MODEL_VERSION} → alias 'champion'
  Image    : ${NEW_IMAGE}
  Rollout  : ${ROLLOUT_NAMESPACE}/${ROLLOUT_NAME}

Monitor canary progress:
  kubectl argo rollouts get rollout ${ROLLOUT_NAME} -n ${ROLLOUT_NAMESPACE} --watch

Promote manually (skip remaining pause steps):
  kubectl argo rollouts promote ${ROLLOUT_NAME} -n ${ROLLOUT_NAMESPACE}

Abort (roll back to stable):
  kubectl argo rollouts abort ${ROLLOUT_NAME} -n ${ROLLOUT_NAMESPACE}
EOF
