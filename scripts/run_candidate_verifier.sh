#!/bin/bash
set -euo pipefail

export BASE_URL="${BASE_URL:-https://tiantianapi.com/v1}"
export API_KEY="${API_KEY:?API_KEY must be set}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/data/banlujie/ckpt/qwen3-0.6b-rm}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/results/candidate_verifier"

cd "$REPO"
python3 eval_candidate_verifier.py \
    --selectors majority_vote random self_rerank clause_prm \
    "$@" 2>&1 | tee results/candidate_verifier/run.log
