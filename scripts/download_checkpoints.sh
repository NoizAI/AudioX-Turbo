#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-HKUSTAudio/AudioX-Turbo}"
LOCAL_DIR="${LOCAL_DIR:-checkpoints}"

mkdir -p "${LOCAL_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[AudioX-Turbo] huggingface-cli was not found." >&2
  echo "[AudioX-Turbo] Install it with: pip install -U \"huggingface_hub[cli]\"" >&2
  exit 127
fi

echo "[AudioX-Turbo] downloading inference checkpoints from ${REPO_ID}"
huggingface-cli download "${REPO_ID}" \
  audiox_turbo/audiox_turbo.ckpt \
  pretransform/vae.ckpt \
  synchformer/synchformer_state_dict.pth \
  --local-dir "${LOCAL_DIR}"

if [[ "${INCLUDE_TRAINING_CKPT:-0}" == "1" ]]; then
  echo "[AudioX-Turbo] downloading training teacher checkpoint from ${REPO_ID}"
  huggingface-cli download "${REPO_ID}" \
    pretrained_ckpt/pretrained_ckpt.ckpt \
    --local-dir "${LOCAL_DIR}"
fi

echo "[AudioX-Turbo] checkpoints are ready under ${LOCAL_DIR}"
