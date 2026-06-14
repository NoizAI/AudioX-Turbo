#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if type module >/dev/null 2>&1; then
  module load cuda/12.1 >/dev/null 2>&1 || module load cuda12.1 >/dev/null 2>&1 || true
fi

export PYTHONPATH="${REPO_ROOT}/.deps:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

# If the CLIP/T5 cache is packaged locally, run offline against it; otherwise let
# transformers download openai/clip-vit-base-patch32 and t5-base from the Hub
# (cached under HF_HOME). Override HF_HUB_OFFLINE / AUDIOX_TURBO_CLIP_MODEL_PATH to force either mode.
CLIP_SNAPSHOT="${HUGGINGFACE_HUB_CACHE}/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
if [[ -d "${CLIP_SNAPSHOT}" ]]; then
  export AUDIOX_TURBO_CLIP_MODEL_PATH="${AUDIOX_TURBO_CLIP_MODEL_PATH:-${CLIP_SNAPSHOT}}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
fi

# Single-node launcher for an 8-GPU host. Keep only GPU 0 and 1 visible by default.
SINGLE_NODE_TOTAL_GPUS="${SINGLE_NODE_TOTAL_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if (( NUM_GPUS > ${#VISIBLE_GPU_LIST[@]} )); then
  echo "[AudioX-Turbo] ERROR: NUM_GPUS=${NUM_GPUS} exceeds visible GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-${NCCL_ASYNC_ERROR_HANDLING}}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-${NCCL_BLOCKING_WAIT}}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/audiox_turbo_distill_4step.json}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/audiox_turbo_dataset.json}"
PRETRANSFORM_CKPT="${PRETRANSFORM_CKPT:-checkpoints/pretransform/vae.ckpt}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/pretrained_ckpt/pretrained_ckpt.ckpt}"

RUN_TAG="${RUN_TAG:-balanced_4step_gan}"
RUN_NAME="${RUN_NAME:-audiox_turbo_dmd_gan_4step}"
SAVE_DIR="${SAVE_DIR:-saved_ckpt/audiox_turbo_distill_4step/${RUN_TAG}}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
PRECISION="${PRECISION:-16-mixed}"
STRATEGY="${STRATEGY:-deepspeed}"

if [[ "${STRATEGY}" == deepspeed* ]]; then
  if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  fi
  if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    echo "[AudioX-Turbo] ERROR: STRATEGY=${STRATEGY} requires a CUDA toolkit with CUDA_HOME/bin/nvcc." >&2
    echo "[AudioX-Turbo] Set CUDA_HOME to a full CUDA installation, or load cuda/12.1 before running this script." >&2
    exit 2
  fi
fi

export TIMESTEP_SAMPLING_PROBS="${AUDIOX_TURBO_TIMESTEP_PROBS:-${TIMESTEP_SAMPLING_PROBS:-0.25,0.25,0.25,0.25}}"
export GAN_DISC_HEAD_MODE="${AUDIOX_TURBO_GAN_DISC_HEAD_MODE:-${GAN_DISC_HEAD_MODE:-all_blocks}}"
export GAN_BACKBONE_NUM_BLOCKS="${AUDIOX_TURBO_GAN_BACKBONE_NUM_BLOCKS:-${GAN_BACKBONE_NUM_BLOCKS:-6}}"
export GAN_BACKBONE_TRAINABLE="${AUDIOX_TURBO_GAN_BACKBONE_TRAINABLE:-${GAN_BACKBONE_TRAINABLE:-false}}"
export GAN_BACKBONE_LORA_R="${AUDIOX_TURBO_GAN_BACKBONE_LORA_R:-${GAN_BACKBONE_LORA_R:-16}}"
export GAN_BACKBONE_LORA_ALPHA="${AUDIOX_TURBO_GAN_BACKBONE_LORA_ALPHA:-${GAN_BACKBONE_LORA_ALPHA:-32}}"

echo "[AudioX-Turbo] repo: ${REPO_ROOT}"
echo "[AudioX-Turbo] single-node host-gpus=${SINGLE_NODE_TOTAL_GPUS} visible-gpus=${CUDA_VISIBLE_DEVICES} nproc=${NUM_GPUS}"
echo "[AudioX-Turbo] model-config: ${MODEL_CONFIG}"
echo "[AudioX-Turbo] dataset-config: ${DATASET_CONFIG}"
echo "[AudioX-Turbo] pretrained-ckpt: ${PRETRAINED_CKPT}"
echo "[AudioX-Turbo] pretransform-ckpt: ${PRETRANSFORM_CKPT}"
echo "[AudioX-Turbo] save-dir: ${SAVE_DIR}"
echo "[AudioX-Turbo] hf-cache: ${HUGGINGFACE_HUB_CACHE}"
echo "[AudioX-Turbo] timestep-probs: ${TIMESTEP_SAMPLING_PROBS}"
echo "[AudioX-Turbo] gan-head=${GAN_DISC_HEAD_MODE} gan-blocks=${GAN_BACKBONE_NUM_BLOCKS} gan-trainable=${GAN_BACKBONE_TRAINABLE}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_audiox_turbo.py \
    --dataset-config "${DATASET_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --name "${RUN_NAME}" \
    --pretransform-ckpt-path "${PRETRANSFORM_CKPT}" \
    --pretrained-ckpt-path "${PRETRAINED_CKPT}" \
    --num-nodes 1 \
    --num-gpus "${NUM_GPUS}" \
    --batch-size "${BATCH_SIZE}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --save-dir "${SAVE_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --strategy "${STRATEGY}"
