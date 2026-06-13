#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/p4q}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"

python "${ROOT}/main.py" \
    --model ViT-B/32 \
    --db_name cifar100 \
    --root "${DATA_ROOT}" \
    --quant \
    --bit_type 4 \
    --calib-iter 10 \
    --zeroshot_prompt \
    --prompt4quant \
    --train_batch 128 \
    --test_batch 128 \
    --coop_epochs 50 \
    --adapter_epochs 50 \
    --load_dir "${CKPT_DIR}" \
    --resume_checkpoint 50 \
    --val_quant \
    --num-workers 0 \
    --devices 1
