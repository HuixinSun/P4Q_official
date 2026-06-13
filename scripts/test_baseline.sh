#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
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
    --train_batch 128 \
    --test_batch 128 \
    --val_quant \
    --num-workers 0 \
    --devices 1
