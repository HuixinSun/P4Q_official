#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/p4q}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"

run_stage() {
    local epochs="$1"
    local resume="$2"
    local extra_args=()

    if [[ "${resume}" -gt 0 ]]; then
        extra_args+=(--load_dir "${SAVE_DIR}" --resume_checkpoint "${resume}")
    fi

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
        --coop_epochs "${epochs}" \
        --adapter_epochs "${epochs}" \
        --save_path "${SAVE_DIR}" \
        --save_freq 10 \
        --val_quant \
        --num-workers 0 \
        --devices 1 \
        "${extra_args[@]}"
}

# Stage schedule matches the paper experiment on CIFAR-100 / ViT-B/32 / 4-bit.
run_stage 1 0
run_stage 5 1
run_stage 10 5
run_stage 20 10
run_stage 50 20
run_stage 100 50
