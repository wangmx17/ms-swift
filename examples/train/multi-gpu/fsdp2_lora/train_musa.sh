#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, retaining the CUDA example's FSDP2 JSON.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/../../../.."
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1}"
export TRAINING_MUSA_ADAPTOR_AUTOLOAD=0
export TRAINING_MUSA_ADAPTOR_ENABLED=0

nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file "./examples/train/multi-gpu/fsdp2_lora/fsdp2.json" \
    swift/cli/sft.py \
    --model "${MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --tuner_type lora \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_checkpointing false \
    --weight_decay 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "${OUTPUT_DIR:-output/fsdp2_musa}" \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
