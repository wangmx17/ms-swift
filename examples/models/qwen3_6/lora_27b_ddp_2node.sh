#!/usr/bin/env bash
set -euo pipefail
# Run once on EACH node. One process per visible accelerator; no model sharding.
export NNODES=2
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
: "${NODE_RANK:?Set NODE_RANK to the zero-based node index}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node address}"
export NODE_RANK MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29536}"
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ || ! "$NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo 'NPROC_PER_NODE must be positive; NODE_RANK must be a nonnegative integer.' >&2
    exit 2
fi
if (( NODE_RANK >= NNODES )); then
    echo 'NODE_RANK must be smaller than NNODES.' >&2
    exit 2
fi
# All nodes must use the same model, dataset, and output path.
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
DATASET="${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh#500}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Qwen3.6-27B-lora-ddp-2node}"
exec swift sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters false \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --attn_impl sdpa \
    --max_length 12288 \
    --packing false \
    --padding_free false \
    --group_by_length true \
    --output_dir "$OUTPUT_DIR" \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    "$@"
