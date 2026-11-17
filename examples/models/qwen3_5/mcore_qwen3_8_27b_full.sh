#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="/home/Megatron-LM:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29661}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-bond0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export MCCL_IB_DISABLE="${MCCL_IB_DISABLE:-0}"
export MCCL_NET_GDR_LEVEL="${MCCL_NET_GDR_LEVEL:-2}"
export MCCL_IB_TIMEOUT="${MCCL_IB_TIMEOUT:-18}"
export MCCL_IB_RETRY_CNT="${MCCL_IB_RETRY_CNT:-7}"
export MCCL_DEBUG="${MCCL_DEBUG:-WARN}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

MODEL="${MODEL:-/mnt_moer/models/Qwen3.8-27B}"
DATASET="${DATASET:-/mnt_moer/wangmx/llava_instruct_150k_preprocessed/llava_instruct_150k_swift.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt_moer/wangmx/msswift_runs/qwen3_8_27b_sft_full_$(date +%Y%m%d_%H%M%S)}"

TP="${TP:-4}"
PP="${PP:-1}"
CP="${CP:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"

mkdir -p "$OUTPUT_DIR"

megatron sft \
    --external_plugins "${REPO_ROOT}/examples/models/qwen3_5/qwen38_token_metrics.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --tuner_type full \
    --bridge_backend mcore-bridge \
    --bf16 true \
    --finetune true \
    --freeze_vit true \
    --freeze_aligner true \
    --max_length 8192 \
    --train_iters "$TRAIN_ITERS" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --context_parallel_size "$CP" \
    --sequence_parallel true \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --optimizer adam \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --attention_backend flash \
    --apply_rope_fusion true \
    --manual_gc true \
    --manual_gc_steps 100 \
    --split_dataset_ratio 0 \
    --logging_steps 1 \
    --save_steps "$TRAIN_ITERS" \
    --save_safetensors true \
    --no_save_optim true \
    --no_save_rng true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --output_dir "$OUTPUT_DIR"
