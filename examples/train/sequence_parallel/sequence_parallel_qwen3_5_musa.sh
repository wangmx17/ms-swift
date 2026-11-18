#!/usr/bin/env bash
set -euo pipefail
export TRAINING_MUSA_ADAPTOR_AUTOLOAD=0
export TORCHADA_FORCE_CUDA_AVAILABLE=1
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=8 \
CELOSS_PARALLEL_SIZE=2048 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --save_steps 50 \
    --max_length 65535 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --logging_steps 1 \
    --use_logits_to_keep false \
    --sequence_parallel_size 4 \
    --padding_free true "$@"
