#!/usr/bin/env bash
set -euo pipefail

# Preserve the official two-device DDP recipe; run one comparison case at a time.
case_name=${1:-fused}
case "$case_name" in
    plain|plain_repeat) optim=adamw_torch ;;
    fused) optim=adamw_torch_fused ;;
    *) echo 'Usage: bash train_musa.sh [plain|fused|plain_repeat]' >&2; exit 2 ;;
esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_root=${OUTPUT_ROOT:-output/fused_adamw_musa}
output_dir="$output_root/$case_name"
if [[ -e "$output_dir" ]]; then
    echo "Output already exists: $output_dir. Choose a fresh OUTPUT_ROOT." >&2
    exit 2
fi
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=2
# Match the recorded validation environment: no adaptor patches.
export TRAINING_MUSA_ADAPTOR_AUTOLOAD=0
export TRAINING_MUSA_ADAPTOR_ENABLED=0
export FUSED_ADAMW_EXPECTED=$optim

swift sft \
    --model "${MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#1000' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$output_dir" \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --optim "$optim" \
    --external_plugins "$script_dir/audit.py" \
    --callbacks fused_adamw_audit
