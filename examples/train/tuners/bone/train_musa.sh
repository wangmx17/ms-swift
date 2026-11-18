#!/usr/bin/env bash
set -euo pipefail
export TRAINING_MUSA_ADAPTOR_AUTOLOAD=0
export TORCHADA_FORCE_CUDA_AVAILABLE=1
python -c "import peft; assert peft.__version__ == '0.20.0', peft.__version__"
MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0} \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type bone \
    --dataset 'swift/self-cognition#1000' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author swift \
    --model_name swift-robot "$@"
