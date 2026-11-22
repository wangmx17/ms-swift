#!/usr/bin/env bash
# Called by mcore_qwen3_5_122b_4node_tuned_launch.sh in each training container.
# Measured on 2026-11-19: tp1_pp1_ep8_mb2_gbs256_confirm20, 26803.9 estimated tokens/s across 32 GPUs.
# Benchmark defaults: 20 steps, adapter-only saving; see docs/source_en/BestPractices/Qwen3_5_122B_A10B_MUSA_LoRA.md.
set -euo pipefail
umask 0002
# Explicitly clear diagnostic launch blocking inherited from a shell.
unset MUSA_LAUNCH_BLOCKING CUDA_LAUNCH_BLOCKING
export MUSA_HOME=/usr/local/musa
export PATH=/usr/local/musa/bin:/usr/local/openmpi/bin:/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/lib64:/usr/local/mtshmem/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MTHREADS_VISIBLE_DEVICES=all MTHREADS_DRIVER_CAPABILITIES=compute,utility
export NNODES=4 NPROC_PER_NODE=8 NODE_RANK="${1:?node rank required}"
export MASTER_ADDR=10.124.31.8 MASTER_PORT="${MASTER_PORT:-29535}"
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024 VIDEO_MAX_TOKEN_NUM=128 FPS_MAX_FRAMES=12
cd /bwykj_ssd/wangmx/ms-swift
export PYTHONPATH="$PWD:/home/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_DIR="${2:?shared output directory required}"
extra=()
if [[ -n "${MAX_STEPS:-}" ]]; then extra+=(--train_iters "$MAX_STEPS"); else extra+=(--num_train_epochs 1); fi
exec python -m swift.cli._megatron.main sft \
    --model /bwykj_ssd/models/Qwen3.5-122B-A10B \
    --save_safetensors true \
    --save_missing_weights true \
    --merge_lora "${MERGE_LORA:-false}" \
    --dataset /bwykj_ssd/datasets/chinatax-qa.jsonl \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --context_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap false \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 2 \
    --global_batch_size 256 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --group_by_length false \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --manual_gc true \
    --manual_gc_steps 100 \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir "$OUTPUT_DIR" \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 4096 \
    --logging_steps 1 --packing true \
    --packing_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel false \
    --nccl_comm_warmup true \
    --attention_backend flash \
    --padding_free true \
    "${extra[@]}"
