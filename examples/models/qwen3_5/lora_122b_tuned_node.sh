#!/usr/bin/env bash
# Qwen3.5-122B-A10B LoRA: four S5000 nodes, packing4096/MB2/GBS256.
# GDN TileLang with validated zero-beta repair and exact saved-state recomputation.
# Native allocator fraction0.90; same full/block41, TE CE, GC and global async.
# Measured pair: 36.51218333333333 -> 34.869444666666666 seconds/step, steps5-10.
# Evidence: /bwykj_ssd/wangmx/review/gdn-tilelang-20261120/allocator-fraction-pair-20261121-131149
# Historical100-step results and caveats are recorded in the optimization stack docs.
set -euo pipefail
umask 0002

# Explicitly clear diagnostic launch blocking inherited from a shell.
unset MUSA_LAUNCH_BLOCKING CUDA_LAUNCH_BLOCKING
export GDN_ALLOCATOR_FRACTION=0.90
export MCCL_DEBUG=WARN NCCL_DEBUG=WARN TORCH_SHOW_CPP_STACKTRACES=1 TORCH_DISABLE_ADDR2LINE=1
export MUSA_HOME=/usr/local/musa
export PATH=/usr/local/musa/bin:/usr/local/openmpi/bin:/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/lib64:/usr/local/mtshmem/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MTHREADS_VISIBLE_DEVICES=all MTHREADS_DRIVER_CAPABILITIES=compute,utility
export NNODES=4 NPROC_PER_NODE=8 NODE_RANK="${1:?node rank required}"
export MASTER_ADDR=10.124.31.8 MASTER_PORT="${MASTER_PORT:-29535}"
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024 VIDEO_MAX_TOKEN_NUM=128 FPS_MAX_FRAMES=12
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"
OPT="$REPO/examples/models/qwen3_5/qwen122_optimizations"
export BWY_RUNTIME_ROOT="${BWY_RUNTIME_ROOT:-/bwykj_ssd/wangmx/qwen122-runtime}"
export PYTHONPATH="$OPT/runtime:$OPT/head${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$PWD:/home/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_DIR="${2:?shared output directory required}"
export PYTHONPATH="$BWY_RUNTIME_ROOT/extension:$PYTHONPATH"
export FLA_EXPERIMENT_MODE=reuse FLA_EXPERIMENT_ROOT="$BWY_RUNTIME_ROOT/fla-configs" FLA_CACHE_MODE=strict
export GDN_CASE=tilelang
export OPT_TRACE="${OPT_TRACE:-0}"
case "${GDN_CASE:?}" in
 baseline) export TRAINING_MUSA_ADAPTOR_DISABLE=megatron.ssm.gated-delta-rule.tilelang,mcore_bridge.ssm.gated-delta-rule.tilelang ;;
 tilelang) unset TRAINING_MUSA_ADAPTOR_DISABLE ;;
 *) exit 2 ;;
esac
extra=()
if [[ -n "${MAX_STEPS:-}" ]]; then extra+=(--train_iters "$MAX_STEPS"); else extra+=(--num_train_epochs 1); fi
exec python -m swift.cli._megatron.main sft \
    --model /bwykj_ssd/models/Qwen3.5-122B-A10B \
    --external_plugins "$OPT/partial/partial_gdn_plugin.py" "$OPT/runtime/allocator_fraction_plugin.py" "$OPT/runtime/qwen122_native_flops_plugin.py" "$OPT/runtime/trace_allocator_plugin.py" "$OPT/runtime/memory_counters_plugin.py" "$OPT/runtime/fla_autotune_cache_plugin.py" "$OPT/runtime/bucket_warmup_plugin.py" \
    --callbacks partial_gdn allocator_fraction native_flops gdn_gradstats_warmup gdn_trace memory_counters fla_autotune_cache \
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
    --recompute_method block \
    --recompute_num_layers 41 \
    --group_by_length false \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --manual_gc true \
    --manual_gc_steps 100 \
    --megatron_extra_kwargs '{"moe_router_fusion":false}' \
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
