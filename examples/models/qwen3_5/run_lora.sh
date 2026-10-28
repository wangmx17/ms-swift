#!/usr/bin/env bash
set -euo pipefail
# Model: Qwen/Qwen3.5-4B; MUSA lora CPT -> SFT -> DPO, 100 steps per stage.
usage() {
    printf '用法：bash %s [all|cpt|sft|dpo] [输入模型目录]\n' "$(basename "$0")"
}
if (( $# > 2 )); then usage >&2; exit 2; fi
STAGE="${1:-all}"
case "$STAGE" in
    -h|--help) usage; exit 0 ;;
    all|cpt|sft|dpo) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ "$STAGE" == dpo && -z "${2:-}" ]]; then
    printf '单独运行 dpo 时，请在第二个参数指定合并后的 SFT 模型目录。\n' >&2
    exit 2
fi
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a QWEN35_DEVICE_LIST <<< "$MUSA_VISIBLE_DEVICES"
if (( ${#QWEN35_DEVICE_LIST[@]} != 1 )); then
    printf '此脚本需要 1 个 GPU，当前 MUSA_VISIBLE_DEVICES=%s\n' "$MUSA_VISIBLE_DEVICES" >&2
    exit 2
fi
# Require the editable installation of this checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
python - "$REPO_ROOT" <<'PY_SOURCE'
import importlib.metadata as metadata
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec('swift')
if spec is None:
    raise SystemExit('Install this checkout first: pip install --no-deps -e .')
source = pathlib.Path(spec.origin).resolve()
expected = (pathlib.Path(sys.argv[1]) / 'swift' / '__init__.py').resolve()
if source != expected:
    raise SystemExit(f'Wrong swift source: {source}; install this checkout with pip install --no-deps -e .')
print(f'ms-swift {metadata.version("ms-swift")}: {source}', flush=True)
PY_SOURCE
MODEL="${2:-${QWEN35_MODEL:-/data/wangmx/models/Qwen3.5-4B}}"
RUN="${QWEN35_RUN_DIR:-/data/wangmx/msswift_runs/qwen35_lora_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "$RUN" ]]; then printf '输出目录已存在：%s\n' "$RUN" >&2; exit 2; fi
mkdir -p "$RUN"/{logs,checkpoints}
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/experiments.exit"' EXIT
export LC_ALL=C
export MASTER_PORT="${MASTER_PORT:-29636}"
export NPROC_PER_NODE=1 OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/wangmx/hf_cache MODELSCOPE_CACHE=/data/wangmx/modelscope_cache
export MPLCONFIGDIR="$RUN/matplotlib" TRITON_CACHE_DIR="$RUN/triton_cache"
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
cd "$REPO_ROOT"
DATA="${QWEN35_DATA_DIR:-/data/wangmx/datasets/qwen35_100steps}"
CKPT="$RUN/checkpoints"
printf '本轮训练产物：%s\n' "$RUN"

COMMON=(--torch_dtype bfloat16 --max_length 2048
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1
    --gradient_accumulation_steps 16 --gradient_checkpointing true
    --attn_impl sdpa --packing false --padding_free false --truncation_strategy right
    --freeze_vit true --freeze_aligner true
    --save_steps 100 --save_only_model true
    --logging_steps 1 --logging_nan_inf_filter false --report_to none
    --dataset_num_proc 4 --dataloader_num_workers 2 --seed 42 --data_seed 42
    --add_version false --warmup_ratio 0.05)
LORA=(--tuner_type lora --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 --target_modules all-linear)

run_cpt() {
    local model="$1"
    printf "\ncpt\n"
    swift pt --model "$model" "${LORA[@]}" \
        --dataset "$DATA/cpt_train.jsonl" --val_dataset "$DATA/cpt_val.jsonl" \
        --output_dir "$CKPT/cpt" --max_steps 100 --eval_steps 50 --save_total_limit 1 "${COMMON[@]}" 2>&1 | tee "$RUN/logs/cpt.log"

    printf "\ncpt_merge\n"
    swift export --model "$model" --adapters "$CKPT/cpt/checkpoint-100" \
        --merge_lora true --output_dir "$CKPT/cpt_merged" 2>&1 | tee "$RUN/logs/cpt_merge.log"
}

run_sft() {
    local model="$1"
    printf "\nsft\n"
    swift sft --model "$model" "${LORA[@]}" \
        --dataset "$DATA/sft_train.jsonl" --val_dataset "$DATA/sft_val.jsonl" \
        --add_non_thinking_prefix true --loss_scale ignore_empty_think \
        --output_dir "$CKPT/sft" --max_steps 100 --eval_steps 50 --save_total_limit 1 "${COMMON[@]}" 2>&1 | tee "$RUN/logs/sft.log"

    printf "\nsft_merge\n"
    swift export --model "$model" --adapters "$CKPT/sft/checkpoint-100" \
        --merge_lora true --output_dir "$CKPT/sft_merged" 2>&1 | tee "$RUN/logs/sft_merge.log"
}

run_dpo() {
    local model="$1"
    printf "\ndpo\n"
    swift rlhf --rlhf_type dpo --rpo_alpha 0.1 --model "$model" "${LORA[@]}" \
        --dataset "$DATA/dpo_train.jsonl" --val_dataset "$DATA/dpo_val.jsonl" \
        --add_non_thinking_prefix true \
        --output_dir "$CKPT/dpo" --max_steps 100 --eval_steps 10 --eval_on_start true --save_total_limit 2 "${COMMON[@]}" 2>&1 | tee "$RUN/logs/dpo.log"

    printf "\ndpo_merge\n"
    swift export --model "$model" --adapters "$CKPT/dpo/checkpoint-100" \
        --merge_lora true --output_dir "$CKPT/dpo_merged" 2>&1 | tee "$RUN/logs/dpo_merge.log"
}

case "$STAGE" in
    all)
        run_cpt "$MODEL"
        run_sft "$CKPT/cpt_merged"
        run_dpo "$CKPT/sft_merged"
        ;;
    cpt) run_cpt "$MODEL" ;;
    sft) run_sft "$MODEL" ;;
    dpo) run_dpo "$MODEL" ;;
esac
printf '\n%s 已完成：%s\n' "$STAGE" "$RUN"
