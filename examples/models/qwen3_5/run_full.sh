#!/usr/bin/env bash
set -euo pipefail
# Model: Qwen/Qwen3.5-4B; MUSA full CPT -> SFT -> DPO.
# Qwen3.5-4B 受控验证基线：默认各阶段 100 步，不代表收敛训练。
# 不带参数运行完整流程；指定 cpt/sft/dpo 只运行该阶段。
# 第二个参数可指定输入模型；单独 dpo 必须传入 SFT 产物。
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
    printf '单独运行 dpo 时，请在第二个参数指定 SFT 模型目录；LoRA 请使用合并后的权重目录。\n' >&2
    exit 2
fi
# 此基线固定 4 卡；可用 MUSA_VISIBLE_DEVICES 指定对应数量的卡号。
QWEN35_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a QWEN35_DEVICE_LIST <<< "$QWEN35_DEVICES"
if (( ${#QWEN35_DEVICE_LIST[@]} != 4 )); then
    printf '此脚本需要 4 个 GPU，当前 MUSA_VISIBLE_DEVICES=%s\n' "$QWEN35_DEVICES" >&2
    exit 2
fi
# Install this checkout first; a script's location does not select the installed package.
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
RUN="${QWEN35_RUN_DIR:-/data/wangmx/msswift_runs/qwen35_full_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "$RUN" ]]; then printf '输出目录已存在：%s\n' "$RUN" >&2; exit 2; fi
mkdir -p "$RUN"/{logs,checkpoints}
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/experiments.exit"' EXIT
export LC_ALL=C
export MASTER_PORT="${MASTER_PORT:-29635}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3}" NPROC_PER_NODE=4 OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/wangmx/hf_cache MODELSCOPE_CACHE=/data/wangmx/modelscope_cache
export MPLCONFIGDIR="$RUN/matplotlib" TRITON_CACHE_DIR="$RUN/triton_cache"
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
cd "$REPO_ROOT"
DATA="${QWEN35_DATA_DIR:-/data/wangmx/datasets/qwen35_100steps}"
DPO_DATA="$DATA"
DPO_STEPS="${DPO_STEPS:-100}"
DPO_LR="${DPO_LR:-1e-5}"
if [[ ! "$DPO_STEPS" =~ ^[1-9][0-9]*$ ]]; then printf 'DPO_STEPS 必须为正整数\n' >&2; exit 2; fi
CKPT="$RUN/checkpoints"
printf '本轮训练产物：%s\n' "$RUN"

COMMON=(--torch_dtype bfloat16 --max_length 2048
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1
    --gradient_accumulation_steps 4 --gradient_checkpointing true --deepspeed zero3
    --attn_impl sdpa --packing false --padding_free false --truncation_strategy right
    --freeze_vit true --freeze_aligner true
    --save_steps 100 --save_only_model true
    --logging_steps 1 --logging_nan_inf_filter false --report_to none
    --dataset_num_proc 4 --dataloader_num_workers 2 --seed 42 --data_seed 42
    --add_version false --warmup_ratio 0.05)

run_cpt() {
    local model="$1"
    printf "\ncpt\n"
    swift pt --model "$model" --tuner_type full --learning_rate 1e-5 \
        --dataset "$DATA/cpt_train.jsonl" --val_dataset "$DATA/cpt_val.jsonl" \
        --output_dir "$CKPT/cpt" --max_steps 100 --eval_steps 50 --save_total_limit 1 "${COMMON[@]}" 2>&1 | tee "$RUN/logs/cpt.log"
}

run_sft() {
    local model="$1"
    printf "\nsft\n"
    swift sft --model "$model" --tuner_type full --learning_rate 1e-5 \
        --dataset "$DATA/sft_train.jsonl" --val_dataset "$DATA/sft_val.jsonl" \
        --add_non_thinking_prefix true --loss_scale ignore_empty_think \
        --output_dir "$CKPT/sft" --max_steps 100 --eval_steps 50 --save_total_limit 1 "${COMMON[@]}" 2>&1 | tee "$RUN/logs/sft.log"
}

run_dpo() {
    local model="$1"
    local -a diagnostics=()
    if [[ -n "${QWEN35_DPO_DIAGNOSTICS:-}" ]]; then
        diagnostics=(--external_plugins "$QWEN35_DPO_DIAGNOSTICS" --callbacks dpo_zero_check)
    fi
    printf "\ndpo\n"
    swift rlhf --rlhf_type dpo --rpo_alpha 0.1 --model "$model" \
        --ref_model "$model" --tuner_type full --learning_rate "$DPO_LR" \
        --dataset "$DPO_DATA/dpo_train.jsonl" --val_dataset "$DPO_DATA/dpo_val.jsonl" \
        --add_non_thinking_prefix true \
        --output_dir "$CKPT/dpo" --max_steps "$DPO_STEPS" --eval_steps 10 --eval_on_start true --save_total_limit 2 "${diagnostics[@]}" "${COMMON[@]}" 2>&1 | tee "$RUN/logs/dpo.log"
}

case "$STAGE" in
    all)
        run_cpt "$MODEL"
        run_sft "$CKPT/cpt/checkpoint-100"
        run_dpo "$CKPT/sft/checkpoint-100"
        ;;
    cpt) run_cpt "$MODEL" ;;
    sft) run_sft "$MODEL" ;;
    dpo) run_dpo "$MODEL" ;;
esac
printf '\n%s 已完成：%s\n' "$STAGE" "$RUN"
