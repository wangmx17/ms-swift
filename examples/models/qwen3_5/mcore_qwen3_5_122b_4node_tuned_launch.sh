#!/usr/bin/env bash
# Historical parallel-strategy measurement on 2026-11-19: tp1_pp1_ep8_mb2_gbs256_confirm20, 26803.9 estimated tokens/s across 32 GPUs.
# Benchmark defaults: 20 steps, adapter-only saving; see docs/source_en/BestPractices/Qwen3_5_122B_A10B_MUSA_LoRA.md.
# Current optimizations: TE cross entropy + manual GC; router fusion disabled.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
[[ $(hostname -s) == worker31008 ]] || { echo 'Run on worker31008' >&2; exit 2; }
if [[ ! -e /.dockerenv ]]; then
    exec docker exec -i -e BWY_RUNTIME_ROOT="${BWY_RUNTIME_ROOT:-/bwykj_ssd/wangmx/qwen122-runtime}" -e MAX_STEPS="${MAX_STEPS-20}" -e MERGE_LORA="${MERGE_LORA:-false}" -e MASTER_PORT="${MASTER_PORT:-29535}" qwen35_test bash "$repo/examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh" "$@"
fi
nodes=(worker31008 worker31013 worker31029 worker31030)
script="$repo/examples/models/qwen3_5/lora_122b_tuned_node.sh"
for node in "${nodes[@]:1}"; do
    ssh -n -o BatchMode=yes "$node" "test -r '$script' && test -r /bwykj_ssd/datasets/chinatax-qa.jsonl" || exit 1
done
[[ ${1:-} != --check ]] || { echo 'Four-node launcher preflight OK'; exit 0; }
steps=${MAX_STEPS-20}
port=${MASTER_PORT:-29535}
merge=${MERGE_LORA:-false}
[[ $merge == true || $merge == false ]] || exit 2
[[ -z $steps || $steps =~ ^[1-9][0-9]*$ ]] && [[ $port =~ ^[0-9]+$ ]] || exit 2
outbase=/bwykj_ssd/wangmx/output
# Remote ranks use the shared storage owner's UID.
run=$(setpriv --reuid=1000 --regid=1001 --clear-groups bash -ec 'mkdir -p "$1"; run="$1/qwen122-lora-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"; mkdir "$run"; printf "%s\n" "$run"' _ "$outbase")
chgrp 1000 "$run"
chmod 2775 "$run"
umask 0002
mkdir -p "$run/logs"
cleanup_failed_run() {
    status=$?
    if ((status != 0)); then
        python "$repo/examples/models/qwen3_5/stop_lora_run.py" "$run"
        for node in "${nodes[@]:1}"; do
            ssh -n -o ConnectTimeout=10 "$node" "/usr/bin/python '$repo/examples/models/qwen3_5/stop_lora_run.py' '$run'" || true
        done
    fi
}
trap cleanup_failed_run EXIT
echo "Training logs: $run/logs"
for rank in 0 1 2 3; do
    if ((rank==0)); then
        BWY_RUNTIME_ROOT="${BWY_RUNTIME_ROOT:-/bwykj_ssd/wangmx/qwen122-runtime}" MERGE_LORA="$merge" MAX_STEPS="$steps" MASTER_PORT="$port" bash "$script" "$rank" "$run" >"$run/logs/rank0.log" 2>&1 &
    else
        ssh -n -o BatchMode=yes "${nodes[$rank]}" "export PATH=/usr/local/bin:\$PATH MAX_STEPS='$steps' MASTER_PORT='$port' MERGE_LORA='$merge' BWY_RUNTIME_ROOT='${BWY_RUNTIME_ROOT:-/bwykj_ssd/wangmx/qwen122-runtime}'; exec bash '$script' '$rank' '$run'" >"$run/logs/rank$rank.log" 2>&1 &
    fi
done
for rank in 0 1 2 3; do wait -n || { echo "Training failed; see $run/logs" >&2; exit 1; }; done

# Safetensors writes mode 0600; allow the shared training group to reload checkpoints.
find "$run" -type f -exec chmod g+r {} +
echo "Training complete: $run"
