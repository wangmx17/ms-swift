#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
HOSTFILE=${HOSTFILE:-$script_dir/hostfile}
SSH_PORT=${SSH_PORT:-62216}
MASTER_PORT=${MASTER_PORT:-29661}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt_moer/wangmx/msswift_runs/qwen3_8_27b_full_1000}
TRAIN=$script_dir/mcore_qwen3_8_27b_full.sh
action=${1:-start}

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
TRAIN_ITERS=${TRAIN_ITERS:-1000}

if [[ ! -f $HOSTFILE ]]; then
  printf '找不到 hostfile：%s\n' "$HOSTFILE" >&2
  exit 1
fi

mapfile -t HOSTS < <(sed 's/#.*//' "$HOSTFILE" | awk 'NF {print $1}')
if [[ ${#HOSTS[@]} -eq 0 ]]; then
  printf 'hostfile 里没有机器：%s\n' "$HOSTFILE" >&2
  exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]}}
NNODES=${#HOSTS[@]}

ssh_to() {
  local host=$1
  shift
  ssh -n -p "$SSH_PORT" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    "root@${host}" "$@"
}

if [[ $action == stop ]]; then
  stop_code=$(cat <<'PY_STOP'
import os
import signal
import sys
from pathlib import Path

train, output = sys.argv[1:]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        env = (proc / 'environ').read_bytes().split(b'\0')
        pid = int(proc.name)
        if (len(args) > 1 and os.path.basename(args[0]) == b'bash'
                and args[1] == os.fsencode(train)
                and os.fsencode('OUTPUT_DIR=' + output) in env
                and os.getpgid(pid) == pid):
            os.killpg(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
PY_STOP
)
  printf -v stop_command '%q ' python3 -c "$stop_code" "$TRAIN" "$OUTPUT_DIR"
  for host in "${HOSTS[@]}"; do
    printf 'stop %s\n' "$host"
    ssh_to "$host" "$stop_command"
  done
  exit 0
fi

if [[ $action != start ]]; then
  printf '用法：bash %s [stop]\n' "$0" >&2
  exit 1
fi

for key in TP PP CP MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_ITERS; do
  if [[ ! ${!key} =~ ^[1-9][0-9]*$ ]]; then
    printf '%s 必须是正整数：%s\n' "$key" "${!key}" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR/logs"
printf 'NNODES=%s MASTER_ADDR=%s MASTER_PORT=%s\n' "$NNODES" "$MASTER_ADDR" "$MASTER_PORT"
for rank in "${!HOSTS[@]}"; do
  host=${HOSTS[$rank]}
  log=$OUTPUT_DIR/logs/${host}.log
  printf 'launch rank %s on %s -> %s\n' "$rank" "$host" "$log"
  ssh_to "$host" "mkdir -p '$OUTPUT_DIR/logs' || exit; cd '$repo_root' || exit; NNODES=$NNODES NODE_RANK=$rank MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT OUTPUT_DIR='$OUTPUT_DIR' TP=$TP PP=$PP CP=$CP MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE TRAIN_ITERS=$TRAIN_ITERS setsid nohup bash '$TRAIN' > '$log' 2>&1 < /dev/null &"
done
printf '已拉起。日志目录：%s/logs\n' "$OUTPUT_DIR"
