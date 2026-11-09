#!/usr/bin/env bash
set -euo pipefail
export TRAINING_MUSA_ADAPTOR_AUTOLOAD=0
export TORCHADA_FORCE_CUDA_AVAILABLE=1
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}
# Keep the version required by the upstream fp8.sh. Never modify site-packages.
python -c "import importlib.metadata as m; assert m.version('transformers') == '5.2.0'"
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
transformers_dir=$(python -c "import importlib.util; print(next(iter(importlib.util.find_spec('transformers').submodule_search_locations)))")
overlay=$(mktemp -d)
trap 'rm -rf -- "$overlay"' EXIT
cp -R "$transformers_dir" "$overlay/transformers"
patch --batch --directory="$overlay" -p1 < "$script_dir/fp8_musa_capability.patch"
PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" swift export \
    --model Qwen/Qwen3.5-4B \
    --quant_method fp8 \
    --output_dir Qwen3.5-4B-FP8 "$@"
# Upstream also provides: swift infer --model Qwen3.5-4B-FP8
# Inference has not been validated by this export test.
