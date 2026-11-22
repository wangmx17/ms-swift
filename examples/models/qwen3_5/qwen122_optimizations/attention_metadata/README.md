# Packed attention metadata reuse (A2)

Repeated group4 eligibility and fallback checks copied identical packed sequence boundaries to the CPU on every attention layer. This bounded cache reuses validated integer spans while preserving the existing attention kernel, packing boundaries, precision, LoRA and RNG behavior. The tuned node script enables it by default.

Cache keys retain tensor identity and check version, shape, stride, offset, dtype, device and total. Maximum 64 entries and 4096 integers per input. Normal PyTorch alias/in-place writes invalidate entries; unversioned/unsupported tensors bypass the cache. External raw-pointer or .data writes bypassing version tracking are unsupported. No activation or gradient is cached.

## Validation and measured effect

Based on PR #25, unchanged 4x8 S5000, MB2/GBS256, packing4096(mean4081.123818), full/block41, TE CE/GC100, global asynchronous execution. 22 CPU cases and 20 GPU output/qkv-gradient/CPU+MUSA RNG comparisons passed bitwise. Local equality is not an end-to-end convergence guarantee.

Separate baseline/candidate traces (6 steps; Python stacks; rank0/rank7) and separate pure timing (10 steps each), full common window5-10:

|Metric|Baseline|Cached|
|---|---:|---:|
|seconds/step|29.139918|28.080979|
|estimated tokens/s/GPU|1120.421510|1162.672802|
|native Megatron TFLOPS/GPU (arithmetic mean logged rates)|64.265325|66.514997|
|allocated peak bytes|73853006848|73863151104|
|reserved peak bytes|75275173888|75434557440|
|whole-card sampled peak MiB|79811|79964|

Throughput +3.7710%; elapsed time -3.63398%. One AB short pair, no statistical confidence or new100-step validation claimed. No TileLang compile, no FLA benchmark in measured window, no OOM/allocation retry, no foreign GPU processes in524 active snapshots.32-rank config/source/binding checks passed. Max ten-step loss difference0.00026417, grad_norm0.00056589. All loss/grad values finite; final loss0.894506 vs0.894770.

Trace step5: original metadata read/copy+sync pairs rank0 352→4; rank7 308→4. Rank0 original host scope1.571s→0.0298s; rank7 1.501s→0.0249s. Inclusive host time includes queued GPU waits and is not an additive speedup estimate. Cache version/check overhead remains; all numerical kernels are still needed. This explains the3.77% end-to-end gain despite98.81% cache hit rate.

The evidence image is a plot of actual trace event counts, not a profiler GUI screenshot. Raw trace locations are in evidence/trace-summary.json. Full remote review: /bwykj_ssd/wangmx/review/gdn-tilelang-20261120/attention-metadata-20261122.

## Reproduce

Use the cumulative tuned launcher from the checkout root:

    bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh

Use MAX_STEPS=10 for the short run. Runtime provisioning and infrastructure remain as documented by the parent PR. No merged PR is assumed.
