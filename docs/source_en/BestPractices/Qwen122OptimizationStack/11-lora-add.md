# perf(examples): fuse LoRA scaling into alpha add

Replace base+delta*4 with torch.add(base,delta,alpha=4), preserving low-rank matrices/dropout.30.018151 to29.325293 s/step (+2.362666%). Ten BF16/FP32 output/gradient and two actual-dropout/RNG cases passed bitwise. Trace removes2136 multiply launches, usesAddAlpha. This is not fusion of the entire LoRA matrix operation.

Stack base: codex/qwen122-10-layer-gdn-recompute-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-11-lora-add-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/11-lora-add.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
