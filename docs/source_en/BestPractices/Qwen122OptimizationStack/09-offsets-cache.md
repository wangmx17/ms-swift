# perf(examples): share bounded host offset reads

Reuse validated host reads across three consumers. Hash-checked generation preserves original guards without vendoring internal source functions.30.180719 to30.131753 s/step,1081.783 to1083.541 estimated tokens/s/GPU (+0.162504%). Trace reads268 to4. Gain small because resolver cache already removes most waits and GPU/communication work remains.

Stack base: codex/qwen122-08-resolver-cache-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-09-offsets-cache-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/09-offsets-cache.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
