# perf(examples): limit internal GDN recomputation to the tail

Keep outer full/block41. Retain GDN handoff state inside checkpointed layers; internally recompute only42,43,45,46,47.30.128579 to30.056351 s/step (+0.240310%,candidate1086.259TGS/GPU). Small gain reflects removing only redundant inner recomputation, with outer work unchanged.

Stack base: codex/qwen122-09-offsets-cache-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-10-layer-gdn-recompute-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/10-layer-gdn-recompute.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
