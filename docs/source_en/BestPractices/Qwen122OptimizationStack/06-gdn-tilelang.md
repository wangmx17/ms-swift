# perf(examples): integrate validated GDN TileLang runtime

Enable repaired GDN TileLang with saved-state recomputation and original fallback guards. Same-allocator36.512183 to34.869445 s/step,894.194 to936.321 estimated tokens/s/GPU (+4.711%). Do not count gradient repair as acceleration. Includes strict captured FLA choices, per-rank JIT paths, memory/native-Megatron metrics and optional stack trace. New shapes can still compile. The pinned internal archive is an explicit prerequisite.

Stack base: codex/qwen122-05-gdn-state-recompute-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-06-gdn-tilelang-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/06-gdn-tilelang.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
