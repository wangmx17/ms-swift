# perf(examples): partition packed GDN with one token permutation

Supported segments use TileLang; other segments retain FLA, original sequence boundaries and restored token order. Retained token-permutation variant improves paired throughput14.456% (candidate approximately1070.882 estimated tokens/s/GPU). Earlier slice-per-segment variant regressed and is excluded.

Stack base: codex/qwen122-06-gdn-tilelang-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-07-partial-gdn-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/07-partial-gdn.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
