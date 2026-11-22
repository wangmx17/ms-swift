# perf(examples): cache validated sequence resolver results

Bounded cache keyed by identity, version and metadata.30.451928 to30.212986 s/step,1072.148541 to1080.627732 estimated tokens/s/GPU (+0.79086%). Trace resolver268 to4 and associated sync536 to8. Gain limited because waits shift into partial dispatch; inclusive CPU time removed is not step-time savings.

Stack base: codex/qwen122-07-partial-gdn-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-08-resolver-cache-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/08-resolver-cache.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
