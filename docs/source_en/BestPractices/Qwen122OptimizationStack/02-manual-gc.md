# perf(examples): schedule Python GC every 100 steps

On top of TE CE:39.883427 to38.501385 s/step (+3.590% throughput). Audit confirms automatic GC disabled. Later100-step validation covers step100 collection; short timing alone does not quantify amortized collection cost.

Stack base: codex/qwen122-01-te-ce-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
