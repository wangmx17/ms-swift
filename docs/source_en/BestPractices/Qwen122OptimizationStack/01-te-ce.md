# perf(examples): use TE cross entropy for Qwen122 LoRA

Existing TE CE: forward/reverse paired40.221646 to39.883427 s/step (+0.848% throughput); logged memory69.22 to57.84 GiB. Separate stack traces confirmed TE CE execution. Small gain reflects limited CE share of whole-step work. Historical stack; do not add this delta to later results.

Stack base: codex/qwen122-00-fix-gdn-zero-beta-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
