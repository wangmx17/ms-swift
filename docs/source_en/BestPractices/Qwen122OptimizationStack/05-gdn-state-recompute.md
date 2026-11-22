# perf(examples): recompute GDN backward handoff tensors

Separate saved-state recomputation patch after zero-beta repair. Nine local equivalence cases passed; approximately321 MiB saved at8192 tokens. Enabled with following integration; no isolated end-to-end throughput result established. Setup verifies exact before/after source hashes.

Stack base: codex/qwen122-04-startup-memory-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
