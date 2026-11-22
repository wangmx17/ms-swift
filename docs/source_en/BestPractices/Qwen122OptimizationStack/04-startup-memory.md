# fix(examples): prepare communicator resources before backward peak

Warm actual gradient-statistics and bucket groups using separate zero tensors before training; never mutate model gradients. Retain allocator fraction0.90 and expandable segments. Wired by following integration; global async unchanged. No independent throughput gain claimed.

Stack base: codex/qwen122-03-block41-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
