# perf(examples): reduce whole-layer recomputation to block41

Matched38.470975 to36.934077 s/step (+4.161%). Trace checkpoint calls192 to164. Block40 OOM; block41 retained. LoRA, packing, microbatch and GBS unchanged.

Stack base: codex/qwen122-02-manual-gc-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
