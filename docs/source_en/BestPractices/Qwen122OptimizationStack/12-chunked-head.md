# perf(examples): recompute frozen output head and CE in chunks

Chunk512 for the already-frozen output head and TE CE; no new parameter freezing. Four numerical cases passed; actual8192-token/248320-vocabulary loss/input gradients bitwise. Throughput approximately unchanged (~29.3545 s/step); reserved peak-1856MiB, whole-card sample-1225MiB, allocation retries43 to0. Retained for memory headroom, not speed.

Stack base: codex/qwen122-11-lora-add-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-12-chunked-head-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/12-chunked-head.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
