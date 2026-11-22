# perf(examples): batch strictly guarded short attention sequences

Group at most4 eligible sequences(length65-256), retaining boundaries and MATE kernels. WarmAB/BA29.331952 to29.159109 s/step,1113.086185 to1119.684081 estimatedTGS/GPU (+0.592757%). Synthetic/actual output,qkv-gradient,RNG checks passed bitwise. Rank7 MATE backward1164 to660; hit rate8.75% limits benefit. Rank0 had no hits; use rank7 evidence. Original layout100-step mean30.228767 s/step,1080.063588TGS/GPU includes two compilations. Loss1.442665 to0.406686,noOOM/retries. No clean-cache or cumulative accuracyAB claim.

Stack base: codex/qwen122-12-chunked-head-20261122

Historical paired evidence; do not add gains across stacks/windows. TGS is estimated as GBS * mean packed length / step seconds / 32, not actual per-batch token counting.

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-13-attention-group4-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/13-attention-group4.png)

Raw traces remain on the server. Figure visualizes aggregate trace summaries, not the profiler UI; absence from the top-event summary does not prove absence of an event.
