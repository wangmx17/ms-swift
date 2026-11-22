# perf(examples): limit internal GDN recomputation to the tail

外层 full/block41 不变，仅末尾 42、43、45、46、47 层启用 GDN 内部重算，其他已被外层重算覆盖的层保留交接状态。单步 30.128579 → 30.056351 秒，提高 0.240310%；只消除重复内层重算，收益有限。

基于：codex/qwen122-09-offsets-cache-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-10-layer-gdn-recompute-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/10-layer-gdn-recompute.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
