# perf(examples): cache validated sequence resolver results

有界 resolver 缓存保留身份、版本与元数据失效检查。单步 30.451928 → 30.212986 秒，TGS 1072.148541 → 1080.627732，提高 0.79086%。resolver 调用 268 → 4，同步 536 → 8；等待转移到其他路径限制收益。

基于：codex/qwen122-07-partial-gdn-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-08-resolver-cache-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/08-resolver-cache.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
