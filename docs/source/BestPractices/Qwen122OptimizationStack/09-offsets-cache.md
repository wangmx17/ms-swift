# perf(examples): share bounded host offset reads

三处消费者共享有界主机 offset 读取缓存，原长度、形状及端点检查保留。单步 30.180719 → 30.131753 秒，TGS 1081.783 → 1083.541，提高 0.162504%；读取 268 → 4。前序缓存已消除大部分等待，整步收益较小。

基于：codex/qwen122-08-resolver-cache-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-09-offsets-cache-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/09-offsets-cache.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
