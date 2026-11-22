# perf(examples): partition packed GDN with one token permutation

支持的序列走 TileLang，其余保持 FLA；一次 token 置换后恢复原顺序和序列边界。保留版本吞吐提高 14.456%，候选约 1070.882 TGS/卡；逐段切片的退化版本未纳入。

基于：codex/qwen122-06-gdn-tilelang-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-07-partial-gdn-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/07-partial-gdn.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
