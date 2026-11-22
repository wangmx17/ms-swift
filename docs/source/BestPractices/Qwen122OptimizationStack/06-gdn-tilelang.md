# perf(examples): integrate validated GDN TileLang runtime

接入带梯度修复和状态重算的 GDN TileLang。相同分配器条件下 36.512183 → 34.869445 秒，估算单卡 TGS 894.194 → 936.321，吞吐提高 4.711%。内部源码归档需要另外提供；补丁不包含完整上游源码。

基于：codex/qwen122-05-gdn-state-recompute-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-06-gdn-tilelang-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/06-gdn-tilelang.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
