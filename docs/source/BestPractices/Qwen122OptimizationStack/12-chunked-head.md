# perf(examples): recompute frozen output head and CE in chunks

对原本已冻结的输出层及 TE CE 按 512 token 分块重算，不新增参数冻结。约 29.3545 秒/步，吞吐持平；reserved 峰值减少 1856 MiB/卡、整卡采样减少 1225 MiB、分配重试 43 → 0。4 组局部校验通过，真实尺寸 loss 和输入梯度逐位一致。

基于：codex/qwen122-11-lora-add-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-12-chunked-head-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/12-chunked-head.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
