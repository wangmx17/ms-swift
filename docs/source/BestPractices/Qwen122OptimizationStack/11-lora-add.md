# perf(examples): fuse LoRA scaling into alpha add

把 base + delta * 4 改为 torch.add(base, delta, alpha=4)，保留 LoRA 矩阵和 dropout。单步 30.018151 → 29.325293 秒，提高 2.362666%；10 组输出及梯度、2 组实际 dropout/RNG 校验逐位一致，trace 减少 2136 次乘法。

基于：codex/qwen122-10-layer-gdn-recompute-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-11-lora-add-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/11-lora-add.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
