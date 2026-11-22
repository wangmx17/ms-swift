# perf(examples): batch strictly guarded short attention sequences

仅对满足严格条件的 65–256 长度序列最多每 4 条分组调用 MATE，保留边界。暖缓存 AB/BA 单步 29.331952 → 29.159109 秒，TGS 1113.086185 → 1119.684081，提高 0.592757%。rank7 的 MATE 反向 1164 → 660 次；命中率 8.75% 限制收益，rank0 无命中。原布局 100 步全程 30.228767 秒、1080.063588 TGS/卡，包含两次编译；loss 1.442665 → 0.406686，无 OOM/分配重试。尚无累计 100 步精度 A/B。

基于：codex/qwen122-12-chunked-head-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

![Trace-derived evidence](https://raw.githubusercontent.com/wangmx17/ms-swift/codex/qwen122-13-attention-group4-20261122/docs/source_en/BestPractices/Qwen122OptimizationStack/evidence/13-attention-group4.png)

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
