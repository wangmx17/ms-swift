# perf(examples): reduce whole-layer recomputation to block41

外层重算调整为 full/block41。单步 38.470975 → 36.934077 秒，吞吐提高约 4.161%，trace 中 checkpoint 调用 192 → 164；block40 曾 OOM。

基于：codex/qwen122-02-manual-gc-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
