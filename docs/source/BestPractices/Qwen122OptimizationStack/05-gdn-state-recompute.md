# perf(examples): recompute GDN backward handoff tensors

GDN 反向交接张量改为重算，保持数值验证门槛。9 组局部校验通过，8192 token 场景节省约 321 MiB；没有独立整模型吞吐对照。

基于：codex/qwen122-04-startup-memory-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
