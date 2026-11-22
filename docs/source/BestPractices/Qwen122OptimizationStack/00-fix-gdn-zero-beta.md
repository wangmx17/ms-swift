# fix(examples): preserve GDN beta-zero gradients

修复 GDN 在 beta=0 时丢失的 beta 梯度。独立缺陷修复，不计性能收益；保持原有 2% 相对 L2 门槛，修复后误差 0.431698%，6 组 packed、3 组 dense 回归通过。

基于：main

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
