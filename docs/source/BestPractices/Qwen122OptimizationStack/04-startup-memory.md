# fix(examples): prepare communicator resources before backward peak

训练开始前用独立零张量预热实际梯度统计及 bucket 通信组，不改模型梯度。分配器比例保留 0.90。作为运行稳定性依赖，未建立独立吞吐收益。

基于：codex/qwen122-03-block41-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
