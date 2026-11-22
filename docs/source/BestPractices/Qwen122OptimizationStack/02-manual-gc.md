# perf(examples): schedule Python GC every 100 steps

使用框架手动 GC，每 100 步回收。叠加 TE 后单步 39.883427 → 38.501385 秒，吞吐提高约 3.590%。短测不能单独覆盖周期回收开销，后续 100 步运行已覆盖第 100 步。

基于：codex/qwen122-01-te-ce-20261122

历史配对数据；测量窗口、软件栈不同的收益不能相加。TGS 按 GBS × packing 均长 ÷ 单步秒数 ÷ 32 估算，不是逐 batch 实测 token 计数。

原始 trace 位于服务器，图是摘要计数可视化，不是 profiler 界面截图；未提供原始事件不代表事件不存在。
