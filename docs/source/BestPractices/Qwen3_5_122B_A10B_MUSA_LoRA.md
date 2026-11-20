# Qwen3.5-122B-A10B 四机 MUSA LoRA SFT

该入口面向四台各8张 S5000 80GiB 的机器，使用 Megatron-SWIFT、chinatax-qa 和4096 packing。当前已测范围内，最高观测配置为 TP1 / PP1 / EP8 / ETP1 / CP1、micro batch2、GBS256，普通DP32、梯度累积4次：`256 = 32 × 2 × 4`。

## 环境与准备

- 已验证依赖包括 numpy1.26.1、datasets4.8.4、pyarrow22.0.0、peft0.18.1、mcore-bridge1.6.5、binpacking1.5.2、future1.0.0。TE沿用MR169的Python修订和配套原生库；Megatron路由掩码、SGLang轴向RoPE导入处理及保存末尾Gloo等待修复为既有环境改动。

## 启动

先按实际环境调整启动脚本中的节点、容器、共享路径和权限配置，再从仓库根目录执行：

```bash
bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh --check
bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh
```

默认20步，只保存LoRA adapter。启动器从宿主机自动进入容器，再通过SSH同时拉起其它三个节点。需要100步时：

```bash
MAX_STEPS=100 bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh
```

`MASTER_PORT`可以覆盖默认端口。`MERGE_LORA=true`请求完整权重合并；本次最佳并行配置只验证了adapter保存，尚未验证完整合并。20步未达到eval_steps=200，因此这两组确认没有执行评估。

日志位于 <output_dir>/logs/rank0.log 至 rank3.log。启动器保留每次独立目录、共享权限和按本次output_dir精确匹配的失败清理，不使用宽泛的pkill。

## 生效参数与异步模式

LoRA为r8/alpha32/all-linear，BF16，full/uniform激活重计算1层，flash后端；max_length和packing_length均为4096，padding_free=true、group_by_length=false。TP1时sequence_parallel=false。

两个全局LAUNCH_BLOCKING变量在节点脚本中unset，moe_shared_expert_overlap=false，保留官方nccl_comm_warmup=true。关闭共享专家重叠是现有异步GPU超时的局部规避，不代表启用全局同步。

## 实验结果

2026-11-19，两组均在四机32卡上完成20步训练和adapter保存，退出0。以下采用第11–20步完整窗口：

| TP/PP/EP | micro | GBS | 累积 | 秒/步 | 全32卡估算tokens/s | 单卡折算 | 整卡窗口采样峰值GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
|1/1/8|2|256|4|39.045301|26803.93|837.62|77.88|
|1/1/8|2|512|8|78.663376|26608.76|831.52|78.49|

train_speed(s/it)为累计平均：`T(k)=k×train_speed(k)`，窗口秒/步为`[T(20)-T(10)]/10`。打包后训练集有1161包，平均4088.154177 tokens，估算吞吐为`GBS×4088.154177/窗口秒每步`，不是逐批精确token计数或仅loss token吞吐。

原TP2/EP4、micro1、GBS16的packing100步，第11–100步为12554.39 tokens/s；当前观测约2.14倍。最终两组只差0.73%，没有跨轮重复方差证明。PP2/4/8的已测组合未超过当前PP1；搜索GBS上限512，结论只覆盖已测组合和full重计算，不代表全局最优或收敛质量最优。

SimuMax快照缺少本模型Gated DeltaNet混合层、PEFT冻结成本和本集群S5000系统配置，本轮只用它核验并行/batch约束，最终选择来自实际训练测量。
