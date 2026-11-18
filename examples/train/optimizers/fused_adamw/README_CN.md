# MUSA fused AdamW 对照脚本

沿用 `d89f151370358b1090bd7ac7df223be222395eca` 的
`examples/train/multi-gpu/ddp/train.sh`：Qwen2.5-7B-Instruct、LoRA rank8/alpha32、
BF16 基座、self-cognition#1000、双卡 DDP、batch1、累积8、1 epoch、lr1e-4、
max_length2048。官方示例是配置来源，不是 CUDA 实测证据。

## 前置条件

- 在支持 MUSA 的环境中从仓库安装 ms-swift（`pip install -e .`），准备两张空闲卡。
- torch_musa 必须包含标准 `torch.optim.AdamW(fused=True)` 支持（上游 !1653）。
  测试环境还使用 TE !168 的兼容修复、datasets 4.8.4、torchao 0.16.0。
  这些是已测试环境信息，不代表其他版本均不支持。
- **此分支的 `swift/trainers/arguments.py` 仍会将 fused 回退成普通 AdamW。**
  在已验证的独立环境中，移除了 `_init` 流程中
  `if is_torch_musa_available() and self.optim == 'adamw_torch_fused':`
  对应的四行回退。运行 fused 组前需要包含这一修改的 ms-swift 源码，或已取消回退的版本。
  本目录不自动修改框架或安装补丁；回调会在训练更新前验证实际 fused 标志，回退时直接报错。
- 禁用 training_musa_adaptor 自动加载与补丁，与原对照环境保持一致。
  本脚本不启用针对普通低精度标量乘法的额外修复。
- 模型与数据默认使用官方 ID；`MODEL` 可指向同一模型的完整本地缓存。
  本目录不包含服务器离线数据映射，离线运行需提前准备框架可识别的同源数据缓存。

## 使用

从仓库根目录运行，每次显式指定一组；三组应使用相同环境、模型、数据和设备。

```bash
export MUSA_VISIBLE_DEVICES=0,1
export MODEL=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_ROOT=output/fused_adamw_musa
bash examples/train/optimizers/fused_adamw/train_musa.sh plain
bash examples/train/optimizers/fused_adamw/train_musa.sh fused
bash examples/train/optimizers/fused_adamw/train_musa.sh plain_repeat
```

固定两进程，不修改官方训练步数、采样规则、保存间隔和其他超参数。
修改仅包括设备/模型缓存/输出路径、显式优化器和观测回调。已有 case 输出目录时拒绝启动；
重新运行请设置新的 `OUTPUT_ROOT`。Swift 可能在指定目录下增加带版本/时间的子目录，
请以实际输出目录（日志及 args.json）为准。

每个实际运行目录的 `optimizer_audit/` 包含每个 rank 的有效参数、优化器类型、
fused 标志、状态 dtype/device、结束步数与日志，以及 rank0 的初始/最终可训练参数。
观测回调会增加 CPU 拷贝与存储开销，不用于性能比较。用实际审计目录执行：

```bash
python examples/train/optimizers/fused_adamw/compare.py \
  /path/to/plain/optimizer_audit /path/to/fused/optimizer_audit \
  --output comparison.json
python examples/train/optimizers/fused_adamw/compare.py \
  /path/to/plain/optimizer_audit /path/to/plain_repeat/optimizer_audit \
  --output repeat.json
```

报告中的 `plain`/`fused` 分别表示第一个参考输入和第二个候选输入，第二次对比的候选是普通重复组。
比较前检查有效参数差异、初始参数、步数及日志 step 是否对齐；报告输出连续误差，
不自动宣称精度通过。审计完成不等于 checkpoint 回读或断点恢复成功。

## 已有运行证据与边界

2026-11-09 原始脚本在 RC5/1104 daily 加上述环境适配后运行：普通、fused、普通重复组
各 63/63 步，退出码均为0并保存 checkpoint-63。可训练 LoRA 参数和优化器动量为 FP32，
基座与 autocast 为 BF16。普通/fused 的最终参数最大绝对差为 0.0012114186，
相对 L2 差约 0.555%，loss 最大差为 0.002970314；普通重复组最终参数逐位一致。
已有 checkpoint 回读通过，但该轮未执行断点恢复，也未执行 CUDA 对照。

整理后的脚本移除了现场路径、下载层映射和额外环境探针；未重新执行三组 GPU 训练。
功能可用与数值验收分开报告；当前差异不能直接归因于 fused 内核或普通标量乘法。
