# Qwen2.5 FSDP2 LoRA on MUSA

在包含 MUSA 基础适配的环境中，从仓库运行：

```bash
MUSA_VISIBLE_DEVICES=6,7 bash examples/train/multi-gpu/fsdp2_lora/train_musa.sh
```

可通过 `MODEL` 指定同源本地模型目录，通过 `OUTPUT_DIR` 指定输出路径。需要可访问 Qwen/Qwen2.5-7B-Instruct 和 swift/self-cognition 数据集或已准备的对应缓存。

脚本保留同目录 CUDA train.sh 的模型、数据、训练参数及 accelerate launch 入口，直接使用原 fsdp2.json（双进程、BF16、FULL_STATE_DICT、FSDP activation checkpointing=true）。不要替换为 swift 内置 fsdp2 预设：其分片保存配置不同。Trainer gradient_checkpointing=false 不等于关闭 FSDP activation checkpointing。

## 验证环境与前置补丁

验证环境：training-suite v2.2.0-rc5-py312-20261104，torch/torch_musa 2.9.0.post1，Transformers 5.16.1，PEFT 0.20.0，Accelerate 1.12.0，torchada 0.1.106，TransformerEngine 2.13.0+9f27388f。包版本号本身不保证包含以下本次任务补丁：

- TE：设备选择优先识别 musa，并同步 torch.distributed.checkpoint.filesystem 已捕获的设备选择函数。
- torchada：将 DeviceFactoryWrapper 注册到 torch.serialization.add_safe_globals，兼容 weights_only=True 加载。

这些补丁由环境提供，脚本不修改安装包。不需要额外的 torch_musa 初始化标志同步、Accelerate FULL adapter 修复或 PyTorch checkpoint kwargs 扫描补丁。本次保留原版 Accelerate 的 preserve_rng_state=False，不修改 Dropout 或重算语义。禁用 training_musa_adaptor 自动加载与已验证环境保持一致。

## 验证范围

2026-11-09，S5000 双卡，固定上游 d5253cbc 的官方配方完成 63/63 步，502.09 秒，训练/保存退出码0。adapter配置、权重、优化器、RNG和trainer状态均生成，两rank可训练参数无NaN/Inf。撤掉初始化同步与显式MCCL清理后，adapter和optimizer文件与前一轮逐字节一致。退出有未显式调用destroy_process_group的MCCL警告，不影响本次正常结束。

上述GPU实测使用同源模型/原始数据缓存路径映射及观测回调；本文件将启动方式整理为可复用脚本，另做命令参数对照和bash语法检查。仅两项任务补丁下的续训尚未单独验证。CUDA默认fused AdamW，而当前ms-swift在MUSA自动回退为非fused AdamW；训练通过不代表CUDA/MUSA数值等价，也不证明preserve_rng_state=True路径无缺陷。
