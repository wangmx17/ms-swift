# MUSA MiSS、Qwen3.5 SP 与 FP8 导出示例

从安装了本仓库的环境调用以下脚本；脚本不切换工作目录。可用 MUSA_VISIBLE_DEVICES 选择空闲卡，末尾参数可覆盖模型和输出路径。adaptor 保留安装但关闭自动加载；使用原生 torchada。

- `bash examples/train/tuners/bone/train_musa.sh`：PEFT **0.20.0**；将既有 bone CLI 路由迁移到 MissConfig，原 BoneConfig 不再使用。Qwen2.5-7B-Instruct、self-cognition#1000、1轮、batch1、GA16、r4；adapter 保存类型是 MISS。旧版 PEFT 不适用于此分支。
- `bash examples/train/sequence_parallel/sequence_parallel_qwen3_5_musa.sh`：Qwen3.5-4B、LongAlpaca-12k、8进程/SP4、batch2/GA8、max_length65535。保持官方显式参数；官方注释日志189步与本环境12000条/3轮得到1125步不同，不能称完整CUDA日志复现。未为凑步数截取数据或设置max_steps。
- `bash examples/export/quantize/mllm/fp8_musa.sh`：要求 Transformers **5.2.0**（测试tokenizers0.22.2）和 `patch` 命令。临时复制Transformers并应用附带平台判断修复；不修改系统安装。修复仅让明确MUSA device_map跳过NVIDIA SM门槛，不改量化算法；CUDA设备仍执行原检查。

## 验证范围（2026-11-09）

基于fork main d89f151370358b1090bd7ac7df223be222395eca，官方示例基于d5253cbc5fe3258f143d1826d4078471fa0fc5dc。RC5镜像，torch/torch_musa2.9.0.post1。

| 项目 | 实测 | 限制 |
|---|---|---|
| Bone→MiSS | 63/63、exit0、189秒；MISS/r4、196张量、优化器63步及有限性审计通过 | TF5.17.0；独立环境移除不兼容的可选torchao0.9.0；没有CUDA数值对照 |
| 长文本SP | 308步loss/grad有限，保存checkpoint-250/300，用户接受本轮完成 | 主动结束而非完整1125步；首次FlashAttention设备探测需torchada开关；有内核编译警告 |
| FP8导出 | 干净环境exit0；200FP8权重和200scale，均有限，重复导出哈希一致 | 依赖附带候选补丁；官方infer尚未执行 |

以上结果验证了脚本对应配方与补丁；提交后的脚本还需在调用者自己的环境复查。只把导出结果当导出验证，不把它等同推理通过。
