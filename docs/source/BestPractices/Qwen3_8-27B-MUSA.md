# Qwen3.8-27B MUSA 四机 SFT

示例包含四机启动入口、单节点训练脚本、有效 token 统计插件和 LLaVA 预处理脚本。适用于已配置好 Qwen3.8、Megatron-LM、mcore-bridge、MUSA 和 FlashAttention 的训练镜像；不是环境安装脚本。四台节点应共享相同代码、模型、数据和输出路径，并配置 SSH 免密登录。

## 准备数据

先下载 LLaVA-Instruct-150K 的原始 JSON，并解压对应 COCO train2014 图片，再执行：

```bash
python3 examples/models/qwen3_5/preprocess_llava_qwen38.py \
  --input /mnt_moer/wangmx/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json \
  --image-dir /mnt_moer/wangmx/llava_instruct_150k_preprocessed/images \
  --output /mnt_moer/wangmx/llava_instruct_150k_preprocessed/llava_instruct_150k_swift.jsonl
```

脚本保留全部字段、对话及 `<image>` 标记，只将 `image` 改成绝对路径并输出 JSONL。支持原始图片名和 `COCO_train2014_` 前缀；检查图片文件存在，不解码图片。输出已存在时拒绝覆盖。

## 启动训练

将 `examples/models/qwen3_5/hostfile` 的四个占位主机名替换为实际 SSH 主机地址，第一行为 master。默认每机 8 卡，SSH 用户为 root，端口为 62216。修改训练脚本中的模型、数据路径默认值以匹配部署；节点间共享路径必须一致。

```bash
bash examples/models/qwen3_5/mcore_qwen3_8_27b_4node_launch.sh
```

默认 TP4、PP1、CP1、SP 开启、micro batch 8、GBS64、BF16、1000 步，序列长度上限 8192；开启 full/uniform 重计算（每组 1 层）、RoPE fusion 和每 100 步 manual GC。学习率 1e-5、cosine、最低 1e-6、预热比例 0.05。SFT 冻结视觉塔和对齐模块，因此不能描述为整个多模态模型全部参数参与训练。测试方案中的 `FORCE_BALANCE=0` 在本验证环境没有对应实现，不是一个实际生效的训练开关。

启动器支持 `HOSTFILE`、`SSH_PORT`、`MASTER_ADDR`、`MASTER_PORT`、`OUTPUT_DIR`、`TP`、`PP`、`CP`、`MICRO_BATCH_SIZE`、`GLOBAL_BATCH_SIZE` 和 `TRAIN_ITERS` 环境变量。每节点输出写入 `OUTPUT_DIR/logs`，框架在其版本目录保存 `args.json`、`logging.jsonl`、权重和 `images/train_loss.png`。

```bash
bash examples/models/qwen3_5/mcore_qwen3_8_27b_4node_launch.sh stop
```

停止命令仅向本脚本路径及 `OUTPUT_DIR` 匹配的独立训练进程组发送 SIGTERM；使用自定义输出目录时，停止命令也需设置相同的 `OUTPUT_DIR`。

## token 和性能统计

插件只支持 PP1、CP1、causal SFT、packing=false、logging_steps=1、eval_iters=0；更改为不支持的组合会报错。按实际消耗的 micro batch 读取 CPU `seq_lens`，在 DP 组汇总，避免重复计算 TP 副本。

- `effective_tokens`：模板及截断后的全局有效输入长度，包含提示、回答、特殊 token 和展开后的图像 token 位置，不含对齐 padding。
- `loss_tokens`：labels 不等于 -100 的位置数。
- `global_samples`：每步全局样本数，校验为 GBS。
- `step_time_s`：相邻日志边界的墙钟时间，取各 rank 最大值；包含加载数据、训练、GC、上一轮日志开销及 token 汇总，不是纯 GPU kernel 时间。
- `cluster_tokens_per_s`：本步有效 token 数除以本步耗时；`tokens_per_s_per_gpu` 再除以总卡数。

本测试最终吞吐对第 101～1000 步的每步吞吐取算术平均；不是总 token 除以总耗时，也不是用 8192×GBS 代替有效 token。

## 验证记录

基于 ms-swift `d89f1513`，4 节点 × 8 卡 X10000 完成 1000 步；第 101～1000 步平均 3.2222 s/step、319.21 tokens/s/GPU。loss 从 1.34467602 到 1.02861476，整体下降并存在波动。该结果仅代表上述环境和配置。

全量预处理通过：157712 条样本、81479 张独立图片，输出与正式训练数据逐字节一致。PR 整理仅对 Python 格式和启动器停止范围做调整；1000 步记录来自整理前的同一训练配置和 token 统计逻辑，停止范围另用隔离进程验证。
