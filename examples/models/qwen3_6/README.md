# Qwen3.6-27B LoRA with ms-swift DDP

## 来源与参数 / Sources and configuration

- Model: https://huggingface.co/Qwen/Qwen3.6-27B
- Official LoRA reference: https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/transformers.sh
- Text dataset reference: https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/mcore.sh
- Checked on 2026-11-20: neither this checkout nor the upstream examples tree has a dedicated `qwen3_6` directory. These are adapted cases, not an official Qwen3.6-27B DDP benchmark. This checkout registers Qwen3.6-27B under the Qwen3.5 architecture/template.

沿用官方 Transformers LoRA 示例的 BF16、lr=1e-4、warmup=0.05、r=8、alpha=32、all-linear、1 epoch 和保存/评估间隔。按需求改为 Qwen3.6-27B、DDP、SDPA、max_length=12288、单卡 batch=1、梯度累积=4。移除 DeepSpeed 和 MoE 参数；显式开启非重入梯度检查点、关闭 DDP unused-parameter 检测，冻结视觉模块，关闭 packing/padding-free。默认数据改为官方 MCore 示例中使用的中文 Alpaca 子集，用于文本 SFT；可通过 DATASET 替换。

The cases retain the reference's LoRA and learning-rate settings, use pure DDP instead of ZeRO-3, and remove MoE-only options. Gradient checkpointing is explicit and non-reentrant. Vision modules are frozen. Packing and padding-free are disabled for this SDPA case. The default dataset is the Chinese Alpaca subset from the official MCore example, not the OCR dataset from the Transformers example.

| Nodes | Processes per node | Batch per device | Accumulation | Global batch |
| --- | --- | --- | --- | --- |
| 1 | 8 | 1 | 4 | 32 |
| 2 | 8 | 1 | 4 | 64 |
| 4 | 8 | 1 | 4 | 128 |

DDP 每卡保存完整模型；增加节点不会降低单卡显存。27B 的 BF16 权重约 54 GB（十进制，粗估），另需激活、LoRA、梯度、优化器和临时空间。12288 是长度上限，不会把短样本补到该长度。SDPA 的内核选择依赖设备/软件栈，不能保证 80 GiB 卡在满长样本下不 OOM；Qwen 的 Gated DeltaNet 层也不会因该选项变成普通 SDPA。尚未进行实际训练或满长显存验证。

DDP replicates the full model on every device. More nodes do not reduce per-device memory. Actual 12288-token memory usage must be measured with the intended backend; the default short dataset is not a full-length stress test. SDPA applies to standard attention, not the model's Gated DeltaNet layers. These scripts have not been validated by a training run.

## 运行 / Run

在每台机器相同的 ms-swift checkout 根目录、已配置好的训练环境中执行。先按该环境要求安装 checkout 和模型依赖；脚本不自动安装或升级包。所有节点使用一致的软件、模型、数据和输出路径；建议使用共享输出目录。每机启动一次脚本，不能只在主节点执行。节点之间需能访问 MASTER_ADDR:MASTER_PORT。通过 MUSA_VISIBLE_DEVICES 或 CUDA_VISIBLE_DEVICES 选择设备，NPROC_PER_NODE 必须与所选设备数量一致。

Run from the checkout root in a prepared training environment on each node. Install the checkout and model dependencies beforehand. Use identical software/model/data on all nodes and a shared output path. Set the backend's visibility variable and match NPROC_PER_NODE to the visible device count. Each invocation starts only the local workers; multi-node cases must be launched on every node.

```bash
# Single node (8 devices by default)
bash examples/models/qwen3_6/lora_27b_ddp_1node.sh

# Two nodes: use an address reachable from both nodes
# node 0
MASTER_ADDR=10.124.31.8 NODE_RANK=0 bash examples/models/qwen3_6/lora_27b_ddp_2node.sh
# node 1
MASTER_ADDR=10.124.31.8 NODE_RANK=1 bash examples/models/qwen3_6/lora_27b_ddp_2node.sh

# Four nodes: execute separately on nodes 0, 1, 2, 3 with the corresponding rank
MASTER_ADDR=10.124.31.8 NODE_RANK=0 bash examples/models/qwen3_6/lora_27b_ddp_4node.sh
MASTER_ADDR=10.124.31.8 NODE_RANK=1 bash examples/models/qwen3_6/lora_27b_ddp_4node.sh
MASTER_ADDR=10.124.31.8 NODE_RANK=2 bash examples/models/qwen3_6/lora_27b_ddp_4node.sh
MASTER_ADDR=10.124.31.8 NODE_RANK=3 bash examples/models/qwen3_6/lora_27b_ddp_4node.sh

# Override model/data/output (use identical values on all nodes)
MODEL=/path/to/Qwen3.6-27B DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/shared/output/qwen36-run01 \
bash examples/models/qwen3_6/lora_27b_ddp_1node.sh
```

额外 CLI 参数通过脚本末尾传入（例如 `--max_steps 2`）；显式覆盖参数会改变上述基线。双机/四机保持单卡 batch 和累积不变，因此全局 batch 增大，并非固定全局 batch 的扩展对比。

Additional CLI arguments can be appended, for example `--max_steps 2`. Overrides change the baseline. The multi-node cases keep per-device batch and accumulation fixed, so they do not represent constant-global-batch scaling experiments.
