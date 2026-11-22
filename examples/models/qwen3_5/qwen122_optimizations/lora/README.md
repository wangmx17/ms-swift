# LoRA残差融合

将LoRA缩放与残差相加合并为torch.add(alpha)，要求BF16和scaling=4。

在仓库根目录设置环境变量：

```bash
export LORA_ADD_MODE=fused
```

将以下参数追加到现有训练命令，保留已有external_plugins和callbacks：

```text
--external_plugins examples/models/qwen3_5/qwen122_optimizations/lora/lora_add_plugin.py
--callbacks lora_add_audit
```

使用GDN整合入口时，本目录的enable.json会自动装载这些配置。
