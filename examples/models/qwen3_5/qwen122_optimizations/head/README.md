# 冻结输出层分块

输出层与TE交叉熵按512 token分块，在反向重算logits。要求输出层冻结、TP1、TE CE及匹配的模型形状；评估走原路径。

在仓库根目录设置环境变量：

```bash
export BWY_CHUNKED_HEAD_MODE=chunk512
export PYTHONPATH="$PWD/examples/models/qwen3_5/qwen122_optimizations/head:${PYTHONPATH:-}"
```

将以下参数追加到现有训练命令，保留已有external_plugins和callbacks：

```text
--external_plugins examples/models/qwen3_5/qwen122_optimizations/head/chunked_head_runtime_plugin.py
--callbacks chunked_head_audit
```

使用GDN整合入口时，本目录的enable.json会自动装载这些配置。
