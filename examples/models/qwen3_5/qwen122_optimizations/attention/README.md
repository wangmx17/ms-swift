# Packed attention分组与缓存

符合条件的packed attention按group4调用MATE，并缓存序列边界。保留形状、dtype、mask与源码版本检查；不符合条件时走原函数。

在仓库根目录设置环境变量：

```bash
export BWY_ATTENTION_GROUP_MODE=group4
export BWY_ATTENTION_METADATA_MODE=cached
```

将以下参数追加到现有训练命令，保留已有external_plugins和callbacks：

```text
--external_plugins examples/models/qwen3_5/qwen122_optimizations/attention_metadata/attention_metadata_plugin.py examples/models/qwen3_5/qwen122_optimizations/attention/attention_group_plugin.py
--callbacks attention_metadata_audit attention_group_audit
```

使用GDN整合入口时，本目录的enable.json会自动装载这些配置。
