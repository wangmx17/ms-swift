# perf(examples): fuse Qwen122 LoRA scaled residual addition

LoRA缩放与残差合并为torch.add(alpha)，限制BF16和scaling=4；通过源码定位替换当前进程方法，不改安装包。

本次仅完成Python语法/AST、shell语法和Git差异检查；按用户要求不再进行训练复测。历史含PR12的结果不能作为本次交付性能或精度结论。合入后需重新验证loss、单步耗时、TGS及显存。未提交trace、截图、日志或测试产物。

本PR直接基于main，不依赖GDN或其他未合入PR。独立使用时，将下列环境变量和CLI参数追加到现有ms-swift启动命令；不要覆盖已有external_plugins/callbacks。

LORA_ADD_MODE=fused
--external_plugins examples/models/qwen3_5/qwen122_optimizations/lora/lora_add_plugin.py
--callbacks lora_add_audit

head插件还需将examples/models/qwen3_5/qwen122_optimizations/head加入PYTHONPATH。若同时合入GDN整合PR，案例入口会读取本目录enable.json自动加载，无需修改公共启动脚本。
