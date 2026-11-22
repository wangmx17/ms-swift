# perf(examples): group packed attention and cache sequence boundaries

合并旧PR25+26：符合条件的packed attention按4组调用MATE，同时缓存packed边界。保留形状、dtype、mask与版本守卫，不符合条件回退原函数。

本次仅完成Python语法/AST、shell语法和Git差异检查；按用户要求不再进行训练复测。历史含PR12的结果不能作为本次交付性能或精度结论。合入后需重新验证loss、单步耗时、TGS及显存。未提交trace、截图、日志或测试产物。

本PR直接基于main，不依赖GDN或其他未合入PR。独立使用时，将下列环境变量和CLI参数追加到现有ms-swift启动命令；不要覆盖已有external_plugins/callbacks。

BWY_ATTENTION_GROUP_MODE=group4 BWY_ATTENTION_METADATA_MODE=cached
--external_plugins examples/models/qwen3_5/qwen122_optimizations/attention_metadata/attention_metadata_plugin.py examples/models/qwen3_5/qwen122_optimizations/attention/attention_group_plugin.py
--callbacks attention_metadata_audit attention_group_audit

head插件还需将examples/models/qwen3_5/qwen122_optimizations/head加入PYTHONPATH。若同时合入GDN整合PR，案例入口会读取本目录enable.json自动加载，无需修改公共启动脚本。
