# perf(examples): chunk frozen Qwen122 output head and TE loss

冻结输出层按512 token分块并在反向重算logits，沿用TE CE。要求head无梯度、TP1、模型形状匹配；推理/评估走原路径。依赖的TE CE已在main（PR27），不依赖GDN。

本次仅完成Python语法/AST、shell语法和Git差异检查；按用户要求不再进行训练复测。历史含PR12的结果不能作为本次交付性能或精度结论。合入后需重新验证loss、单步耗时、TGS及显存。未提交trace、截图、日志或测试产物。

本PR直接基于main，不依赖GDN或其他未合入PR。独立使用时，将下列环境变量和CLI参数追加到现有ms-swift启动命令；不要覆盖已有external_plugins/callbacks。

BWY_CHUNKED_HEAD_MODE=chunk512
--external_plugins examples/models/qwen3_5/qwen122_optimizations/head/chunked_head_runtime_plugin.py
--callbacks chunked_head_audit

head插件还需将examples/models/qwen3_5/qwen122_optimizations/head加入PYTHONPATH。若同时合入GDN整合PR，案例入口会读取本目录enable.json自动加载，无需修改公共启动脚本。
