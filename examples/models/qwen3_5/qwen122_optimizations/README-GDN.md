# perf(examples): consolidate original TileLang GDN runtime and memory optimizations

合并旧PR16–22：分配器fraction0.90、实际通信组预热、原始TileLang接入、packing部分分流、resolver/offset缓存以及tail5内部重算。PR16+17及其后端依赖在同一个PR中交付。删除PR12零beta修复、两份补丁源码目录和原补丁SHA依赖。内部重算改为案例内autograd适配器，调用原始前向/反向，安装包和扩展源码不写入补丁。

原始算子的零beta边界行为保持原样；本PR不声称修复该边界。offset读取仍使用带源码校验的进程内函数替换，不修改site-packages。LoRA/head/attention可独立安装，其enable.json存在时此入口才加载。

本次仅完成Python语法/AST、shell语法和Git差异检查；按用户要求不再进行训练复测。历史含PR12的结果不能作为本次交付性能或精度结论。合入后需重新验证loss、单步耗时、TGS及显存。未提交trace、截图、日志或测试产物。

准备与启动（worker31008，四机共享仓库、镜像与SSH已配置）：

python examples/models/qwen3_5/qwen122_optimizations/prepare_runtime.py --archive /path/to/torch_musa_op_extension-master.tar.gz --runtime /bwykj_ssd/wangmx/qwen122-runtime-clean
BWY_RUNTIME_ROOT=/bwykj_ssd/wangmx/qwen122-runtime-clean MAX_STEPS=3 bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh

归档来源：https://sh-code.mthreads.com/ai/torch_musa_op_extension ，本案例归档SHA256为2d3c65f34d05bf35bb50364a4c5d0af09538750f05adf39afe5c61102d64f727。使用原测试环境留存归档；任意最新master归档不能替代。准备程序拒绝旧修补runtime，请使用新目录。FLA配置是原环境实际选出的kernel配置，保留源码/运行时校验，不是trace附件。

默认TP1/PP1/EP8、MB2/GBS256、packing4096、full/block41、tail5内部重算、TE CE、GC100、shared-expert-overlap=false及全局异步。LoRA/head/attention各目录enable.json与本入口配合自动加载；未合入相应PR时不会加载。可移走对应enable.json临时禁用该项。实际参数以node脚本为准。TFLOPS沿用Megatron原生估算，不做LoRA修正。
