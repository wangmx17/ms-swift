# Qwen3.5-122B GDN TileLang

提供GDN TileLang接入、packing分流、元数据缓存、通信预热及内部重算。实现仅在当前训练进程中加载。

## 准备与启动

四机需共享仓库和runtime目录，并预先配置容器间SSH。使用与当前适配代码匹配的训练环境，源码不匹配时校验会停止执行。

```bash
python examples/models/qwen3_5/qwen122_optimizations/prepare_runtime.py --archive /path/to/torch_musa_op_extension-master.tar.gz --runtime /bwykj_ssd/wangmx/qwen122-runtime-clean
BWY_RUNTIME_ROOT=/bwykj_ssd/wangmx/qwen122-runtime-clean MAX_STEPS=3 bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh
```

扩展来源：https://sh-code.mthreads.com/ai/torch_musa_op_extension 。归档SHA256必须为`2d3c65f34d05bf35bb50364a4c5d0af09538750f05adf39afe5c61102d64f727`，请使用新的runtime目录。

默认TP1/PP1/EP8、MB2/GBS256、packing4096、full/block41、tail5内部重算、TE CE、GC100、shared-expert-overlap=false，保留全局异步。实际参数以node脚本为准。

LoRA、head、attention目录中的`enable.json`存在时，入口自动加载对应模块；移走对应文件可禁用该项。
