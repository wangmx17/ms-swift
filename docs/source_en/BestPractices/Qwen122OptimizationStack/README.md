# Qwen122 LoRA reproduction / 四机复现

All PRs remain open; final branch contains the full stack. 所有 PR 保持打开，最后分支包含全部优化与独立修复。

## Environment / 环境

4x8 S5000 80GiB; driver3.3.10-server; existing qwen35_test containers. Torch/torch_musa2.9.0.post1, MCore0.19.1+5be962670, TE2.13.0+496599c8, training-musa-adaptor2.2.0, mcore-bridge1.6.3, mate0.2.8.post2. Shared /bwykj_ssd and container SSH aliases worker31008/13/29/30 required.

Model /bwykj_ssd/models/Qwen3.5-122B-A10B and data /bwykj_ssd/datasets/chinatax-qa.jsonl are external. Obtain authorized torch_musa_op_extension archive separately, SHA256 2d3c65f34d05bf35bb50364a4c5d0af09538750f05adf39afe5c61102d64f727. Source/ABI guards deliberately reject mismatches. 仓库不提供模型、数据及完整内部扩展源码。

## Prepare / 准备

From final shared checkout inside worker31008 container:

    export BWY_RUNTIME_ROOT=/bwykj_ssd/wangmx/qwen122-runtime
    python examples/models/qwen3_5/qwen122_optimizations/prepare_runtime.py --archive /path/to/authorized/source.tar.gz --runtime /bwykj_ssd/wangmx/qwen122-runtime
    chmod -R a+rX /bwykj_ssd/wangmx/qwen122-runtime
    mkdir -p /bwykj_ssd/wangmx/qwen122-runtime/cache-ranks
    chmod 2777 /bwykj_ssd/wangmx/qwen122-runtime/cache-ranks
    bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh --check

Use fresh runtime when advancing patch stages. FLA choices bundled/fingerprinted; TileLang machine code not bundled. Cold/new shapes can compile beyond warmup. Keep per-rank writable caches separate between concurrent jobs. 升级补丁使用新运行目录；新形状仍可能编译。

## Launch / 启动

    bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh

Default20 steps. For1000 steps after verifying idle GPUs and unused port:

    MAX_STEPS=1000 MASTER_PORT=29535 nohup bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh > /bwykj_ssd/wangmx/qwen122-1000-launch.log 2>&1 < /dev/null &

Launcher prints Training logs; rank0.log through rank3.log contain metrics. Per-device counters/binding reports below versioned output. Global async; overlap/router false. TP1/PP1/EP8/ETP1/CP1/DP32, MB2/GBS256, BF16 LoRA r8 alpha32, packing4096, full/block41, TE CE, GC100. Eval/save interval200, adapter-only saving. 1000步会覆盖短测未覆盖的周期评估及保存。

## Results / 结果边界

Historical warm AB/BA:29.159109s,1119.684081 estimated TGS/GPU. Original layout100-step overall30.228767s,1080.063588TGS;11-100:29.970672s,1089.364656TGS. Two new-shape compilations. Allocated69.151GiB/reserved70.625GiB, whole-card78.468GiB; noOOM/retries; loss1.442665 to0.406686. Packaged-checkout validation is reported separately in PRs.

TGS=GBS*mean packed length/step seconds/32; mean4081.123818 for this data/cache. Not actual per-batch token counts. TFLOPS follows customer-approved native Megatron estimator; no custom LoRA correction or MFU. 各优化收益不可跨软件栈和窗口相加。

Local numerical tests preserve original tolerances. No cumulative100-step accuracy A/B. Trace-derived plots visualize summary events, not profiler UI screenshots or step timing. 局部等价校验不等于累计训练轨迹完全相同。
