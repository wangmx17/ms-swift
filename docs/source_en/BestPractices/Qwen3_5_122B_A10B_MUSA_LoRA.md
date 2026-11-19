# Four-node MUSA LoRA SFT for Qwen3.5-122B-A10B

This example targets four nodes with eight S5000 80GiB GPUs each, using Megatron-SWIFT, chinatax-qa and 4096 packing. The highest observed throughput among the tested configurations uses TP1 / PP1 / EP8 / ETP1 / CP1, micro batch2 and GBS256. Standard DP is 32 and gradient accumulation is 4: `256 = 32 × 2 × 4`.

## Environment and prerequisites

- Node aliases are worker31008, worker31013, worker31029 and worker31030; the container is wmx_megatron. Aliases must resolve inside the containers and SSH aliases must enter the corresponding training containers without interactive authentication. Remote users need GPU device access.
- All nodes read the shared /bwykj_ssd mount. The repository is /bwykj_ssd/wangmx/ms-swift, the model is /bwykj_ssd/models/Qwen3.5-122B-A10B and the dataset is /bwykj_ssd/datasets/chinatax-qa.jsonl.
- The master address is 10.124.31.8 and MASTER_PORT defaults to 29535. Training nodes must be able to communicate with each other.
- Validation used registry.mthreads.com/public/training-suite:v2.2.0-rc5-py312-20261105 with driver 3.3.5-server and existing project compatibility fixes. The stock image alone is not the complete validated environment. This PR does not install dependencies or patch frameworks.
- Validated dependencies include numpy1.26.1, datasets4.8.4, pyarrow22.0.0, peft0.18.1, mcore-bridge1.6.5, binpacking1.5.2 and future1.0.0. The environment retains TE MR169 Python revisions with the 1105 native libraries, Megatron routing-mask fixes, the SGLang axial-RoPE import fix, and the save-end Gloo barrier fix.
- The master container runs as root. setpriv creates outputs using shared-storage UID1000/GID1001, and shared GID1000 grants cross-node write access. Adapt these IDs and permissions for other clusters.

## Launch

On the worker31008 host or inside its wmx_megatron container:

```bash
cd /bwykj_ssd/wangmx/ms-swift
bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh --check
bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh
```

The default is 20 steps with adapter-only saving. The host entry automatically enters the local container and launches the other three nodes over SSH. For 100 steps:

```bash
MAX_STEPS=100 bash examples/models/qwen3_5/mcore_qwen3_5_122b_4node_tuned_launch.sh
```

MASTER_PORT overrides the default port. MERGE_LORA=true requests full weight merging; this tuned parallel configuration has only been verified through adapter saving. The 20-step confirmations did not reach eval_steps=200, so evaluation was not performed.

Logs are /bwykj_ssd/wangmx/output/qwen122-lora-YYYYMMDD-HHMMSS/logs/rank0.log through rank3.log. The launcher retains independent timestamp directories, shared permissions, and failure cleanup matched to the current output_dir instead of broad pkill patterns.

## Effective settings and asynchronous execution

LoRA uses r8/alpha32/all-linear, BF16, full/uniform activation recomputation of one layer and the flash backend. max_length and packing_length are 4096, with padding_free=true and group_by_length=false. sequence_parallel=false is required for TP1.

The node script unsets both global LAUNCH_BLOCKING variables, sets moe_shared_expert_overlap=false, and keeps the official nccl_comm_warmup=true option. Disabling shared-expert overlap is the existing local workaround for asynchronous GPU timeouts; it does not enable global synchronous execution.

## Experiment results

On 2026-11-19 both configurations completed 20 training steps and adapter saving across all 32 GPUs with exit0. The complete steps 11–20 window is reported below:

| TP/PP/EP | micro | GBS | Accumulation | Seconds/step | Estimated tokens/s, all 32 GPUs | Per-GPU equivalent | Sampled whole-card peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
|1/1/8|2|256|4|39.045301|26803.93|837.62|77.88|
|1/1/8|2|512|8|78.663376|26608.76|831.52|78.49|

train_speed(s/it) is cumulative: `T(k)=k×train_speed(k)` and window seconds/step is `[T(20)-T(10)]/10`. The training set contains 1161 packed sequences averaging 4088.154177 tokens. Throughput is estimated as `GBS×4088.154177/window_seconds_per_step`; this is not exact per-batch or loss-token counting.

The original TP2/EP4, micro1, GBS16 packing 100 run reached 12554.39 tokens/s over steps 11–100, giving an observed ratio of about 2.14. The final confirmations differ by only 0.73%, without repeated-run variance estimates. Tested PP2/4/8 combinations did not exceed this PP1 configuration. GBS exploration stopped at 512 with full recomputation retained; the result is limited to tested points and does not establish a global optimum or convergence-quality optimum.

The inspected SimuMax snapshot lacks the model's hybrid Gated DeltaNet layers, PEFT frozen-parameter costs and a measured S5000 system configuration. It was used for parallelism/batch constraints; final selection uses actual training measurements.
