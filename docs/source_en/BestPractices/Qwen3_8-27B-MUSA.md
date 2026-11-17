# Qwen3.8-27B MUSA four-node SFT

These examples provide a four-node launcher, training script, effective-token metrics plugin and LLaVA preprocessing. Use an existing training image with Qwen3.8, Megatron-LM, mcore-bridge, MUSA and FlashAttention configured. They do not install that environment. All nodes need shared code, model, dataset and output paths, plus passwordless SSH.

## Prepare data

Download the original LLaVA-Instruct-150K JSON and extract the corresponding COCO train2014 images, then run:

```bash
python3 examples/models/qwen3_5/preprocess_llava_qwen38.py \
  --input /mnt_moer/wangmx/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json \
  --image-dir /mnt_moer/wangmx/llava_instruct_150k_preprocessed/images \
  --output /mnt_moer/wangmx/llava_instruct_150k_preprocessed/llava_instruct_150k_swift.jsonl
```

All fields, conversations and `<image>` markers are preserved. Only `image` is resolved to an absolute path, and records are written as JSONL. Original filenames and `COCO_train2014_` prefixes are supported. Image files are checked for existence, not decoded. Existing outputs are never overwritten.

## Run training

Replace the four placeholder hosts in `examples/models/qwen3_5/hostfile` with actual SSH hosts; the first host is the master. Defaults are eight GPUs per node, SSH user root and port 62216. Adjust model/data defaults in the training script for your deployment. Paths must be shared consistently across nodes.

```bash
bash examples/models/qwen3_5/mcore_qwen3_8_27b_4node_launch.sh
```

Defaults: TP4, PP1, CP1, SP enabled, micro batch 8, GBS64, BF16, 1000 steps and a maximum sequence length of 8192. Full/uniform recomputation uses one layer per group. RoPE fusion and manual GC every 100 steps are enabled. Learning rate is 1e-5 with cosine decay, minimum 1e-6 and a 0.05 warmup fraction. The vision tower and aligner are frozen during SFT; not every multimodal parameter is trained. The test plan's `FORCE_BALANCE=0` has no implementation in the validated environment and is not an effective training switch.

The launcher accepts `HOSTFILE`, `SSH_PORT`, `MASTER_ADDR`, `MASTER_PORT`, `OUTPUT_DIR`, `TP`, `PP`, `CP`, `MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE` and `TRAIN_ITERS` environment variables. Per-node stdout goes to `OUTPUT_DIR/logs`; the framework saves `args.json`, `logging.jsonl`, weights and `images/train_loss.png` in its versioned output directory.

```bash
bash examples/models/qwen3_5/mcore_qwen3_8_27b_4node_launch.sh stop
```

Stop sends SIGTERM only to a session-leading process group whose launcher script path and `OUTPUT_DIR` match this run. Set the same `OUTPUT_DIR` when stopping a run with a custom output location.

## Token and performance metrics

The plugin requires PP1, CP1, causal SFT, packing=false, logging_steps=1 and eval_iters=0; unsupported settings fail explicitly. It reads CPU `seq_lens` for consumed micro batches and sums within each DP group without double-counting TP replicas.

- `effective_tokens`: global post-template, post-truncation input lengths, including prompts, answers, special tokens and expanded image-token positions; excludes alignment padding.
- `loss_tokens`: number of label positions other than -100.
- `global_samples`: global sample count, checked against GBS.
- `step_time_s`: wall time between logging boundaries, using the maximum rank interval; includes data loading, training, GC, previous logging work and token aggregation. This is not pure GPU kernel time.
- `cluster_tokens_per_s`: effective tokens divided by step time; `tokens_per_s_per_gpu` additionally divides by world size.

The final throughput for this test is the arithmetic mean of per-step throughput over steps 101–1000, not total tokens divided by total time or a nominal 8192×GBS token count.

## Validation

On ms-swift `d89f1513`, four nodes with eight X10000 GPUs each completed 1000 steps. Steps 101–1000 averaged 3.2222 s/step and 319.21 tokens/s/GPU. Loss changed from 1.34467602 to 1.02861476 with a downward trend and normal fluctuations. Results apply only to this environment and configuration.

Full preprocessing produced 157712 records referencing 81479 unique images, byte-identical to the training dataset. PR preparation only formats Python and narrows the launcher's stop scope. The 1000-step evidence predates those changes but uses the same training configuration and token-counting logic; the stop behavior is validated with isolated processes.
