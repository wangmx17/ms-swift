# Moore Threads MUSA Support

ms-swift supports Moore Threads GPUs through [torch_musa](https://github.com/MooreThreads/torch_musa) and [torchada](https://github.com/MooreThreads/torchada). torchada redirects the `torch.cuda.*` APIs to `torch.musa`, and swift imports it automatically when both torch_musa and torchada are installed, so most training/inference scripts run on MUSA without modification.

This document uses **Qwen3.5-4B** as an example to demonstrate LoRA and full-parameter fine-tuning, inference, and LoRA merging with ms-swift on MTT S5000.

## 1. Environment Setup

### 1.1 Base Environment

Pull the Moore Threads training image (which ships torch_musa, DeepSpeed, flash-attention, MT-TransformerEngine, etc.) and start a container with the following command:

```bash
IMAGE_NAME=sh-harbor.mthreads.com/mcctest/training-suite:v2.1.7-rc4
docker pull ${IMAGE_NAME}

CONTAINER_NAME=swift_test
docker run -itd --privileged --network=host \
    --env MTHREADS_VISIBLE_DEVICES=all \
    --shm-size=80g \
    -v /data:/data \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME} \
    /bin/bash

docker exec -it ${CONTAINER_NAME} /bin/bash
```

Software versions used for verification in this document:

| Component | Version |
|---|---|
| GPU / Driver | MTT S5000 (80GB) × 8 / Driver 3.3.8-server |
| MUSA Toolkits | 4.3.8 |
| python | 3.10.12 |
| torch / torch_musa | 2.7.1 / 2.7.1.post1 |
| torchada | 0.1.90 |
| transformers | 5.16.1 |
| peft | 0.18.1 |
| deepspeed | 0.19.3 (MUSA build shipped in the image) |

### 1.2 Install ms-swift

Qwen3.5 requires `transformers>=5.2.0`, while the image ships an older transformers (4.57). To avoid breaking the environment that other components in the image (vLLM, sglang, etc.) depend on, we recommend creating a virtual environment that inherits the system packages and installing ms-swift there:

```bash
# Reuse torch/torch_musa/deepspeed from the image; only override the packages that need upgrading in the venv
python -m venv --without-pip --system-site-packages /home/venv_swift
source /home/venv_swift/bin/activate

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
python -m pip install -e . -r requirements/musa.txt qwen_vl_utils decord
# The image ships torchao 0.9.0 (required by sglang); peft>=0.19 requires torchao>=0.16, otherwise LoRA fails
python -m pip install peft==0.18.1
```

### 1.3 Environment Check

- Confirm that torch_musa detects the GPUs:

```bash
python -c "import torch, torch_musa; print(torch.musa.is_available(), torch.musa.device_count())"
# output: True 8
```

- Confirm that swift imports correctly (torchada is enabled automatically):

```bash
python -c "import swift, torch; print(swift.__version__, torch.cuda.device_count())"
# output: 4.6.0.dev0 8
```

- Check GPU utilization and memory usage: `mthreads-gmi`

```
---------------------------------------------------------------------
    mthreads-gmi:2.3.3           Driver Version:3.3.8-server
---------------------------------------------------------------------
ID   Name                 |PCIe                |%GPU  Mem
     Device Type   Perf   |Pcie Lane Width     |Temp  MPC Capable
                                               |      ECC Mode
+-------------------------------------------------------------------+
0    MTT S5000            |00000000:18:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |47C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
...
```

## 2. Examples

Notes:
- Use `MUSA_VISIBLE_DEVICES` to select the visible GPUs; for multi-GPU training set `NPROC_PER_NODE`, and swift launches the processes via torchrun automatically.
- `--model Qwen/Qwen3.5-4B` below downloads the model from ModelScope automatically; it can also be replaced by a local model directory.
- The datasets follow the ms-swift examples (alpaca Chinese/English data plus self-cognition data); `--model_author`/`--model_name` are used for self-cognition training.

### 2.1 Single-GPU LoRA Fine-tuning of Qwen3.5-4B

```bash
PYTORCH_MUSA_ALLOC_CONF='expandable_segments:True' \
MUSA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --model_author swift \
    --model_name swift-robot \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --group_by_length true \
    --output_dir output/Qwen3.5-4B-lora \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
```

The single GPU uses about 15GiB of memory, and 1 epoch (94 steps) takes about 9 minutes.

### 2.2 Multi-GPU Full-parameter Fine-tuning of Qwen3.5-4B (DeepSpeed ZeRO-3)

The image ships a MUSA build of DeepSpeed, so `--deepspeed zero3` can be used directly:

```bash
PYTORCH_MUSA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MUSA_VISIBLE_DEVICES=0,1,2,3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --tuner_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --model_author swift \
    --model_name swift-robot \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --group_by_length true \
    --output_dir output/Qwen3.5-4B-full \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero3
```

With 4-GPU ZeRO-3, each GPU uses about 26.5GiB of memory, and 1 epoch (93 steps) takes about 7 minutes. To use 8 GPUs, set `NPROC_PER_NODE=8` and `MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

### 2.3 Inference and LoRA Merging

Inference with the LoRA weights (transformers backend):

```bash
MUSA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/Qwen3.5-4B-lora/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --enable_thinking false \
    --stream true \
    --max_new_tokens 2048
```

Inference with the full-parameter checkpoint:

```bash
MUSA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/Qwen3.5-4B-full/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --enable_thinking false \
    --stream true \
    --max_new_tokens 2048
```

Inference results after self-cognition training:

```
[QUERY] 你是谁？
[RESPONSE] 我是一个由swift开发的人工智能助手，名为swift-robot。我被设计用来回答您的问题，提供信息，并进行有趣的对话。如果您有任何问题或需要帮助，请随时向我提问。

[QUERY] who are you?
[RESPONSE] I am an artificial intelligence assistant named swift-robot, trained by swift. My purpose is to provide assistance, answer questions, and engage in conversation with users. ...
```

Merge the LoRA weights (written to `checkpoint-xxx-merged`):

```bash
MUSA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/Qwen3.5-4B-lora/vx-xxx/checkpoint-xxx \
    --merge_lora true
```

## 3. CPT → SFT → DPO validation with the new image

Use `registry.mthreads.com/mcctest/training-suite:v2.1.7-rc4-qwen35-qwen38-20260930` with visible MUSA devices and mounted model, dataset and output directories. Run these commands inside the container.
The image includes ms-swift 4.0.2, datasets 3.0.0 and PEFT 0.20.0, but no TRL. Install the cloned source and the image-specific incremental requirements:

```bash
git clone https://github.com/wangmx17/ms-swift.git
cd ms-swift
python -m pip install --no-deps --no-build-isolation -e . -r requirements/qwen35_musa.txt
python -c 'import swift; print(swift.__version__, swift.__file__)'
```

The module path should point to this checkout's `swift/__init__.py`. Changing directories alone does not select the installed source; both scripts check the source path before training.
This installation preserves the image's MUSA PyTorch, Transformers 5.2.0, TE and FlashAttention. It installs datasets 4.8.4, PEFT 0.18.1, TRL 0.29.1 and this checkout's ms-swift. It is not a complete environment recipe for arbitrary base images.

Prepare `/data/wangmx/models/Qwen3.5-4B` and six files under `/data/wangmx/datasets/qwen35_100steps`: `cpt_train.jsonl`, `cpt_val.jsonl`, `sft_train.jsonl`, `sft_val.jsonl`, `dpo_train.jsonl` and `dpo_val.jsonl`. Original files are retained; preprocessing still truncates to length 2048.
Override the model and dataset paths with `QWEN35_MODEL` and `QWEN35_DATA_DIR`, or supply a single-stage input model as the second positional argument. `QWEN35_RUN_DIR` accepts an output directory that does not already exist.

```bash
# Check that the requested GPUs are idle. Start each command in its own terminal or background job.
MUSA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29635 bash run_full.sh all
MUSA_VISIBLE_DEVICES=4 MASTER_PORT=29636 bash run_lora.sh all
```

| Setting | All three full-tuning stages | All three LoRA stages |
|---|---|---|
| GPUs / micro batch / gradient accumulation | 4 / 1 / 4 | 1 / 1 / 16 |
| Learning rate | 1e-5 | 1e-4 |
| Maximum length / default steps | 2048 / 100 per stage | 2048 / 100 per stage |
| DeepSpeed | ZeRO-3 | Disabled |
| Shared settings | BF16, SDPA, gradient checkpointing, warmup 0.05, packing/padding_free disabled | Same |
| DPO | beta defaults to 0.1, rpo_alpha=0.1 | Same |
| LoRA | N/A | rank=8, alpha=32, all-linear |

Full tuning freezes the vision tower and aligner and passes each final model to the next stage. LoRA merges after each stage, using the merged model as the base for a new adapter in the next stage.
The scripts also accept `cpt`, `sft` and `dpo` for a single stage. Standalone DPO requires an SFT model path; LoRA requires the merged SFT model. Each run records logs, checkpoints and its exit code in `experiments.exit`.

The DPO loss configuration follows the official CUDA Qwen2.5 examples: DPO preference loss plus 0.1 times the chosen-response NLL. The model, length, batch and attention settings deliberately differ from those independent examples. This recipe validates the pipeline; it does not establish NVIDIA numerical parity.
The 100-step budget checks startup, training, saving and stage chaining, not convergence or model quality.
