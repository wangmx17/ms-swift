# MUSA fused AdamW comparison

These scripts preserve the official two-device Qwen2.5-7B-Instruct LoRA DDP recipe
at `d89f151370358b1090bd7ac7df223be222395eca` (`examples/train/multi-gpu/ddp/train.sh`):
rank 8/alpha 32, BF16 base model, self-cognition#1000, batch 1, accumulation 8,
one epoch, learning rate 1e-4 and max length 2048. This is a configuration source,
not evidence of a CUDA run. See [Chinese instructions](README_CN.md).

## Requirements

Install the checkout with `pip install -e .` in a MUSA environment and select two
idle devices. Standard `torch.optim.AdamW(fused=True)` requires torch_musa support
(upstream !1653). The recorded environment also used TE !168 compatibility changes,
datasets 4.8.4 and torchao 0.16.0; other combinations have not been certified here.

**This branch still falls back from fused to ordinary AdamW in
`swift/trainers/arguments.py`.** The validated isolated checkout removed the
four-line block starting with
`if is_torch_musa_available() and self.optim == 'adamw_torch_fused':`.
Use that validated source change or a version that no longer falls back before
running the fused case. These scripts do not patch or install framework code.
The audit callback rejects an unexpected optimizer or fused flag before updates.

The launcher disables training_musa_adaptor autoload and patches, matching the
recorded environment. No additional low-precision scalar multiplication fix is
applied. MODEL may point to a complete local cache of the same model. The default
dataset ID and preprocessing are retained; the original server-specific offline
transport is not included. Prepare a compatible cache before running offline.

## Run and compare

Run from the repository root, keeping environment, model, dataset and devices
identical across cases:

```bash
export MUSA_VISIBLE_DEVICES=0,1
export MODEL=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_ROOT=output/fused_adamw_musa
bash examples/train/optimizers/fused_adamw/train_musa.sh plain
bash examples/train/optimizers/fused_adamw/train_musa.sh fused
bash examples/train/optimizers/fused_adamw/train_musa.sh plain_repeat
```

The process count stays at two; sampling, epochs, checkpoint intervals and other
training hyperparameters remain unchanged. Adaptations are device/model-cache/
output paths, explicit optimizer selection and an observation callback. Existing
case output directories are rejected: choose a fresh OUTPUT_ROOT to rerun.
Swift may append a version/timestamp directory; use the actual output path from
the logs and args.json.

Each run's `optimizer_audit/` contains per-rank effective arguments, optimizer
groups, state tensor dtypes/devices, final steps and logs, plus rank-zero initial
and final trainable parameters. CPU copies and serialization add overhead, so
these runs are not performance benchmarks.

```bash
python examples/train/optimizers/fused_adamw/compare.py \
  /path/to/plain/optimizer_audit /path/to/fused/optimizer_audit \
  --output comparison.json
python examples/train/optimizers/fused_adamw/compare.py \
  /path/to/plain/optimizer_audit /path/to/plain_repeat/optimizer_audit \
  --output repeat.json
```

Report keys `plain` and `fused` label the reference and candidate inputs,
respectively; the second comparison uses the ordinary repeat as candidate.
Review effective argument differences, initial parameters, steps and log alignment
before interpreting errors. The report provides continuous error metrics without
an acceptance threshold. Audit completion does not establish checkpoint reload
or resume correctness.

## Recorded results and limitations

On 2026-11-09, the original scripts ran all three cases for 63/63 steps, exited
with zero and saved checkpoint-63 on the adapted RC5/1104 daily environment.
Trainable LoRA parameters and optimizer moments were FP32, with a BF16 base model
and BF16 autocast. Ordinary versus fused final parameters had max absolute error
0.0012114186 and relative L2 error approximately 0.555%; maximum logged loss
difference was 0.002970314. Ordinary repeat parameters were bitwise identical.
Checkpoint reload was checked, but resume and CUDA comparisons were not run.

The packaged scripts remove site-specific paths, download transport and extra
environment probes. The three GPU runs have not been repeated with this packaging.
Functional success does not establish numerical acceptance or identify the fused
kernel or ordinary scalar multiplication as the source of the remaining gap.
