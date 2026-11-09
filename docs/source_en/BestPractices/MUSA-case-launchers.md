# MUSA MiSS, Qwen3.5 sequence parallelism and FP8 export

Run the three `*_musa.sh` examples from an environment with this checkout installed. They do not change directories. Set MUSA_VISIBLE_DEVICES to idle GPUs; trailing CLI arguments can override model/output paths. The installed adaptor is retained with autoload disabled; torchada handles MUSA.

Bone's existing CLI route now uses MissConfig and requires PEFT 0.20.0. The official Qwen2.5-7B recipe retains 1000 examples, one epoch, batch 1, accumulation 16 and rank 4. Saved adapters have type MISS. Older PEFT versions are not supported by this route.

The SP launcher retains the upstream Qwen3.5-4B/LongAlpaca explicit settings: eight processes, SP4, batch 2, accumulation 8 and length 65535. The upstream attached log has 189 steps, whereas this environment derives 1125 from 12000 examples and three default epochs. This is not an exact reproduction of that CUDA log.

FP8 export requires Transformers 5.2.0 (tested with tokenizers 0.22.2) and the patch utility. Its temporary package overlay applies the included MUSA capability fix without changing site-packages or quantization arithmetic. CUDA device maps retain the NVIDIA capability check.

## Validation, 2026-11-09

Runtime fork d89f151370358b1090bd7ac7df223be222395eca; official recipes d5253cbc5fe3258f143d1826d4078471fa0fc5dc; RC5 image with torch/torch_musa 2.9.0.post1.

- MiSS: 63/63, exit 0, 189 seconds; MISS/r4 weights and optimizer step 63 passed artifact audits. TF5.17.0; incompatible optional torchao0.9.0 removed only in its isolated container.
- SP: stopped intentionally after 308 steps, with finite loss/gradients and checkpoints 250/300. User accepted this partial run; neither a full 1125-step run nor CUDA-189-step equivalence is claimed. Device detection required the torchada switch; compilation warnings are retained.
- FP8 export: exit 0, 200 finite FP8 tensors and 200 scales, identical hashes on repeated export with the candidate patch. Upstream inference remains untested.

These are MUSA functional results, not CUDA numerical comparisons. Recheck the submitted launchers in the target environment.
