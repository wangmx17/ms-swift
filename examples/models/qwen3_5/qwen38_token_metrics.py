"""Qwen SFT input-token throughput; PP=CP=1, one log per optimizer step.

Counts post-template/post-truncation sequence lengths before alignment padding.
Includes prompt, answer, special tokens and expanded image-token positions.
The separate loss_tokens count includes only labels != -100.
"""
import json
import os
import time
import torch
import torch.distributed as dist
from megatron.core import mpu
from pathlib import Path

from swift.megatron.callbacks import megatron_callbacks_map
from swift.megatron.callbacks.print import PrintCallback
from swift.megatron.trainers.base import BaseMegatronTrainer
from swift.utils import get_current_device


def _get_batch_with_token_count(self, data_iterator, vp_stage=None):
    # Count only batches consumed by the training loop, never prefetched batches.
    data = next(data_iterator)
    if getattr(self, '_input_token_capture', False):
        lengths = data.get('seq_lens')
        if not isinstance(lengths, (list, tuple)) or not lengths:
            raise RuntimeError('Token metrics requires CPU seq_lens from the Megatron collator')
        lengths = [int(n) for n in lengths]
        if min(lengths) <= 0:
            raise RuntimeError('Invalid seq_lens for token metrics')
        labels = data['labels']
        if labels.device.type != 'cpu':
            raise RuntimeError('Token metrics expects labels before device transfer')
        supervised = int(torch.count_nonzero(labels != -100))
        self._input_token_counts[0] += sum(lengths)
        self._input_token_counts[1] += supervised
        self._input_token_counts[2] += len(lengths)
        if self._input_token_audit:
            self._input_token_lengths.extend(lengths)
    return self._prepare_batch(data, vp_stage)


class InputTokenPrintCallback(PrintCallback):

    def on_train_begin(self):
        a = self.args
        if (a.pipeline_model_parallel_size != 1 or a.context_parallel_size != 1 or a.logging_steps != 1
                or a.task_type != 'causal_lm' or a.packing or a.eval_iters != 0):
            raise RuntimeError('Token metrics supports causal SFT with PP1/CP1, packing=false, '
                               'logging_steps=1 and eval_iters=0 only')
        super().on_train_begin()
        self.trainer._input_token_capture = False
        self.trainer._input_token_audit = os.environ.get('QWEN_TOKEN_AUDIT_DIR', '')
        self.world_size = dist.get_world_size()
        self.last_iteration = self.state.iteration
        # Align the initial measurement boundary; wall-clock offsets do not matter.
        dist.barrier()
        self.last_time = time.perf_counter()

    def on_step_begin(self):
        self.trainer._input_token_counts = [0, 0, 0]
        self.trainer._input_token_lengths = []
        self.trainer._input_token_capture = True

    def on_step_end(self):
        self.trainer._input_token_capture = False
        super().on_step_end()

    def on_log(self, logs):
        if self.state.iteration != self.last_iteration + 1:
            raise RuntimeError('Token metrics requires exactly one log per training step')
        counts = torch.tensor(self.trainer._input_token_counts, dtype=torch.int64, device=get_current_device())
        # Each TP rank reduces its own DP group: TP replicas are never added together.
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
        effective, supervised, samples = [int(n) for n in counts.tolist()]
        if samples != self.args.global_batch_size or not 0 <= supervised <= effective:
            raise RuntimeError(f'Invalid global token counts: {effective}, {supervised}, {samples}')
        now = time.perf_counter()
        local_seconds = now - self.last_time
        duration = torch.tensor(local_seconds, dtype=torch.float32, device=get_current_device())
        # Slowest rank interval; includes data loading, training, GC, prior log work,
        # and token-count collection. This reduction's overhead enters the next step.
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        seconds = float(duration.item())
        if seconds <= 0 or seconds < local_seconds * (1 - 1e-6):
            raise RuntimeError('Invalid MAX reduction for step interval')
        self.last_time = now
        self.last_iteration = self.state.iteration
        logs.update(
            effective_tokens=effective,
            loss_tokens=supervised,
            global_samples=samples,
            step_time_s=seconds,
            cluster_tokens_per_s=effective / seconds,
            tokens_per_s_per_gpu=effective / seconds / self.world_size)
        audit_dir = self.trainer._input_token_audit
        if audit_dir and mpu.get_tensor_model_parallel_rank() == 0:
            p = Path(audit_dir)
            p.mkdir(parents=True, exist_ok=True)
            row = dict(
                iteration=self.state.iteration,
                rank=dist.get_rank(),
                dp_rank=mpu.get_data_parallel_rank(),
                seq_lens=self.trainer._input_token_lengths,
                local_counts=self.trainer._input_token_counts)
            with (p / f'dp{row["dp_rank"]}.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        super().on_log(logs)


BaseMegatronTrainer.get_batch = _get_batch_with_token_count
megatron_callbacks_map['print'] = InputTokenPrintCallback
