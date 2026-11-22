"""Preallocate actual gradient-statistics and coalesced DDP RS resources.

Only runs before training. Reduces an independent zero scalar, never model
parameters/gradients. Avoids lazy MCCL allocation at the first backward peak.
"""
import json
import os
from pathlib import Path
import torch
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

class GradStatsWarmup(MegatronCallback):
    def warm_gradient_buckets(self):
        from torch.distributed.distributed_c10d import _coalescing_manager
        reports = []
        for model in self.trainer.wrapped_models:
            assert hasattr(model, 'bucket_groups'), type(model)
            for name in ['bucket_groups', 'expert_parallel_bucket_groups']:
                for bg in getattr(model, name):
                    assert bg.ddp_config.use_distributed_optimizer
                    assert not bg.ddp_config.reduce_scatter_with_fp32_accumulation
                    group = bg.intra_distributed_optimizer_instance_group
                    size = torch.distributed.get_world_size(group)
                    rank = torch.distributed.get_rank(group)
                    sizes = [b.grad_data.numel() for b in bg.buckets]
                    dtypes = [str(b.grad_data.dtype) for b in bg.buckets]
                    versions = [b.grad_data._version for b in bg.buckets]
                    row = {'kind': name, 'size': size,
                           'ranks': torch.distributed.get_process_group_ranks(group),
                           'numel': sizes, 'dtypes': dtypes,
                           'dummy_bytes': sum(b.grad_data.numel()*b.grad_data.element_size() for b in bg.buckets)}
                    print('[bucket-warmup begin]', row, flush=True)
                    # Separate zeros, same shapes and coalesced RS as DDP. Never
                    # call start_grad_sync or write to actual parameter/gradient buffers.
                    inputs = [torch.zeros_like(b.grad_data) for b in bg.buckets]
                    outputs = [x.view(-1).chunk(size)[rank] for x in inputs]
                    op = (torch.distributed.ReduceOp.AVG if bg.ddp_config.average_in_collective
                          else torch.distributed.ReduceOp.SUM)
                    with _coalescing_manager(group, async_ops=False):
                        for inp, out in zip(inputs, outputs):
                            torch.distributed.reduce_scatter_tensor(out, inp, op=op, group=group)
                    torch.cuda.synchronize()
                    assert all(torch.count_nonzero(x).item() == 0 for x in outputs)
                    assert versions == [b.grad_data._version for b in bg.buckets]
                    reports.append(row)
                    del inp, out, inputs, outputs
        assert reports, 'No DDP gradient bucket groups found'
        # Release only temporary dummy tensors' allocator cache at startup.
        torch.cuda.empty_cache()
        return reports

    def on_train_begin(self):
        optimizer = self.trainer.optimizer
        optimizers = getattr(optimizer, 'chained_optimizers', [optimizer])
        groups = []
        for opt in optimizers:
            group = opt.get_grad_stats_parallel_group()
            group = torch.distributed.group.WORLD if group is None else group
            if not any(group is old for old in groups):
                groups.append(group)
        dummy = torch.zeros(1, device=torch.device("musa", torch.cuda.current_device()), dtype=torch.float32)
        before = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
        reports = []
        for group in groups:
            torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM, group=group)
            reports.append({'world': group is torch.distributed.group.WORLD,
                            'size': torch.distributed.get_world_size(group),
                            'ranks': torch.distributed.get_process_group_ranks(group)})
        # Startup only, consistent with the existing communicator warm-up.
        torch.cuda.synchronize()
        assert dummy.item() == 0.0
        bucket_reports = self.warm_gradient_buckets()
        row = {'rank': int(os.environ['RANK']), 'groups': reports, 'bucket_groups': bucket_reports, 'before': before,
               'after': {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()},
               'launch_blocking': [os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']]}
        (Path(self.args.output_dir)/f'gradstats-warmup-rank{row["rank"]}.json').write_text(json.dumps(row, indent=2))
        print(f'[gradstats-warmup rank={row["rank"]}] {reports}', flush=True)

megatron_callbacks_map['gdn_gradstats_warmup'] = GradStatsWarmup
