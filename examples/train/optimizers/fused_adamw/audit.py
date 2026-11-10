import json
import os
import torch
from collections import Counter
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map


class FusedAdamWAudit(TrainerCallback):
    """Reject silent optimizer fallback and save evidence without altering updates."""

    def write(self, args, name, value):
        root = Path(args.output_dir) / 'optimizer_audit'
        root.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get('RANK', '0')
        (root / f'{name}.rank{rank}.json').write_text(json.dumps(value, indent=2, default=str))

    def save_parameters(self, args, model, stage):
        if args.process_index == 0:
            params = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
            torch.save(params, Path(args.output_dir) / 'optimizer_audit' / f'{stage}.pt')

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        opt = getattr(optimizer, 'optimizer', optimizer)
        expected = os.environ['FUSED_ADAMW_EXPECTED'] == 'adamw_torch_fused'
        if not isinstance(opt, torch.optim.AdamW):
            raise RuntimeError(f'Expected torch.optim.AdamW, got {type(opt)}')
        if not opt.param_groups or any(bool(group.get('fused')) != expected for group in opt.param_groups):
            raise RuntimeError('Optimizer fused flag differs from request; check the Swift MUSA fallback.')
        self.write(
            args, 'begin', {
                'args': args.to_dict(),
                'optimizer_class': str(type(opt)),
                'groups': [{
                    k: v
                    for k, v in group.items() if k != 'params'
                } for group in opt.param_groups],
                'trainable_dtypes': dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)),
            })
        self.save_parameters(args, model, 'initial')

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        if state.global_step == 1:
            opt = getattr(optimizer, 'optimizer', optimizer)
            self.write(
                args, 'state', {
                    'tensor_types':
                    dict(
                        Counter(f'{key}:{value.dtype}:{value.device.type}' for entry in opt.state.values()
                                for key, value in entry.items() if isinstance(value, torch.Tensor)))
                })

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.write(args, 'end', {'global_step': state.global_step, 'log_history': state.log_history})
        self.save_parameters(args, model, 'final')


callbacks_map['fused_adamw_audit'] = FusedAdamWAudit
