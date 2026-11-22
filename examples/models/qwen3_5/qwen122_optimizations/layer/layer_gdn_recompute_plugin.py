"""Prepared G0d: retain handoffs inside block41, recompute them in the last five GDN layers.

Process-local candidate only. Requires accepted exactness evidence for the two
GDN implementations and a separate controlled training experiment.
"""
from collections import Counter
from contextvars import ContextVar
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


class LayerGDNRecomputeAudit(MegatronCallback):
    def on_train_begin(self):
        args = self.args
        assert args.pipeline_model_parallel_size == 1
        assert args.recompute_granularity == 'full' and args.recompute_method == 'block'
        assert args.recompute_num_layers == 41
        assert not getattr(args, 'fp8', None) and not getattr(args, 'fp4', None)
        modules = [m for model in self.trainer.unwrapped_models for m in model.modules()
                   if type(m).__module__ == 'mcore_bridge.model.modules.gated_delta_net'
                   and type(m).__name__ == 'GatedDeltaNet']
        layer_numbers = sorted(m.layer_number for m in modules)
        assert layer_numbers == [i for i in range(1, 49) if i % 4]
        tail = {42, 43, 45, 46, 47}
        assert {i for i in layer_numbers if i > 41} == tail
        mode = os.environ['GDN_INTERNAL_RECOMPUTE_MODE']
        assert mode in ('all', 'tail5')
        # Use the same audit wrapper in both sides of the controlled experiment.
        selected = set(layer_numbers) if mode == 'all' else tail
        cls = type(modules[0])
        assert all(type(m) is cls for m in modules)

        backend = importlib.import_module(
            'torch_musa_op_extension.attention.tilelang.flash_linear_attention.gdn_chunk_bwd')
        expected_recompute = '4d1c8e243a4cc09aad2edeed8c63ca753131e6644ae78ff5e5d34b0dddf07a54'
        assert hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest() == expected_recompute
        path = Path(os.environ['GDN_RETAINED_STATE_SOURCE'])
        expected_retained = '6138f45abd028d7296d7af1607b1c65d0172905ec7b4ed572c7b951f37443dce'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_retained
        spec = importlib.util.spec_from_file_location(backend.__package__+'.retained_for_layer_scope', path)
        retained = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retained)
        # This module is imported after the inherited offsets callback. Preserve
        # its already-installed read implementation in both layer modes.
        assert retained.varlen_lengths is backend.varlen_lengths

        front = importlib.import_module('torch_musa_op_extension.attention.gated_delta_net')
        ops = front._REGISTRY.resolve('tilelang')
        assert ops.gdn is backend.gdn_chunk, 'Unexpected pre-existing backend wrapper'
        current_layer = ContextVar('bwy_gdn_layer', default=None)
        self.calls = Counter()

        def choose(*inputs, **kwargs):
            layer = current_layer.get()
            assert layer in layer_numbers, 'GDN call outside audited model-layer context'
            use_recompute = layer in selected
            kind = 'recompute' if use_recompute else 'retain'
            self.calls[(layer, kind, torch.is_grad_enabled())] += 1
            fn = backend.gdn_chunk if use_recompute else retained.gdn_chunk
            with torch.profiler.record_function('gdn_state_'+kind):
                return fn(*inputs, **kwargs)

        front._REGISTRY._impls['tilelang'] = front.BackendOps(gdn=choose, gdn_fwd=ops.gdn_fwd)
        original_forward = cls.forward

        @functools.wraps(original_forward)
        def layer_forward(module, *inputs, **kwargs):
            token = current_layer.set(module.layer_number)
            try:
                return original_forward(module, *inputs, **kwargs)
            finally:
                current_layer.reset(token)

        cls.forward = layer_forward
        self.root = Path(self.args.output_dir)
        self.rank = int(os.environ['RANK'])
        (self.root/f'gdn-layer-recompute-rank{self.rank}.json').write_text(json.dumps({
            'rank': self.rank, 'mode': mode, 'layers': layer_numbers, 'recompute_layers': sorted(selected),
            'retained_source': str(path), 'retained_sha256': expected_retained,
            'recompute_sha256': expected_recompute, 'outer_recompute': 'full/block41 unchanged',
            'plugin_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'launch_blocking': [os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']],
            'registry_installed': front._REGISTRY.resolve('tilelang').gdn is choose,
            'layer_wrapper_installed': cls.forward is layer_forward,
            'module_count': len(modules),
            'shared_varlen_binding': retained.varlen_lengths is backend.varlen_lengths,
            'varlen_source': backend.varlen_lengths.__code__.co_filename,
        }, indent=2))

    def on_step_begin(self):
        self.calls.clear()

    def on_step_end(self):
        row = {'step': self.state.iteration, 'calls': [
            dict(layer=k[0], mode=k[1], grad_enabled=k[2], count=v)
            for k, v in sorted(self.calls.items())]}
        with (self.root/f'gdn-layer-calls-rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')


megatron_callbacks_map['layer_gdn_recompute_audit'] = LayerGDNRecomputeAudit
