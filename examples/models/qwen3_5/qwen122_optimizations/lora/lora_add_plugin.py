"""Isolated L1 candidate: scaling=4 LoRA pointwise mul+add -> add(alpha)."""
import ast
import hashlib
import inspect
import json
import linecache
import os
import textwrap
from pathlib import Path
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


class LoraAddAudit(MegatronCallback):
    def on_train_begin(self):
        from mcore_bridge.tuners.lora import LoraParallelLinear
        mode = os.environ.get('LORA_ADD_MODE', 'native')
        assert mode in ['native', 'fused'], mode
        assert not os.environ.get('MUSA_LAUNCH_BLOCKING')
        assert not os.environ.get('CUDA_LAUNCH_BLOCKING')
        modules = [m for model in self.trainer.unwrapped_models
                   for m in model.modules() if isinstance(m, LoraParallelLinear)]
        assert modules, 'No actual LoraParallelLinear modules'
        for m in modules:
            assert all(float(v) == 4.0 for v in m.scaling.values()), m.scaling
        dtypes = sorted({str(p.dtype) for m in modules
                         for group in [m.lora_A, m.lora_B]
                         for layer in group.values() for p in layer.parameters()})
        assert dtypes == ['torch.bfloat16'], dtypes
        original = LoraParallelLinear.forward
        assert not original.__closure__, 'Unexpected closure around original forward'
        source = textwrap.dedent(inspect.getsource(original))
        needle = '            lora_result = lora_result * scaling\n            result = result + lora_result\n'
        assert source.count(needle) == 1, 'Current LoRA source does not match reviewed target'
        candidate = source.replace(needle, '            result = torch.add(result, lora_result, alpha=scaling)\n')
        ast.parse(candidate)
        filename = str(Path(self.args.output_dir)/'lora_add_effective_forward.py')
        if int(os.environ['RANK']) == 0:
            Path(filename).write_text(candidate)
        # Keep the generated method visible in Python-stack traces without
        # changing the installed package on any node.
        linecache.cache[filename] = (len(candidate), None, candidate.splitlines(True), filename)
        namespace = {}
        exec(compile(candidate, filename, 'exec'), original.__globals__, namespace)
        patched = namespace['forward']
        patched.__module__ = original.__module__
        patched.__qualname__ = original.__qualname__
        if mode == 'fused':
            LoraParallelLinear.forward = patched
        row = {'rank': int(os.environ['RANK']), 'module_count': len(modules),
               'mode': mode, 'lora_dtypes': dtypes,
               'launch_blocking': [os.environ.get(k) for k in
                                   ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']],
               'plugin_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scaling': 4.0, 'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'candidate_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
               'effective_sha256': hashlib.sha256((candidate if mode == 'fused' else source).encode()).hexdigest(),
               'candidate_source': filename}
        (Path(self.args.output_dir)/f'lora-add-rank{row["rank"]}.json').write_text(json.dumps(row, indent=2))


megatron_callbacks_map['lora_add_audit'] = LoraAddAudit
