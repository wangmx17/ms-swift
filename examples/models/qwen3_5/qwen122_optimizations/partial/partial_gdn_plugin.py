"""Isolated G0c2 permutation trial callback; not enabled by the formal entry."""
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

ROOT = Path(__file__).parent
# Load the exact reviewed planner/dispatcher adjacent to this trial callback.
for name in ['partial_gdn_plan', 'token_permutation_plan', 'partial_gdn_dispatch']:
    assert name not in sys.modules, ('Unexpected preloaded experimental module', name)
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
make_partial_dispatcher = sys.modules['partial_gdn_dispatch'].make_partial_dispatcher


class PartialGdn(MegatronCallback):
    def on_train_begin(self):
        import mcore_bridge.model.modules.gated_delta_net as bridge
        import megatron.core.ssm.gated_delta_net as mcore
        from training_musa_adaptor.ops import gated_delta_rule as official
        self.root = Path(self.args.output_dir)
        self.rank = int(os.environ['RANK'])
        self.calls = {}
        bindings = []
        for module in [bridge, mcore]:
            original = module.chunk_gated_delta_rule
            assert getattr(original, official.MARKER, False), module.__name__
            assert not getattr(original, '_bwy_partial_gdn', False)
            candidate = make_partial_dispatcher(original)
            key = module.__name__
            self.calls[key] = 0

            def observe(fn, name):
                @functools.wraps(fn)
                def call(*args, **kwargs):
                    self.calls[name] += 1
                    return fn(*args, **kwargs)
                call._bwy_partial_gdn = True
                return call

            installed = observe(candidate, key)
            module.chunk_gated_delta_rule = installed
            bindings.append(dict(module=key, installed=module.chunk_gated_delta_rule is installed,
                                 tilelang_marker=bool(getattr(installed, official.MARKER, False)),
                                 partial_marker=installed._bwy_partial_gdn))
        row = dict(rank=self.rank, bindings=bindings,
                   sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__), ROOT/'partial_gdn_plan.py', ROOT/'token_permutation_plan.py', ROOT/'partial_gdn_dispatch.py']},
                   launch_blocking=[os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']])
        (self.root/f'partial-effective-rank{self.rank}.json').write_text(json.dumps(row, indent=2))

    def on_step_begin(self):
        self.calls = dict.fromkeys(self.calls, 0)

    def on_step_end(self):
        with (self.root/f'partial-calls-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.state.iteration, calls=self.calls))+'\n')


megatron_callbacks_map['partial_gdn'] = PartialGdn
