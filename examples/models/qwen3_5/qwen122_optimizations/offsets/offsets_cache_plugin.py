"""G1b candidate: share bounded host-offset reads across original consumers."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

ROOT = Path(__file__).parent


def load(name, filename):
    assert name not in sys.modules, name
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


implementation = load('bwy_shared_offsets_cache', 'offsets_cache.py')
installer = load('bwy_shared_offsets_install', 'offsets_cache_install.py')


class SharedOffsetsCache(MegatronCallback):

    def on_train_begin(self):
        import mcore_bridge.model.modules.gated_delta_net as bridge
        import megatron.core.ssm.gated_delta_net as mcore
        assert not getattr(bridge.chunk_gated_delta_rule, '_bwy_partial_gdn',
                           False), 'Must run before partial_gdn callback'
        assert not getattr(mcore.chunk_gated_delta_rule, '_bwy_partial_gdn', False)
        assert 'partial_gdn' in megatron_callbacks_map
        partial_globals = megatron_callbacks_map['partial_gdn'].on_train_begin.__globals__
        self.cache = implementation.OffsetsCache(max_entries=64, max_elements=4096)
        self.restore, self.report, self.channel_counts = installer.install(self.cache, ROOT, partial_globals)
        self.root = Path(self.args.output_dir)
        self.rank = int(os.environ['RANK'])
        self.first_step = True
        self.report.update(
            rank=self.rank,
            max_entries=64,
            max_elements=4096,
            launch_blocking=[os.getenv(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']],
            files={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in ROOT.iterdir() if p.suffix in ['.py', '.json']
            })
        (self.root / f'offsets-effective-rank{self.rank}.json').write_text(json.dumps(self.report, indent=2))

    def on_step_begin(self):
        if self.first_step:
            import mcore_bridge.model.modules.gated_delta_net as bridge
            import megatron.core.ssm.gated_delta_net as mcore
            observed = []
            expected = str(ROOT / 'partial_cached.py')
            for module in [bridge, mcore]:
                fn = module.chunk_gated_delta_rule
                assert getattr(fn, '_bwy_partial_gdn', False)
                chain = []
                while True:
                    chain.append(getattr(getattr(fn, '__code__', None), 'co_filename', None))
                    if not hasattr(fn, '__wrapped__'):
                        break
                    fn = fn.__wrapped__
                assert expected in chain, (module.__name__, chain)
                observed.append(dict(module=module.__name__, code_files=chain))
            (self.root / f'offsets-bindings-rank{self.rank}.json').write_text(json.dumps(observed, indent=2))
            self.first_step = False
        self.cache.counts.clear()
        for key in self.channel_counts:
            self.channel_counts[key] = 0

    def on_step_end(self):
        with (self.root / f'offsets-calls-rank{self.rank}.jsonl').open('a') as stream:
            stream.write(
                json.dumps(
                    dict(
                        step=self.state.iteration,
                        counts=dict(self.cache.counts),
                        channels=self.channel_counts,
                        entries=len(self.cache.entries))) + '\n')


megatron_callbacks_map['shared_offsets_cache'] = SharedOffsetsCache
