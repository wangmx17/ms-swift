"""Isolated G1a experiment; installed only by an explicit external-plugin path."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

path = Path(__file__).with_name('resolver_cache.py')
spec = importlib.util.spec_from_file_location('bwy_resolver_cache_impl', path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)


class ResolverCacheAudit(MegatronCallback):

    def on_train_begin(self):
        self.root = Path(self.args.output_dir)
        self.rank = int(os.environ['RANK'])
        classes = {}
        for model in self.trainer.unwrapped_models:
            for module in model.modules():
                if (type(module).__module__ == 'mcore_bridge.model.modules.gated_delta_net'
                        and hasattr(type(module), '_resolve_cu_seqlens')):
                    classes[type(module)] = True
        assert len(classes) == 1, list(classes)
        self.caches = []
        reports = []
        for cls in classes:
            original = cls._resolve_cu_seqlens
            source = inspect.getsource(original)
            # Reject an already patched resolver rather than stacking wrappers.
            assert 'seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]' in source
            source_sha = hashlib.sha256(source.encode()).hexdigest()
            assert source_sha == '17b2b774a3d7d01aba0ded1e495c01a7b1ba678a4b27cc131c82a6d13f0853b9', source_sha
            cache = impl.ResolverCache(original, max_entries=16)
            cls._resolve_cu_seqlens = staticmethod(cache)
            assert cls._resolve_cu_seqlens is cache
            self.caches.append(cache)
            reports.append({
                'class': cls.__name__,
                'original_source_sha256': source_sha,
                'installed': cls._resolve_cu_seqlens is cache
            })
        row = {
            'rank':
            self.rank,
            'classes':
            reports,
            'implementation_sha256':
            hashlib.sha256(path.read_bytes()).hexdigest(),
            'plugin_sha256':
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'launch_blocking': [os.getenv('MUSA_LAUNCH_BLOCKING'),
                                os.getenv('CUDA_LAUNCH_BLOCKING')],
            'max_entries':
            16,
            'invalidation':
            'object identity, input/output version and metadata',
            'limitation':
            'Untracked external .data/storage writes are outside this cache contract; '
            'training sequence metadata is read-only.'
        }
        (self.root / f'resolver-effective-rank{self.rank}.json').write_text(json.dumps(row, indent=2))

    def on_step_begin(self):
        for cache in self.caches:
            cache.counts.clear()

    def on_step_end(self):
        row = {
            'step': self.state.iteration,
            'caches': [{
                'counts': dict(cache.counts),
                'entries': len(cache.entries)
            } for cache in self.caches]
        }
        with (self.root / f'resolver-calls-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')


megatron_callbacks_map['resolver_cache_audit'] = ResolverCacheAudit
