"""Capture real tuning choices and reuse them through FLA's STRICT cache API.

No kernel choices are invented. Unmatched keys retain official autotuning.
This plugin is isolated to a reviewed experiment and does not edit FLA.
"""
import functools
import gc
import hashlib
import inspect
import json
import os
import time
import torch
from pathlib import Path

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_dict(config):
    # FLA's loader in this Triton 3.2 build reconstructs these three fields.
    # Reject any choice that would lose extra semantics in that round trip.
    assert getattr(config, 'num_ctas', 1) == 1
    assert getattr(config, 'pre_hook', None) is None
    assert getattr(config, 'maxnreg', None) is None
    assert getattr(config, 'ir_override', None) is None
    for field in ['num_buffers_warp_spec', 'num_consumer_groups', 'reg_dec_producer', 'reg_inc_consumer']:
        assert getattr(config, field, 0) == 0, field
    import triton
    rebuilt = triton.Config(dict(config.kwargs), num_warps=config.num_warps, num_stages=config.num_stages)
    assert rebuilt.all_kwargs() == config.all_kwargs(), 'FLA loader would change kernel options'
    return {
        'kwargs': dict(config.kwargs),
        'num_warps': config.num_warps,
        'num_stages': config.num_stages,
        'num_ctas': 1
    }


class FlaAutotuneCache(MegatronCallback):

    def on_train_begin(self):
        import triton
        from fla.ops.utils import cache
        from triton.runtime.autotuner import Autotuner

        self.rank = int(os.environ['RANK'])
        self.root = Path(self.args.output_dir)
        self.cache = cache
        self.mode = os.environ['FLA_EXPERIMENT_MODE']
        self.directory = Path(os.environ['FLA_EXPERIMENT_ROOT']) / f'rank-{self.rank}'
        self.lookups = self.hits = self.benches = 0
        self.bench_seconds = 0.0
        self.runtime = {
            'triton': triton.__version__,
            'torch': torch.__version__,
            'cache_source': inspect.getfile(cache),
            'cache_source_sha256': digest(inspect.getfile(cache)),
            'autotuner_source_sha256': digest(inspect.getfile(Autotuner))
        }
        if self.mode == 'reuse':
            manifest = json.loads((self.directory / 'manifest.json').read_text())
            assert manifest['runtime'] == self.runtime, 'Cache runtime fingerprint mismatch'
            for filename, sha in manifest['sources'].items():
                assert digest(filename) == sha, ('Kernel source changed', filename)
            for name, sha in manifest['files'].items():
                assert digest(self.directory / name) == sha, ('Cache file changed', name)
            os.environ['FLA_CONFIG_DIR'] = str(self.directory)
            cache.FLA_CACHE_MODE = cache.FlaCacheMode.STRICT
            cache.load_config_file.cache_clear()
        else:
            assert self.mode == 'capture'
            assert cache.FLA_CACHE_MODE == cache.FlaCacheMode.DISABLED, cache.FLA_CACHE_MODE

        owner = self
        original_lookup = cache.CachedAutotuner.maybe_load_cached_config

        @functools.wraps(original_lookup)
        def lookup(tuner, key):
            with torch.profiler.record_function('fla_autotune_cache_lookup'):
                result = original_lookup(tuner, key)
            owner.lookups += 1
            hit = key.autotune_key in tuner.cache
            owner.hits += int(hit)
            owner.write({'event': 'lookup', 'kernel': tuner.kernel_name, 'key': key.autotune_key, 'hit': hit})
            return result

        original_bench = Autotuner._bench

        @functools.wraps(original_bench)
        def bench(tuner, *args, **kwargs):
            start = time.perf_counter()
            try:
                with torch.profiler.record_function('triton_autotune_benchmark'):
                    return original_bench(tuner, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                owner.benches += 1
                owner.bench_seconds += elapsed
                owner.write({'event': 'bench', 'kernel': tuner.base_fn.__name__, 'seconds': elapsed})

        cache.CachedAutotuner.maybe_load_cached_config = lookup
        Autotuner._bench = bench
        self.write({
            'event': 'installed',
            'mode': self.mode,
            'directory': str(self.directory),
            'official_mode': cache.FLA_CACHE_MODE.value,
            'runtime': self.runtime,
            'launch_blocking': [os.environ.get(k) for k in ['MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING']]
        })

    def write(self, row):
        row.update(iteration=self.state.iteration, time=time.time())
        with (self.root / f'fla-cache-events-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')

    def on_step_end(self):
        self.write({
            'event': 'step_end',
            'lookups': self.lookups,
            'hits': self.hits,
            'bench_calls': self.benches,
            'bench_seconds': self.bench_seconds
        })

    def on_train_end(self):
        if self.mode != 'capture':
            return
        # Read the selected configs after all timed steps; never tune extra
        # kernels or retain model tensors to construct this cache.
        self.directory.mkdir(parents=True, exist_ok=False)
        files, sources, by_name = {}, {}, {}
        for tuner in gc.get_objects():
            if not isinstance(tuner, self.cache.CachedAutotuner) or not tuner.cache:
                continue
            filename = inspect.getfile(tuner.base_fn)
            sources[filename] = digest(filename)
            name = tuner.kernel_name
            data = by_name.setdefault(name, {
                'kernel_name': name,
                'triton_version': self.runtime['triton'],
                'autotune_entries': {}
            })
            for key, config in tuner.cache.items():
                entry = {'autotune_key': list(key), 'config': config_dict(config)}
                h = self.cache.AutotuneKey.key_hash(key)
                if h in data['autotune_entries']:
                    assert data['autotune_entries'][h] == entry, ('Conflicting kernel/key choices', name, key)
                data['autotune_entries'][h] = entry
        assert by_name, 'No actual CachedAutotuner choices captured'
        for name, data in by_name.items():
            path = self.directory / (name + '.json')
            parsed = self.cache.KernelConfigFile.from_dict(path, data)
            assert parsed is not None
            for entry in data['autotune_entries'].values():
                key = self.cache.AutotuneKey(tuple(entry['autotune_key']))
                assert parsed.lookup_exact(key) == entry
            path.write_text(json.dumps(data, indent=2))
            files[path.name] = digest(path)
        manifest = {
            'rank': self.rank,
            'runtime': self.runtime,
            'sources': sources,
            'files': files,
            'source_run': str(self.root),
            'entries': sum(len(d['autotune_entries']) for d in by_name.values())
        }
        (self.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        self.write({'event': 'captured', 'files': len(files), 'entries': manifest['entries']})


megatron_callbacks_map['fla_autotune_cache'] = FlaAutotuneCache
