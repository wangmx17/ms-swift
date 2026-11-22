"""Process-local metadata-read replacement with pinned original source.

Only three host read expressions change. Original shape/endpoint/length guards
remain verbatim. Install before the existing partial-dispatch callback builds
its closures; restore is available to isolated numerical probes.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import linecache
import sys
from pathlib import Path


def install(cache, prepared: Path, partial_globals: dict):
    manifest = json.loads((prepared / 'offsets-source-manifest.json').read_text())
    official = importlib.import_module('training_musa_adaptor.ops.gated_delta_rule')
    varlen = importlib.import_module('torch_musa_op_extension.attention.tilelang.flash_linear_attention.gdn_varlen')
    partial = importlib.import_module('partial_gdn_dispatch')
    originals = {
        'guard': official._supported_by_tilelang,
        'varlen': varlen.varlen_lengths,
        'partial': partial.make_partial_dispatcher
    }
    assert partial_globals['make_partial_dispatcher'] is originals['partial']
    assert not getattr(originals['guard'], '_bwy_offsets_cache', False)
    replacements = {}
    channel_counts = {key: 0 for key in originals}
    for key, original in originals.items():
        spec = manifest[key]
        source = inspect.getsource(original)
        assert hashlib.sha256(source.encode()).hexdigest() == spec['original_source_sha256'], key
        assert hashlib.sha256(Path(
            inspect.getfile(original)).read_bytes()).hexdigest() == spec['original_file_sha256'], key
        candidate = prepared / spec['candidate_file']
        code = 'from __future__ import annotations' + chr(10) + source.replace(spec['old_expression'],
                                                                               spec['new_expression'])
        assert hashlib.sha256(code.encode()).hexdigest() == spec['candidate_sha256'], key
        assert source.count(spec['old_expression']) == 1
        assert code == 'from __future__ import annotations\n' + source.replace(spec['old_expression'],
                                                                               spec['new_expression']), key
        linecache.cache[str(candidate)] = (len(code), None, code.splitlines(keepends=True), str(candidate))
        namespace = dict(original.__globals__)

        def read_for(channel):
            reader = (lambda tensor: tensor.tolist()) if channel == 'varlen' else (
                lambda tensor: tensor.detach().cpu().tolist())

            def read(tensor):
                channel_counts[channel] += 1
                return cache.read(tensor, reader)

            return read

        namespace['_bwy_offsets_read'] = read_for(key)
        exec(compile(code, str(candidate), 'exec', dont_inherit=True), namespace)
        fn = namespace[original.__name__]
        fn._bwy_offsets_cache = True
        replacements[key] = fn
    changes = []

    def replace(namespace, name, expected, new):
        assert namespace[name] is expected, (name, 'Unexpected binding')
        changes.append((namespace, name, expected, new))
        namespace[name] = new

    replace(vars(official), '_supported_by_tilelang', originals['guard'], replacements['guard'])
    aliases = []
    # Include aliases imported by both forward and backward implementations.
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith('torch_musa_op_extension.'):
            continue
        for key, value in list(vars(module).items()):
            if value is originals['varlen']:
                replace(vars(module), key, value, replacements['varlen'])
                aliases.append(name + ':' + key)
    assert aliases
    replace(vars(partial), 'make_partial_dispatcher', originals['partial'], replacements['partial'])
    replace(partial_globals, 'make_partial_dispatcher', originals['partial'], replacements['partial'])

    def restore():
        for namespace, name, old, new in reversed(changes):
            assert namespace[name] is new, (name, 'Another mutation prevents restoration')
            namespace[name] = old

    report = {
        'varlen_aliases': aliases,
        'replacement_sources': {
            key: str(prepared / manifest[key]['candidate_file'])
            for key in replacements
        },
        'original_source_sha256': {
            key: manifest[key]['original_source_sha256']
            for key in replacements
        },
        'candidate_sha256': {
            key: manifest[key]['candidate_sha256']
            for key in replacements
        }
    }
    return restore, report, channel_counts
