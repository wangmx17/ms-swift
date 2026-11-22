"""Prepared revision: one token permutation instead of per-sequence slicing/cat.

Not installed in training. No relaxation of the existing TileLang guard.
The same outer math, q/k normalization, scale and token order are retained.
"""
import functools
from partial_gdn_plan import partition
from token_permutation_plan import token_permutation_plan


def make_partial_dispatcher(existing):
    import torch
    from training_musa_adaptor.ops import gated_delta_rule as official

    assert getattr(existing, official.MARKER, False)
    original = existing.__wrapped__
    front, dense_guard, varlen_guard = official._tilelang_stack()

    @functools.wraps(existing)
    def dispatched(*args, **kwargs):
        call = official._normalized_call(args, kwargs)
        if call is None or call.get('initial_state') is not None or call.get('output_final_state', False):
            return existing(*args, **kwargs)
        q, k, v, g, beta = [call[x] for x in ['q', 'k', 'v', 'g', 'beta']]
        cu = call.get('cu_seqlens')
        if (not all(torch.is_tensor(x) for x in [q, k, v, g, beta, cu]) or q.ndim != 4 or k.ndim != 4 or v.ndim != 4
                or g.ndim != 3 or beta.ndim != 3 or q.shape[0] != 1 or q.shape != k.shape
                or any(x.shape[:2] != q.shape[:2] for x in [v, g, beta]) or g.shape != beta.shape
                or g.shape[2] != v.shape[2] or cu.ndim != 1 or cu.dtype not in (torch.int32, torch.int64)):
            return existing(*args, **kwargs)
        with torch.profiler.record_function('gdn_partial_partition'):
            offsets = cu.detach().cpu().tolist()
            if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != q.shape[1]:
                return existing(*args, **kwargs)
            lengths = [b - a for a, b in zip(offsets, offsets[1:])]
            groups = partition(lengths, v.shape[2])
        if groups is None:
            return existing(*args, **kwargs)

        # Validate metadata through the original guard before gathering large
        # activations. Views retain the original dtype/device/head dimensions.
        for backend, ids in groups:
            if backend == 'tilelang':
                first = ids[0]
                probe = dict(call)
                for name in ['q', 'k', 'v', 'g', 'beta']:
                    probe[name] = call[name][:, offsets[first]:offsets[first + 1]]
                probe['cu_seqlens'] = torch.tensor([0, lengths[first]], device=cu.device, dtype=cu.dtype)
                if not official._supported_by_tilelang(probe, dense_guard, varlen_guard):
                    return existing(*args, **kwargs)
                if not varlen_guard([lengths[i]
                                     for i in ids], v.shape[2], k.shape[3], v.shape[3], call.get('chunk_size', 64)):
                    return existing(*args, **kwargs)

        # The plan is a bijection: each original token occurs exactly once.
        # Gather once per input, then expose each whole backend group as a view.
        # This avoids a full-size SliceBackward buffer for every sequence.
        with torch.profiler.record_function('gdn_partial_permute_inputs'):
            permutation, inverse, group_offsets = token_permutation_plan(offsets, groups)
            indices = torch.tensor(permutation, device=cu.device, dtype=torch.long)
            restore = torch.tensor(inverse, device=cu.device, dtype=torch.long)
            ordered = {name: torch.index_select(call[name], 1, indices) for name in ['q', 'k', 'v', 'g', 'beta']}
        outputs = []
        for gi, (backend, ids) in enumerate(groups):
            with torch.profiler.record_function('gdn_partial_' + backend):
                group = dict(call)
                left, right = group_offsets[gi:gi + 2]
                for name in ['q', 'k', 'v', 'g', 'beta']:
                    group[name] = ordered[name].narrow(1, left, right - left)
                ends = [0]
                for i in ids:
                    ends.append(ends[-1] + lengths[i])
                group['cu_seqlens'] = torch.tensor(ends, device=cu.device, dtype=cu.dtype)
                if backend == 'tilelang':
                    output, state = front(
                        *(group[x] for x in ['q', 'k', 'v', 'g', 'beta']),
                        backend='tilelang',
                        scale=group.get('scale'),
                        initial_state=None,
                        output_final_state=False,
                        use_qk_l2norm_in_kernel=group.get('use_qk_l2norm_in_kernel', False),
                        cu_seqlens=group['cu_seqlens'])
                else:
                    output, state = original(**group)
                assert state is None
                outputs.append(output)
        with torch.profiler.record_function('gdn_partial_restore_order'):
            return torch.index_select(torch.cat(outputs, dim=1), 1, restore), None

    setattr(dispatched, official.MARKER, True)
    return dispatched
