"""Bounded memoization of immutable-between-layers packed sequence metadata.

Retains at most 16 small integer metadata tensors, never activations. Every hit
checks object identity and PyTorch mutation versions of both input and output.
Unversioned inference tensors and non-tensor inputs use the original function.
The original validation, normalization, and exception behavior runs on misses.
"""
import torch
from collections import Counter, OrderedDict


class ResolverCache:

    def __init__(self, original, max_entries=16):
        self.original = original
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.counts = Counter()

    @staticmethod
    def stamp(tensor):
        if tensor is None:
            return None
        return (tensor._version, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype,
                tensor.device)

    def __call__(self, padded, actual, total_seq_len, cp_size=1):
        selected = padded if padded is not None else actual
        if (type(selected) is not torch.Tensor or selected.requires_grad
                or selected.dtype not in (torch.int32, torch.int64) or selected.numel() > 4096):
            self.counts['bypass'] += 1
            return self.original(padded, actual, total_seq_len, cp_size)
        try:
            stamp = self.stamp(selected)
        except RuntimeError:  # Inference tensors have no mutation counter.
            self.counts['unversioned'] += 1
            return self.original(padded, actual, total_seq_len, cp_size)
        key = (id(selected), total_seq_len, cp_size)
        old = self.entries.get(key)
        if old is not None:
            src, src_stamp, result, result_stamp = old
            if src is selected and src_stamp == stamp and self.stamp(result) == result_stamp:
                self.counts['hit'] += 1
                self.entries.move_to_end(key)
                return result
            self.counts['invalidated'] += 1
            del self.entries[key]
        self.counts['miss'] += 1
        result = self.original(padded, actual, total_seq_len, cp_size)
        try:
            result_stamp = self.stamp(result)
        except RuntimeError:
            self.counts['unversioned_result'] += 1
            return result
        self.entries[key] = (selected, stamp, result, result_stamp)
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return result
