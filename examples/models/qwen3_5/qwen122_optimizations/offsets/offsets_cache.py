"""Bounded reuse of read-only integer sequence offsets, never activations.

This caches only host values from a successful original read. Caller-visible
lists are fresh copies; input identity/version/metadata determine reuse.
External writes that bypass PyTorch's version counter are outside the contract.
"""
import torch
from collections import Counter, OrderedDict


class OffsetsCache:

    def __init__(self, max_entries=64, max_elements=4096):
        assert max_entries > 0 and max_elements > 0
        self.max_entries = max_entries
        self.max_elements = max_elements
        self.entries = OrderedDict()
        self.counts = Counter()

    @staticmethod
    def stamp(value):
        return (value._version, tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype,
                value.device)

    def read(self, value, original_read):
        if (type(value) is not torch.Tensor or value.ndim != 1 or value.dtype not in (torch.int32, torch.int64)
                or value.requires_grad or value.numel() > self.max_elements):
            self.counts['bypass'] += 1
            return original_read(value)
        try:
            stamp = self.stamp(value)
        except RuntimeError:
            self.counts['unversioned'] += 1
            return original_read(value)
        key = id(value)
        previous = self.entries.get(key)
        if previous is not None:
            owner, old_stamp, offsets = previous
            if owner is value and old_stamp == stamp:
                self.entries.move_to_end(key)
                self.counts['hit'] += 1
                return list(offsets)
            del self.entries[key]
            self.counts['invalidated'] += 1
        self.counts['miss'] += 1
        # Preserve the actual reader and its exceptions on misses. All cached
        # values are immutable Python integers, never caller-owned lists.
        result = original_read(value)
        assert isinstance(result, list) and all(type(x) is int for x in result)
        self.entries[key] = (value, stamp, tuple(result))
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return list(result)
