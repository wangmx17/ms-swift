"""Cache validated packed boundaries, never activations or gradients.

Only versioned, plain int tensors qualify. PyTorch writes (including writes
through normal aliases) invalidate entries. External/raw-pointer or .data
writes that bypass the version counter are outside this opt-in contract.
"""
import torch
from collections import Counter, OrderedDict


class PackedSpansCache:

    def __init__(self, original, max_entries=64, max_elements=4096):
        self.original = original
        self.max_entries = max_entries
        self.max_elements = max_elements
        self.entries = OrderedDict()
        self.counts = Counter()

    def stamp(self, x):
        if x is None:
            return None
        if (type(x) is not torch.Tensor or x.ndim != 1 or x.dtype not in (torch.int32, torch.int64) or x.requires_grad
                or x.numel() > self.max_elements):
            raise TypeError('not cacheable')
        return (x._version, tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.dtype, x.device)

    def __call__(self, cumulative, padded, total):
        try:
            if cumulative is None or type(total) is not int:
                raise TypeError('not cacheable')
            stamp = (self.stamp(cumulative), self.stamp(padded))
        except (TypeError, RuntimeError):
            self.counts['bypass'] += 1
            return self.original(cumulative, padded, total)
        key = (id(cumulative), id(padded), total)
        previous = self.entries.get(key)
        if previous is not None:
            c, p, old_stamp, spans = previous
            if c is cumulative and p is padded and old_stamp == stamp:
                self.entries.move_to_end(key)
                self.counts['hit'] += 1
                return list(spans)
            del self.entries[key]
            self.counts['invalidated'] += 1
        self.counts['miss'] += 1
        result = self.original(cumulative, padded, total)
        # The audited original returns a list of immutable integer triples.
        if not (type(result) is list
                and all(type(s) is tuple and len(s) == 3 and all(type(v) is int for v in s) for s in result)):
            self.counts['unexpected_result'] += 1
            return result
        self.entries[key] = (cumulative, padded, stamp, tuple(result))
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
            self.counts['evicted'] += 1
        return list(result)
