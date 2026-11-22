"""CPU metadata plan for an isolated partial-GDN candidate; no kernel changes."""


def partition(lengths, heads):
    """Keep every sequence intact and ordered within its backend group.

    This is not a shape guard: the caller must still apply the original
    backend guard to each materialized group before executing any kernel.
    """
    if not lengths or any(type(x) is not int or x <= 0 for x in lengths):
        return None
    if type(heads) is not int or heads <= 0 or heads > 4096:
        return None
    limit = 4096 // heads
    long = [i for i, length in enumerate(lengths) if length > 64]
    short = [i for i, length in enumerate(lengths) if length <= 64]
    if not long or (not short and len(long) <= limit):
        return None  # Entire call already has one appropriate backend.
    groups = [('tilelang', long[start:start + limit]) for start in range(0, len(long), limit)]
    if short:
        groups.append(('fla', short))
    return groups
