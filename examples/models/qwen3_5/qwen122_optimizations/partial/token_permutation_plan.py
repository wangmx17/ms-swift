"""Construct the exact token bijection for already-validated backend groups."""


def token_permutation_plan(offsets, groups):
    if not offsets or offsets[0] != 0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError('Expected non-empty positive sequence intervals from zero')
    sequence_ids = [i for _, ids in groups for i in ids]
    if sorted(sequence_ids) != list(range(len(offsets)-1)):
        raise ValueError('Backend groups must contain each sequence exactly once')
    permutation = []
    group_offsets = [0]
    for _, ids in groups:
        for i in ids:
            permutation.extend(range(offsets[i], offsets[i+1]))
        group_offsets.append(len(permutation))
    assert len(permutation) == offsets[-1]
    inverse = [0]*len(permutation)
    for grouped, original in enumerate(permutation):
        inverse[original] = grouped
    return permutation, inverse, group_offsets
