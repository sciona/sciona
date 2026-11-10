"""Uniform-group, then uniform-member sampling with explicit random state."""
import numpy as np


def sample_indices(group_sizes, batch_size, random_state):
    """Return aligned int64 group/member indices, sampling with replacement.

Groups have equal probability regardless of their sizes; conditional on a
group, every member has equal probability. All group draws precede member
draws. The supplied NumPy RandomState is advanced in place; callers retain it
for reproducibility. Membership/order/eligibility belong to the caller.
"""
    sizes = np.asarray(group_sizes)
    if (sizes.ndim != 1 or not sizes.size or sizes.dtype.kind not in 'iu'
            or np.any(sizes <= 0) or np.any(sizes > np.iinfo(np.intp).max)):
        raise ValueError('Nonempty positive integer group sizes required')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive integer batch size required')
    if not isinstance(random_state, np.random.RandomState):
        raise ValueError('Explicit NumPy RandomState required')
    groups = random_state.choice(len(sizes), size=batch_size).astype(np.int64)
    members = np.array([random_state.choice(int(sizes[g]), size=1)[0]
                        for g in groups], dtype=np.int64)
    return groups, members
