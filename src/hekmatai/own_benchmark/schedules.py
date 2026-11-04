"""Context schedules: lists of (context index, number of episodes).

Seed sets are fixed here. Development seeds are the only ones allowed for
tuning; evaluation seeds are used once the decision rules are written down.
"""

from __future__ import annotations

import numpy as np

DEV_SEEDS = list(range(1000, 1010))
EVAL_SEEDS = list(range(0, 20))

Schedule = list[tuple[int, int]]


def regular_revisit(block_len: int = 30) -> Schedule:
    """A B A C B A: every context returns, regular block length."""
    return [(c, block_len) for c in (0, 1, 0, 2, 1, 0)]


def irregular_revisit(seed: int) -> Schedule:
    """Same order, block lengths drawn from 12..60 episodes (differs per seed)."""
    rng = np.random.default_rng(10_000 + seed)
    return [(c, int(rng.integers(12, 61))) for c in (0, 1, 0, 2, 1, 0)]


def new_contexts(block_len: int = 30) -> Schedule:
    """Mostly new contexts, one return at the end."""
    return [(c, block_len) for c in (0, 1, 2, 3, 4, 5, 0)]


def get_schedule(name: str, seed: int) -> Schedule:
    if name == "regular":
        return regular_revisit()
    if name == "irregular":
        return irregular_revisit(seed)
    if name == "new":
        return new_contexts()
    raise ValueError(f"unknown schedule {name!r}")


SCHEDULE_NAMES = ("regular", "irregular", "new")
