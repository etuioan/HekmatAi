"""Metrics of the own benchmark.

adaptation_cost  : total steps over the first K episodes of a block
asymptote        : mean steps over the last K episodes of a block
revisit_cost     : adaptation_cost of a block whose context occurred before
first_cost       : adaptation_cost of a context's first block
forward_transfer : computed in analysis (first_cost against the A3 baseline)
recognition      : (A2 only) share of episodes whose inferred model maps to the true context
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

K = 10


def block_table(episodes: list[dict]) -> list[dict]:
    """episodes: dicts with block, context, occurrence, steps."""
    by_block: dict[int, list[dict]] = defaultdict(list)
    for e in episodes:
        by_block[e["block"]].append(e)
    rows = []
    for b, eps in sorted(by_block.items()):
        steps = [e["steps"] for e in eps]
        rows.append(
            {
                "block": b,
                "context": eps[0]["context"],
                "occurrence": eps[0]["occurrence"],
                "adaptation_cost": float(np.sum(steps[:K])),
                "asymptote": float(np.mean(steps[-K:])),
            }
        )
    return rows


def recognition_accuracy(episodes: list[dict]) -> float | None:
    pairs = [(e["context"], e["model"]) for e in episodes if e["model"] is not None]
    if not pairs:
        return None
    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ctx, model in pairs:
        votes[model][ctx] += 1
    mapping = {m: max(v, key=v.get) for m, v in votes.items()}
    return float(np.mean([mapping[m] == ctx for ctx, m in pairs]))
