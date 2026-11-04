"""Prints mean and 95% interval (normal approximation over seeds) of revisit cost.

    python -m hekmatai.own_benchmark.summarize results/own_benchmark/dev/<version>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _ci(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    half = 1.96 * float(a.std(ddof=1)) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    return mean, half


def main(path: str) -> None:
    rows = list(csv.DictReader((Path(path) / "blocks.csv").open()))
    per_seed: dict[tuple, list[float]] = defaultdict(list)  # (schedule, variant, kind, seed) -> costs
    for r in rows:
        kind = "revisit" if int(r["occurrence"]) > 1 else "first"
        per_seed[(r["schedule"], r["variant"], kind, r["seed"])].append(float(r["adaptation_cost"]))
    agg: dict[tuple, list[float]] = defaultdict(list)
    for (schedule, variant, kind, _seed), costs in per_seed.items():
        agg[(schedule, variant, kind)].append(float(np.mean(costs)))
    print(f"{'schedule':10} {'variant':14} {'kind':8} {'mean cost (steps in first 10 episodes)':>42}")
    for key in sorted(agg):
        mean, half = _ci(agg[key])
        print(f"{key[0]:10} {key[1]:14} {key[2]:8} {mean:12.1f} +- {half:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
