"""The only contract between the agent and any world.

The agent knows nothing about benchmarks. A world hands it integer
observations and rewards. Only agents that declare ``uses_oracle = True``
(the oracle upper bound A1o) ever receive a context label.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    name: str
    uses_oracle: bool

    def begin_episode(self, oracle_context: int | None = None) -> None: ...

    def act(self, obs: int) -> int: ...

    def observe(self, obs: int, action: int, reward: float, next_obs: int, done: bool) -> None: ...

    def end_episode(self) -> None: ...

    @property
    def context_id(self) -> int | None:
        """Index of the model currently in use, or None if the agent has no contexts."""
        ...

    @property
    def n_models(self) -> int: ...


def greedy_with_ties(q_row: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(q_row >= q_row.max() - 1e-12)
    return int(rng.choice(best))
