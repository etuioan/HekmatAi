"""Four-rooms grid world with switching contexts.

A context fixes (a) the goal cell and (b) how the four nominal actions map to
real moves. The observation is only the cell index: no context label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 9
N_STATES = SIZE * SIZE
N_ACTIONS = 4
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

_WALL_ROW = SIZE // 2
_WALL_COL = SIZE // 2
_DOORS = {(_WALL_ROW, 2), (_WALL_ROW, 6), (2, _WALL_COL), (6, _WALL_COL)}


def _is_wall(r: int, c: int) -> bool:
    if (r, c) in _DOORS:
        return False
    return r == _WALL_ROW or c == _WALL_COL


FREE_CELLS = [r * SIZE + c for r in range(SIZE) for c in range(SIZE) if not _is_wall(r, c)]


@dataclass(frozen=True)
class Context:
    goal: tuple[int, int]
    action_map: tuple[int, int, int, int]  # nominal action -> real move index


CONTEXTS = [
    Context((8, 8), (0, 1, 2, 3)),
    Context((0, 8), (1, 0, 3, 2)),
    Context((8, 0), (2, 3, 0, 1)),
    Context((0, 0), (3, 2, 1, 0)),
    Context((8, 8), (0, 1, 3, 2)),
    Context((0, 8), (1, 0, 2, 3)),
]


class RoomsWorld:
    max_steps = 200

    def __init__(self, seed: int = 0, slip: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.slip = slip
        self.context_index = 0
        self._pos = (0, 0)
        self._t = 0

    @property
    def n_states(self) -> int:
        return N_STATES

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def set_context(self, index: int) -> None:
        self.context_index = index

    @property
    def goal_cell(self) -> int:
        r, c = CONTEXTS[self.context_index].goal
        return r * SIZE + c

    def reset(self) -> int:
        starts = [s for s in FREE_CELLS if s != self.goal_cell]
        s = int(self.rng.choice(starts))
        self._pos = divmod(s, SIZE)
        self._t = 0
        return s

    def step(self, action: int) -> tuple[int, float, bool]:
        ctx = CONTEXTS[self.context_index]
        if self.rng.random() < self.slip:
            action = int(self.rng.integers(N_ACTIONS))
        dr, dc = MOVES[ctx.action_map[action]]
        r, c = self._pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE and not _is_wall(nr, nc):
            self._pos = (nr, nc)
        self._t += 1
        s = self._pos[0] * SIZE + self._pos[1]
        at_goal = s == self.goal_cell
        done = at_goal or self._t >= self.max_steps
        return s, (1.0 if at_goal else 0.0), done
