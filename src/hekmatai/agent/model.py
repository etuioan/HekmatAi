"""Tabular predictive world model (M1) with optimistic planning.

One instance stores one context's knowledge (M5 in the multi-model variants).
Unknown (state, action) pairs are valued optimistically so the agent explores
them (R-max style); no separate exploration mechanism is used.
"""

from __future__ import annotations

import numpy as np

from .config import AgentConfig


class WorldModel:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        s, a = cfg.n_states, cfg.n_actions
        self.trans = np.zeros((s, a, s))
        self.rew_sum = np.zeros((s, a))
        self.done_sum = np.zeros((s, a))
        self.n = np.zeros((s, a))
        self.q = np.ones((s, a))  # optimistic until planned

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.trans[s, a, s2] += 1.0
        self.rew_sum[s, a] += r
        self.done_sum[s, a] += float(done)
        self.n[s, a] += 1.0

    def log_lik(self, s: int, a: int, s2: int) -> float | None:
        """Smoothed log-probability of a transition, or None if (s, a) was never seen."""
        n = self.n[s, a]
        if n == 0:
            return None
        c = self.trans[s, a, s2]
        p = (c + self.cfg.smoothing_alpha) / (n + self.cfg.smoothing_alpha * self.cfg.smoothing_support)
        return float(np.log(p))

    def plan(self) -> None:
        cfg = self.cfg
        known = self.n > 0
        denom = np.where(known, self.n, 1.0)
        t_hat = self.trans / denom[:, :, None]
        r_hat = self.rew_sum / denom
        cont = 1.0 - self.done_sum / denom
        q = np.where(known, r_hat, 1.0)
        for _ in range(cfg.planning_sweeps):
            v = q.max(axis=1)
            q_new = r_hat + cfg.gamma * cont * (t_hat @ v)
            q = np.where(known, q_new, 1.0)
        self.q = q
