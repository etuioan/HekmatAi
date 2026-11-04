"""Agent variants of the experiment plan.

A0  replay baseline (GDumb-like: greedy class-balanced buffer, model rebuilt from it)
A1  fixed context rule (context slot follows a fixed schedule guess)
A1o oracle (true context label; upper bound)
A2  learned context inference (M8) with one model per context (M5)
A3  single model, no contexts (ablation)
"""

from __future__ import annotations

import numpy as np

from .config import AgentConfig
from .interface import greedy_with_ties
from .model import WorldModel


class _Base:
    name = "base"
    uses_oracle = False

    def __init__(self, cfg: AgentConfig | None = None, seed: int = 0):
        self.cfg = cfg or AgentConfig()
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._episode = 0

    # -- subclasses provide the active model --
    def _active(self) -> WorldModel:
        raise NotImplementedError

    @property
    def context_id(self) -> int | None:
        return None

    @property
    def n_models(self) -> int:
        return 1

    def begin_episode(self, oracle_context: int | None = None) -> None:
        self._step = 0
        self._active().plan()

    def act(self, obs: int) -> int:
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(self.cfg.n_actions))
        return greedy_with_ties(self._active().q[obs], self.rng)

    def observe(self, obs: int, action: int, reward: float, next_obs: int, done: bool) -> None:
        self._step += 1
        if self._step % self.cfg.replan_every == 0:
            self._active().plan()

    def end_episode(self) -> None:
        self._episode += 1


class SingleModelAgent(_Base):
    """A3: one model for everything."""

    name = "A3_single"

    def __init__(self, cfg=None, seed=0):
        super().__init__(cfg, seed)
        self.model = WorldModel(self.cfg)

    def _active(self):
        return self.model

    def observe(self, obs, action, reward, next_obs, done):
        self.model.update(obs, action, reward, next_obs, done)
        super().observe(obs, action, reward, next_obs, done)


class ReplayBaselineAgent(_Base):
    """A0: GDumb-like. Greedy class-balanced buffer (class = state-action pair);
    the model is rebuilt from the buffer only, never kept across replans."""

    name = "A0_replay"

    def __init__(self, cfg=None, seed=0):
        super().__init__(cfg, seed)
        self.buffer: dict[tuple[int, int], list[tuple[int, float, bool]]] = {}
        self.size = 0
        self.model = WorldModel(self.cfg)

    def _rebuild(self) -> None:
        m = WorldModel(self.cfg)
        for (s, a), items in self.buffer.items():
            for s2, r, d in items:
                m.update(s, a, r, s2, d)
        self.model = m

    def _active(self):
        return self.model

    def begin_episode(self, oracle_context=None):
        self._step = 0
        self._rebuild()
        self.model.plan()

    def observe(self, obs, action, reward, next_obs, done):
        key = (obs, action)
        self.buffer.setdefault(key, []).append((next_obs, reward, done))
        self.size += 1
        if self.size > self.cfg.replay_capacity:
            biggest = max(self.buffer, key=lambda k: len(self.buffer[k]))
            items = self.buffer[biggest]
            items.pop(int(self.rng.integers(len(items))))
            self.size -= 1
        self._step += 1
        if self._step % self.cfg.replan_every == 0:
            self._rebuild()
            self.model.plan()


class FixedRuleAgent(_Base):
    """A1: context slot follows a fixed rule (switch every N episodes, cycle K slots)."""

    name = "A1_fixed"

    def __init__(self, cfg=None, seed=0):
        super().__init__(cfg, seed)
        self.models = [WorldModel(self.cfg) for _ in range(self.cfg.fixed_slots)]
        self._slot = 0

    @property
    def context_id(self):
        return self._slot

    @property
    def n_models(self):
        return len(self.models)

    def _active(self):
        return self.models[self._slot]

    def begin_episode(self, oracle_context=None):
        self._slot = (self._episode // self.cfg.fixed_block_len) % self.cfg.fixed_slots
        super().begin_episode(oracle_context)

    def observe(self, obs, action, reward, next_obs, done):
        self._active().update(obs, action, reward, next_obs, done)
        super().observe(obs, action, reward, next_obs, done)


class OracleAgent(_Base):
    """A1o: one model per true context; receives the label. Upper bound only."""

    name = "A1o_oracle"
    uses_oracle = True

    def __init__(self, cfg=None, seed=0):
        super().__init__(cfg, seed)
        self.models: dict[int, WorldModel] = {}
        self._ctx = 0

    @property
    def context_id(self):
        return self._ctx

    @property
    def n_models(self):
        return len(self.models)

    def _active(self):
        return self.models[self._ctx]

    def begin_episode(self, oracle_context=None):
        assert oracle_context is not None, "oracle agent needs the context label"
        self._ctx = oracle_context
        self.models.setdefault(self._ctx, WorldModel(self.cfg))
        super().begin_episode(oracle_context)

    def observe(self, obs, action, reward, next_obs, done):
        self._active().update(obs, action, reward, next_obs, done)
        super().observe(obs, action, reward, next_obs, done)


class ContextInferenceAgent(_Base):
    """A2: infers the context from recent transitions.

    Keeps one model per inferred context. After each step it scores the last
    `window` transitions under every model that knows enough of them; the best
    model becomes active. If even the best explains them poorly, a new model is
    created (accommodation). Transitions are committed to the active model with
    a lag of `window` steps (and flushed at episode end) so that the first
    steps after a switch do not contaminate the previous context's model.
    """

    name = "A2_inference"

    def __init__(self, cfg=None, seed=0):
        super().__init__(cfg, seed)
        self.models: list[WorldModel] = [WorldModel(self.cfg)]
        self._active_idx = 0
        self._window: list[tuple[int, int, float, int, bool]] = []

    @property
    def context_id(self):
        return self._active_idx

    @property
    def n_models(self):
        return len(self.models)

    def _active(self):
        return self.models[self._active_idx]

    def begin_episode(self, oracle_context=None):
        self._window = []
        super().begin_episode(oracle_context)

    def _score(self, model: WorldModel) -> tuple[float, int, int]:
        """(mean log-likelihood, number of known transitions, number of distinct surprising ones).

        Surprises are counted per distinct (state, action, next state) triple, so one
        wall bump repeated eight times is one piece of evidence, not eight.
        """
        total, known = 0.0, 0
        surprising: set[tuple[int, int, int]] = set()
        cutoff = float(np.log(self.cfg.surprise_prob))
        for s, a, _r, s2, _d in self._window:
            ll = model.log_lik(s, a, s2)
            if ll is not None:
                total += ll
                known += 1
                if ll < cutoff:
                    surprising.add((s, a, s2))
        return (total / known if known else 0.0), known, len(surprising)

    def _infer(self) -> None:
        cfg = self.cfg
        scores = [self._score(m) for m in self.models]
        # A model is refuted by several clearly surprising transitions, not by one or two slips.
        eligible = [(sc, i) for i, (sc, k, _u) in enumerate(scores) if k >= cfg.min_known]
        consistent = [(sc, i) for (sc, i) in eligible if scores[i][2] < cfg.surprise_min]
        active_known = scores[self._active_idx][1]
        active_refuted = scores[self._active_idx][2] >= cfg.surprise_min
        if consistent:
            _, best_idx = max(consistent)
            if best_idx != self._active_idx and active_refuted:
                self._active_idx = best_idx
                self._active().plan()
            return
        # No model with enough overlap explains the recent transitions. Create a new one only
        # if the active model had enough knowledge to be refuted; an empty or fresh active
        # model is simply still learning.
        if active_refuted and active_known >= cfg.min_known:
            self.models.append(WorldModel(self.cfg))
            self._active_idx = len(self.models) - 1
            for s, a, r, s2, d in self._window:  # these transitions belong to the new context
                self._active().update(s, a, r, s2, d)
            self._window = []
            self._active().plan()

    def observe(self, obs, action, reward, next_obs, done):
        self._window.append((obs, action, reward, next_obs, done))
        self._infer()
        while len(self._window) > self.cfg.window:
            s, a, r, s2, d = self._window.pop(0)
            self._active().update(s, a, r, s2, d)
        super().observe(obs, action, reward, next_obs, done)

    def end_episode(self) -> None:
        for s, a, r, s2, d in self._window:
            self._active().update(s, a, r, s2, d)
        self._window = []
        super().end_episode()


VARIANTS = {
    "A0_replay": ReplayBaselineAgent,
    "A1_fixed": FixedRuleAgent,
    "A1o_oracle": OracleAgent,
    "A2_inference": ContextInferenceAgent,
    "A3_single": SingleModelAgent,
}
