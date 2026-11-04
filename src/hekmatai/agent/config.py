"""Agent hyperparameters.

Rule (docs: experiment plan): these values may only be tuned on the
development seeds of the own benchmark. Reference runs (arc_reference) read
them and must not change them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    n_states: int = 81
    n_actions: int = 4
    gamma: float = 0.95
    epsilon: float = 0.02
    replan_every: int = 10
    planning_sweeps: int = 60
    # A0 replay baseline
    replay_capacity: int = 2000
    # A1 fixed context rule: switch slot every `fixed_block_len` episodes, cycling `fixed_slots` slots
    fixed_block_len: int = 30
    fixed_slots: int = 4
    # A2 learned context inference
    window: int = 8
    min_known: int = 3
    surprise_prob: float = 0.05  # a transition with smoothed probability below this is 'surprising'
    surprise_min: int = 3  # surprising transitions needed in the window to refute the active model
    smoothing_alpha: float = 0.1
    smoothing_support: float = 5.0
