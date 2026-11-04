import numpy as np

from hekmatai.agent import VARIANTS, AgentConfig, agent_version
from hekmatai.own_benchmark.run import run_one


def test_all_variants_run_and_the_oracle_reaches_the_goal_reliably():
    # Smoke test only. Weak baselines (A0, A3) mix contexts and are expected to be worse;
    # no ranking claim is made here.
    for name in VARIANTS:
        eps = run_one(name, "regular", seed=1001)
        assert len(eps) == 6 * 30
        share = sum(e["reached_goal"] for e in eps) / len(eps)
        assert share > (0.8 if name == "A1o_oracle" else 0.3), (name, share)


def test_run_is_deterministic_per_seed():
    a = run_one("A2_inference", "irregular", seed=1002)
    b = run_one("A2_inference", "irregular", seed=1002)
    assert [e["steps"] for e in a] == [e["steps"] for e in b]


def test_only_the_oracle_receives_labels():
    uses = {n: cls().uses_oracle for n, cls in VARIANTS.items()}
    assert uses == {n: (n == "A1o_oracle") for n in VARIANTS}


def test_inference_agent_builds_several_models_on_switching_contexts():
    eps = run_one("A2_inference", "new", seed=1003)
    assert eps[-1]["n_models"] >= 3


def test_inference_agent_does_not_explode_in_models_on_revisits():
    eps = run_one("A2_inference", "regular", seed=1004)
    assert eps[-1]["n_models"] <= 8  # 3 true contexts; a few spurious ones tolerated


def test_agent_version_is_stable_within_a_run():
    assert agent_version() == agent_version()
    assert len(agent_version()) == 12


def test_config_is_frozen():
    cfg = AgentConfig()
    try:
        cfg.gamma = 0.5  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("AgentConfig must be immutable")
