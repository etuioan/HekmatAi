import numpy as np

from hekmatai.own_benchmark.world import CONTEXTS, FREE_CELLS, MOVES, SIZE, RoomsWorld


def _reachable(goal: int, action_map) -> set[int]:
    """Cells from which the goal can be reached with the nominal actions."""
    from hekmatai.own_benchmark import world as w

    # reverse search over real moves (action_map is a permutation, so all moves exist)
    seen = {goal}
    frontier = [goal]
    while frontier:
        cur = frontier.pop()
        r, c = divmod(cur, SIZE)
        for dr, dc in MOVES:
            pr, pc = r - dr, c - dc
            if 0 <= pr < SIZE and 0 <= pc < SIZE and not w._is_wall(pr, pc):
                p = pr * SIZE + pc
                if p not in seen:
                    seen.add(p)
                    frontier.append(p)
    return seen


def test_goal_reachable_from_every_free_cell_in_every_context():
    for ctx in CONTEXTS:
        goal = ctx.goal[0] * SIZE + ctx.goal[1]
        assert goal in FREE_CELLS
        assert set(FREE_CELLS) <= _reachable(goal, ctx.action_map)


def test_contexts_are_distinguishable_by_dynamics_or_goal():
    keys = {(c.goal, c.action_map) for c in CONTEXTS}
    assert len(keys) == len(CONTEXTS)
    # all action maps are permutations
    for c in CONTEXTS:
        assert sorted(c.action_map) == [0, 1, 2, 3]


def test_world_is_deterministic_for_a_seed():
    def trace(seed):
        w = RoomsWorld(seed=seed)
        w.set_context(1)
        s = w.reset()
        out = [s]
        rng = np.random.default_rng(0)
        for _ in range(30):
            s, r, d = w.step(int(rng.integers(4)))
            out.append((s, r, d))
            if d:
                break
        return out

    assert trace(3) == trace(3)


def test_observation_carries_no_context_label():
    w = RoomsWorld(seed=0)
    s = w.reset()
    assert isinstance(s, int) and 0 <= s < w.n_states
