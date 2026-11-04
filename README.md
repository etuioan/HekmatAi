# HekmatAi

HekmatAi is an experimental agent. The research question is whether a small combination of mechanisms helps an agent that meets changing contexts: **M8** (inferring which context it is in) together with **M5** (keeping one model per context so that new contexts do not overwrite old ones), both on top of a predictive world model (**M1**).

This repository is deliberately split into three parts that must never be merged into one score.

| Part | Folder | Role |
|---|---|---|
| A | `src/hekmatai/agent/` | The agent itself. It knows nothing about any test. |
| B | `src/hekmatai/own_benchmark/` | Our own test: a small grid world with switching contexts. |
| C | `src/hekmatai/arc_reference/` | External reference, ARC-AGI-3 (public demo games only). Placeholder, not implemented yet. |

## Separation rules

This is a Ports-and-Adapters layout.

- `agent` imports neither `own_benchmark` nor `arc_reference`.
- `own_benchmark` and `arc_reference` may import `agent`, but never each other.
- The agent never receives a context label unless it is the oracle variant `A1o`, which is an upper bound and is marked `uses_oracle`.
- There is no combined score across B and C.
- The agent is frozen (identified by `agent_version()`, a hash of its sources) before any reference run. ARC-AGI-3 is never used for tuning.
- `tests/test_separation.py` checks the import direction automatically.

## Agent variants

| Name | Idea |
|---|---|
| `A0_replay` | GDumb-like baseline: class-balanced replay buffer, model rebuilt from it |
| `A1_fixed` | Fixed context rule (time-based slots) |
| `A1o_oracle` | Oracle context labels (upper bound) |
| `A2_inference` | Learned context inference (M8) with one model per context (M5) |
| `A3_single` | One model for everything (ablation) |

The world model is currently a learned tabular transition and reward model with optimistic planning. A successor-representation map is a possible later replacement; it is not implemented.

## Own test (Part B)

A 9x9 four-rooms grid. A context is a goal cell plus a permutation of the four actions. The observation is the cell index only. Schedules: `regular` (A B A C B A), `irregular` (random block lengths) and `new` (mostly new contexts, one return). The main measures are the adaptation cost on the first visit of a context and on a revisit.

Seeds are split: `DEV_SEEDS` (1000..1009) are for developing and tuning, `EVAL_SEEDS` (0..19) are for the final comparison only.

## Running

```bash
pip install -e ".[dev]"
pytest
python -m hekmatai.own_benchmark.run --seeds dev
python -m hekmatai.own_benchmark.summarize results/own_benchmark/dev/<agent_version>
```

The evaluation run (`--seeds eval`) refuses to start until `similarity_margin_steps` in `src/hekmatai/own_benchmark/config/decision_rules.json` has been set. That value must be fixed before looking at evaluation results.

Results are written to `results/` (ignored by git).

## Status

Scaffold and Part B are in place. Nothing has been evaluated on the evaluation seeds. Part C is a placeholder.
