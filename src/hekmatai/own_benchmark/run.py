"""Runner of the own benchmark.

    python -m hekmatai.own_benchmark.run --seeds dev
    python -m hekmatai.own_benchmark.run --seeds eval   # refuses until decision_rules.json is filled in

Writes episodes.csv, blocks.csv and meta.json under results/own_benchmark/<seeds>/<agent version>/.
"""

from __future__ import annotations

import argparse
import csv
import json
from importlib import resources
from pathlib import Path

from hekmatai.agent import VARIANTS, AgentConfig, agent_version

from .metrics import block_table, recognition_accuracy
from .schedules import DEV_SEEDS, EVAL_SEEDS, SCHEDULE_NAMES, get_schedule
from .world import RoomsWorld


def run_one(variant: str, schedule_name: str, seed: int, cfg: AgentConfig | None = None) -> list[dict]:
    cfg = cfg or AgentConfig()
    world = RoomsWorld(seed=seed)
    agent = VARIANTS[variant](cfg, seed=seed)
    schedule = get_schedule(schedule_name, seed)
    seen: dict[int, int] = {}
    episodes: list[dict] = []
    for block, (ctx, n_eps) in enumerate(schedule):
        seen[ctx] = seen.get(ctx, 0) + 1
        world.set_context(ctx)
        for _ in range(n_eps):
            obs = world.reset()
            agent.begin_episode(oracle_context=ctx if agent.uses_oracle else None)
            steps, ret, done = 0, 0.0, False
            while not done:
                action = agent.act(obs)
                nxt, reward, done = world.step(action)
                agent.observe(obs, action, reward, nxt, done)
                obs, ret, steps = nxt, ret + reward, steps + 1
            model_id = agent.context_id
            agent.end_episode()
            episodes.append(
                {
                    "variant": variant,
                    "schedule": schedule_name,
                    "seed": seed,
                    "block": block,
                    "context": ctx,
                    "occurrence": seen[ctx],
                    "steps": steps,
                    "reached_goal": int(ret > 0),
                    "model": model_id,
                    "n_models": agent.n_models,
                }
            )
    return episodes


def _load_rules() -> dict:
    text = resources.files("hekmatai.own_benchmark").joinpath("config/decision_rules.json").read_text()
    return json.loads(text)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", choices=["dev", "eval"], default="dev")
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--schedules", nargs="*", default=list(SCHEDULE_NAMES))
    p.add_argument("--out", default="results/own_benchmark")
    args = p.parse_args(argv)

    if args.seeds == "eval":
        rules = _load_rules()
        if rules.get("similarity_margin_steps") is None:
            raise SystemExit(
                "Refusing the evaluation run: write similarity_margin_steps into "
                "own_benchmark/config/decision_rules.json first (decision rules must be fixed before the first evaluation)."
            )
    seeds = DEV_SEEDS if args.seeds == "dev" else EVAL_SEEDS
    version = agent_version()
    out = Path(args.out) / args.seeds / version
    out.mkdir(parents=True, exist_ok=True)

    all_eps: list[dict] = []
    all_blocks: list[dict] = []
    for variant in args.variants:
        for schedule in args.schedules:
            for seed in seeds:
                eps = run_one(variant, schedule, seed)
                all_eps.extend(eps)
                rec = recognition_accuracy(eps) if variant == "A2_inference" else None
                for row in block_table(eps):
                    all_blocks.append(
                        {"variant": variant, "schedule": schedule, "seed": seed, "recognition": rec, **row}
                    )
    _write_csv(out / "episodes.csv", all_eps)
    _write_csv(out / "blocks.csv", all_blocks)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "part": "B own_benchmark",
                "seeds": args.seeds,
                "seed_list": seeds,
                "agent_version": version,
                "agent_config": AgentConfig().__dict__,
                "variants": args.variants,
                "schedules": args.schedules,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
