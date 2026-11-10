"""Generate an ImageNet-100 JSONL grid or run one zero-based row; never submits jobs."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from .config import METHODS, BACKBONES, TREATMENTS, CONDITIONS, load_config


def build_grid(
    *,
    backbones=BACKBONES,
    methods=METHODS,
    treatments=TREATMENTS,
    conditions=tuple(CONDITIONS),
    seeds=(0, 1, 2),
    config=None,
    **overrides,
):
    seen = set()
    for backbone, method, treatment, condition, seed in itertools.product(
        backbones, methods, treatments, conditions, seeds
    ):
        cfg = load_config(
            config,
            backbone=backbone,
            method=method,
            treatment=treatment,
            condition=condition,
            seed=seed,
            **overrides,
        )
        name = cfg.run_name()
        if name in seen:
            raise ValueError(f"Duplicate experiment: {name}")
        seen.add(name)
        yield {"index": len(seen) - 1, "name": name, "config": asdict(cfg)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", type=Path, default=Path("runs/imagenet100_grid.jsonl"))
    generate.add_argument("--config", type=Path)
    for name, choices in (
        ("backbones", BACKBONES),
        ("methods", METHODS),
        ("treatments", TREATMENTS),
        ("conditions", tuple(CONDITIONS)),
    ):
        generate.add_argument("--" + name, choices=choices, nargs="+", default=choices)
    generate.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    generate.add_argument("--data-root")
    generate.add_argument("--num-workers", type=int)
    generate.add_argument("--noise-sampling", choices=("fixed", "stochastic"))
    execute = commands.add_parser("run")
    execute.add_argument("--grid", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--output-root", type=Path, default=Path("runs/imagenet100"))
    execute.add_argument("--dry-run", action="store_true")
    args = vars(p.parse_args())
    if args.pop("command") == "generate":
        output = args.pop("output")
        rows = list(build_grid(**args))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Wrote {len(rows)} experiments to {output}; no jobs started.")
    else:
        rows = [json.loads(line) for line in args["grid"].read_text().splitlines() if line.strip()]
        selected = [row for row in rows if row["index"] == args["index"]]
        if len(selected) != 1:
            p.error("Index must identify exactly one row")
        cfg = load_config(**selected[0]["config"])
        if args["dry_run"]:
            print(json.dumps(cfg.resolved(), indent=2))
        else:
            from .train import run

            run(cfg, args["output_root"] / cfg.run_name())


if __name__ == "__main__":
    main()
