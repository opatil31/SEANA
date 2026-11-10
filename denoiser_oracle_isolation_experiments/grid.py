"""Generate or execute a portable grid for the rebuttal isolation experiments."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from cifar10.config import BACKBONES, CONDITIONS, METHODS
from .config import TABLE_ARMS, TABLE_METHODS, ARM_TREATMENTS, load_config


def build_grid(
    methods=TABLE_METHODS,
    backbones=("resnet18",),
    conditions=tuple(CONDITIONS),
    arms=TABLE_ARMS,
    seeds=(0,),
    **overrides,
):
    seen = set()
    for method, backbone, condition, arm, seed in itertools.product(
        methods, backbones, conditions, arms, seeds
    ):
        cfg = load_config(
            method=method, backbone=backbone, condition=condition, arm=arm, seed=seed, **overrides
        )
        if cfg.run_name() in seen:
            raise ValueError("Duplicate experiment requested")
        seen.add(cfg.run_name())
        yield {"index": len(seen) - 1, "name": cfg.run_name(), "config": asdict(cfg)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, default=Path("runs/isolation_grid.jsonl"))
    generate.add_argument("--methods", nargs="+", choices=METHODS, default=TABLE_METHODS)
    generate.add_argument("--backbones", nargs="+", choices=BACKBONES, default=["resnet18"])
    generate.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    generate.add_argument("--arms", nargs="+", choices=ARM_TREATMENTS, default=TABLE_ARMS)
    generate.add_argument("--seeds", nargs="+", type=int, default=[0])
    generate.add_argument("--data-root", default="data/cifar10")
    generate.add_argument("--num-workers", type=int, default=4)
    execute = sub.add_parser("run")
    execute.add_argument("--grid", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--output-root", type=Path, default=Path("runs/denoiser_oracle_isolation"))
    execute.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.command == "generate":
        rows = list(
            build_grid(
                args.methods,
                args.backbones,
                args.conditions,
                args.arms,
                args.seeds,
                data_root=args.data_root,
                num_workers=args.num_workers,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Wrote {len(rows)} runs to {args.output}; no jobs submitted")
    else:
        rows = [json.loads(line) for line in args.grid.read_text().splitlines() if line.strip()]
        selected = [row for row in rows if row["index"] == args.index]
        if len(selected) != 1:
            p.error("Index must select exactly one grid row")
        cfg = load_config(**selected[0]["config"])
        if args.dry_run:
            print(json.dumps(cfg.resolved(), indent=2))
        else:
            from .train import run

            run(cfg, args.output_root / cfg.run_name())


if __name__ == "__main__":
    main()
