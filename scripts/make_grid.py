"""Write portable JSONL run configurations; never starts training."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from cifar10.config import BACKBONES, METHODS, TREATMENTS, CONDITIONS, load_config


def build_grid(
    backbones=BACKBONES,
    methods=METHODS,
    treatments=TREATMENTS,
    conditions=tuple(CONDITIONS),
    seeds=(0, 1, 2),
    noise_sampling="stochastic",
    data_root="data/cifar10",
    num_workers=4,
):
    seen = set()
    for backbone, method, treatment, condition, seed in itertools.product(
        backbones, methods, treatments, conditions, seeds
    ):
        cfg = load_config(
            backbone=backbone,
            method=method,
            treatment=treatment,
            condition=condition,
            seed=seed,
            noise_sampling=noise_sampling,
            data_root=data_root,
            num_workers=num_workers,
        )
        name = cfg.run_name()
        if name in seen:
            raise ValueError(f"Duplicate experiment: {name}")
        seen.add(name)
        yield {"index": len(seen) - 1, "name": name, "config": asdict(cfg)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/cifar10_grid.jsonl"))
    p.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--treatments", nargs="+", choices=TREATMENTS, default=TREATMENTS)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=tuple(CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--noise-sampling", choices=("stochastic", "fixed"), default="stochastic")
    p.add_argument("--data-root", default="data/cifar10")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    rows = list(
        build_grid(
            args.backbones,
            args.methods,
            args.treatments,
            args.conditions,
            args.seeds,
            args.noise_sampling,
            args.data_root,
            args.num_workers,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} experiments to {args.output}; no jobs started.")


if __name__ == "__main__":
    main()
