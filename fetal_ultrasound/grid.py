"""Generate main, patient-validation or validation-only strength-sweep experiments."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from .config import load_config, METHODS, BACKBONES, TREATMENTS, MAIN_TREATMENTS


def make_grid(
    *,
    recipe="main",
    methods=METHODS,
    backbones=("resnet18",),
    treatments=None,
    seeds=(0, 1, 2),
    sigmas=(0.16, 0.24, 0.32, 0.4),
    sigma_multipliers=None,
    manifest=None,
):
    if recipe not in ("main", "validation", "sigma_sweep"):
        raise ValueError("Unknown fetal recipe")
    treatments = treatments or (("seana",) if recipe == "sigma_sweep" else MAIN_TREATMENTS)
    if recipe == "sigma_sweep" and tuple(treatments) != ("seana",):
        raise ValueError("The strength sweep varies SEANA only")
    if sigma_multipliers is not None and recipe != "sigma_sweep":
        raise ValueError("Rule multipliers belong to the sigma_sweep recipe")
    rows, seen = [], set()
    strengths = (
        (sigma_multipliers if sigma_multipliers is not None else sigmas)
        if recipe == "sigma_sweep"
        else (None,)
    )
    for method, backbone, treatment, seed, strength in itertools.product(
        methods, backbones, treatments, seeds, strengths
    ):
        values = dict(
            method=method,
            backbone=backbone,
            treatment=treatment,
            seed=seed,
            manifest=manifest,
            cache_estimates=treatment != "baseline",
        )
        if recipe != "main":
            values.update(
                split_mode="official_validation",
                split_seed=0,
                probe_selection="validation_macro_f1",
                evaluate_test=False,
            )
        if strength is not None:
            if sigma_multipliers is None:
                values.update(sigma_vst=strength)
            else:
                values.update(sigma_mode="unsupervised", sigma_multiplier=strength)
        cfg = load_config(**values)
        signature = json.dumps(asdict(cfg), sort_keys=True)
        if signature in seen:
            raise ValueError("Duplicate grid recipe")
        seen.add(signature)
        rows.append(dict(index=len(rows), recipe=recipe, run_name=cfg.run_name(), config=asdict(cfg)))
    if not rows:
        raise ValueError("Empty grid")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--recipe", choices=("main", "validation", "sigma_sweep"), default="main")
    generate.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    generate.add_argument("--backbones", nargs="+", choices=BACKBONES, default=("resnet18",))
    generate.add_argument("--treatments", nargs="+", choices=TREATMENTS)
    generate.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    generate.add_argument("--sigmas", nargs="+", type=float, default=(0.16, 0.24, 0.32, 0.4))
    generate.add_argument("--sigma-multipliers", nargs="+", type=float)
    generate.add_argument("--manifest")
    generate.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--grid", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--output-root", type=Path, default=Path("runs/fetal"))
    execute.add_argument("--device", choices=("auto", "cpu", "cuda"))
    execute.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args())
    if args.pop("command") == "generate":
        output = args.pop("output")
        rows = make_grid(**args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Wrote {len(rows)} fetal runs; no jobs submitted.")
    else:
        rows = [json.loads(line) for line in args["grid"].read_text().splitlines() if line.strip()]
        selected = [r for r in rows if r["index"] == args["index"]]
        if len(selected) != 1:
            parser.error("Index must select exactly one row")
        cfg = load_config(
            **{**selected[0]["config"], **({"device": args["device"]} if args["device"] else {})}
        )
        if args["dry_run"]:
            print(json.dumps(cfg.resolved(), indent=2))
        else:
            from .train import run

            run(cfg, args["output_root"] / cfg.run_name())


if __name__ == "__main__":
    main()
