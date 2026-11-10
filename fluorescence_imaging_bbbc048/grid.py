"""Generate portable BBBC048 experiment grids and execute a single zero-based row."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from .config import load_config, METHODS, BACKBONES, TREATMENTS, SPLITS, VIEW_ORDERS


def make_grid(
    *,
    config=None,
    methods=METHODS,
    backbones=("resnet18",),
    treatments=TREATMENTS,
    seeds=(0, 1, 2),
    split_modes=("random",),
    view_orders=None,
    sigmas=None,
    manifest=None,
    probe_inputs=None,
):
    rows, seen = [], set()
    for method, backbone, treatment, seed, split in itertools.product(
        methods, backbones, treatments, seeds, split_modes
    ):
        vst = treatment in ("seana", "vst_only", "resample_only")
        orders = (view_orders or (None,)) if vst else ("paper",)
        strengths = sigmas if sigmas and treatment in ("seana", "resample_only") else (None,)
        for order, sigma in itertools.product(orders, strengths):
            overrides = dict(
                method=method,
                backbone=backbone,
                treatment=treatment,
                seed=seed,
                split_mode=split,
                view_order=order,
                sigma_vst=sigma,
                manifest=manifest,
                probe_inputs=probe_inputs,
            )
            if treatment == "baseline":
                overrides.update(cache_estimates=False)
            cfg = load_config(config, **overrides)
            signature = json.dumps(asdict(cfg), sort_keys=True)
            if signature in seen:
                raise ValueError("Repeated recipe in requested grid")
            seen.add(signature)
            rows.append(dict(index=len(rows), config=asdict(cfg), run_name=cfg.run_name()))
    if not rows:
        raise ValueError("Empty experiment grid")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--config", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    for name, choices, default in (
        ("methods", METHODS, METHODS),
        ("backbones", BACKBONES, ("resnet18",)),
        ("treatments", TREATMENTS, TREATMENTS),
        ("split_modes", SPLITS, ("random",)),
        ("view_orders", VIEW_ORDERS, None),
    ):
        generate.add_argument("--" + name.replace("_", "-"), nargs="+", choices=choices, default=default)
    generate.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    generate.add_argument("--sigmas", nargs="+", type=float)
    generate.add_argument("--manifest")
    generate.add_argument("--probe-inputs", choices=("historical", "raw"))
    execute = sub.add_parser("run")
    execute.add_argument("--grid", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--output-root", type=Path, default=Path("runs/bbbc048"))
    execute.add_argument("--dry-run", action="store_true")
    execute.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = vars(p.parse_args())
    if args.pop("command") == "generate":
        output = args.pop("output")
        rows = make_grid(**args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Wrote {len(rows)} BBBC048 runs; no jobs submitted.")
    else:
        rows = [json.loads(line) for line in args["grid"].read_text().splitlines() if line.strip()]
        selected = [r for r in rows if r["index"] == args["index"]]
        if len(selected) != 1:
            p.error("Index must select exactly one grid row")
        row = selected[0]
        cfg = load_config(**{**row["config"], **({"device": args["device"]} if args["device"] else {})})
        if args["dry_run"]:
            print(json.dumps(cfg.resolved(), indent=2))
        else:
            from .train import run

            run(cfg, args["output_root"] / cfg.run_name())


if __name__ == "__main__":
    main()
