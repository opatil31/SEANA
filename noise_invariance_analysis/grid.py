"""Create an analysis-only checkpoint manifest or run one row; never trains or submits jobs."""

import argparse
from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import re

from cifar10.config import BACKBONES, CONDITIONS, METHODS
from cifar10.train import sha256_file
from .config import load_config


def parse_legacy_name(path, backbone):
    stem = Path(path).stem
    match = re.fullmatch(r"(byol|simclr|vicreg|mocov2|moco|lejepa)_(.+)_s(\d+)(?:_ep\d+)?", stem)
    if not match:
        raise ValueError(f"Unrecognized legacy filename; use analyze with explicit metadata: {stem}")
    method, rest, seed = match.groups()
    condition = next((c for c in CONDITIONS if rest == c or rest.startswith(c + "_")), None)
    if condition is None:
        raise ValueError(f"No CIFAR condition in filename: {stem}")
    return {
        "method": "mocov2" if method == "moco" else method,
        "condition": condition,
        "arm": rest[len(condition) :].lstrip("_") or "baseline",
        "training_seed": int(seed),
        "backbone": backbone,
    }


def discover_runs(root, include_smoke=False):
    paths = []
    for path in sorted(Path(root).rglob("backbone.pt")):
        result_path = path.parent / "metrics.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        config = result.get("config", {})
        allowed = ("complete", "smoke_not_benchmark") if include_smoke else ("complete",)
        if result.get("status") not in allowed or config.get("dataset", "cifar10") != "cifar10":
            continue
        if config.get("method") not in METHODS:
            continue
        if result.get("artifacts", {}).get("backbone.pt") != sha256_file(path):
            raise ValueError(f"Backbone differs from completed run's recorded artifact: {path}")
        paths.append((path, None))
    return paths


def write_manifest(checkpoints, output, cfg):
    output = Path(output)
    rows, seen = [], set()
    for path, legacy in checkpoints:
        path = Path(path).resolve()
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError(f"Duplicate checkpoint bytes in manifest: {path.name}")
        seen.add(digest)
        rows.append(
            {
                "index": len(rows),
                "checkpoint": os.path.relpath(path, output.parent.resolve()),
                "checkpoint_sha256": digest,
                "legacy_metadata": legacy,
                "config": asdict(cfg),
            }
        )
    if not rows:
        raise ValueError("No eligible checkpoints found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Find completed release-format CIFAR runs")
    discover.add_argument("--runs", type=Path, required=True)
    discover.add_argument("--include-smoke", action="store_true")
    legacy = sub.add_parser("legacy", help="Import captured-style filenames, with an explicit architecture")
    legacy.add_argument("--models", nargs="+", required=True, help="Quoted checkpoint globs")
    legacy.add_argument("--backbone", choices=BACKBONES, required=True)
    for generate in (discover, legacy):
        generate.add_argument("--output", type=Path, required=True)
        generate.add_argument("--config", type=Path)
        generate.add_argument("--data-root")
    execute = sub.add_parser("run")
    execute.add_argument("--grid", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    execute.add_argument("--output-root", type=Path, default=Path("runs/noise_invariance_analysis"))
    execute.add_argument("--dry-run", action="store_true")
    execute.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = p.parse_args()
    if args.command == "run":
        rows = [json.loads(line) for line in args.grid.read_text().splitlines() if line.strip()]
        selected = [r for r in rows if r["index"] == args.index]
        if len(selected) != 1:
            p.error("Index must identify exactly one manifest row")
        row = selected[0]
        values = dict(row["config"])
        if args.device is not None:
            values["device"] = args.device
        cfg = load_config(**values)
        if args.dry_run:
            print(json.dumps({**row, "config": asdict(cfg)}, indent=2))
            return
        from .analyze import run_analysis

        run_analysis(
            (args.grid.parent / row["checkpoint"]).resolve(),
            args.output_root / f"row_{args.index:04d}_{row['checkpoint_sha256'][:12]}",
            cfg,
            row["legacy_metadata"],
            expected_checkpoint_sha256=row["checkpoint_sha256"],
        )
    else:
        cfg = load_config(args.config, data_root=args.data_root)
        if args.command == "discover":
            checkpoints = discover_runs(args.runs, args.include_smoke)
        else:
            paths = sorted({Path(path) for pattern in args.models for path in glob.glob(pattern)})
            checkpoints = [(path, parse_legacy_name(path, args.backbone)) for path in paths]
        rows = write_manifest(checkpoints, args.output, cfg)
        print(f"Wrote {len(rows)} frozen-backbone analyses; no jobs started.")


if __name__ == "__main__":
    main()
