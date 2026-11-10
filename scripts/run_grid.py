"""Run one zero-based JSONL grid row (suitable for a Slurm array)."""

import argparse
import json
from pathlib import Path

from cifar10.config import load_config
from cifar10.train import run_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/cifar10"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    selected = None
    with args.grid.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["index"] == args.index:
                if selected is not None:
                    raise ValueError(f"Duplicate index {args.index}")
                selected = row
    if selected is None:
        p.error(f"Index {args.index} not found")
    cfg = load_config(**selected["config"])
    if args.dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
    else:
        # Recompute the name instead of trusting a path from the manifest.
        run_experiment(cfg, args.output_root / cfg.run_name())


if __name__ == "__main__":
    main()
