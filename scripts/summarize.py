"""Summarize completed runs without pooling different protocols or recipes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    groups = {}
    for path in sorted(Path(root).rglob("metrics.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            continue
        cfg = result["config"]
        protocol = {
            key: value
            for key, value in cfg.items()
            if key not in ("seed", "data_root", "download", "device", "num_workers")
        }
        # Code and dependency changes can change scientific behavior. Do not
        # aggregate them silently even when their config files look identical.
        protocol["source_sha256"] = result["runtime"]["source_sha256"]
        protocol["packages"] = result["runtime"]["packages"]
        protocol["dataset_identity"] = result.get("dataset")
        signature = json.dumps(protocol, sort_keys=True)
        group = groups.setdefault(signature, {"config": cfg, "seeds": {}, "paths": []})
        if cfg["seed"] in group["seeds"]:
            raise ValueError(f"Duplicate seed {cfg['seed']} for the same recipe: {path}")
        group["seeds"][cfg["seed"]] = result["metrics"]["probe_final_clean_acc"]
        group["paths"].append(str(path))
    rows = []
    for signature, group in groups.items():
        values = list(group["seeds"].values())
        cfg = group["config"]
        rows.append(
            {
                "recipe_id": hashlib.sha256(signature.encode()).hexdigest()[:12],
                "dataset": cfg.get("dataset", "cifar10"),
                **{k: cfg[k] for k in ("backbone", "method", "condition", "treatment", "noise_sampling")},
                "probe_train_inputs": "noisy",
                "probe_test_inputs": "clean",
                "probe_protocol": cfg.get("probe_protocol", "raw_train_clean_test_legacy"),
                "probe_train_preprocessing": cfg.get("probe_train_preprocessing", "none"),
                "probe_test_preprocessing": cfg.get("probe_test_preprocessing", "none"),
                "n2n_implementation": (
                    cfg.get("n2n_implementation", "legacy_project_n2n") if cfg["treatment"] == "n2n" else ""
                ),
                "metric": "final_head",
                "n_seeds": len(values),
                "seeds": ",".join(map(str, sorted(group["seeds"]))),
                "mean_accuracy": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else "",
                "source_metrics": "|".join(group["paths"]),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs/cifar10"))
    p.add_argument("--output", type=Path, default=Path("runs/cifar10_summary.csv"))
    args = p.parse_args()
    rows = summarize(args.runs)
    if not rows:
        p.error("No completed benchmark runs found; smoke tests are excluded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} recipe summaries to {args.output}")


if __name__ == "__main__":
    main()
