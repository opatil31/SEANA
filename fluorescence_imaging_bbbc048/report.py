"""Report captured BBBC048 runs or summarize compatible new experiments."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .config import METHODS, TREATMENTS


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to report")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_captured(root=None):
    root = Path(root) if root else Path(__file__).parent / "captured_results"
    manifest = json.loads((root / "manifest.json").read_text())
    rows, seen = [], set()
    for entry in manifest["files"]:
        path = (root / entry["file"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Captured CSV must be within captured_results")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["bytes"]:
            raise ValueError("Captured CSV hash/size mismatch")
        if entry["sha256"] in seen:
            raise ValueError("Duplicate captured file bytes")
        seen.add(entry["sha256"])
        captured = list(csv.DictReader(raw.decode().splitlines()))
        if len(captured) != entry["rows"]:
            raise ValueError("Captured row count mismatch")
        for row in captured:
            method = "mocov2" if row["ssl_method"] == "moco" else row["ssl_method"]
            if method not in METHODS or row["dataset"] != "BBBC048" or int(row["epochs"]) < 2:
                raise ValueError("Unexpected method/dataset or preparation run in capture")
            active = [
                name
                for name in ("use_seana", "use_denoise", "use_n2n_denoise", "use_nsp_denoise")
                if row[name] == "True"
            ]
            if len(active) > 1:
                raise ValueError("Ambiguous captured treatment")
            treatment = (
                {
                    "use_denoise": "vst_only",
                    "use_n2n_denoise": "n2n",
                    "use_nsp_denoise": "nsp",
                    "use_seana": "seana",
                }.get(active[0], "baseline")
                if active
                else "baseline"
            )
            if treatment == "seana" and row.get("vst_type") == "identity":
                treatment = "resample_only"
            metrics = {}
            for name in (
                "probe_final_acc",
                "probe_final_balanced_acc",
                "probe_final_macro_f1",
                "probe_best_acc",
                "probe_best_balanced_acc",
            ):
                value = float(row[name]) if row.get(name) else None
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"Invalid captured metric: {name}")
                metrics[name] = value
            if row.get("probe_final_confusion"):
                confusion = json.loads(row["probe_final_confusion"])
                if len(confusion) != 7 or any(
                    len(r) != 7 or any(not isinstance(v, int) or v < 0 for v in r) for r in confusion
                ):
                    raise ValueError("Invalid captured confusion matrix")
                support = list(map(sum, confusion))
                predictions = [sum(r[c] for r in confusion) for c in range(7)]
                present = [i for i in range(7) if support[i]]
                recalculated = dict(
                    probe_final_acc=sum(confusion[i][i] for i in range(7)) / sum(support),
                    probe_final_balanced_acc=statistics.mean(confusion[i][i] / support[i] for i in present),
                    probe_final_macro_f1=statistics.mean(
                        2 * confusion[i][i] / (support[i] + predictions[i]) for i in present
                    ),
                )
                for name, value in recalculated.items():
                    if metrics[name] is None or not math.isclose(metrics[name], value, abs_tol=2e-7):
                        raise ValueError(f"Captured confusion disagrees with {name}")
                if (
                    row.get("probe_final_per_class_support")
                    and json.loads(row["probe_final_per_class_support"]) != support
                ):
                    raise ValueError("Captured class support disagrees with confusion matrix")
            rows.append(
                dict(
                    capture_id=entry["sha256"],
                    name=entry["source_name"],
                    method=method,
                    treatment=treatment,
                    seed=int(row["seed"]),
                    split_mode=row.get("split_mode", "unknown"),
                    **metrics,
                    seana_sigma_vst=row.get("seana_sigma_vst", ""),
                    seana_view_order=row.get("seana_view_order", ""),
                    seana_share_denoise=row.get("seana_share_denoise", "unrecorded"),
                    byol_momentum_start=row.get("byol_momentum_start", ""),
                    byol_momentum_end=row.get("byol_momentum_end", ""),
                    source_csv=entry["file"],
                    origins="|".join(entry["origins"]),
                    recorded_fields=json.dumps(row, sort_keys=True),
                )
            )
    return rows


def captured_report(output_dir, root=None):
    rows = read_captured(root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Report directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_run.csv", rows)
    coverage = []
    for method in METHODS:
        for treatment in TREATMENTS:
            subset = [r for r in rows if r["method"] == method and r["treatment"] == treatment]
            coverage.append(
                dict(
                    method=method,
                    treatment=treatment,
                    captured_records=len(subset),
                    seeds="|".join(map(str, sorted({r["seed"] for r in subset}))),
                    status="captured_variants_not_pooled" if subset else "no_captured_result",
                )
            )
    write_csv(output / "coverage.csv", coverage)
    summary = dict(
        origin="captured_results_not_new_reproduction",
        unique_records=len(rows),
        methods=dict(Counter(r["method"] for r in rows)),
        metrics="final accuracy, balanced accuracy and macro-F1; historical test-epoch maxima separate",
        aggregation="one row per distinct captured CSV; no pooling across source versions or variants",
        missing_results="no scores inferred from the paper or other methods",
    )
    (output / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_runs(root):
    groups = {}
    for path in sorted(Path(root).rglob("metrics.json")):
        result = json.loads(path.read_text())
        if result.get("experiment") != "fluorescence_imaging_bbbc048" or result.get("status") != "complete":
            continue
        cfg = result["config"]
        protocol = {
            k: v
            for k, v in cfg.items()
            if k
            not in (
                "seed",
                "effective_split_seed",
                "manifest",
                "device",
                "num_workers",
                "treatment_checkpoint",
            )
        }
        protocol.update(
            dataset_fingerprint=result["dataset"]["fingerprint"],
            runtime=result["runtime"],
            input_treatment_sha256=result.get("input_treatment_sha256"),
        )
        signature = json.dumps(protocol, sort_keys=True)
        group = groups.setdefault(signature, dict(config=cfg, seeds={}, paths=[]))
        if cfg["seed"] in group["seeds"]:
            raise ValueError(f"Duplicate seed for the same BBBC048 recipe: {path}")
        group["seeds"][cfg["seed"]] = result["metrics"]["final"]
        group["paths"].append(str(path))
    rows = []
    for signature, group in groups.items():
        cfg = group["config"]
        row = dict(
            recipe_id=hashlib.sha256(signature.encode()).hexdigest()[:12],
            method=cfg["method"],
            backbone=cfg["backbone"],
            treatment=cfg["treatment"],
            n2n_implementation=(
                cfg.get("n2n_implementation", "legacy_project_n2n") if cfg["treatment"] == "n2n" else ""
            ),
            split_mode=cfg["split_mode"],
            probe_preprocessing=cfg["probe_preprocessing"],
            view_order=cfg["view_order"],
            sigma_d=cfg["sigma_d"],
            sigma_vst=cfg["sigma_vst"],
            split_seed=cfg["split_seed"],
            n_seeds=len(group["seeds"]),
            seeds="|".join(map(str, sorted(group["seeds"]))),
        )
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [v[metric] for v in group["seeds"].values()]
            row["final_" + metric + "_mean"] = statistics.mean(values)
            row["final_" + metric + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        row["source_metrics"] = "|".join(group["paths"])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    captured = sub.add_parser("captured")
    captured.add_argument("--output-dir", type=Path, required=True)
    runs = sub.add_parser("runs")
    runs.add_argument("--runs", type=Path, default=Path("runs/bbbc048"))
    runs.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "captured":
        print(json.dumps(captured_report(args.output_dir), indent=2))
    else:
        rows = summarize_runs(args.runs)
        if not rows:
            p.error("No complete BBBC048 benchmark runs; smoke outputs are excluded")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output, rows)
        print(f"Wrote {len(rows)} compatible recipe summaries")


if __name__ == "__main__":
    main()
