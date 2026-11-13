import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .config import TABLE_ARMS, TABLE_METHODS, ARM_TREATMENTS, EXPERIMENT
from .psnr import HIGH_CONDITIONS, ESTIMATORS
from cifar10.config import CONDITIONS


def write_csv(path, rows, fields=None):
    if not rows:
        return
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_captured(root=None):
    root = Path(root) if root else Path(__file__).parent / "captured_results"
    manifest = json.loads((root / "manifest.json").read_text())
    records, seen = [], set()
    for entry in manifest["records"]:
        path = (root / entry["file"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Captured file must be inside its result directory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Captured CSV hash mismatch: {entry['file']}")
        rows = list(csv.DictReader(data.decode().splitlines()))
        key = tuple(entry[k] for k in ("condition", "method", "arm", "seed"))
        if key in seen or len(rows) != 1:
            raise ValueError(f"Duplicate or invalid captured result: {key}")
        seen.add(key)
        row = rows[0]
        if row["ssl_method"] != entry["method"] or int(row["seed"]) != entry["seed"]:
            raise ValueError(f"Captured CSV identity mismatch: {entry['file']}")
        if "--probe_train_on_noisy" not in entry["recorded_arguments"]:
            raise ValueError("Captured run does not identify noisy probe training")
        values = {name: float(row[name]) for name in ("probe_best_clean_acc", "probe_final_clean_acc")}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError("Invalid captured accuracy")
        records.append({**entry, **values})
    return records


def read_captured_psnr(root=None):
    root = Path(root) if root else Path(__file__).parent / "captured_results"
    data = (root / "psnr_log_excerpt.txt").read_bytes()
    provenance = json.loads((root / "psnr_provenance.json").read_text())
    if hashlib.sha256(data).hexdigest() != provenance["excerpt_sha256"]:
        raise ValueError("Captured PSNR excerpt hash mismatch")
    rows = []
    for line in data.decode().splitlines():
        parts = line.split()
        if parts and parts[0] in CONDITIONS:
            if len(parts) != 8:
                raise ValueError("Unexpected PSNR summary row")
            values = list(map(float, parts[1:]))
            rows.append(
                dict(
                    condition=parts[0],
                    sigma_d=values[0],
                    noisy=values[1],
                    n2n=values[2],
                    nsp=values[3],
                    drunet_raw=values[4],
                    drunet_vst=values[5],
                    drunet_vst_minus_nsp=values[6],
                )
            )
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Captured PSNR table is incomplete")
    return rows


def captured_tables(output_dir, root=None):
    records, psnr = read_captured(root), read_captured_psnr(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lookup = {(r["condition"], r["method"], r["arm"], r["seed"]): r for r in records}
    for metric in ("best", "final"):
        key = f"probe_{metric}_clean_acc"
        table, controls = [], []
        for condition in CONDITIONS:
            for method in TABLE_METHODS:
                row = {"condition": condition, "method": method, "seed": 0}
                for arm in ARM_TREATMENTS:
                    if (condition, method, arm, 0) not in lookup:
                        raise ValueError(f"Missing captured cell: {condition}/{method}/{arm}")
                    row[arm] = f"{100 * lookup[condition, method, arm, 0][key]:.2f}"
                table.append({name: row[name] for name in ("condition", "method", "seed", *TABLE_ARMS)})
                controls.append(row)
        write_csv(output / f"table2_captured_{metric}.csv", table)
        write_csv(output / f"all_controls_captured_{metric}.csv", controls)
    write_csv(output / "psnr_captured_all_conditions.csv", psnr)
    write_csv(output / "table1_captured.csv", [row for row in psnr if row["condition"] in HIGH_CONDITIONS])
    summary = {
        "origin": "captured_runs_not_new_reproduction",
        "ssl_files": len(records),
        "ssl_seeds": sorted({r["seed"] for r in records}),
        "accuracy_units": "percent",
        "table2_oracle": "oracle_postcrop",
        "historical_oracle": "oracle_precrop",
        "table2_primary_metric": "probe_best_clean_acc",
        "psnr_units": "dB",
        "psnr_precision": "two decimals as printed in source log",
    }
    with (output / "provenance.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def new_run_tables(runs, output_dir, metric="best", psnr_runs=None):
    """Keep incompatible recipes/code separate and never aggregate smoke outputs."""
    if metric not in ("best", "final"):
        raise ValueError("metric must be best or final")
    metric_key = f"probe_{metric}_clean_acc"
    groups = {}
    for path in sorted(Path(runs).rglob("metrics.json")):
        result = json.loads(path.read_text())
        cfg = result.get("config", {})
        if result.get("status") != "complete" or cfg.get("experiment") != EXPERIMENT:
            continue
        value = result["metrics"].get(metric_key)
        if value is None:
            raise ValueError(f"{metric_key} absent in {path}; select --metric final or enable best reporting")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid metric in {path}")
        skip = {
            "arm",
            "treatment",
            "resampling",
            "oracle_order",
            "seed",
            "data_root",
            "download",
            "device",
            "num_workers",
            "treatment_checkpoint",
            # Derived from the treatment under the shared probe_protocol.
            "probe_train_preprocessing",
        }
        protocol = {key: val for key, val in cfg.items() if key not in skip}
        protocol["source_sha256"] = result["runtime"]["source_sha256"]
        protocol["packages"] = result["runtime"]["packages"]
        signature = json.dumps(protocol, sort_keys=True)
        group = groups.setdefault(signature, {"config": cfg, "arms": {}, "train_preprocessing": {}})
        group["train_preprocessing"][cfg["arm"]] = cfg.get("probe_train_preprocessing", "none")
        arm = group["arms"].setdefault(cfg["arm"], {})
        if cfg["seed"] in arm:
            raise ValueError(f"Duplicate seed/arm for compatible recipe: {path}")
        arm[cfg["seed"]] = (value * 100, str(path))
    summaries, tables = [], []
    for signature, group in groups.items():
        cfg = group["config"]
        identity = {
            "recipe_id": hashlib.sha256(signature.encode()).hexdigest()[:12],
            "backbone": cfg["backbone"],
            "condition": cfg["condition"],
            "method": cfg["method"],
            "noise_sampling": cfg["noise_sampling"],
            "metric": metric_key,
            "n2n_implementation": cfg.get("n2n_implementation", "legacy_project_n2n"),
            "probe_protocol": cfg.get("probe_protocol", "raw_train_clean_test_legacy"),
            "probe_test_preprocessing": cfg.get("probe_test_preprocessing", "none"),
        }
        table = {**identity, **{arm: "" for arm in ARM_TREATMENTS}}
        for arm, seeds in group["arms"].items():
            values = [value[0] for value in seeds.values()]
            mean = statistics.mean(values)
            summaries.append(
                {
                    **identity,
                    "arm": arm,
                    "probe_train_preprocessing": group["train_preprocessing"][arm],
                    "n_seeds": len(values),
                    "seeds": ",".join(map(str, sorted(seeds))),
                    "mean_accuracy_pct": mean,
                    "sample_std_pct": statistics.stdev(values) if len(values) > 1 else "",
                    "source_metrics": "|".join(value[1] for value in seeds.values()),
                }
            )
            table[arm] = f"{mean:.2f}"
        tables.append(table)
    psnr_rows = []
    if psnr_runs:
        for path in sorted(Path(psnr_runs).rglob("psnr_metrics.json")):
            result = json.loads(path.read_text())
            if result.get("status") != "complete" or result.get("experiment") != EXPERIMENT + "_psnr":
                continue
            # Keep PSNR runs separate: no pooling of different evaluation draws,
            # fitted checkpoints, image counts or training recipes.
            psnr_rows.append(
                {
                    "condition": result["config"]["condition"],
                    "seed": result["config"]["seed"],
                    "eval_images": result["evaluation"]["images"],
                    "n2n_implementation": result.get("fits", {})
                    .get("n2n", {})
                    .get("config", {})
                    .get("n2n_implementation", "legacy_project_n2n"),
                    **{name: result["estimators"][name]["psnr_db"] for name in ESTIMATORS},
                    "source_metrics": str(path),
                }
            )
    if not summaries and not psnr_rows:
        raise ValueError("No completed isolation benchmark runs; smoke results are excluded")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ssl_summary.csv", summaries)
    write_csv(output / "table2_new_runs.csv", tables)
    write_csv(output / "table1_new_runs.csv", psnr_rows)
    return {"ssl_recipe_arm_groups": len(summaries), "psnr_runs": len(psnr_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    captured = sub.add_parser("captured")
    captured.add_argument("--source", type=Path)
    captured.add_argument("--output-dir", type=Path, default=Path("runs/isolation_captured_tables"))
    new = sub.add_parser("runs")
    new.add_argument("--runs", type=Path, default=Path("runs/denoiser_oracle_isolation"))
    new.add_argument("--psnr-runs", type=Path)
    new.add_argument("--metric", choices=("best", "final"), default="best")
    new.add_argument("--output-dir", type=Path, default=Path("runs/isolation_new_tables"))
    args = parser.parse_args()
    if args.mode == "captured":
        result = captured_tables(args.output_dir, args.source)
    else:
        result = new_run_tables(args.runs, args.output_dir, args.metric, args.psnr_runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
