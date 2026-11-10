"""Rebuild captured Tables 3/4 with accurate metric names, or collect new analyses."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from cifar10.config import CONDITIONS
from .metrics import anova_from_moments

CAPTURE_METRICS = ("noise_variance_fraction", "s_w", "s_b", "effective_rank_entropy", "content_reid")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_directory(path):
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def read_captured(root=None):
    root = Path(root) if root else Path(__file__).parent / "captured_results"
    manifest = json.loads((root / "manifest.json").read_text())
    captures = {}
    for entry in manifest["files"]:
        path = (root / entry["file"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Captured files must be inside the capture directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Captured CSV hash mismatch: {entry['file']}")
        rows = list(csv.DictReader(raw.decode().splitlines()))
        if len(rows) != entry["rows"]:
            raise ValueError("Captured row count mismatch")
        seen, records = set(), []
        for row in rows:
            if row["name"] in seen:
                raise ValueError("Duplicate captured checkpoint identity")
            seen.add(row["name"])
            n, k = int(row["n_images"]), int(row["k_reps"])
            s_w, s_b = float(row["s_w"]), float(row["s_b"])
            decomposition = anova_from_moments(s_w, s_b, n, k)
            ndv = float(row["ndv"])
            if not math.isclose(ndv, decomposition["noise_variance_fraction"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Captured ndv disagrees with S_W/(S_W+S_B)")
            chance = float(row["chance"])
            if not 0 < chance <= 1:
                raise ValueError("Invalid captured retrieval chance")
            m = round(1 / chance)
            if m < 2 or m > n // 2 or not math.isclose(chance, 1 / m):
                raise ValueError("Invalid captured retrieval gallery size")
            rank, content, noise = (
                float(row["eff_rank"]),
                float(row["content_reid"]),
                float(row["noise_reid"]),
            )
            if not math.isfinite(rank) or rank < 0 or not math.isfinite(content) or not 0 <= content <= 1:
                raise ValueError("Invalid captured rank or retrieval accuracy")
            if row["cond"].startswith("pg_"):
                if not math.isnan(noise):
                    raise ValueError("Poisson-Gaussian noise re-ID must be unavailable")
                noise = None
            elif not math.isfinite(noise) or not 0 <= noise <= 1:
                raise ValueError("Invalid noise re-ID")
            if row["prepool"] not in ("True", "False") or not row["seed"].startswith("s"):
                raise ValueError("Invalid captured metadata")
            records.append(
                {
                    "name": row["name"],
                    "condition": row["cond"],
                    "method": row["method"],
                    "arm": row["arm"],
                    "training_seed": int(row["seed"][1:]),
                    "epoch_suffix": row["ep"],
                    "n_images": n,
                    "k_reps": k,
                    "retrieval_images": m,
                    "prepool": row["prepool"] == "True",
                    **decomposition,
                    "noise_variance_fraction": ndv,
                    "effective_rank_entropy": rank,
                    "participation_ratio": None,
                    "noise_reid": noise,
                    "content_reid": content,
                    "retrieval_chance": chance,
                    "source_csv": entry["file"],
                }
            )
        if entry["role"] in captures:
            raise ValueError("Duplicate capture role")
        captures[entry["role"]] = records
    main = {r["name"]: r for r in captures["main"]}
    for row in captures["gauss_high_subset"]:
        comparable = {k: v for k, v in row.items() if k != "source_csv"}
        if comparable != {k: v for k, v in main[row["name"]].items() if k != "source_csv"}:
            raise ValueError("Gaussian subset differs from the main capture")
    return captures


def member_keys(rows):
    return {(row["method"], row["training_seed"], row["epoch_suffix"]) for row in rows}


def members_text(rows):
    return "|".join(
        f"{method}:s{seed}{':' + epoch if epoch else ''}" for method, seed, epoch in sorted(member_keys(rows))
    )


def table3(rows, matched=False):
    table = []
    for condition in CONDITIONS:
        baseline = [r for r in rows if r["condition"] == condition and r["arm"] == "baseline"]
        seana = [r for r in rows if r["condition"] == condition and r["arm"] == "seana"]
        common = member_keys(baseline) & member_keys(seana)
        if matched:
            baseline = [r for r in baseline if (r["method"], r["training_seed"], r["epoch_suffix"]) in common]
            seana = [r for r in seana if (r["method"], r["training_seed"], r["epoch_suffix"]) in common]
        if not baseline or not seana:
            raise ValueError(f"Missing baseline/SEANA rows for {condition}")
        b = statistics.mean(r["noise_variance_fraction"] for r in baseline)
        s = statistics.mean(r["noise_variance_fraction"] for r in seana)
        table.append(
            {
                "condition": condition,
                "baseline_noise_variance_fraction": b,
                "seana_noise_variance_fraction": s,
                "relative_change_pct": 100 * (s / b - 1) if b else None,
                "baseline_n_checkpoints": len(baseline),
                "seana_n_checkpoints": len(seana),
                "baseline_members": members_text(baseline),
                "seana_members": members_text(seana),
                "aggregation": "matched_method_seed" if matched else "all_available_checkpoints",
            }
        )
    return table


def table4(rows, matched=False):
    arms = ("baseline", "nsp", "seana")
    selected = {arm: [r for r in rows if r["condition"] == "gauss_high" and r["arm"] == arm] for arm in arms}
    common = set.intersection(*(member_keys(selected[arm]) for arm in arms))
    table = []
    for arm, records in selected.items():
        if matched:
            records = [r for r in records if (r["method"], r["training_seed"], r["epoch_suffix"]) in common]
        if not records:
            raise ValueError(f"Missing Gaussian-high decomposition for {arm}")
        table.append(
            {
                "condition": "gauss_high",
                "arm": arm,
                **{metric: statistics.mean(r[metric] for r in records) for metric in CAPTURE_METRICS},
                "n_checkpoints": len(records),
                "members": members_text(records),
                "aggregation": "matched_method_seed" if matched else "all_available_checkpoints",
            }
        )
    return table


def captured_tables(output_dir, root=None):
    captures = read_captured(root)
    rows = captures["main"]
    output = output_directory(output_dir)
    write_csv(output / "table3_captured.csv", table3(rows))
    write_csv(output / "table4_captured.csv", table4(rows))
    write_csv(output / "table3_matched.csv", table3(rows, matched=True))
    write_csv(output / "table4_matched.csv", table4(rows, matched=True))
    write_csv(output / "per_checkpoint_captured.csv", rows)
    write_csv(output / "sigma_sweep_captured.csv", captures["sigma_sweep"])
    summary = {
        "origin": "captured_results_not_new_reproduction",
        "main_checkpoints": len(rows),
        "overlapping_gauss_high_rows_not_double_counted": len(captures["gauss_high_subset"]),
        "additional_sigma_sweep_checkpoints": len(captures["sigma_sweep"]),
        "arms": dict(sorted(Counter(r["arm"] for r in rows).items())),
        "primary_metric": "noise_variance_fraction = S_W/(S_W+S_B) = 1-eta_squared",
        "captured_eff_rank": "effective_rank_entropy of per-image mean covariance",
        "participation_ratio": "not recoverable from the stored scalar ranks",
        "icc_1_1_trace": "derived per checkpoint from stored S_W, S_B, N and K; distinct from ndv",
        "aggregation": "arithmetic means of per-checkpoint metrics; counts and members included",
        "retrieval_images": sorted({r["retrieval_images"] for r in rows}),
        "evaluation_seed": "not recorded in CSV; captured script default is 0",
        "checkpoint_hashes": "not recorded in historical CSVs",
    }
    (output / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def collect_runs(root):
    rows, seen = [], set()
    for path in sorted(Path(root).rglob("analysis.json")):
        result = json.loads(path.read_text())
        if result.get("experiment") != "noise_invariance_analysis" or result.get("status") != "complete":
            continue
        checkpoint = result["checkpoint"]
        protocol = {
            k: v
            for k, v in result["config"].items()
            if k
            not in ("data_root", "download", "device", "batch_size", "retrieval_chunk_size", "save_features")
        }
        identity = {
            "protocol": protocol,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "dataset": result["dataset"],
            "evaluation": result["evaluation"],
            "runtime_sources": result["runtime"]["source_sha256"],
            "packages": result["runtime"]["packages"],
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate checkpoint analysis: {path}")
        seen.add(digest)
        rows.append(
            {
                "analysis_id": digest[:16],
                **{
                    k: checkpoint[k]
                    for k in ("method", "condition", "arm", "backbone", "training_seed", "checkpoint_sha256")
                },
                "analysis_seed": result["config"]["seed"],
                "n_images": result["dataset"]["n_images"],
                "k_reps": result["evaluation"]["k_reps"],
                "retrieval_images": result["evaluation"]["retrieval_images"],
                "feature_space": result["evaluation"]["feature_space"],
                **result["metrics"],
                "source_analysis": str(path),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    captured = sub.add_parser("captured")
    captured.add_argument("--output-dir", type=Path, required=True)
    runs = sub.add_parser("runs")
    runs.add_argument("--runs", type=Path, default=Path("runs/noise_invariance_analysis"))
    runs.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.command == "captured":
        print(json.dumps(captured_tables(args.output_dir), indent=2))
    else:
        rows = collect_runs(args.runs)
        if not rows:
            p.error("No completed analyses; smoke results are excluded")
        output = output_directory(args.output_dir)
        write_csv(output / "per_checkpoint.csv", rows)
        print(f"Wrote {len(rows)} analyses; checkpoints are kept separate, with no automatic pooling.")


if __name__ == "__main__":
    main()
