"""Choose strength by validation macro-F1, then evaluate only selected probes on test."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from cifar10.train import sha256_file


def select(runs, output, seeds=(0, 1, 2)):
    output = Path(output)
    groups = {}
    for path in sorted(Path(runs).rglob("metrics.json")):
        result = json.loads(path.read_text())
        if result.get("experiment") != "fetal_ultrasound" or result.get("status") != "complete":
            continue
        cfg = result["requested_config"]
        if cfg["split_mode"] != "official_validation" or cfg["treatment"] != "seana":
            continue
        if result.get("test_evaluated") or result["metrics"].get("test") is not None:
            raise ValueError("Strength selection requires candidates whose test evaluation was deferred")
        if cfg["seed"] not in seeds:
            continue
        validation = result["metrics"].get("validation")
        if not validation:
            raise ValueError("Candidate has no validation result")
        score = validation["overall"]["macro_f1"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation macro-F1")
        members = result["dataset"]["patients"]
        if any(
            set(members[a]) & set(members[b])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError("Patient overlap in candidate metadata")
        protocol = {
            k: v
            for k, v in cfg.items()
            if k
            not in (
                "seed",
                "sigma_vst",
                "sigma_multiplier",
                "manifest",
                "device",
                "num_workers",
                "treatment_checkpoint",
            )
        }
        protocol.update(
            dataset_fingerprint=result["dataset"]["fingerprint"],
            split_sha256=result["dataset"]["split_sha256"],
            runtime=result["runtime"],
        )
        key = json.dumps(protocol, sort_keys=True)
        strength = cfg["sigma_multiplier"] if cfg["sigma_mode"] == "unsupervised" else cfg["sigma_vst"]
        candidates = groups.setdefault(key, {})
        runs_at_strength = candidates.setdefault(strength, {})
        if cfg["seed"] in runs_at_strength:
            raise ValueError("Duplicate seed for the same candidate")
        runs_at_strength[cfg["seed"]] = dict(
            seed=cfg["seed"],
            validation_macro_f1=score,
            run=os.path.relpath(path.parent.resolve(), output.parent.resolve()),
            metrics_sha256=sha256_file(path),
            artifacts=result["artifacts"],
            effective_sigma_vst=result["config"]["sigma_vst"],
        )
    selections = []
    for key, candidates in sorted(groups.items()):
        summaries = []
        for strength, members in sorted(candidates.items()):
            if set(members) != set(seeds):
                raise ValueError(f"Incomplete seed coverage for strength {strength}: {sorted(members)}")
            values = [r["validation_macro_f1"] for r in members.values()]
            summaries.append(
                dict(
                    strength=strength,
                    mean=statistics.mean(values),
                    sample_std=statistics.stdev(values) if len(values) > 1 else None,
                    n=len(values),
                )
            )
        winner = min(summaries, key=lambda row: (-row["mean"], row["strength"]))
        selections.append(
            dict(
                protocol=json.loads(key),
                candidates=summaries,
                selected=winner,
                runs=[candidates[winner["strength"]][seed] for seed in sorted(seeds)],
            )
        )
    if not selections:
        raise ValueError("No completed validation-only SEANA candidates")
    result = dict(
        schema_version=1,
        objective="validation.overall.macro_f1",
        seeds=list(seeds),
        tie_break="smaller strength",
        test_used_for_selection=False,
        selections=selections,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def evaluate_selected(selection, output, device="cpu", manifest=None):
    from .evaluate import evaluate_run

    selection, output = Path(selection), Path(output)
    payload = json.loads(selection.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("objective") != "validation.overall.macro_f1"
        or payload.get("test_used_for_selection") is not False
    ):
        raise ValueError("Expected a validation-only selection manifest")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Selected-test output directory is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, selected in enumerate(payload["selections"]):
        for item in selected["runs"]:
            run = (selection.parent / item["run"]).resolve()
            if sha256_file(run / "metrics.json") != item["metrics_sha256"]:
                raise ValueError("Candidate changed after validation selection")
            saved = json.loads((run / "metrics.json").read_text())
            if saved["artifacts"] != item["artifacts"] or saved.get("test_evaluated"):
                raise ValueError("Selected candidate provenance mismatch")
            evaluation = evaluate_run(run, device, manifest, "test")
            row = dict(
                selection_sha256=sha256_file(selection),
                method=selected["protocol"]["method"],
                backbone=selected["protocol"]["backbone"],
                seed=item["seed"],
                strength=selected["selected"]["strength"],
                source_metrics_sha256=item["metrics_sha256"],
                **evaluation,
            )
            with (output / f"group{i}_seed{item['seed']}.json").open("x") as stream:
                json.dump(row, stream, indent=2)
                stream.write("\n")
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--runs", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    choose.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--selection", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "select":
        result = select(args.runs, args.output, args.seeds)
        print(f"Selected {len(result['selections'])} recipes using validation macro-F1 only.")
    else:
        result = evaluate_selected(args.selection, args.output_dir, args.device, args.manifest)
        print(f"Evaluated {len(result)} selected probes on the untouched test patients.")


if __name__ == "__main__":
    main()
