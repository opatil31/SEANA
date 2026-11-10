"""Evaluate a saved fetal probe, including anatomy and ultrasound-machine groups."""

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

from .config import Config, load_config


def evaluate_run(run, device="cpu", manifest=None, split="test"):
    import torch
    from cifar10.models import build_backbone
    from cifar10.data import make_loader
    from cifar10.train import sha256_file
    from .data import build_datasets, eval_transform
    from .views import load_treatment
    from .metrics import evaluate_predictions

    run = Path(run)
    result = json.loads((run / "metrics.json").read_text())
    if result.get("experiment") != "fetal_ultrasound" or result.get("status") not in (
        "complete",
        "smoke_not_benchmark",
    ):
        raise ValueError("Expected a completed fetal experiment")
    recorded = result["config"]
    cfg = load_config(**{f.name: recorded[f.name] for f in fields(Config)})
    if manifest is not None:
        cfg = replace(cfg, manifest=str(manifest))
    if cfg.smoke:
        torch.set_num_threads(2)
    splits, dataset = build_datasets(cfg)
    if dataset != result["dataset"] or split not in ("validation", "test") or not len(splits[split]):
        raise ValueError("Dataset/split differs from the saved run or is empty")
    needed = ["backbone.pt", "probe.pt"] + (["treatment.pt"] if cfg.probe_preprocessing() != "none" else [])
    for name in needed:
        if sha256_file(run / name) != result["artifacts"][name]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    backbone_payload = torch.load(run / "backbone.pt", map_location="cpu", weights_only=True)
    probe_payload = torch.load(run / "probe.pt", map_location="cpu", weights_only=True)
    for payload, kind in ((backbone_payload, "backbone"), (probe_payload, "linear_probe")):
        if (
            payload.get("format_version") != 1
            or payload.get("kind") != kind
            or payload.get("config") != recorded
            or payload.get("dataset") != dataset
        ):
            raise ValueError("Checkpoint metadata mismatch")
    if probe_payload["backbone_sha256"] != result["artifacts"]["backbone.pt"]:
        raise ValueError("Probe belongs to a different backbone")
    backbone, dim = build_backbone(cfg.backbone)
    if any(p["feature_dim"] != dim for p in (backbone_payload, probe_payload)):
        raise ValueError("Checkpoint feature dimension mismatch")
    backbone.load_state_dict(backbone_payload["state_dict"], strict=True)
    head = torch.nn.Linear(dim, 6)
    head.load_state_dict(probe_payload["state_dict"], strict=True)
    backbone.eval().requires_grad_(False).to(device)
    head.eval().requires_grad_(False).to(device)
    treatment = None
    if cfg.probe_preprocessing() != "none":
        treatment, digest = load_treatment(run / "treatment.pt", cfg, dataset, device)
        if probe_payload["treatment_sha256"] != digest:
            raise ValueError("Probe belongs to a different fitted denoiser")
    transform, labels, predictions = eval_transform(cfg), [], []
    with torch.inference_mode():
        for image, label in make_loader(
            splits[split], replace(cfg, batch_size=cfg.probe_batch_size), shuffle=False
        ):
            if treatment is not None:
                image = treatment.clean_estimate(image.to(device)).cpu()
            x = torch.stack([transform(item) for item in image]).to(device)
            predictions.extend(head(backbone(x).flatten(1)).argmax(1).cpu().tolist())
            labels.extend(label.tolist())
    return dict(
        status=result["status"],
        split=split,
        probe_checkpoint_selection=result["probe_checkpoint_selection"],
        metrics=evaluate_predictions(labels, predictions, splits[split].records),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--split", choices=("test", "validation"), default="test")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = evaluate_run(args.run, args.device, args.manifest, args.split)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
