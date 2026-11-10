"""Reevaluate the saved final probe on its recorded BBBC048 held-out cells."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .config import Config, load_config


def evaluate_run(run, device="cpu", manifest=None):
    import torch
    from cifar10.models import build_backbone
    from cifar10.data import make_loader
    from cifar10.train import sha256_file
    from .data import build_datasets, eval_transform
    from .probe import confusion_metrics
    from .views import load_treatment

    run = Path(run)
    result = json.loads((run / "metrics.json").read_text())
    if result.get("experiment") != "fluorescence_imaging_bbbc048" or result.get("status") not in (
        "complete",
        "smoke_not_benchmark",
    ):
        raise ValueError("Expected a completed BBBC048 experiment")
    recorded = result["config"]
    values = {f.name: recorded[f.name] for f in fields(Config)}
    values["device"] = device
    if manifest is not None:
        values["manifest"] = str(manifest)
    cfg = load_config(**values)
    if cfg.smoke:
        torch.set_num_threads(2)
    _, test, dataset = build_datasets(cfg)
    if dataset != result["dataset"]:
        raise ValueError("Dataset contents or split differ from the saved run")
    paths = ["backbone.pt", "probe.pt"]
    if cfg.probe_preprocessing() != "none":
        paths.append("treatment.pt")
    for name in paths:
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
        raise ValueError("Checkpoint feature dimensions differ")
    backbone.load_state_dict(backbone_payload["state_dict"], strict=True)
    head = torch.nn.Linear(dim, 7)
    head.load_state_dict(probe_payload["state_dict"], strict=True)
    backbone.eval().requires_grad_(False).to(device)
    head.eval().requires_grad_(False).to(device)
    treatment = None
    if cfg.probe_preprocessing() != "none":
        treatment, digest = load_treatment(run / "treatment.pt", cfg, dataset, device)
        if probe_payload["treatment_sha256"] != digest:
            raise ValueError("Probe belongs to a different denoiser/VST")
    transform, confusion = eval_transform(cfg), torch.zeros((7, 7), dtype=torch.long)
    with torch.inference_mode():
        for image, label in make_loader(test, cfg, shuffle=False):
            if treatment is not None:
                image = treatment.clean_estimate(image.to(device)).cpu()
            x = torch.stack([transform(item) for item in image]).to(device)
            prediction = head(backbone(x).flatten(1)).argmax(1).cpu()
            confusion += torch.bincount(7 * label + prediction, minlength=49).reshape(7, 7)
    return dict(
        status=result["status"],
        probe_preprocessing=cfg.probe_preprocessing(),
        final=confusion_metrics(confusion),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--manifest", type=Path)
    args = p.parse_args()
    print(json.dumps(evaluate_run(args.run, args.device, args.manifest), indent=2))


if __name__ == "__main__":
    main()
