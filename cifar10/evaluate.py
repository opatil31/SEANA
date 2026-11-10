"""Reload a saved backbone and final linear probe for clean-test evaluation."""

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

from .config import Config, load_config
from .train import sha256_file


def evaluate_run(
    run_dir,
    device_name="cpu",
    data_root=None,
    download=False,
    *,
    config_type=Config,
    config_loader=load_config,
    dataset_builder=None,
):
    import torch
    from torch import nn
    from .data import build_datasets, make_loader
    from .models import build_backbone
    from .probe import evaluate

    run = Path(run_dir)
    backbone_checkpoint = torch.load(run / "backbone.pt", map_location="cpu", weights_only=True)
    probe_checkpoint = torch.load(run / "probe.pt", map_location="cpu", weights_only=True)
    if backbone_checkpoint.get("kind") != "backbone" or probe_checkpoint.get("kind") != "linear_probe":
        raise ValueError("Expected release-format backbone and linear-probe checkpoints")
    if probe_checkpoint["backbone_sha256"] != sha256_file(run / "backbone.pt"):
        raise ValueError("Probe was trained with a different backbone checkpoint")
    if backbone_checkpoint["config"] != probe_checkpoint["config"]:
        raise ValueError("Backbone/probe configuration mismatch")
    if probe_checkpoint["config"].get("probe_test_preprocessing", "none") != "none":
        raise ValueError("This evaluator requires clean test inputs without denoising")
    if backbone_checkpoint["config"].get("dataset", "cifar10") != config_type.dataset_name:
        raise ValueError("Dataset mismatch: use the evaluation command for this checkpoint's dataset")
    known = {field.name for field in fields(config_type)}
    values = {key: value for key, value in backbone_checkpoint["config"].items() if key in known}
    values.update(device=device_name, num_workers=0, download=download)
    if data_root is not None:
        values["data_root"] = data_root
    cfg = config_loader(**values)
    if cfg.smoke:
        torch.set_num_threads(2)
    device = torch.device(device_name)
    backbone, dim = build_backbone(cfg.backbone)
    if dim != backbone_checkpoint["feature_dim"] or dim != probe_checkpoint["feature_dim"]:
        raise ValueError("Feature dimension mismatch")
    backbone.load_state_dict(backbone_checkpoint["state_dict"])
    head = nn.Linear(dim, cfg.num_classes)
    head.load_state_dict(probe_checkpoint["state_dict"])
    datasets = (dataset_builder or build_datasets)(cfg)
    if backbone_checkpoint.get("dataset") != probe_checkpoint.get("dataset"):
        raise ValueError("Backbone/probe dataset identity mismatch")
    recorded_dataset = backbone_checkpoint.get("dataset")
    if recorded_dataset is not None and recorded_dataset != datasets.get("metadata"):
        raise ValueError("Evaluation dataset differs from the checkpoint's recorded dataset")
    if (run / "dataset.json").is_file():
        if json.loads((run / "dataset.json").read_text()) != datasets.get("metadata"):
            raise ValueError("Evaluation dataset differs from the recorded training-run dataset")
    probe_cfg = replace(cfg, batch_size=getattr(cfg, "probe_batch_size", cfg.batch_size))
    loader = make_loader(datasets["probe_test"], probe_cfg, shuffle=False)
    accuracy = evaluate(backbone.to(device), head.to(device), loader, device)
    return {
        "probe_final_clean_acc": accuracy,
        "test_inputs": "clean",
        "train_preprocessing": probe_checkpoint["config"].get("probe_train_preprocessing", "none"),
        "test_preprocessing": "none",
        "status": "smoke_not_benchmark" if cfg.smoke else "complete",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-root")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run, args.device, args.data_root, args.download), indent=2))


if __name__ == "__main__":
    main()
