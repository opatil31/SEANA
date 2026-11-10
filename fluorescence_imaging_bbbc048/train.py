"""Pretrain BBBC048 SSL and fit a frozen seven-phase linear classifier."""

import argparse
from dataclasses import fields, replace
import json
import math
import os
from pathlib import Path
import time

from cifar10.train import seed_all, sha256_file, write_json, runtime_metadata
from .config import Config, load_config, smoke_overrides, METHODS, BACKBONES, TREATMENTS, SPLITS, VIEW_ORDERS


def pretrain(model, criterion, treatment, loader, cfg, device):
    import torch
    from lightly.models.utils import update_momentum
    from cifar10.models import ssl_loss

    if not len(loader):
        raise ValueError("No full SSL batches; reduce batch_size")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_type(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    total_steps, history = cfg.epochs * len(loader), []
    warmup = min(cfg.warmup_epochs * len(loader), total_steps - 1)
    for epoch in range(cfg.epochs):
        model.train()
        criterion.train()
        total = 0.0
        momentum = cfg.moco_momentum
        if cfg.method == "byol":
            fraction = epoch / max(1, cfg.epochs - 1)
            momentum = (
                cfg.byol_momentum_end
                - (cfg.byol_momentum_end - cfg.byol_momentum) * (1 + math.cos(math.pi * fraction)) / 2
            )
        for index, (image, _) in enumerate(loader):
            step = epoch * len(loader) + index
            factor = 1.0
            if cfg.lr_schedule == "cosine":
                factor = (
                    (step + 1) / warmup
                    if warmup and step < warmup
                    else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                )
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * factor
            x0, x1 = treatment(image.to(device), input_is_estimate=cfg.cache_estimates)
            if cfg.method in ("byol", "mocov2"):
                update_momentum(model.backbone, model.backbone_momentum, momentum)
                update_momentum(model.projection_head, model.projection_head_momentum, momentum)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                if cfg.method == "mocov2" and not cfg.moco_shuffle_bn:
                    loss = criterion(model(x0), model.forward_momentum(x1))
                else:
                    loss = ssl_loss(model, criterion, cfg.method, x0, x1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite SSL loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters, cfg.clip_grad_norm or float("inf"), error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        history.append(
            dict(
                epoch=epoch + 1,
                loss=total / len(loader),
                lr=cfg.lr * factor,
                momentum=momentum if cfg.method in ("byol", "mocov2") else None,
            )
        )
        print(f"[ssl] epoch={epoch + 1} loss={total / len(loader):.6f}")
    return history


def run(cfg, output_dir):
    import torch
    from cifar10.data import make_loader
    from .data import build_datasets, TransformedDataset, eval_transform
    from .models import build_ssl
    from .probe import train_probe
    from .views import ViewTreatment, load_treatment, fit_identity, cache_estimates

    cfg.validate()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process per experiment")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run directory: {output}")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if cfg.smoke:
        torch.set_num_threads(2)
    started = time.time()
    seed_all(cfg.seed)
    train, test, dataset = build_datasets(cfg)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg.resolved())
    write_json(output / "dataset.json", dataset)
    runtime = runtime_metadata(("fluorescence_imaging_bbbc048",))
    try:
        from importlib.metadata import version

        runtime["packages"]["tifffile"] = version("tifffile")
    except ImportError:
        runtime["packages"]["tifffile"] = None
    input_treatment = None
    if cfg.treatment_checkpoint:
        treatment, input_treatment = load_treatment(cfg.treatment_checkpoint, cfg, dataset, device)
    else:
        treatment = ViewTreatment(cfg, device)
        loader = make_loader(train, replace(cfg, batch_size=cfg.fit_batch_size), shuffle=True, seed_offset=1)
        treatment.fit(loader, device)
    treatment_hash = None
    if cfg.treatment != "baseline":
        torch.save(
            dict(
                format_version=1,
                kind="bbbc048_treatment",
                config=cfg.resolved(),
                fit_identity=fit_identity(cfg, dataset),
                state_dict=treatment.state_dict(),
                fit_provenance=getattr(treatment, "fit_provenance", dict(seed=cfg.seed, runtime=runtime)),
            ),
            output / "treatment.pt",
        )
        treatment_hash = sha256_file(output / "treatment.pt")
    train_estimates = None

    def estimates(base, split):
        identity = dict(
            dataset_fingerprint=dataset["fingerprint"],
            split_sha256=dataset["split_sha256"][split],
            treatment_sha256=treatment_hash,
        )
        return cache_estimates(
            base, treatment, device, output / "cache" / f"{split}.npy", identity, cfg.fit_batch_size
        )

    if cfg.cache_estimates:
        train_estimates = estimates(train, "train")
    seed_all(cfg.seed)  # fitted-treatment duration does not change initialization/shuffle
    ssl_loader = make_loader(
        train_estimates if train_estimates is not None else train, cfg, shuffle=True, drop_last=True
    )
    model, criterion, dim = build_ssl(cfg)
    model, criterion = model.to(device), criterion.to(device)
    ssl_history = pretrain(model, criterion, treatment, ssl_loader, cfg, device)
    write_json(output / "ssl_history.json", ssl_history)
    common = dict(format_version=1, config=cfg.resolved(), dataset=dataset, feature_dim=dim)
    torch.save(
        dict(**common, kind="backbone", state_dict=model.backbone.state_dict()), output / "backbone.pt"
    )
    torch.save(
        dict(**common, kind="ssl", state_dict=model.state_dict(), loss_state_dict=criterion.state_dict()),
        output / "ssl.pt",
    )
    # Probe caches use the exact fitted treatment, never test images for fitting.
    if cfg.probe_preprocessing() != "none":
        probe_train_data = train_estimates if train_estimates is not None else estimates(train, "train")
        probe_test_data = estimates(test, "test")
    else:
        probe_train_data, probe_test_data = train, test
    transform = eval_transform(cfg)
    probe_cfg = replace(cfg, batch_size=cfg.probe_batch_size)
    train_loader = make_loader(
        TransformedDataset(probe_train_data, transform), probe_cfg, shuffle=True, seed_offset=10000
    )
    test_loader = make_loader(
        TransformedDataset(probe_test_data, transform), probe_cfg, shuffle=False, seed_offset=20000
    )
    seed_all(cfg.seed + 10000)
    head, metrics, history = train_probe(model.backbone, dim, train_loader, test_loader, cfg, device)
    torch.save(
        dict(
            **common,
            kind="linear_probe",
            backbone_sha256=sha256_file(output / "backbone.pt"),
            treatment_sha256=treatment_hash if cfg.probe_preprocessing() != "none" else None,
            state_dict=head.state_dict(),
        ),
        output / "probe.pt",
    )
    write_json(output / "probe_history.json", history)
    result = dict(
        schema_version=1,
        experiment="fluorescence_imaging_bbbc048",
        status="smoke_not_benchmark" if cfg.smoke else "complete",
        config=cfg.resolved(),
        dataset=dataset,
        metrics=metrics,
        runtime=runtime,
        device=str(device),
        seconds=time.time() - started,
        probe_checkpoint_selection="final_epoch",
        input_treatment_sha256=input_treatment,
        artifacts={p.name: sha256_file(p) for p in sorted(output.glob("*.pt"))},
    )
    write_json(output / "metrics.json", result)
    print(
        json.dumps(
            {"output": str(output), "status": result["status"], "final_probe": metrics["final"]}, indent=2
        )
    )
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    choices = dict(
        method=METHODS,
        backbone=BACKBONES,
        treatment=TREATMENTS,
        split_mode=SPLITS,
        view_order=VIEW_ORDERS,
        device=("auto", "cpu", "cuda"),
        probe_inputs=("historical", "raw"),
    )
    for field in fields(Config):
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=None)
        else:
            value_type = int if field.name == "split_seed" else type(field.default)
            p.add_argument(name, type=value_type, choices=choices.get(field.name))
    return p


def main(argv=None):
    p = parser()
    args = vars(p.parse_args(argv))
    config, output, dry_run = args.pop("config"), args.pop("output_dir"), args.pop("dry_run")
    if args.get("smoke"):
        args.update(smoke_overrides())
    try:
        cfg = load_config(config, **args)
    except (ValueError, TypeError) as error:
        p.error(str(error))
    if dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
    else:
        run(cfg, output or Path("runs/bbbc048") / cfg.run_name())


if __name__ == "__main__":
    main()
