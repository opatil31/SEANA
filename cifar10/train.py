"""Run CIFAR-10 SSL pretraining followed by a frozen linear probe.

Run from the repository root: python -m cifar10.train --help
"""

import argparse
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

from .config import METHODS, BACKBONES, TREATMENTS, CONDITIONS, load_config


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime_metadata(extra_source_roots=()):
    import seana

    packages = {}
    for name in (
        "seana-ssl",
        "torch",
        "torchvision",
        "lightly",
        "deepinv",
        "numpy",
        "pillow",
        "datasets",
        "pyarrow",
        "huggingface-hub",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parent.parent
    sources = [
        Path(seana.__file__).resolve(),
        *sorted((root / "cifar10").glob("*.py")),
        *sorted((root / "denoisers").glob("*.py")),
    ]
    if "_seana_api" in sys.modules:
        sources.append(Path(sys.modules["_seana_api"].__file__).resolve())
    for source_root in extra_source_roots:
        sources.extend(sorted((root / source_root).rglob("*.py")))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "source_sha256": {
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else f"installed/{path.name}": sha256_file(path)
            for path in sources
        },
    }


def ssl_learning_rate(cfg, epoch):
    if cfg.warmup_epochs and epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / cfg.warmup_epochs
    if cfg.lr_schedule == "cosine":
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs - 1)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr


def pretrain(model, criterion, view_treatment, loader, cfg, device):
    import torch
    from lightly.models.utils import update_momentum
    from .models import ssl_loss

    parameters = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    model.train()
    criterion.train()
    history = []
    for epoch in range(cfg.epochs):
        rate = ssl_learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        total = count = 0
        momentum = cfg.moco_momentum
        if cfg.method == "byol":
            fraction = epoch / max(1, cfg.epochs - 1)
            momentum = 1.0 - (1.0 - cfg.byol_momentum) * (math.cos(math.pi * fraction) + 1) / 2
        for (x0, x1), _ in loader:
            x0 = view_treatment(x0.to(device))
            x1 = view_treatment(x1.to(device))
            if cfg.method in ("byol", "mocov2"):
                update_momentum(model.backbone, model.backbone_momentum, m=momentum)
                update_momentum(model.projection_head, model.projection_head_momentum, m=momentum)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = ssl_loss(model, criterion, cfg.method, x0, x1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite SSL loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
        if not count:
            raise ValueError("SSL loader has no full batch; reduce batch_size")
        history.append({"epoch": epoch + 1, "loss": total / count, "lr": rate})
        print(f"[ssl] epoch={epoch + 1} loss={total / count:.6f} lr={rate:.6g}")
    return history


def run_experiment(
    cfg,
    output_dir,
    *,
    dataset_builder=None,
    treatment_builder=None,
    model_builder=None,
    pretrain_fn=None,
    extra_source_roots=(),
):
    import torch
    from .data import build_datasets, make_loader
    from .models import build_ssl
    from .probe import train_probe
    from .views import ViewTreatment

    cfg.validate()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("One process per experiment is supported; do not use distributed launch.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg.resolved())
    started = time.time()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use CPU or CUDA; MPS is not validated for these experiments")
    if cfg.smoke:
        torch.set_num_threads(2)
    seed_all(cfg.seed)
    datasets = (dataset_builder or build_datasets)(cfg)
    if "metadata" in datasets:
        write_json(output / "dataset.json", datasets["metadata"])
    ssl_loader = make_loader(datasets["ssl"], cfg, shuffle=True, drop_last=True)
    fit_cfg = replace(cfg, batch_size=getattr(cfg, "fit_batch_size", cfg.batch_size))
    raw_loader = make_loader(datasets["fit"], fit_cfg, shuffle=True, drop_last=True, seed_offset=1)
    treatment = (treatment_builder or ViewTreatment)(cfg, device)
    treatment.fit(ssl_loader, raw_loader, device)
    if cfg.treatment != "baseline":
        torch.save(
            {
                "format_version": 1,
                "kind": "view_treatment",
                "config": cfg.resolved(),
                "state_dict": treatment.state_dict(),
            },
            output / "treatment.pt",
        )

    # Restart the pretraining RNG/loader stream so fitting duration cannot change
    # model initialization or shuffle order between treatment arms.
    seed_all(cfg.seed)
    ssl_loader = make_loader(datasets["ssl"], cfg, shuffle=True, drop_last=True)
    model, criterion, feature_dim = (model_builder or build_ssl)(cfg)
    model, criterion = model.to(device), criterion.to(device)
    ssl_history = (pretrain_fn or pretrain)(model, criterion, treatment, ssl_loader, cfg, device)
    write_json(output / "ssl_history.json", ssl_history)
    backbone_path = output / "backbone.pt"
    torch.save(
        {
            "format_version": 1,
            "kind": "backbone",
            "config": cfg.resolved(),
            "dataset": datasets.get("metadata"),
            "feature_dim": feature_dim,
            "state_dict": model.backbone.state_dict(),
        },
        backbone_path,
    )
    torch.save(
        {
            "format_version": 1,
            "kind": "ssl",
            "config": cfg.resolved(),
            "state_dict": model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
        },
        output / "ssl.pt",
    )
    # Reuse the fitted N2N/NSP only on noisy probe-training inputs. The clean
    # test path bypasses all treatments, including during optional epoch checks.
    train_preprocess = None
    if cfg.probe_train_preprocessing() != "none":
        treatment.eval().requires_grad_(False)
        train_preprocess = treatment.denoise_probe_inputs
    seed_all(cfg.seed + 10000)
    probe_cfg = replace(cfg, batch_size=getattr(cfg, "probe_batch_size", cfg.batch_size))
    train_loader = make_loader(datasets["probe_train"], probe_cfg, shuffle=True, seed_offset=10000)
    test_loader = make_loader(datasets["probe_test"], probe_cfg, shuffle=False, seed_offset=20000)
    head, metrics, probe_history = train_probe(
        model.backbone,
        feature_dim,
        train_loader,
        test_loader,
        cfg,
        device,
        train_preprocess=train_preprocess,
    )
    torch.save(
        {
            "format_version": 1,
            "kind": "linear_probe",
            "config": cfg.resolved(),
            "dataset": datasets.get("metadata"),
            "feature_dim": feature_dim,
            "backbone_sha256": sha256_file(backbone_path),
            # Provenance only: clean-test evaluation does not load this treatment.
            "training_treatment_sha256": (
                sha256_file(output / "treatment.pt") if train_preprocess is not None else None
            ),
            "state_dict": head.state_dict(),
        },
        output / "probe.pt",
    )
    write_json(output / "probe_history.json", probe_history)
    result = {
        "schema_version": 1,
        "status": "smoke_not_benchmark" if cfg.smoke else "complete",
        "config": cfg.resolved(),
        "metrics": metrics,
        "runtime": runtime_metadata(extra_source_roots),
        "device": str(device),
        "seconds": time.time() - started,
        "probe_checkpoint_selection": "final_epoch",
        "artifacts": {path.name: sha256_file(path) for path in sorted(output.glob("*.pt"))},
    }
    if getattr(treatment, "input_checkpoint_sha256", None):
        result["input_treatment_sha256"] = treatment.input_checkpoint_sha256
    if "metadata" in datasets:
        result["dataset"] = datasets["metadata"]
    write_json(output / "metrics.json", result)
    print(f"Clean-test accuracy (final head): {100 * metrics['probe_final_clean_acc']:.2f}%")
    print(f"Saved run: {output}")
    return result


def parser(*, conditions=CONDITIONS):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="JSON recipe; CLI options override it")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--backbone", choices=BACKBONES)
    p.add_argument("--treatment", choices=TREATMENTS)
    p.add_argument("--condition", choices=conditions)
    for name in (
        "seed",
        "epochs",
        "probe_epochs",
        "batch_size",
        "num_workers",
        "seana_fit_steps",
        "nsp_fit_steps",
        "n2n_fit_steps",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "probe_lr", "sigma_d", "sigma_vst"):
        p.add_argument("--" + name.replace("_", "-"), type=float)
    p.add_argument("--noise-sampling", choices=("fixed", "stochastic"))
    p.add_argument("--data-root")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    p.add_argument("--drunet-weights", help="Local DRUNet checkpoint, or download")
    for name in ("download", "amp", "probe_report_best"):
        p.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--smoke", action="store_true", help="Tiny fake-data pipeline check, not a benchmark")
    p.add_argument(
        "--dry-run", action="store_true", help="Print fully resolved config without importing torch"
    )
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    config_path = args.pop("config")
    output = args.pop("output_dir")
    dry_run = args.pop("dry_run")
    smoke = args.pop("smoke")
    if smoke:
        args.update(
            smoke=True,
            epochs=1,
            probe_epochs=1,
            batch_size=4,
            num_workers=0,
            seana_fit_steps=1,
            nsp_fit_steps=1,
            n2n_fit_steps=1,
            n2n_base_ch=8,
            nsp_base_ch=8,
            nsp_num_module=0,
            denoiser_patch_batch_size=2,
            seana_patch_batch_size=2,
            lejepa_num_slices=8,
            moco_queue_size=32,
            warmup_epochs=0,
        )
    try:
        cfg = load_config(config_path, **args)
    except (ValueError, TypeError) as error:
        parser().error(str(error))
    if dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
        return
    output = output or Path("runs") / ("smoke" if cfg.smoke else "cifar10") / cfg.run_name()
    run_experiment(cfg, output)


if __name__ == "__main__":
    main()
