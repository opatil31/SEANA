"""Run ImageNet-100 SSL, then a frozen 100-class noisy-train / clean-validation probe."""

import argparse
import json
import math
from pathlib import Path

from cifar10.train import run_experiment, parser as base_parser
from .config import CONDITIONS, load_config, smoke_overrides


def make_optimizer(model, cfg):
    import torch
    from lightly.utils.lars import LARS

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg.optimizer == "lars":
        return LARS(groups, lr=cfg.lr, momentum=0.9, trust_coefficient=0.001)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9)


def learning_rate(cfg, step, steps_per_epoch):
    warmup = cfg.warmup_epochs * steps_per_epoch
    if warmup and step < warmup:
        return cfg.lr * (0.001 + 0.999 * step / warmup)
    if cfg.lr_schedule == "cosine":
        progress = (step - warmup) / max(1, cfg.epochs * steps_per_epoch - warmup)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr


def pretrain(model, criterion, treatment, loader, cfg, device):
    import torch
    from lightly.models.utils import update_momentum
    from cifar10.models import ssl_loss

    if not len(loader):
        raise ValueError("SSL loader has no full batch; reduce batch_size")
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    parameters = [p for p in model.parameters() if p.requires_grad]
    model.train()
    criterion.train()
    history = []
    for epoch in range(cfg.epochs):
        total = 0.0
        for index, (image, _) in enumerate(loader):
            step = epoch * len(loader) + index
            rate = learning_rate(cfg, step, len(loader))
            for group in optimizer.param_groups:
                group["lr"] = rate
            x0, x1 = treatment(image.to(device))
            if cfg.method == "mocov2":
                update_momentum(model.backbone, model.backbone_momentum, cfg.moco_momentum)
                update_momentum(model.projection_head, model.projection_head_momentum, cfg.moco_momentum)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
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
            if cfg.method == "byol":
                fraction = step / max(1, cfg.epochs * len(loader) - 1)
                momentum = 1 - (1 - cfg.byol_momentum) * (1 + math.cos(math.pi * fraction)) / 2
                update_momentum(model.backbone, model.backbone_momentum, momentum)
                update_momentum(model.projection_head, model.projection_head_momentum, momentum)
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total / len(loader), "lr": rate})
        print(f"[ssl] epoch={epoch + 1} loss={total / len(loader):.6f} lr={rate:.6g}")
    return history


def run(cfg, output_dir):
    from .data import build_datasets
    from .models import build_ssl
    from .views import ViewTreatment

    return run_experiment(
        cfg,
        output_dir,
        dataset_builder=build_datasets,
        model_builder=build_ssl,
        treatment_builder=ViewTreatment,
        pretrain_fn=pretrain,
        extra_source_roots=("imagenet100",),
    )


def parser():
    p = base_parser(conditions=CONDITIONS)
    p.description = __doc__
    for name in (
        "image_size",
        "source_size",
        "fit_batch_size",
        "probe_batch_size",
        "max_train_samples",
        "max_val_samples",
        "nsp_patch_size",
        "n2n_patch_size",
        "seana_patch_size",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    p.add_argument("--seana-type", choices=("conditional", "global"))
    p.add_argument("--color-jitter", action=argparse.BooleanOptionalAction, default=None)
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    path, output = args.pop("config"), args.pop("output_dir")
    dry_run, smoke = args.pop("dry_run"), args.pop("smoke")
    if smoke:
        args.update(smoke_overrides())
    try:
        cfg = load_config(path, **args)
    except (ValueError, TypeError) as error:
        parser().error(str(error))
    if dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
    else:
        run(cfg, output or Path("runs/imagenet100") / cfg.run_name())


if __name__ == "__main__":
    main()
