"""Portable ImageNet-100 recipes; importing this module does not import torch."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import ClassVar

from cifar10.config import Config as BaseConfig
from cifar10.config import METHODS as METHODS, BACKBONES as BACKBONES, TREATMENTS as TREATMENTS

HF_DATASET = "clane9/imagenet-100"
HF_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"
CONDITIONS = {
    "gauss_low": dict(family="gaussian", sigma=0.1, sigma_d=0.10, sigma_vst=0.04),
    "gauss_high": dict(family="gaussian", sigma=0.5, sigma_d=0.50, sigma_vst=0.20),
    "shot_low": dict(family="shot", peak=30.0, sigma_d=0.13, sigma_vst=0.05),
    "shot_high": dict(family="shot", peak=5.0, sigma_d=0.32, sigma_vst=0.13),
    "speckle_low": dict(family="speckle", sigma=0.2, sigma_d=0.10, sigma_vst=0.04),
    "speckle_high": dict(family="speckle", sigma=0.6, sigma_d=0.30, sigma_vst=0.12),
}

METHOD_DEFAULTS = {
    "byol": dict(lr=0.45, proj_hidden=4096, proj_out=256),
    "simclr": dict(lr=0.3, proj_hidden=2048, proj_out=128),
    "vicreg": dict(lr=0.3, proj_hidden=8192, proj_out=8192, proj_layers=3),
    # Extensions of the released CIFAR methods, not captured ImageNet runs.
    "mocov2": dict(
        lr=0.03, optimizer="sgd", weight_decay=1e-4, warmup_epochs=0, proj_hidden=2048, proj_out=128
    ),
    "lejepa": dict(lr=5e-4, optimizer="adamw", weight_decay=5e-4, proj_hidden=2048, proj_out=128),
}


@dataclass
class Config(BaseConfig):
    dataset_name: ClassVar[str] = "imagenet100"
    num_classes: ClassVar[int] = 100
    condition_specs: ClassVar[dict] = CONDITIONS
    optimizers: ClassVar[tuple] = ("sgd", "adamw", "lars")
    backbone: str = "resnet50"
    data_root: str = "data/imagenet100"
    epochs: int = 100
    optimizer: str = "lars"
    lr: float = 0.3
    weight_decay: float = 1e-6
    lr_schedule: str = "cosine"
    warmup_epochs: int = 10
    image_size: int = 224
    source_size: int = 256
    fit_batch_size: int = 64
    probe_batch_size: int = 256
    probe_schedule_interval: str = "step"
    max_train_samples: int = 0
    max_val_samples: int = 0
    proj_hidden: int = 8192
    proj_out: int = 8192
    proj_layers: int = 3
    simclr_temperature: float = 0.1
    seana_type: str = "conditional"
    seana_fit_steps: int = 20000
    seana_patch_size: int = 64
    seana_calibration_batches: int = 50
    seana_per_image_norm: bool = True
    cond_hidden_dim: int = 64
    cond_delta_scale: float = 0.1
    n2n_fit_steps: int = 20000
    n2n_patch_size: int = 64
    nsp_patch_size: int = 160
    nsp_tds: int = 5
    color_jitter: bool = True

    def validate(self):
        for name in (
            "image_size",
            "source_size",
            "fit_batch_size",
            "probe_batch_size",
            "proj_hidden",
            "proj_out",
            "proj_layers",
            "seana_patch_size",
            "seana_calibration_batches",
            "cond_hidden_dim",
            "n2n_patch_size",
            "nsp_patch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        super().validate()
        if self.image_size < 32 or self.image_size > self.source_size or self.source_size % 8:
            raise ValueError("Require 32 <= image_size <= source_size, with source_size divisible by 8")
        if max(self.seana_patch_size, self.n2n_patch_size, self.nsp_patch_size) > self.source_size:
            raise ValueError("Denoiser fitting patches cannot exceed source_size")
        if self.seana_patch_size % 8 or self.n2n_patch_size < 2 or self.n2n_patch_size % 2:
            raise ValueError("SEANA patches must divide by 8; N2N patches must be even and at least 2")
        if self.seana_type not in ("conditional", "global"):
            raise ValueError("seana_type must be conditional or global")
        if self.source_size % self.nsp_ids:
            raise ValueError("NSP ids must divide source_size for full-image inference")
        if self.proj_layers < 2 or self.probe_schedule_interval not in ("step", "epoch"):
            raise ValueError("Invalid projector depth or probe schedule")
        if not math.isfinite(self.cond_delta_scale) or self.cond_delta_scale < 0:
            raise ValueError("cond_delta_scale must be finite and nonnegative")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in (self.max_train_samples, self.max_val_samples)
        ):
            raise ValueError("Sample limits must be nonnegative integers (0 means all)")
        if self.download:
            raise ValueError("Prepare ImageNet-100 first with python -m imagenet100.prepare")
        return self

    def resolved(self):
        data = super().resolved()
        data.update(
            view_pipeline="treatment_then_spatial_then_photometric",
            data_format="huggingface_save_to_disk",
            probe_evaluation_split="validation",
            lars_implementation="lightly.utils.lars.LARS",
            ssl_schedule_interval="step",
            seana_cache=False,
        )
        return data

    def run_name(self):
        return "imagenet100_" + super().run_name()


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    values = {**METHOD_DEFAULTS.get(values.get("method", "vicreg"), {}), **values}
    known = {f.name for f in fields(Config)}
    if set(values) - known:
        raise ValueError(f"Unknown configuration fields: {sorted(set(values) - known)}")
    return Config(**values).validate()


def smoke_overrides():
    return dict(
        smoke=True,
        epochs=1,
        probe_epochs=1,
        batch_size=4,
        fit_batch_size=4,
        probe_batch_size=4,
        num_workers=0,
        image_size=32,
        source_size=32,
        seana_fit_steps=1,
        seana_patch_size=32,
        seana_calibration_batches=1,
        n2n_fit_steps=1,
        n2n_patch_size=32,
        n2n_base_ch=8,
        nsp_fit_steps=1,
        nsp_patch_size=32,
        nsp_tds=4,
        nsp_base_ch=8,
        nsp_num_module=0,
        denoiser_patch_batch_size=2,
        seana_patch_batch_size=2,
        lejepa_num_slices=8,
        moco_queue_size=32,
        warmup_epochs=0,
        proj_hidden=64,
        proj_out=32,
    )
