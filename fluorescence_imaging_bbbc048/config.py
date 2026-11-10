"""BBBC048 recipes, independent of the training dependencies."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar

from cifar10.config import METHODS, BACKBONES

PHASES = ("G1", "S", "G2", "prophase", "metaphase", "anaphase", "telophase")
TREATMENTS = ("baseline", "seana", "vst_only", "resample_only", "nsp", "n2n")
SPLITS = ("random", "acquisition_order", "acquisition_order_global")
VIEW_ORDERS = ("paper", "shared", "pre_crop")
METHOD_DEFAULTS = {
    "vicreg": dict(proj_hidden=2048, proj_out=2048),
    "lejepa": dict(batch_size=256, fit_batch_size=256, lr_schedule="cosine", warmup_epochs=10),
    "mocov2": dict(sigma_vst=0.6),
}


@dataclass
class Config:
    dataset_name: ClassVar[str] = "bbbc048"
    num_classes: ClassVar[int] = 7
    method: str = "vicreg"
    backbone: str = "resnet18"
    treatment: str = "baseline"
    manifest: str = "data/bbbc048/manifest.json"
    split_mode: str = "random"
    train_fraction: float = 0.8
    split_seed: int | None = None  # None retains the captured coupling to training seed
    seed: int = 0
    device: str = "auto"
    num_workers: int = 4
    batch_size: int = 128
    fit_batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-6
    lr_schedule: str = "none"
    warmup_epochs: int = 0
    clip_grad_norm: float = 0.0
    amp: bool = False
    image_size: int = 64
    crop_scale_min: float = 0.8
    cj_strength: float = 0.0
    cj_prob: float = 0.0
    gaussian_blur: float = 0.0
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    proj_hidden: int = 512
    proj_out: int = 128
    proj_layers: int = 2
    byol_momentum: float = 0.99
    byol_momentum_end: float = 1.0
    simclr_temperature: float = 0.5
    moco_temperature: float = 0.2
    moco_momentum: float = 0.999
    moco_queue_size: int = 4096
    moco_shuffle_bn: bool = False  # absent from the captured microscopy loop
    lejepa_num_slices: int = 1024
    lejepa_num_points: int = 17
    lejepa_sigreg_weight: float = 0.05
    lejepa_objective: str = "legacy"  # captured inv + lambda*SIGReg weighting
    probe_epochs: int = 100
    probe_batch_size: int = 256
    probe_lr: float = 1e-3
    probe_weight_decay: float = 0.0
    probe_inputs: str = "historical"  # denoise train AND test for vst_only/nsp/n2n
    probe_report_best: bool = False
    view_order: str = "paper"
    sigma_d: float = 0.039
    sigma_vst: float = 0.8
    seana_knots: int = 128
    seana_fit_steps: int = 5000
    seana_fit_lr: float = 1e-2
    seana_patch_size: int = 64
    seana_patch_batch_size: int = 4
    seana_stride: int = 4
    seana_calibration_batches: int = 50
    drunet_weights: str = "download"
    n2n_fit_steps: int = 20000
    n2n_base_ch: int = 48
    n2n_fit_lr: float = 3e-4
    n2n_gamma: float = 2.0
    n2n_patch_size: int = 64
    nsp_fit_steps: int = 20000
    nsp_base_ch: int = 128
    nsp_num_module: int = 9
    nsp_ids: int = 2
    nsp_tds: int = 4
    nsp_patch_size: int = 64
    nsp_fit_lr: float = 1e-4
    denoiser_patch_batch_size: int = 8
    treatment_checkpoint: str = ""
    cache_estimates: bool = False
    smoke: bool = False

    def validate(self):
        choices = dict(
            method=METHODS,
            backbone=BACKBONES,
            treatment=TREATMENTS,
            split_mode=SPLITS,
            view_order=VIEW_ORDERS,
            device=("auto", "cpu", "cuda"),
            optimizer=("adam", "adamw"),
            lr_schedule=("none", "cosine"),
            probe_inputs=("historical", "raw"),
            lejepa_objective=("legacy", "upstream"),
        )
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        nonnegative_ints = {"seed", "num_workers", "warmup_epochs", "nsp_num_module"}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(field.default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{field.name} must be boolean")
            elif isinstance(field.default, int):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < (0 if field.name in nonnegative_ints else 1)
                ):
                    raise ValueError(f"Invalid integer {field.name}")
            elif isinstance(field.default, float):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid number {field.name}")
        if self.split_seed is not None and (
            isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int) or self.split_seed < 0
        ):
            raise ValueError("split_seed must be a nonnegative integer or null")
        if self.batch_size < 2 or self.warmup_epochs >= self.epochs:
            raise ValueError("SSL needs batch_size >= 2 and warmup_epochs < epochs")
        if not 0 < self.train_fraction < 1 or not 0 < self.crop_scale_min <= 1:
            raise ValueError("Invalid train fraction or crop scale")
        if self.cj_prob > 1 or self.gaussian_blur > 1 or self.lejepa_sigreg_weight >= 1:
            raise ValueError("Probabilities/LeJEPA weight must be within their bounds")
        if any(not 0 < v <= 1 for v in (self.byol_momentum, self.byol_momentum_end, self.moco_momentum)):
            raise ValueError("EMA momenta must be in (0,1]")
        for name in (
            "lr",
            "probe_lr",
            "normalize_std",
            "seana_fit_lr",
            "n2n_fit_lr",
            "nsp_fit_lr",
            "simclr_temperature",
            "moco_temperature",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size < 32 or self.image_size % 8 or self.seana_patch_size % 8:
            raise ValueError("Image size must be >=32; image and SEANA patch sizes must divide by 8")
        if self.n2n_patch_size < 2 or self.n2n_patch_size % 2:
            raise ValueError("N2N patch size must be even and at least 2")
        if (
            self.nsp_tds <= self.nsp_ids
            or self.nsp_patch_size % self.nsp_tds
            or self.nsp_base_ch % 2
            or (2 * self.nsp_base_ch) % self.nsp_ids**2
        ):
            raise ValueError("Invalid NSP subsampling, patch size or channels")
        if self.seana_knots < 2 or self.seana_stride > self.seana_patch_size or self.proj_layers < 2:
            raise ValueError("Invalid VST or projection settings")
        if self.lejepa_num_points < 3 or self.lejepa_num_points % 2 != 1:
            raise ValueError("LeJEPA integration needs an odd number of points >=3")
        if (
            self.cache_estimates
            and self.treatment in ("seana", "vst_only", "resample_only")
            and self.view_order != "shared"
        ):
            raise ValueError("Cached SEANA estimates require view_order=shared")
        if self.treatment == "baseline" and (self.treatment_checkpoint or self.cache_estimates):
            raise ValueError("Baseline has no treatment checkpoint or estimate cache")
        return self

    def probe_preprocessing(self):
        return (
            self.treatment
            if self.probe_inputs == "historical" and self.treatment in ("vst_only", "nsp", "n2n")
            else "none"
        )

    def resolved(self):
        from denoisers import N2N_IMPLEMENTATION

        return {
            **asdict(self),
            **({"n2n_implementation": N2N_IMPLEMENTATION} if self.treatment == "n2n" else {}),
            "schema_version": 1,
            "dataset": self.dataset_name,
            "num_classes": self.num_classes,
            "phases": list(PHASES),
            "effective_split_seed": self.seed if self.split_seed is None else self.split_seed,
            "probe_preprocessing": self.probe_preprocessing(),
            "evaluation_images": "acquired_noisy",
            "synthetic_corruption": False,
        }

    def run_name(self):
        digest = hashlib.sha256(json.dumps(self.resolved(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.method}_{self.backbone}_{self.split_mode}_{self.treatment}_s{self.seed}_{digest}"


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    values = {**METHOD_DEFAULTS.get(values.get("method", "vicreg"), {}), **values}
    extra = set(values) - {f.name for f in fields(Config)}
    if extra:
        raise ValueError(f"Unknown settings: {sorted(extra)}")
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
        seana_fit_steps=1,
        seana_patch_size=32,
        seana_patch_batch_size=2,
        seana_calibration_batches=1,
        n2n_fit_steps=1,
        n2n_patch_size=32,
        n2n_base_ch=8,
        nsp_fit_steps=1,
        nsp_patch_size=32,
        nsp_base_ch=8,
        nsp_num_module=0,
        denoiser_patch_batch_size=2,
        lejepa_num_slices=8,
        proj_hidden=64,
        proj_out=32,
        moco_queue_size=32,
        warmup_epochs=0,
    )
