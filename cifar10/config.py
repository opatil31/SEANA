"""Explicit CIFAR-10 recipes, independent of the training dependencies."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar
from denoisers import N2N_IMPLEMENTATION

METHODS = ("vicreg", "simclr", "byol", "mocov2", "lejepa")
BACKBONES = ("resnet18", "resnet50")
TREATMENTS = ("baseline", "seana", "nsp", "n2n")

METHOD_DEFAULTS = {
    "vicreg": {},
    "byol": {"clip_grad_norm": 0.0},
    "simclr": {"clip_grad_norm": 0.0},
    "mocov2": {"lr": 0.03, "weight_decay": 0.0001, "lr_schedule": "cosine", "clip_grad_norm": 0.0},
    "lejepa": {
        "lr": 0.0005,
        "weight_decay": 0.0005,
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "warmup_epochs": 10,
        "clip_grad_norm": 0.0,
    },
}

# sigma_d and sigma_vst retain the captured high-/low-noise grid values.
# Low-noise sigma_d values were inferred in that grid, not independently tuned.
CONDITIONS = {
    "gauss_low": dict(family="gaussian", sigma=0.1, sigma_d=0.1, sigma_vst=0.1),
    "gauss_high": dict(family="gaussian", sigma=0.5, sigma_d=0.5, sigma_vst=0.5),
    "pg_low": dict(family="poisson_gaussian", peak=30.0, sigma=0.01, sigma_d=0.1, sigma_vst=0.05),
    "pg_high": dict(family="poisson_gaussian", peak=5.0, sigma=0.2, sigma_d=0.2, sigma_vst=0.1),
    "speckle_low": dict(family="speckle", sigma=0.2, sigma_d=0.1, sigma_vst=0.05),
    "speckle_high": dict(family="speckle", sigma=0.6, sigma_d=0.1, sigma_vst=0.05),
}


@dataclass
class Config:
    dataset_name: ClassVar[str] = "cifar10"
    num_classes: ClassVar[int] = 10
    condition_specs: ClassVar[dict] = CONDITIONS
    optimizers: ClassVar[tuple] = ("sgd", "adamw")
    backbone: str = "resnet18"
    method: str = "vicreg"
    treatment: str = "baseline"
    condition: str = "gauss_low"
    seed: int = 0
    data_root: str = "data/cifar10"
    download: bool = False
    device: str = "auto"
    num_workers: int = 4
    batch_size: int = 256
    epochs: int = 200
    lr: float = 0.06
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    lr_schedule: str = "none"
    warmup_epochs: int = 0
    amp: bool = False
    clip_grad_norm: float = 1.0
    # The corrected captured grids redraw noise on each access. Fixed corruption
    # remains selectable for the paper's offline-corruption protocol.
    noise_sampling: str = "stochastic"
    probe_epochs: int = 100
    probe_lr: float = 0.1
    probe_weight_decay: float = 0.0
    # Source images. N2N/NSP denoise the noisy training inputs before the backbone.
    probe_train_inputs: str = "noisy"
    probe_test_inputs: str = "clean"
    # Default final head is chosen by epoch count, never by test-set accuracy.
    # Optional historical per-epoch evaluation records the maximum separately.
    probe_report_best: bool = False
    byol_momentum: float = 0.99
    simclr_temperature: float = 0.5
    moco_temperature: float = 0.2
    moco_momentum: float = 0.999
    moco_queue_size: int = 65536
    lejepa_num_slices: int = 1024
    lejepa_num_points: int = 17
    lejepa_sigreg_weight: float = 0.05
    lejepa_objective: str = "upstream"
    seana_knots: int = 128
    seana_fit_steps: int = 5000
    seana_fit_lr: float = 0.01
    seana_patch_batch_size: int = 16
    seana_stride: int = 4
    sigma_d: float | None = None
    sigma_vst: float | None = None
    drunet_weights: str = "download"
    n2n_fit_steps: int = 10000
    n2n_base_ch: int = 48
    n2n_fit_lr: float = 0.0003
    n2n_gamma: float = 2.0
    nsp_fit_steps: int = 20000
    nsp_base_ch: int = 128
    nsp_num_module: int = 9
    nsp_ids: int = 2
    nsp_tds: int = 4
    nsp_fit_lr: float = 0.0001
    denoiser_patch_batch_size: int = 8
    smoke: bool = False

    def validate(self):
        for field, allowed in (
            ("method", METHODS),
            ("backbone", BACKBONES),
            ("treatment", TREATMENTS),
            ("condition", self.condition_specs),
            ("noise_sampling", ("fixed", "stochastic")),
            ("optimizer", self.optimizers),
            ("lr_schedule", ("none", "cosine")),
            ("lejepa_objective", ("upstream", "legacy")),
        ):
            if getattr(self, field) not in allowed:
                raise ValueError(f"{field} must be one of {tuple(allowed)}")
        if self.probe_train_inputs != "noisy" or self.probe_test_inputs != "clean":
            raise ValueError("This release uses noisy-train / clean-test linear probes.")
        for field in (
            "epochs",
            "probe_epochs",
            "batch_size",
            "moco_queue_size",
            "seana_fit_steps",
            "n2n_fit_steps",
            "nsp_fit_steps",
            "seana_patch_batch_size",
            "denoiser_patch_batch_size",
            "lejepa_num_slices",
            "seana_knots",
            "seana_stride",
            "nsp_ids",
            "nsp_tds",
            "nsp_base_ch",
            "n2n_base_ch",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for SSL BatchNorm and VICReg")
        if self.num_workers < 0 or self.seed < 0 or self.warmup_epochs < 0:
            raise ValueError("workers, seed and warmup_epochs must be nonnegative")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be less than epochs")
        if not 0 < self.byol_momentum <= 1 or not 0 < self.moco_momentum <= 1:
            raise ValueError("EMA momenta must be in (0, 1]")
        if not 0 <= self.lejepa_sigreg_weight < 1:
            raise ValueError("lejepa_sigreg_weight must be in [0, 1)")
        if self.lejepa_num_points < 3 or self.lejepa_num_points % 2 != 1:
            raise ValueError("lejepa_num_points must be odd and at least 3")
        nsp_patch_size = getattr(self, "nsp_patch_size", 32)
        if self.nsp_tds <= self.nsp_ids or nsp_patch_size % self.nsp_tds:
            raise ValueError("NSP requires ids < tds and tds must divide its training patch size")
        if self.nsp_base_ch % 2 or (self.nsp_base_ch * 2) % self.nsp_ids**2:
            raise ValueError("NSP channels must be even and 2*channels divisible by ids**2")
        if self.nsp_num_module < 0 or self.seana_stride > getattr(self, "seana_patch_size", 32):
            raise ValueError("Invalid denoiser channels/modules or SEANA stride")
        for field in (
            "lr",
            "probe_lr",
            "n2n_fit_lr",
            "nsp_fit_lr",
            "seana_fit_lr",
            "simclr_temperature",
            "moco_temperature",
        ):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        for field in (
            "sigma_d",
            "sigma_vst",
            "weight_decay",
            "probe_weight_decay",
            "clip_grad_norm",
            "n2n_gamma",
        ):
            value = getattr(self, field)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{field} must be finite and nonnegative")
        return self

    def probe_train_preprocessing(self):
        return self.treatment if self.treatment in ("n2n", "nsp") else "none"

    def resolved(self):
        data = asdict(self)
        data["noise"] = self.condition_specs[self.condition].copy()
        data["dataset"] = self.dataset_name
        data["num_classes"] = self.num_classes
        for name in ("sigma_d", "sigma_vst"):
            data[name] = data[name] if data[name] is not None else data["noise"][name]
        data["schema_version"] = 1
        # A shared protocol identity keeps comparison arms together, while
        # distinguishing this recipe from earlier probes trained on raw inputs.
        data["probe_protocol"] = "denoise_train_clean_test_v1"
        data["probe_train_preprocessing"] = self.probe_train_preprocessing()
        data["probe_test_preprocessing"] = "none"
        if self.treatment == "n2n":
            data["n2n_implementation"] = N2N_IMPLEMENTATION
        return data

    def run_name(self):
        # Include all settings to avoid mixing incompatible experiments.
        digest = hashlib.sha256(json.dumps(self.resolved(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.backbone}_{self.method}_{self.condition}_{self.treatment}_s{self.seed}_{digest}"


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    values = {**METHOD_DEFAULTS.get(values.get("method", "vicreg"), {}), **values}
    known = {field.name for field in fields(Config)}
    extra = set(values) - known
    if extra:
        raise ValueError(f"Unknown configuration fields: {sorted(extra)}")
    return Config(**values).validate()
