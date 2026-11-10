"""Portable fetal recipes; the historical and validation protocols remain explicit."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar

from fluorescence_imaging_bbbc048.config import Config as MicroscopyConfig
from fluorescence_imaging_bbbc048.config import smoke_overrides as microscopy_smoke
from cifar10.config import METHODS as METHODS, BACKBONES as BACKBONES

CLASSES = ("Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other")
TREATMENTS = ("baseline", "seana", "nsp", "n2n")
MAIN_TREATMENTS = ("baseline", "nsp", "seana")
SPLITS = ("official", "official_validation")
METHOD_DEFAULTS = {
    "simclr": dict(sigma_vst=0.16),
    "vicreg": dict(sigma_vst=0.16, proj_hidden=2048, proj_out=2048),
    "byol": dict(sigma_vst=0.12),
    "mocov2": dict(sigma_vst=0.04),
    "lejepa": dict(sigma_vst=0.24, batch_size=256, lr_schedule="cosine", warmup_epochs=10),
}


@dataclass
class Config(MicroscopyConfig):
    dataset_name: ClassVar[str] = "fetal_planes_db"
    num_classes: ClassVar[int] = 6
    manifest: str = "data/fetal/manifest.json"
    split_mode: str = "official"
    split_seed: int | None = 0
    validation_fraction: float = 0.1
    image_size: int = 128
    region_size: int = 192
    view_order: str = "shared"
    fit_batch_size: int = 32
    sigma_d: float = 0.08
    sigma_vst: float = 0.16
    sigma_mode: str = "fixed"
    sigma_multiplier: float = 1.0
    sigma_rule_max_batches: int = 50
    seana_fit_lr: float = 1e-3
    seana_stride: int = 32
    probe_selection: str = "final"
    evaluate_test: bool = True
    probe_report_best: bool = False

    def validate(self):
        # Reuse common numerical and network checks, with this dataset's split vocabulary.
        common = {f.name: getattr(self, f.name) for f in fields(MicroscopyConfig)}
        common["split_mode"] = "random"
        MicroscopyConfig(**common).validate()
        if self.split_mode not in SPLITS or self.treatment not in TREATMENTS:
            raise ValueError("Unsupported fetal split or treatment")
        if self.split_seed is None or not 0 < self.validation_fraction < 1:
            raise ValueError("A fixed patient split seed and validation fraction in (0,1) are required")
        if self.region_size < self.image_size:
            raise ValueError("Native region must be at least as large as the SSL view")
        if any(type(v) is not int or v < 1 for v in (self.region_size, self.sigma_rule_max_batches)):
            raise ValueError("Invalid region size or sigma calibration batch count")
        if not math.isfinite(self.sigma_multiplier) or not isinstance(self.evaluate_test, bool):
            raise ValueError("Invalid multiplier or evaluation flag")
        if self.view_order != "shared" or self.cj_prob or self.gaussian_blur:
            raise ValueError("Fetal recipes use shared native estimates, with no color jitter or blur")
        if self.sigma_mode not in ("fixed", "unsupervised") or self.sigma_multiplier <= 0:
            raise ValueError("Invalid sigma rule or multiplier")
        if self.sigma_mode != "fixed" and self.treatment != "seana":
            raise ValueError("The VST-space sigma rule requires SEANA")
        if self.probe_selection not in ("final", "validation_macro_f1"):
            raise ValueError("Invalid probe selection")
        if self.probe_selection != "final" and self.split_mode != "official_validation":
            raise ValueError("Validation selection requires a validation partition")
        if self.probe_report_best:
            raise ValueError("Fetal release runs do not select or track maxima over test epochs")
        return self

    def resolved(self):
        return dict(
            **asdict(self),
            schema_version=1,
            dataset=self.dataset_name,
            num_classes=6,
            classes=list(CLASSES),
            probe_preprocessing=self.probe_preprocessing(),
            evaluation_images="acquired_ultrasound",
            synthetic_corruption=False,
            native_crop=True,
            vertical_flip_probability=0.0,
        )

    def run_name(self):
        digest = hashlib.sha256(json.dumps(self.resolved(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.method}_{self.backbone}_{self.split_mode}_{self.treatment}_s{self.seed}_{digest}"


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    values = {**METHOD_DEFAULTS.get(values.get("method", "vicreg"), {}), **values}
    extra = set(values) - {f.name for f in fields(Config)}
    if extra:
        raise ValueError(f"Unknown fetal settings: {sorted(extra)}")
    return Config(**values).validate()


def smoke_overrides():
    return {**microscopy_smoke(), "region_size": 40, "seana_stride": 16, "sigma_rule_max_batches": 2}
