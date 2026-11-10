"""Controlled denoiser estimates, independent resampling and oracle data paths."""

import hashlib
import io
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from cifar10.data import (
    SyntheticNoise,
    build_datasets as build_base_datasets,
    normalize_imagenet,
    unnormalize_imagenet,
    ssl_transforms,
)
from cifar10.views import ViewTreatment
from denoisers import check_n2n_checkpoint


def fit_signature(config):
    """Settings that determine fitted weights; ignores the subsequent resampling arm."""
    fields = [
        "treatment",
        "condition",
        "noise",
        "noise_sampling",
        "seed",
        "smoke",
        "batch_size",
        "num_workers",
    ]
    if config["treatment"] in ("n2n", "nsp"):
        fields += ["method", "denoiser_patch_batch_size"]
        prefix = config["treatment"] + "_"
        fields += [name for name in config if name.startswith(prefix)]
    elif config["treatment"] == "seana":
        fields += [
            "seana_knots",
            "seana_fit_steps",
            "seana_fit_lr",
            "seana_patch_batch_size",
            "seana_stride",
            "sigma_d",
            "drunet_weights",
        ]
    return {
        "dataset": config.get("dataset", "cifar10"),
        **{name: config[name] for name in sorted(set(fields))},
    }


def load_fitted_treatment(treatment, path, config):
    data = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if payload.get("kind") != "view_treatment" or payload.get("format_version") != 1:
        raise ValueError("Expected a release-format view_treatment checkpoint")
    check_n2n_checkpoint(payload["config"])
    if fit_signature(payload["config"]) != fit_signature(config.resolved()):
        raise ValueError("Treatment checkpoint fitting recipe does not match this run")
    treatment.load_state_dict(payload["state_dict"], strict=True)
    treatment.input_checkpoint_sha256 = hashlib.sha256(data).hexdigest()
    return treatment.eval().requires_grad_(False)


class OracleDataset(Dataset):
    def __init__(self, base, cfg):
        self.base, self.cfg = base, cfg
        self.transforms = ssl_transforms(cfg.method)
        self.to_tensor, self.to_pil = T.ToTensor(), T.ToPILImage()
        self.noise = SyntheticNoise(cfg.condition, "stochastic", cfg.seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        clean, label = self.base[idx]
        if self.cfg.arm == "oracle_precrop":
            clean_tensor = self.to_tensor(clean)
            views = tuple(transform(self.to_pil(self.noise(clean_tensor))) for transform in self.transforms)
        else:
            # Only the oracle SSL views can access clean training images. Probe
            # training still uses the ordinary noisy dataset from build_datasets.
            views = tuple(transform(clean) for transform in self.transforms)
        return views, label


def build_datasets(cfg):
    datasets = build_base_datasets(cfg)
    if cfg.arm.startswith("oracle_"):
        datasets["ssl"] = OracleDataset(datasets["ssl"].base, cfg)
    return datasets


class IsolationTreatment(ViewTreatment):
    def __init__(self, cfg, device):
        super().__init__(cfg, device, initialize_from_checkpoint=bool(cfg.treatment_checkpoint))
        self.known_noise = SyntheticNoise(cfg.condition, "stochastic", cfg.seed)

    def fit(self, ssl_loader, raw_loader, device):
        if self.cfg.treatment_checkpoint:
            return load_fitted_treatment(self, self.cfg.treatment_checkpoint, self.cfg)
        return super().fit(ssl_loader, raw_loader, device)

    @torch.no_grad()
    def forward(self, normalized_view):
        arm = self.cfg.arm
        if arm == "oracle_postcrop":
            clean_view = unnormalize_imagenet(normalized_view).clamp(0, 1)
            return normalize_imagenet(self.known_noise(clean_view))
        if arm.endswith("resample"):
            image = unnormalize_imagenet(normalized_view).clamp(0, 1)
            if self.vst is not None:
                # Hold T -> DRUNet -> T^-1 fixed and change only the noise draw.
                estimate = self.vst(image, self.denoiser).clamp(0, 1)
            else:
                estimate = self.denoiser(image).clamp(0, 1)
            return normalize_imagenet(self.known_noise(estimate))
        return super().forward(normalized_view)
