"""Microscopy views and train-split-bound fitted treatment checkpoints."""

from dataclasses import replace
import io
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from cifar10.train import sha256_file, write_json
from cifar10.views import ViewTreatment as BaseTreatment
from seana import augmentation
from denoisers import N2N_IMPLEMENTATION, check_n2n_checkpoint


VST_ARMS = ("seana", "vst_only", "resample_only")


def fit_identity(cfg, dataset):
    kind = "learned_vst" if cfg.treatment in ("seana", "vst_only") else cfg.treatment
    fields = {
        "learned_vst": (
            "sigma_d",
            "seana_knots",
            "seana_fit_steps",
            "seana_fit_lr",
            "seana_patch_size",
            "seana_patch_batch_size",
            "seana_stride",
            "seana_calibration_batches",
            "fit_batch_size",
        ),
        "resample_only": ("sigma_d", "seana_knots"),
        "n2n": (
            "n2n_base_ch",
            "n2n_fit_steps",
            "n2n_fit_lr",
            "n2n_gamma",
            "n2n_patch_size",
            "denoiser_patch_batch_size",
            "fit_batch_size",
        ),
        "nsp": (
            "nsp_base_ch",
            "nsp_num_module",
            "nsp_ids",
            "nsp_tds",
            "nsp_fit_steps",
            "nsp_fit_lr",
            "nsp_patch_size",
            "denoiser_patch_batch_size",
            "fit_batch_size",
        ),
    }
    return dict(
        kind=kind,
        **({"n2n_implementation": N2N_IMPLEMENTATION} if cfg.treatment == "n2n" else {}),
        dataset_fingerprint=dataset["fingerprint"],
        training_split_sha256=dataset["split_sha256"]["train"],
        fit_settings={k: getattr(cfg, k) for k in fields.get(kind, ())},
        smoke=cfg.smoke,
    )


class ViewTreatment(BaseTreatment):
    def __init__(self, cfg, device, *, initialize_from_checkpoint=False):
        super().__init__(
            replace(cfg, treatment="seana") if cfg.treatment in VST_ARMS else cfg,
            device,
            initialize_from_checkpoint=initialize_from_checkpoint,
        )
        self.cfg = cfg
        spatial = [
            T.RandomResizedCrop(
                cfg.image_size, scale=(cfg.crop_scale_min, 1), ratio=(0.9, 1.1), antialias=True
            ),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
        if cfg.gaussian_blur:
            kernel = max(3, int(0.1 * cfg.image_size)) | 1
            spatial.append(T.RandomApply([T.GaussianBlur(kernel)], p=cfg.gaussian_blur))
        photo = []
        if cfg.cj_prob and cfg.cj_strength:
            photo.append(
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.8 * cfg.cj_strength,
                            contrast=0.8 * cfg.cj_strength,
                            saturation=0.8 * cfg.cj_strength,
                            hue=0.2 * cfg.cj_strength,
                        )
                    ],
                    p=cfg.cj_prob,
                )
            )
        photo.append(T.Normalize((cfg.normalize_mean,) * 3, (cfg.normalize_std,) * 3))
        self.spatial, self.photometric = T.Compose(spatial), T.Compose(photo)

    def fit(self, loader, device):
        cfg = self.cfg
        if cfg.treatment in ("seana", "vst_only"):
            self.denoiser.eval().requires_grad_(False)
            self.vst.train()
            self.vst.calibrate_from_dataloader(loader, max_batches=cfg.seana_calibration_batches)
            self.vst.fit_dataset(
                loader,
                self.denoiser,
                total_steps=cfg.seana_fit_steps,
                patch_size=cfg.seana_patch_size,
                patch_batch_size=cfg.seana_patch_batch_size,
                stride=cfg.seana_stride,
                lr=cfg.seana_fit_lr,
            )
        elif cfg.treatment == "n2n":
            from denoisers.n2n import train_n2n_denoiser

            train_n2n_denoiser(
                self.denoiser,
                loader,
                device,
                total_steps=cfg.n2n_fit_steps,
                patch_size=cfg.n2n_patch_size,
                patch_batch_size=cfg.denoiser_patch_batch_size,
                lr=cfg.n2n_fit_lr,
                gamma=cfg.n2n_gamma,
                unnormalize_fn=None,
                patch_augment_fn=augmentation,
            )
        elif cfg.treatment == "nsp":
            from denoisers.nsp import train_nsp_denoiser

            train_nsp_denoiser(
                self.denoiser,
                loader,
                device,
                total_steps=cfg.nsp_fit_steps,
                patch_size=cfg.nsp_patch_size,
                patch_batch_size=cfg.denoiser_patch_batch_size,
                lr=cfg.nsp_fit_lr,
                tds=cfg.nsp_tds,
                seed=cfg.seed,
                unnormalize_fn=None,
            )
        # Identity VST: deliberately no calibration or fitting, range remains [0,1].
        self.eval().requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        return self

    @torch.no_grad()
    def clean_estimate(self, image):
        if self.vst is not None:
            return self.vst(image, self.denoiser).clamp(0, 1)
        return image if self.denoiser is None else self.denoiser(image).clamp(0, 1)

    @torch.no_grad()
    def renoise(self, estimate, sigma):
        if sigma == 0:
            return estimate
        transformed = self.vst.spline1(self.vst._normalize(estimate))
        noisy = transformed + sigma * torch.randn_like(transformed)
        return self.vst._denormalize(self.vst.spline1(noisy, inverse=True)).clamp(0, 1)

    @torch.no_grad()
    def forward(self, image, *, input_is_estimate=False):
        cfg = self.cfg
        sigma = 0.0 if cfg.treatment == "vst_only" else cfg.sigma_vst
        if input_is_estimate and cfg.treatment in VST_ARMS and cfg.view_order != "shared":
            raise ValueError("Precomputed VST estimates require shared view order")
        if cfg.treatment in VST_ARMS and cfg.view_order == "pre_crop":
            sources = self.vst.augment_pair(image, self.denoiser, sigma_vst=sigma, amp=False)
        elif cfg.treatment in VST_ARMS and cfg.view_order == "paper":
            sources = (image, image)
        else:
            estimate = image if input_is_estimate else self.clean_estimate(image)
            sources = (estimate, estimate)
        result = []
        for source in sources:
            cropped = torch.stack([self.spatial(x) for x in source]).clamp(0, 1)
            if cfg.treatment in VST_ARMS:
                if cfg.view_order == "paper":
                    cropped = self.vst.augment(cropped, self.denoiser, sigma_vst=sigma, amp=False).clamp(0, 1)
                elif cfg.view_order == "shared":
                    cropped = self.renoise(cropped, sigma)
            result.append(torch.stack([self.photometric(x) for x in cropped]))
        return tuple(result)


def load_treatment(path, cfg, dataset, device):
    raw = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != "bbbc048_treatment":
        raise ValueError("Expected a BBBC048 treatment checkpoint")
    check_n2n_checkpoint(payload["config"])
    if payload["fit_identity"] != fit_identity(cfg, dataset):
        raise ValueError("Treatment was fitted with a different training split, dataset or fitting recipe")
    treatment = ViewTreatment(cfg, device, initialize_from_checkpoint=True)
    treatment.load_state_dict(payload["state_dict"], strict=True)
    treatment.eval().requires_grad_(False)
    treatment.fit_provenance = payload["fit_provenance"]
    return treatment, hashlib.sha256(raw).hexdigest()


class CachedCells(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        if sha256_file(self.path) != self.meta["sha256"]:
            raise ValueError("Clean-estimate cache hash mismatch")
        self.images = None

    def __len__(self):
        return len(self.meta["labels"])

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return torch.from_numpy(np.array(self.images[index], copy=True)), self.meta["labels"][index]


def cache_estimates(base, treatment, device, path, identity, batch_size):
    """Create a per-run disk cache after fitting, without changing model weights."""
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels, count, output = [], 0, None
    for images, target in DataLoader(base, batch_size=batch_size, shuffle=False):
        clean = treatment.clean_estimate(images.to(device)).cpu().numpy()
        if output is None:
            output = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(len(base), *clean.shape[1:])
            )
        output[count : count + len(clean)] = clean
        labels.extend(target.tolist())
        count += len(clean)
    if count != len(base) or not count:
        raise ValueError("Incomplete/empty estimate cache")
    output.flush()
    del output
    write_json(
        path.with_suffix(".json"),
        dict(schema_version=1, identity=identity, labels=labels, sha256=sha256_file(path)),
    )
    return CachedCells(path)
