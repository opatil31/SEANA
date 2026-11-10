"""Verified native regions, split by patient before fitting any learned component."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

from cifar10.train import sha256_file
from .config import CLASSES
from .prepare import validate_manifest
from .splits import partitions, split_metadata


class FetalDataset(Dataset):
    def __init__(self, manifest):
        manifest = Path(manifest)
        self.payload = json.loads(manifest.read_text())
        validate_manifest(self.payload)
        self.records = self.payload["records"]
        self.fingerprint = self.payload["fingerprint"]
        self.path = manifest.parent / "images.npy"
        if sha256_file(self.path) != self.payload["images_sha256"]:
            raise ValueError("Prepared images hash mismatch")
        self.images = None
        self.region_size = self.payload["shape"][1]

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "images": None}

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if list(self.images.shape) != self.payload["shape"] or self.images.dtype != np.uint8:
                raise ValueError("Prepared array shape/dtype mismatch")
        crop = np.array(self.images[index], copy=True)
        if hashlib.sha256(crop.tobytes()).hexdigest() != self.records[index]["crop_sha256"]:
            raise ValueError("Prepared crop bytes changed")
        image = torch.from_numpy(crop).float().div_(255).unsqueeze(0).repeat(3, 1, 1)
        return image, self.records[index]["label"]


class FakeFetal(Dataset):
    def __init__(self, region_size):
        self.region_size = region_size
        self.fingerprint = f"fake_fetal_v1_region{region_size}"
        self.records = [
            dict(
                image_id=f"image{i}",
                patient_id=f"patient{i // 6}",
                label=i % 6,
                plane=CLASSES[i % 6],
                machine=("Aloka", "Voluson E6", "Voluson S10")[i % 3],
                official_split="train" if i < 30 else "test",
            )
            for i in range(42)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        x = torch.rand(
            1, self.region_size, self.region_size, generator=torch.Generator().manual_seed(80000 + index)
        )
        return x.repeat(3, 1, 1), self.records[index]["label"]


class PatientSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.records = [dataset.records[i] for i in indices]


def build_datasets(cfg):
    base = FakeFetal(cfg.region_size) if cfg.smoke else FetalDataset(cfg.manifest)
    if base.region_size != cfg.region_size:
        raise ValueError("Prepared region geometry differs from the experiment recipe")
    indices = partitions(base.records, cfg.split_mode, cfg.validation_fraction, cfg.split_seed)
    if not cfg.smoke:
        for plan in base.payload["prepared_splits"]:
            if (plan["mode"], plan["validation_fraction"], plan["seed"]) == (
                cfg.split_mode,
                cfg.validation_fraction,
                cfg.split_seed,
            ):
                indices = plan["indices"]
    metadata = dict(
        dataset="fake_fetal" if cfg.smoke else "fetal_planes_db",
        fingerprint=base.fingerprint,
        classes=list(CLASSES),
        region_size=cfg.region_size,
        split_mode=cfg.split_mode,
        split_seed=cfg.split_seed,
        validation_fraction=cfg.validation_fraction,
        **split_metadata(base.records, indices),
    )
    return {name: PatientSubset(base, ids) for name, ids in indices.items()}, metadata


def eval_transform(cfg):
    return T.Compose(
        [
            T.CenterCrop(cfg.image_size),
            T.Normalize((cfg.normalize_mean,) * 3, (cfg.normalize_std,) * 3),
        ]
    )


class TransformedDataset(Dataset):
    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.transform(image), label
