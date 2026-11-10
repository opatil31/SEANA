"""One noisy source per image, raw fitting images, and separate clean validation."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms as T

from cifar10.data import MEAN, STD, SyntheticNoise as BaseNoise
from .config import CONDITIONS


class SyntheticNoise(BaseNoise):
    def __init__(self, condition, sampling="stochastic", seed=0):
        if sampling not in ("fixed", "stochastic"):
            raise ValueError("sampling must be fixed or stochastic")
        self.spec, self.sampling, self.seed = CONDITIONS[condition], sampling, seed

    def __call__(self, x, idx=None):
        if self.spec["family"] != "shot":
            return super().__call__(x, idx)
        generator = None
        if self.sampling == "fixed":
            if idx is None:
                raise ValueError("Fixed noise requires an image index")
            generator = torch.Generator(device=x.device).manual_seed(self.seed + int(idx))
        peak = self.spec["peak"]
        return (torch.poisson(x * peak, generator=generator) / peak).clamp(0, 1)


def split_metadata(splits):
    """Validate both label vocabularies without decoding the image columns."""
    info = {}
    for name in ("train", "validation"):
        if name not in splits or len(splits[name]) == 0:
            raise ValueError(f"Missing or empty {name} split")
        ds = splits[name]
        if not {"image", "label"} <= set(ds.features):
            raise ValueError("ImageNet data must have image and label columns")
        names = getattr(ds.features["label"], "names", None)
        if names is None or len(names) != 100 or len(set(names)) != 100:
            raise ValueError("ImageNet-100 requires exactly 100 distinct ClassLabel names")
        labels = set(ds["label"])
        if not labels or min(labels) < 0 or max(labels) >= 100:
            raise ValueError("Labels must be in [0, 99]")
        info[name] = {"images": len(ds), "fingerprint": ds._fingerprint, "classes": list(names)}
    if info["train"]["classes"] != info["validation"]["classes"]:
        raise ValueError("Train/validation class-to-index mappings differ")
    return info


class HFImages(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[int(idx)]
        return row["image"].convert("RGB"), int(row["label"])


def load_bases(cfg):
    if cfg.smoke:
        train = torchvision.datasets.FakeData(
            size=8, image_size=(3, cfg.source_size, cfg.source_size), num_classes=100, random_offset=cfg.seed
        )
        val = torchvision.datasets.FakeData(
            size=8,
            image_size=(3, cfg.source_size, cfg.source_size),
            num_classes=100,
            random_offset=100000 + cfg.seed,
        )
        return train, val, {"source": "fake_data", "train_images": 8, "validation_images": 8}
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise ImportError("Install requirements-imagenet100.txt before loading ImageNet-100") from error
    root = Path(cfg.data_root)
    manifest_path = root / "seana_dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared dataset missing at {root}; run python -m imagenet100.prepare")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("dataset") != "imagenet100" or manifest.get("schema_version") != 1:
        raise ValueError("Not a release-format ImageNet-100 dataset")
    splits = load_from_disk(str(root))
    info = split_metadata(splits)
    if info != manifest["splits"]:
        raise ValueError("Prepared dataset fingerprints/classes differ from its manifest")
    bases = []
    for name, maximum in (("train", cfg.max_train_samples), ("validation", cfg.max_val_samples)):
        base = HFImages(splits[name])
        if maximum:
            base = Subset(base, range(min(maximum, len(base))))
        bases.append(base)
    return *bases, {
        "source": manifest["source"],
        "splits": info,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "train_images": len(bases[0]),
        "validation_images": len(bases[1]),
    }


class ImageDataset(Dataset):
    def __init__(self, base, *, resize, crop, noise=None, normalize=False):
        self.base, self.noise = base, noise
        self.pre = T.Compose(
            [T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(crop), T.ToTensor()]
        )
        self.normalize = T.Normalize(MEAN, STD) if normalize else torch.nn.Identity()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        x = self.pre(image.convert("RGB"))
        if self.noise is not None:
            x = self.noise(x, idx)
        return self.normalize(x), label


def build_datasets(cfg):
    train, val, metadata = load_bases(cfg)
    noise = SyntheticNoise(cfg.condition, cfg.noise_sampling, cfg.seed)
    return {
        "ssl": ImageDataset(train, resize=cfg.source_size, crop=cfg.source_size, noise=noise),
        "fit": ImageDataset(train, resize=cfg.source_size, crop=cfg.source_size, noise=noise),
        "probe_train": ImageDataset(
            train, resize=cfg.source_size, crop=cfg.image_size, noise=noise, normalize=True
        ),
        "probe_test": ImageDataset(val, resize=cfg.source_size, crop=cfg.image_size, normalize=True),
        "metadata": metadata,
    }
