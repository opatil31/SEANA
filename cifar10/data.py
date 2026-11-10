"""Synthetic corruption and separate SSL/probe data paths for CIFAR-10."""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from lightly.transforms.byol_transform import BYOLView1Transform, BYOLView2Transform
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.vicreg_transform import VICRegTransform
from lightly.transforms.moco_transform import MoCoV2Transform

from .config import CONDITIONS

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def unnormalize_imagenet(x):
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    return x * std + mean


def normalize_imagenet(x):
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


class SyntheticNoise:
    def __init__(self, condition, sampling="stochastic", seed=0):
        self.spec = CONDITIONS[condition]
        if sampling not in ("fixed", "stochastic"):
            raise ValueError("sampling must be fixed or stochastic")
        self.sampling = sampling
        self.seed = seed

    def __call__(self, x, idx=None):
        generator = None
        if self.sampling == "fixed":
            if idx is None:
                raise ValueError("Fixed noise requires an image index")
            generator = torch.Generator(device=x.device).manual_seed(self.seed + int(idx))
        spec = self.spec
        if spec["family"] == "poisson_gaussian":
            y = torch.poisson(x * spec["peak"], generator=generator) / spec["peak"]
            y = y + spec["sigma"] * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        else:
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            y = x + spec["sigma"] * eps if spec["family"] == "gaussian" else x * (1 + spec["sigma"] * eps)
        return y.clamp(0, 1)


def ssl_transforms(method):
    if method == "byol":
        return (
            BYOLView1Transform(input_size=32, gaussian_blur=0.0),
            BYOLView2Transform(input_size=32, gaussian_blur=0.0),
        )
    transform = {
        "vicreg": VICRegTransform,
        "simclr": SimCLRTransform,
        "mocov2": MoCoV2Transform,
        "lejepa": VICRegTransform,
    }[method]
    # Two views for all models. LeJEPA uses the project's two-view adaptation.
    multi = transform(input_size=32, gaussian_blur=0.0)
    return tuple(multi.transforms[:2])


class SSLDataset(Dataset):
    def __init__(self, base, noise, transforms):
        self.base, self.noise, self.transforms = base, noise, transforms
        self.to_tensor, self.to_pil = T.ToTensor(), T.ToPILImage()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        noisy = self.noise(self.to_tensor(image), idx)
        # Match the captured runner's 8-bit PIL conversion before augmentation.
        noisy = self.to_pil(noisy)
        return tuple(transform(noisy) for transform in self.transforms), label


class ProbeDataset(Dataset):
    def __init__(self, base, *, train, noise=None, raw=False):
        if not train and noise is not None:
            raise ValueError("The released linear probe evaluates clean test images")
        self.base, self.noise = base, noise
        self.to_tensor = T.ToTensor()
        transforms = []
        if train and not raw:
            transforms += [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
        if not raw:
            transforms += [T.Normalize(MEAN, STD)]
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        x = self.to_tensor(image)
        if self.noise is not None:
            x = self.noise(x, idx)
        return self.transform(x), label


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, cfg, *, shuffle, drop_last=False, seed_offset=0):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=drop_last,
        pin_memory=str(cfg.device).startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.seed + seed_offset),
    )


def build_datasets(cfg):
    if cfg.smoke:
        train = torchvision.datasets.FakeData(
            size=8, image_size=(3, 32, 32), num_classes=10, random_offset=cfg.seed
        )
        test = torchvision.datasets.FakeData(
            size=8, image_size=(3, 32, 32), num_classes=10, random_offset=100000 + cfg.seed
        )
    else:
        train = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=cfg.download)
        test = torchvision.datasets.CIFAR10(cfg.data_root, train=False, download=cfg.download)
    noise = SyntheticNoise(cfg.condition, cfg.noise_sampling, cfg.seed)
    # N2N/NSP probe-training denoising runs on the device in train_probe, after
    # this dataset's spatial transforms. Test images bypass all treatments.
    return {
        "ssl": SSLDataset(train, noise, ssl_transforms(cfg.method)),
        "fit": ProbeDataset(train, train=True, noise=noise, raw=True),
        "probe_train": ProbeDataset(train, train=True, noise=noise),
        "probe_test": ProbeDataset(test, train=False),
    }
