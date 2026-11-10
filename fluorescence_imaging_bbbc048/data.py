"""Cell identity splits and TIFF-safe loading, preserving microscopy intensity scale."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

from .config import PHASES
from .prepare import fingerprint
from .splits import split_indices as split_indices, select_split, validate_splits


def read_image(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Image bytes changed since preparation: {Path(path).name}")
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        import tifffile

        array = tifffile.imread(io.BytesIO(data))
    else:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            array = np.array(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] <= 4 and min(array.shape[1:]) > 8:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] < 1:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / np.iinfo(array.dtype).max
    else:
        array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite image pixels")
        if array.max() > 1:
            array /= array.max()
    return torch.from_numpy(np.ascontiguousarray(np.clip(array, 0, 1).transpose(2, 0, 1)))


class CellDataset(Dataset):
    def __init__(self, manifest):
        path = Path(manifest)
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version") != 1
            or payload.get("dataset") != "bbbc048"
            or payload.get("phases") != list(PHASES)
        ):
            raise ValueError("Invalid BBBC048 manifest or phase vocabulary")
        self.records = payload["records"]
        self.layout = payload["layout"]
        if (
            self.layout not in ("merged", "channels")
            or fingerprint(self.records, self.layout) != payload["fingerprint"]
        ):
            raise ValueError("Manifest fingerprint mismatch")
        self.fingerprint = payload["fingerprint"]
        self.prepared_splits = payload.get("prepared_splits", [])
        validate_splits(self.records, self.prepared_splits)
        self.root = (path.parent / payload["source_root"]).resolve()
        seen, seen_paths = set(), set()
        for row in self.records:
            identity = row["object_id"]
            if not identity.isdigit() or identity in seen or row["label"] not in range(7):
                raise ValueError("Invalid/duplicate cell identity or label")
            seen.add(identity)
            if len(row["files"]) != (1 if self.layout == "merged" else 3):
                raise ValueError("Invalid channel count in manifest")
            for file in row["files"]:
                resolved = (self.root / file["path"]).resolve()
                if self.root not in resolved.parents or resolved in seen_paths:
                    raise ValueError("Images must be distinct paths within the dataset root")
                seen_paths.add(resolved)
        if not self.records:
            raise ValueError("Empty manifest")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        channels = [read_image(self.root / f["path"], f["sha256"]) for f in row["files"]]
        image = channels[0] if self.layout == "merged" else torch.cat([c[:1] for c in channels])
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 2:
            image = torch.cat((image, torch.zeros_like(image[:1])))
        return image[:3].contiguous(), row["label"]


class FakeCells(Dataset):
    def __init__(self):
        self.records = [dict(object_id=str(i), label=i % 7) for i in range(28)]
        self.fingerprint = "fake_cells_v1"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.rand((3, 66, 66), generator=torch.Generator().manual_seed(70000 + index)), self.records[
            index
        ]["label"]


def build_datasets(cfg):
    dataset = FakeCells() if cfg.smoke else CellDataset(cfg.manifest)
    seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
    train, test = select_split(
        dataset.records, cfg.split_mode, cfg.train_fraction, seed, getattr(dataset, "prepared_splits", ())
    )
    identities = {
        name: [dataset.records[i]["object_id"] for i in indices]
        for name, indices in (("train", train), ("test", test))
    }
    metadata = dict(
        dataset="fake_cells" if cfg.smoke else "bbbc048",
        fingerprint=dataset.fingerprint,
        phases=list(PHASES),
        split_mode=cfg.split_mode,
        split_seed=seed,
        train_fraction=cfg.train_fraction,
        identities=identities,
        split_sha256={
            name: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for name, ids in identities.items()
        },
    )
    return Subset(dataset, train), Subset(dataset, test), metadata


def eval_transform(cfg):
    return T.Compose(
        [
            T.Resize(cfg.image_size, antialias=True),
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
