"""Frozen-backbone analysis settings; independent of training recipes."""

from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass
class Config:
    data_root: str = "data/cifar10"
    download: bool = False
    device: str = "auto"
    n_images: int = 2000
    k_reps: int = 8
    retrieval_images: int = 500
    batch_size: int = 512
    retrieval_chunk_size: int = 256
    seed: int = 0
    prepool: bool = False
    save_features: bool = False
    smoke: bool = False

    def validate(self):
        for name, minimum in (
            ("n_images", 4),
            ("k_reps", 2),
            ("retrieval_images", 2),
            ("batch_size", 1),
            ("retrieval_chunk_size", 1),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Use auto, cpu or cuda")
        return self


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown analysis settings: {sorted(unknown)}")
    return Config(**values).validate()
