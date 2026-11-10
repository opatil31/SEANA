"""Native crop views with shared estimates and independent VST-space resampling."""

import hashlib
import io
from pathlib import Path

import torch
from torchvision import transforms as T

from fluorescence_imaging_bbbc048.views import ViewTreatment as MicroscopyTreatment
from fluorescence_imaging_bbbc048.views import fit_identity, cache_estimates


class ViewTreatment(MicroscopyTreatment):
    def __init__(self, cfg, device, *, initialize_from_checkpoint=False):
        super().__init__(cfg, device, initialize_from_checkpoint=initialize_from_checkpoint)
        self.spatial = T.Compose([T.RandomCrop(cfg.image_size), T.RandomHorizontalFlip(p=0.5)])


def load_treatment(path, cfg, dataset, device):
    raw = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != "fetal_treatment":
        raise ValueError("Expected a fetal treatment checkpoint")
    if payload["fit_identity"] != fit_identity(cfg, dataset):
        raise ValueError("Treatment belongs to a different training split, dataset or fitting recipe")
    treatment = ViewTreatment(cfg, device, initialize_from_checkpoint=True)
    treatment.load_state_dict(payload["state_dict"], strict=True)
    treatment.eval().requires_grad_(False)
    treatment.fit_provenance = payload["fit_provenance"]
    return treatment, hashlib.sha256(raw).hexdigest()


__all__ = ["ViewTreatment", "load_treatment", "fit_identity", "cache_estimates"]
