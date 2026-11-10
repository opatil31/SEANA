"""ImageNet-sized projection heads and the five shared SSL objectives."""

import torch
from torch import nn
from torch.nn import functional as F
from lightly.loss import NTXentLoss, VICRegLoss

from cifar10.models import build_backbone, BYOL, SimCLRModel, VICRegModel, MoCoModel, LeJEPAModel
from cifar10.lejepa import LeJEPALoss


class StableNegativeCosineSimilarity(nn.Module):
    def forward(self, prediction, target):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return (
                -(F.normalize(prediction.float(), dim=-1) * F.normalize(target.float(), dim=-1))
                .sum(-1)
                .mean()
            )


def build_ssl(cfg):
    backbone, dim = build_backbone(cfg.backbone)
    if cfg.method == "byol":
        model = BYOL(backbone, dim, cfg.proj_hidden, cfg.proj_out)
        loss = StableNegativeCosineSimilarity()
    elif cfg.method == "simclr":
        model = SimCLRModel(backbone, dim, cfg.proj_hidden, cfg.proj_out)
        loss = NTXentLoss(temperature=cfg.simclr_temperature)
    elif cfg.method == "vicreg":
        model = VICRegModel(backbone, dim, cfg.proj_hidden, cfg.proj_out, cfg.proj_layers)
        loss = VICRegLoss()
    elif cfg.method == "mocov2":
        model = MoCoModel(backbone, dim, cfg.proj_hidden, cfg.proj_out)
        loss = NTXentLoss(
            temperature=cfg.moco_temperature, memory_bank_size=(cfg.moco_queue_size, cfg.proj_out)
        )
    elif cfg.method == "lejepa":
        model = LeJEPAModel(backbone, dim, cfg.proj_hidden, cfg.proj_out)
        loss = LeJEPALoss(
            cfg.lejepa_num_slices, cfg.lejepa_num_points, cfg.lejepa_sigreg_weight, cfg.lejepa_objective
        )
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
    return model, loss, dim
