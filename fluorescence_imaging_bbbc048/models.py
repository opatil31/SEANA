"""Captured microscopy projection sizes and objectives, using the shared encoders."""

from lightly.loss import NTXentLoss, NegativeCosineSimilarity, VICRegLoss
from cifar10.models import build_backbone, BYOL, SimCLRModel, VICRegModel, MoCoModel, LeJEPAModel
from cifar10.lejepa import LeJEPALoss


def build_ssl(cfg):
    backbone, dim = build_backbone(cfg.backbone)
    if cfg.method == "byol":
        model, loss = BYOL(backbone, dim, cfg.proj_hidden, cfg.proj_out), NegativeCosineSimilarity()
    elif cfg.method == "simclr":
        model, loss = (
            SimCLRModel(backbone, dim, cfg.proj_hidden, cfg.proj_out),
            NTXentLoss(temperature=cfg.simclr_temperature),
        )
    elif cfg.method == "vicreg":
        model, loss = VICRegModel(backbone, dim, cfg.proj_hidden, cfg.proj_out, cfg.proj_layers), VICRegLoss()
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
        raise ValueError(f"Unknown SSL method: {cfg.method}")
    return model, loss, dim
