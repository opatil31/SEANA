"""SSL encoders from the captured CIFAR and microscopy runners."""

import copy
import torch
from torch import nn
import torchvision
from lightly.models.utils import deactivate_requires_grad
from lightly.models.modules.heads import (
    BYOLProjectionHead,
    BYOLPredictionHead,
    SimCLRProjectionHead,
    VICRegProjectionHead,
    MoCoProjectionHead,
)
from lightly.loss import NTXentLoss, NegativeCosineSimilarity, VICRegLoss
from .lejepa import LeJEPALoss


class BYOL(nn.Module):
    """BYOL online + momentum encoders (Lightly example-style).

    Exposes:
      - backbone, projection_head, prediction_head
      - backbone_momentum, projection_head_momentum
      - forward(x) -> p
      - forward_momentum(x) -> z (detached)

    Target-encoder parameters are updated by EMA in the SSL training loop.
    """

    def __init__(
        self, backbone: nn.Module, feat_dim: int = 512, proj_hidden: int = 1024, proj_out: int = 256
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(feat_dim, proj_hidden, proj_out)
        self.prediction_head = BYOLPredictionHead(proj_out, proj_hidden, proj_out)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        return z.detach()


class SimCLRModel(nn.Module):
    def __init__(self, backbone, feat_dim=512, proj_hidden=512, proj_out=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(feat_dim, proj_hidden, proj_out)

    def forward(self, x):
        return self.projection_head(self.backbone(x).flatten(1))


class VICRegModel(nn.Module):
    """VICReg using lightly's VICRegProjectionHead.

    Retains the captured runner's two-layer, 2048/2048 projection head.
    """

    def __init__(self, backbone, feat_dim=512, proj_hidden=2048, proj_out=2048, num_layers=2):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=feat_dim,
            hidden_dim=proj_hidden,
            output_dim=proj_out,
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.projection_head(self.backbone(x).flatten(1))


class MoCoModel(nn.Module):
    """MoCoV2 query/key encoders; the NT-Xent criterion owns the negative queue."""

    def __init__(
        self, backbone: nn.Module, feature_dim: int, proj_hidden_dim: int = 512, proj_out_dim: int = 128
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(feature_dim, proj_hidden_dim, proj_out_dim)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def _bb(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h.flatten(start_dim=1) if h.dim() > 2 else h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Query branch (gradients flow)."""
        return self.projection_head(self._bb(x))

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Key branch (momentum encoder, detached)."""
        h = self.backbone_momentum(x)
        if h.dim() > 2:
            h = h.flatten(start_dim=1)
        return self.projection_head_momentum(h).detach()

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._bb(x)


class LeJEPAModel(nn.Module):
    """ResNet backbone + SimCLR-style MLP projector. model(x) returns the projection z (both
    LeJEPA terms act on z); forward_features(x) returns the backbone embedding used by the
    frozen linear probe. No predictor or momentum encoder is used."""

    def __init__(
        self, backbone: nn.Module, feature_dim: int, proj_hidden_dim: int = 512, proj_out_dim: int = 128
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),  # unconstrained output (BN-free)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        if h.dim() > 2:
            h = h.flatten(start_dim=1)
        return self.projection_head(h)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        if h.dim() > 2:
            h = h.flatten(start_dim=1)
        return h


def build_backbone(name):
    """Retain the captured ImageNet-style stem, including its 7x7 conv/maxpool."""
    if name not in ("resnet18", "resnet50"):
        raise ValueError(f"Unsupported backbone: {name}")
    net = getattr(torchvision.models, name)(weights=None)
    dim = net.fc.in_features
    return nn.Sequential(*list(net.children())[:-1]), dim


def build_ssl(cfg):
    backbone, dim = build_backbone(cfg.backbone)
    if cfg.method == "byol":
        model, loss = BYOL(backbone, feat_dim=dim), NegativeCosineSimilarity()
    elif cfg.method == "simclr":
        model, loss = SimCLRModel(backbone, feat_dim=dim), NTXentLoss(temperature=cfg.simclr_temperature)
    elif cfg.method == "vicreg":
        model, loss = VICRegModel(backbone, feat_dim=dim), VICRegLoss()
    elif cfg.method == "mocov2":
        model = MoCoModel(backbone, feature_dim=dim)
        # Exactly one key batch is enqueued per optimization step.
        loss = NTXentLoss(temperature=cfg.moco_temperature, memory_bank_size=(cfg.moco_queue_size, 128))
    elif cfg.method == "lejepa":
        model = LeJEPAModel(backbone, feature_dim=dim)
        loss = LeJEPALoss(
            cfg.lejepa_num_slices, cfg.lejepa_num_points, cfg.lejepa_sigreg_weight, cfg.lejepa_objective
        )
    else:
        raise ValueError(f"Unknown SSL method: {cfg.method}")
    return model, loss, dim


def ssl_loss(model, criterion, method, x0, x1):
    if method == "byol":
        p0, p1 = model(x0), model(x1)
        with torch.no_grad():
            z0, z1 = model.forward_momentum(x0), model.forward_momentum(x1)
        return 0.5 * (criterion(p0, z1) + criterion(p1, z0))
    if method == "mocov2":
        q = model(x0)
        # Single-device shuffle/unshuffle. This does not provide cross-device
        # ShuffleBN; this release deliberately supports one process per run.
        with torch.no_grad():
            order = torch.randperm(x1.size(0), device=x1.device)
            k = model.forward_momentum(x1[order])[order.argsort()]
        return criterion(q, k)
    z0, z1 = model(x0), model(x1)
    # Characteristic functions and variance/covariance statistics stay in FP32,
    # even when the outer training loop uses autocast for the encoders.
    with torch.autocast(device_type=z0.device.type, enabled=False):
        if method == "lejepa":
            return criterion(torch.stack((z0, z1)).float())
        return criterion(z0.float(), z1.float())
