"""Fit a treatment for SSL views and, for N2N/NSP, probe-training images."""

from pathlib import Path
import torch
from torch import nn

from seana import GlobalNoise2VST, DeepInvDRUNetWrapper
from .data import normalize_imagenet, unnormalize_imagenet


class ViewTreatment(nn.Module):
    def __init__(self, cfg, device, *, initialize_from_checkpoint=False):
        super().__init__()
        self.cfg = cfg
        self.vst = None
        self.denoiser = None
        self.sigma_vst = cfg.resolved()["sigma_vst"]
        if cfg.treatment == "seana":
            self.vst = GlobalNoise2VST(
                nb_knots=cfg.seana_knots, inverse=True, data_min=0.0, data_max=1.0, clamp=True
            ).to(device)
            if cfg.smoke:
                # Explicitly labelled test fixture; no pretrained weights needed.
                self.denoiser = nn.AvgPool2d(3, stride=1, padding=1).to(device)
            else:
                from deepinv.models import DRUNet

                if (
                    not initialize_from_checkpoint
                    and cfg.drunet_weights != "download"
                    and not Path(cfg.drunet_weights).is_file()
                ):
                    raise FileNotFoundError(f"DRUNet weights not found: {cfg.drunet_weights}")
                drunet = DRUNet(
                    pretrained=None if initialize_from_checkpoint else cfg.drunet_weights, device=device
                )
                self.denoiser = DeepInvDRUNetWrapper(drunet, sigma_denoiser=cfg.resolved()["sigma_d"]).to(
                    device
                )
        elif cfg.treatment == "n2n":
            from denoisers.n2n import Neighbor2NeighborUNet

            self.denoiser = Neighbor2NeighborUNet(base_ch=cfg.n2n_base_ch).to(device)
        elif cfg.treatment == "nsp":
            from denoisers.nsp import build_nsp_denoiser

            self.denoiser = build_nsp_denoiser(
                device, base_ch=cfg.nsp_base_ch, num_module=cfg.nsp_num_module, ids=cfg.nsp_ids
            )

    def fit(self, ssl_loader, raw_loader, device):
        cfg = self.cfg
        if cfg.treatment == "seana":
            self.denoiser.eval().requires_grad_(False)
            self.vst.train()
            self.vst.fit_dataset(
                raw_loader,
                denoiser=self.denoiser,
                total_steps=cfg.seana_fit_steps,
                patch_size=32,
                patch_batch_size=cfg.seana_patch_batch_size,
                stride=cfg.seana_stride,
                lr=cfg.seana_fit_lr,
            )
        elif cfg.treatment == "n2n":
            from denoisers.n2n import train_n2n_denoiser

            train_n2n_denoiser(
                self.denoiser,
                raw_loader,
                device,
                total_steps=cfg.n2n_fit_steps,
                patch_size=32,
                patch_batch_size=cfg.denoiser_patch_batch_size,
                lr=cfg.n2n_fit_lr,
                gamma=cfg.n2n_gamma,
                unnormalize_fn=None,
            )
        elif cfg.treatment == "nsp":
            from denoisers.nsp import train_nsp_denoiser

            train_nsp_denoiser(
                self.denoiser,
                ssl_loader,
                device,
                total_steps=cfg.nsp_fit_steps,
                patch_size=32,
                patch_batch_size=cfg.denoiser_patch_batch_size,
                lr=cfg.nsp_fit_lr,
                tds=cfg.nsp_tds,
                seed=cfg.seed,
                unnormalize_fn=unnormalize_imagenet,
            )
        self.eval().requires_grad_(False)
        return self

    @torch.no_grad()
    def denoise_probe_inputs(self, normalized_images):
        """Denoise probe-training inputs without SSL transforms or resampling.

        Probe datasets already apply their spatial transforms and normalization.
        Undo normalization for the denoiser, then restore it for the backbone.
        Clean test inputs never pass through this method.
        """
        if self.cfg.treatment not in ("n2n", "nsp") or self.denoiser is None:
            raise ValueError("Probe-training denoising requires a fitted N2N or NSP treatment")
        image = unnormalize_imagenet(normalized_images).clamp(0, 1)
        return normalize_imagenet(self.denoiser(image).clamp(0, 1))

    @torch.no_grad()
    def forward(self, normalized_view):
        if self.cfg.treatment == "baseline":
            return normalized_view
        image = unnormalize_imagenet(normalized_view).clamp(0, 1)
        if self.vst is not None:
            image = self.vst.augment(image, denoiser=self.denoiser, sigma_vst=self.sigma_vst, amp=False)
        else:
            image = self.denoiser(image).clamp(0, 1)
        return normalize_imagenet(image)
