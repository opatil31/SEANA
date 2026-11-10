"""Fit on unaltered noisy images; treat full images before crop and color transforms."""

import torch
from torchvision import transforms as T

from cifar10.data import normalize_imagenet
from cifar10.views import ViewTreatment as BaseTreatment
from seana import ConditionalGlobalNoise2VST, augmentation


class PhotometricTransform:
    """Captured v10 per-image color operations, including its whole-image inversion.

    The historical flag was named solarize, but it inverts every pixel rather
    than thresholding. Retain that operation explicitly for source fidelity.
    """

    def __init__(self, *, color_jitter=True, invert=False):
        self.color_jitter, self.invert = color_jitter, invert

    @torch.no_grad()
    def __call__(self, x):
        batch, device = x.size(0), x.device
        shape = (batch, 1, 1, 1)
        if self.color_jitter:
            apply = (torch.rand(batch, device=device) < 0.8).view(shape)
            brightness = torch.empty(shape, device=device).uniform_(0.6, 1.4)
            jittered = (x * brightness).clamp(0, 1)
            mean = jittered.mean(dim=(1, 2, 3), keepdim=True)
            contrast = torch.empty(shape, device=device).uniform_(0.6, 1.4)
            jittered = (mean + contrast * (jittered - mean)).clamp(0, 1)
            gray = 0.299 * jittered[:, 0:1] + 0.587 * jittered[:, 1:2] + 0.114 * jittered[:, 2:3]
            saturation = torch.empty(shape, device=device).uniform_(0.8, 1.2)
            jittered = (gray + saturation * (jittered - gray)).clamp(0, 1)
            x = torch.where(apply, jittered, x)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
        apply_gray = (torch.rand(batch, device=device) < 0.2).view(shape)
        x = torch.where(apply_gray, gray, x)
        if self.invert:
            apply_invert = (torch.rand(batch, device=device) < 0.2).view(shape)
            x = torch.where(apply_invert, 1 - x, x)
        return x


class ViewTreatment(BaseTreatment):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        if cfg.treatment == "seana" and cfg.seana_type == "conditional":
            self.vst = ConditionalGlobalNoise2VST(
                nb_knots=cfg.seana_knots,
                inverse=True,
                data_min=0,
                data_max=1,
                clamp=True,
                cond_hidden_dim=cfg.cond_hidden_dim,
                cond_delta_scale=cfg.cond_delta_scale,
            ).to(device)
        self.spatial = T.Compose(
            [
                T.RandomResizedCrop(
                    cfg.image_size,
                    scale=(0.08, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(),
            ]
        )
        self.photometric = (
            PhotometricTransform(color_jitter=cfg.color_jitter),
            PhotometricTransform(color_jitter=cfg.color_jitter, invert=True),
        )

    def fit(self, ssl_loader, raw_loader, device):
        cfg = self.cfg
        if cfg.treatment == "seana":
            self.denoiser.eval().requires_grad_(False)
            self.vst.train()
            self.vst.calibrate_from_dataloader(raw_loader, max_batches=cfg.seana_calibration_batches)
            self.vst.fit_dataset(
                raw_loader,
                denoiser=self.denoiser,
                total_steps=cfg.seana_fit_steps,
                patch_size=cfg.seana_patch_size,
                patch_batch_size=cfg.seana_patch_batch_size,
                stride=cfg.seana_stride,
                lr=cfg.seana_fit_lr,
            )
        elif cfg.treatment == "n2n":
            from denoisers.n2n import train_n2n_denoiser

            train_n2n_denoiser(
                self.denoiser,
                ssl_loader,
                device,
                total_steps=cfg.n2n_fit_steps,
                patch_size=cfg.n2n_patch_size,
                patch_batch_size=cfg.denoiser_patch_batch_size,
                lr=cfg.n2n_fit_lr,
                gamma=cfg.n2n_gamma,
                unnormalize_fn=None,
                patch_augment_fn=augmentation,
            )
        elif cfg.treatment == "nsp":
            from denoisers.nsp import train_nsp_denoiser

            train_nsp_denoiser(
                self.denoiser,
                ssl_loader,
                device,
                total_steps=cfg.nsp_fit_steps,
                patch_size=cfg.nsp_patch_size,
                patch_batch_size=cfg.denoiser_patch_batch_size,
                lr=cfg.nsp_fit_lr,
                tds=cfg.nsp_tds,
                seed=cfg.seed,
                unnormalize_fn=None,
            )
        return self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, image):
        """Return two normalized views of the single full-resolution source."""
        if self.cfg.treatment == "seana":
            sources = self.vst.augment_pair(
                image,
                self.denoiser,
                sigma_vst=self.sigma_vst,
                per_image_norm=self.cfg.seana_per_image_norm,
                clamp=True,
                amp=False,
            )
        else:
            source = image if self.denoiser is None else self.denoiser(image).clamp(0, 1)
            sources = (source, source)
        views = []
        for source, photometric in zip(sources, self.photometric):
            cropped = torch.stack([self.spatial(x) for x in source])
            views.append(normalize_imagenet(photometric(cropped)))
        return tuple(views)
