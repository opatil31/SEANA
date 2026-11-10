"""Official Neighbor2Neighbor objective and sampler, integrated with SEANA datasets.

Adapted from TaoHuang2018/Neighbor2Neighbor (CVPR 2021), commit
e66389ebef1e64d306d8fcb95f096cf427243452.
Copyright (c) 2021, Tao Huang. BSD-3-Clause.
See third_party/neighbor2neighbor-LICENSE.txt and n2n_upstream.json.

Reflection padding supports small/odd image sizes. Training uses project-specific patch/step budgets, with the upstream regularization ramp and LR drops expressed in update counts.
"""

import math
import random

import torch

from cifar10.data import unnormalize_imagenet
from ._n2n_unet import UNet


class Neighbor2NeighborUNet(UNet):
    """Upstream non-blind-spot U-Net with reflection padding to multiples of 32."""

    def __init__(self, in_ch=3, base_ch=48):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (in_ch, base_ch)):
            raise ValueError("N2N channel counts must be positive integers")
        super().__init__(in_nc=in_ch, out_nc=in_ch, n_feature=base_ch, blindspot=False)

    def forward(self, image):
        if image.ndim != 4 or min(image.shape) < 1:
            raise ValueError("N2N expects a nonempty BCHW tensor")
        height, width = image.shape[-2:]
        # Index-based reflection matches numpy.pad(mode='reflect') even when
        # the pad equals/exceeds the input size, e.g. CIFAR's 16x16 subimages.
        for dimension, size in ((-2, height), (-1, width)):
            target = ((size + 31) // 32) * 32
            if target != size:
                index = torch.arange(target, device=image.device)
                if size == 1:
                    index.zero_()
                else:
                    period = 2 * (size - 1)
                    index = index.remainder(period)
                    index = torch.minimum(index, period - index)
                image = image.index_select(dimension, index)
        return super().forward(image)[..., :height, :width]


def _subimage_shape(image):
    if image.ndim != 4 or min(image.shape) < 1 or any(s % 2 for s in image.shape[-2:]):
        raise ValueError("Neighbor subsampling needs nonempty BCHW images with even height and width")
    return image.shape


def generate_mask_pair(image, *, generator=None):
    """Choose one of eight ordered adjacent pairs independently in every 2x2 block.

    A block's choice is shared across channels, as in the official implementation.
    The ordinary seeded PyTorch RNG works on both CPU and CUDA.
    """
    n, _, h, w = _subimage_shape(image)
    count = n * (h // 2) * (w // 2)
    adjacent = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.long,
        device=image.device,
    )
    choices = torch.randint(8, (count,), device=image.device, generator=generator)
    offsets = 4 * torch.arange(count, device=image.device)
    pairs = adjacent[choices] + offsets[:, None]
    first = torch.zeros(count * 4, dtype=torch.bool, device=image.device)
    second = torch.zeros_like(first)
    first[pairs[:, 0]], second[pairs[:, 1]] = True, True
    return first, second


def generate_subimages(image, mask):
    """Apply the same block mask to every channel, preserving gradients."""
    n, c, h, w = _subimage_shape(image)
    if mask.dtype != torch.bool or mask.shape != (n * h * w,) or mask.device != image.device:
        raise ValueError("N2N mask must be a flat boolean mask on the image device")
    blocks = image.reshape(n, c, h // 2, 2, w // 2, 2)
    blocks = blocks.permute(0, 2, 4, 3, 5, 1).reshape(-1, c)
    return blocks[mask].reshape(n, h // 2, w // 2, c).permute(0, 3, 1, 2).contiguous()


def neighbor2neighbor_loss(denoiser, noisy, regularization_weight, *, mask_pair=None):
    """Official reconstruction + detached full-image correction loss.

    The full-image prediction is used only to estimate neighbor content
    differences. Gradients pass through the prediction from the first subimage.
    """
    if not math.isfinite(regularization_weight) or regularization_weight < 0:
        raise ValueError("N2N regularization weight must be finite and nonnegative")
    first, second = generate_mask_pair(noisy) if mask_pair is None else mask_pair
    source = generate_subimages(noisy, first)
    target = generate_subimages(noisy, second)
    with torch.no_grad():
        estimate = denoiser(noisy)
        correction = generate_subimages(estimate, first) - generate_subimages(estimate, second)
    residual = denoiser(source) - target
    return residual.square().mean() + regularization_weight * (residual - correction).square().mean()


def train_n2n_denoiser(
    denoiser,
    dataloader,
    device,
    total_steps=20000,
    patch_size=32,
    patch_batch_size=8,
    lr=3e-4,
    gamma=2.0,
    log_every=500,
    unnormalize_fn=unnormalize_imagenet,
    patch_augment_fn=None,
):
    """Fit official N2N on noisy training images, ignoring labels.

    gamma is the final regularization weight (upstream increase_ratio), not the
    LR decay factor. The weight is gamma * step / total_steps. Adam's LR is
    halved at 20/40/60/80% of the update budget. Patch sizes and update budgets
    belong to the dataset recipes rather than the upstream ImageNet benchmark.
    """
    for name, value in (
        ("total_steps", total_steps),
        ("patch_size", patch_size),
        ("patch_batch_size", patch_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if patch_size < 2 or patch_size % 2:
        raise ValueError("N2N patch_size must be even and at least 2")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("N2N requires positive finite LR and nonnegative finite gamma")
    denoiser.train().requires_grad_(True)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    milestones = sorted({int(total_steps * f) for f in (0.2, 0.4, 0.6, 0.8)} - {0, total_steps})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    data_iter = iter(dataloader)
    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration as error:
                raise ValueError("N2N fitting loader is empty") from error
        z = batch if isinstance(batch, torch.Tensor) else batch[0]
        if isinstance(z, (tuple, list)):
            z = z[0]
        z = z.to(device)
        z = unnormalize_fn(z) if unnormalize_fn is not None else z
        if z.ndim != 4 or min(z.shape) < 1 or not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("N2N fitting needs finite floating-point BCHW images")
        batch_size, channels, height, width = z.shape
        if patch_size > min(height, width):
            raise ValueError("N2N patch_size exceeds training image size")
        # Do not clamp observations: Gaussian noise can legitimately leave [0,1].
        patches = torch.empty(
            patch_batch_size, channels, patch_size, patch_size, device=device, dtype=z.dtype
        )
        for index in range(patch_batch_size):
            sample = random.randrange(batch_size)
            top = random.randrange(height - patch_size + 1)
            left = random.randrange(width - patch_size + 1)
            patches[index] = z[sample, :, top : top + patch_size, left : left + patch_size]
        if patch_augment_fn is not None:
            patches = torch.cat([patch_augment_fn(p.unsqueeze(0), k=random.randint(0, 7)) for p in patches])
        optimizer.zero_grad(set_to_none=True)
        weight = gamma * step / total_steps
        loss = neighbor2neighbor_loss(denoiser, patches, weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite N2N loss at step {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        if log_every > 0 and step % log_every == 0:
            print(f"[n2n][step {step:05d}/{total_steps}] loss={loss.item():.6f} lambda={weight:.4f}")
    denoiser.eval()
    return denoiser
