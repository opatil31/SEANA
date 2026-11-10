"""nsp_denoiser.py

Next-Scale Prediction (NSP) self-supervised denoiser, for use as a stronger
denoising baseline alongside Neighbor2Neighbor in the SEANA CIFAR-10 experiment.

Reference
---------
  "Next-Scale Prediction: A Self-Supervised Approach for Real-World Image
   Denoising", CVPR 2026 (arXiv:2512.21038).
  Reference implementation: https://github.com/XLearning-SCU/2026-CVPR-NSP

Method recap
------------
NSP decouples *noise decorrelation* from *detail preservation* by learning a
cross-scale mapping instead of a same-scale one:

  - Partition the image into non-overlapping p x p blocks (p = `tds`).
  - In each block, pick a randomly-placed CONSECUTIVE t x t patch (t = `ids`).
    Gathering those t*t pixels across all blocks gives the *high-scale target*
    at resolution (H/p * t) x (W/p * t).
  - The remaining p*p - t*t pixels of each block are distributed into
    p*p - t*t *low-scale sub-images*, each at resolution (H/p) x (W/p).
  - A blind-spot network with a PixelShuffle(t) in its tail maps each low-scale
    sub-image to the high-scale target; the objective is a plain L1 loss.

  Because the target pixels are pixel-disjoint from every input sub-image, the
  target noise is independent of the input noise given the clean signal -- the
  same argument that makes Noise2Self / Neighbor2Neighbor unbiased.  The blind
  spot and the aggressive p-fold decimation additionally buy robustness to
  *spatially correlated* noise, which is what NSP was designed for.

  At inference the image is decomposed by t (not p) into t*t sub-images, each is
  passed through the network (which upsamples by t, restoring full resolution),
  and the t*t reconstructions are averaged.

Adaptation to CIFAR-10 (deliberate, documented deviations)
----------------------------------------------------------
  * `tds` defaults to **4**, not the paper's 5.  CIFAR-10 is 32x32 and 5 does
    not divide 32, whereas 4 does (8x8 sub-images -> 16x16 targets).  The
    paper's p=5 exists to decorrelate *spatially correlated* real sensor noise;
    the SEANA CIFAR-10 corruptions are pixelwise independent by construction,
    so the extra decimation buys nothing here.  Pass --nsp_tds 5 (with a 30x30
    crop) to run the literal paper setting.
  * The network sees 8x8 sub-images during training versus 32x32 in the paper
    (160x160 patches at p=5).  This is a hard ceiling set by CIFAR-10's
    resolution, and is a genuine handicap for NSP on this benchmark.
  * Training length is set by --nsp_fit_steps (default 20k) rather than the
    paper's 900 x 400 = 360k iterations, to stay within the per-run budget of
    the SSL grid.  A StepLR decay is used so the short budget converges, pass
    --nsp_const_lr for the paper's constant 1e-4.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralMaskedConv2d(nn.Conv2d):
    """Conv2d whose kernel centre tap is pinned to zero (the blind spot)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask = torch.ones_like(self.weight)
        _, _, kH, kW = self.weight.shape
        mask[:, :, kH // 2, kW // 2] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The reference mutates storage in-place (`self.weight.data *= self.mask`).
        # Multiplying functionally produces identical outputs -- the centre tap is
        # zero whenever it is used -- without touching parameter storage.
        return self._conv_forward(x, self.weight * self.mask, self.bias)


class DCl(nn.Module):
    """Residual dilated-conv module."""

    def __init__(self, stride: int, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride, bias=True),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DC_branchl(nn.Module):
    """One dilated blind-spot branch: masked conv, then `num_module` DCl blocks."""

    def __init__(self, stride: int, in_ch: int, num_module: int):
        super().__init__()
        ly = [
            CentralMaskedConv2d(
                in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1, bias=True
            ),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.GELU(),
        ]
        ly += [DCl(stride, in_ch) for _ in range(num_module)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True), nn.GELU()]
        self.body = nn.Sequential(*ly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DBSNl(nn.Module):
    """Dilated Blind-Spot Network with a PixelShuffle(ids) scale transition.

    Maps [B, C, H, W] -> [B, C, H*ids, W*ids].  The PixelShuffle sits between
    the 1x1 blocks near the tail so the blind-spot property is preserved.
    """

    def __init__(
        self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, num_module: int = 9, ids: int = 2
    ):
        super().__init__()
        assert base_ch % 2 == 0, "base channel should be divisible by 2"
        assert (base_ch * 2) % (ids * ids) == 0, (
            f"base_ch*2={base_ch * 2} must be divisible by ids^2={ids * ids}"
        )

        self.ids = int(ids)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=True),
            nn.GELU(),
        )

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        mid = base_ch * 2
        post = mid // (ids * ids)
        self.tail = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.PixelShuffle(ids),  # [B,2D,H,W] -> [B,2D/ids^2,H*ids,W*ids]
            nn.Conv2d(post, post, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(post, post, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(post, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        return self.tail(torch.cat([br1, br2], dim=1))


# =====================================================================
# Cross-scale training-pair construction
# =====================================================================


def build_cross_scale_pair(
    img: torch.Tensor,
    p: int,
    t: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one NSP cross-scale training pair.

    Vectorised equivalent of the reference `generate_mask_pair` +
    `generate_imgs_pair` for the `ng=1` setting (one target group per block),
    which is the configuration used in the paper.

    Args:
        img: [B, C, H, W] float tensor; H and W must be divisible by `p`.
        p:   block size (`tds`).
        t:   target scale (`ids`); the target patch is t x t and the network
             upsamples by t.
        rng: numpy Generator, for reproducible target placement.

    Returns:
        lr: [B*(p*p - t*t), C, H//p,     W//p    ]  low-scale sub-images
        hr: [B*(p*p - t*t), C, H//p * t, W//p * t]  the matching high-scale
            target, repeated once per sub-image.
    """
    B, C, H, W = img.shape
    if H % p or W % p:
        raise ValueError(f"image {H}x{W} not divisible by tds={p}")
    if t > p:
        raise ValueError(f"ids={t} must be <= tds={p}")

    hp, wp = H // p, W // p
    p2, t2 = p * p, t * t
    nblk = B * hp * wp

    # --- per-block permutation of the p*p intra-block positions -------------
    # First t*t entries: a randomly placed CONSECUTIVE t x t patch (the
    # sampling strategy the NSP paper found best -- it preserves the relative
    # spatial arrangement of the target).  Remaining entries: shuffled, and
    # each becomes one low-scale sub-image.
    r0 = rng.integers(0, p - t + 1, size=nblk)
    c0 = rng.integers(0, p - t + 1, size=nblk)
    ii, jj = np.meshgrid(np.arange(t), np.arange(t), indexing="ij")
    offs = (ii * p + jj).reshape(-1)  # [t*t]
    front = (r0 * p + c0)[:, None] + offs[None, :]  # [nblk, t*t]

    keep = np.ones((nblk, p2), dtype=bool)
    np.put_along_axis(keep, front, False, axis=1)
    tail = np.tile(np.arange(p2, dtype=np.int64), (nblk, 1))[keep].reshape(nblk, p2 - t2)
    tail = np.take_along_axis(tail, rng.random((nblk, p2 - t2)).argsort(axis=1), axis=1)

    rd = np.concatenate([front.astype(np.int64), tail], axis=1)  # [nblk, p*p]
    rd_t = torch.from_numpy(rd).to(img.device)

    # --- gather one pixel per block per permutation slot --------------------
    x = img.view(B, C, hp, p, wp, p).permute(0, 1, 2, 4, 3, 5).reshape(B, C, hp, wp, p2)
    idx = rd_t.view(B, hp, wp, p2).unsqueeze(1).expand(B, C, hp, wp, p2)
    sub = torch.gather(x, 4, idx)  # [B, C, hp, wp, p*p]

    # --- high-scale target, laid out to match PixelShuffle(t) ---------------
    hr = sub[..., :t2].view(B, C, hp, wp, t, t)
    hr = hr.permute(0, 1, 2, 4, 3, 5).reshape(B, C, hp * t, wp * t)

    # --- low-scale inputs, one per remaining slot ---------------------------
    lr = sub[..., t2:].permute(0, 4, 1, 2, 3).reshape(B * (p2 - t2), C, hp, wp)
    hr = hr.unsqueeze(1).expand(B, p2 - t2, C, hp * t, wp * t).reshape(B * (p2 - t2), C, hp * t, wp * t)

    return lr.contiguous(), hr.contiguous()


# =====================================================================
# Inference: decompose by t, denoise, average
# =====================================================================


def _pixel_unshuffle(x: torch.Tensor, f: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """[B,C,H,W] -> [B*f*f, C, H/f, W/f], reflect-padding to a multiple of f."""
    sz = (x.shape[-2], x.shape[-1])
    p_h, p_w = [math.ceil(s / f) * f - s for s in sz]
    if p_h or p_w:
        x = F.pad(x, (0, p_w, 0, p_h), mode="reflect")
    B, C, H, W = x.shape
    h, w = H // f, W // f
    x = x.contiguous().view(B, C, h, f, w, f)
    y = x.permute(0, 3, 5, 1, 2, 4).contiguous().view(B * f * f, C, h, w)
    return y, sz


class NSPDenoiser(nn.Module):
    """Drop-in denoiser wrapper: ``forward(x in [0,1]) -> denoised x in [0,1]``.

    Matches the call signature of the N2N denoiser used elsewhere in the
    experiment, so the two are interchangeable in the SSL view pipeline.
    """

    def __init__(self, net: DBSNl, ids: int = 2, clamp: bool = True):
        super().__init__()
        self.net = net
        self.ids = int(ids)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        f = self.ids
        y, sz = _pixel_unshuffle(x, f)
        out = self.net(y)  # [B*f*f, C, H, W]
        B = out.shape[0] // (f * f)
        out = out.view(B, f * f, *out.shape[1:]).mean(dim=1)
        out = out[..., : sz[0], : sz[1]]
        return out.clamp(0.0, 1.0) if self.clamp else out


def build_nsp_denoiser(
    device: torch.device,
    in_ch: int = 3,
    base_ch: int = 128,
    num_module: int = 9,
    ids: int = 2,
) -> NSPDenoiser:
    net = DBSNl(in_ch=in_ch, out_ch=in_ch, base_ch=base_ch, num_module=num_module, ids=ids)
    wrapper = NSPDenoiser(net, ids=ids).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[nsp] DBSNl base_ch={base_ch} num_module={num_module} ids={ids} -> {n_par:,} params")
    return wrapper


# =====================================================================
# Training
# =====================================================================


def train_nsp_denoiser(
    denoiser: NSPDenoiser,
    dataloader,
    device: torch.device,
    total_steps: int = 20000,
    patch_size: int = 32,
    patch_batch_size: int = 8,
    lr: float = 1e-4,
    tds: int = 4,
    const_lr: bool = False,
    seed: int = 0,
    log_every: int = 500,
    unnormalize_fn=None,
) -> NSPDenoiser:
    """Train NSP on noisy images drawn from the SSL dataloader.
    """
    net = denoiser.net
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = (
        None
        if const_lr
        else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, total_steps // 3), gamma=0.1)
    )

    rng = np.random.default_rng(seed)
    t = denoiser.ids

    data_iter = iter(dataloader)
    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # NoisyCIFAR10SSL yields ((x0, x1), label); use view 0, as N2N does.
        z = batch[0][0] if isinstance(batch[0], (tuple, list)) else batch[0]
        z = z.to(device)
        z_01 = unnormalize_fn(z).clamp(0, 1) if unnormalize_fn is not None else z.clamp(0, 1)

        B, C, H, W = z_01.shape
        patches = torch.empty(patch_batch_size, C, patch_size, patch_size, device=device)
        for b in range(patch_batch_size):
            n = random.randint(0, B - 1)
            i = random.randint(0, H - patch_size)
            j = random.randint(0, W - patch_size)
            patches[b] = z_01[n, :, i : i + patch_size, j : j + patch_size]

        lr_in, hr_tgt = build_cross_scale_pair(patches, p=tds, t=t, rng=rng)

        opt.zero_grad(set_to_none=True)
        pred = net(lr_in)
        loss = F.l1_loss(pred, hr_tgt)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        if log_every > 0 and step % log_every == 0:
            print(f"[nsp][step {step:05d}/{total_steps}] l1={loss.item():.6f}")

    net.eval()
    denoiser.eval()
    return denoiser
