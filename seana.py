"""SEANA variance-stabilizing transforms and noise-aligned views.

Extracted from the shared research implementation; see docs/provenance.md.
The core requires PyTorch. A denoiser is passed explicitly.
"""

from typing import Iterable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from random import randint
from contextlib import nullcontext

# Public library facade; its references to the core are imported lazily.
from _seana_api import SEANA as SEANA, __version__ as __version__


def _as_4d(z: torch.Tensor):
    if z.dim() == 3:
        return z.unsqueeze(0), True
    if z.dim() == 4:
        return z, False
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(z.shape)}")


def _clamp_to_data_range(x: torch.Tensor, data_min: torch.Tensor, data_range: torch.Tensor):
    mn = float(data_min.item())
    mx = mn + float(data_range.item())
    return x.clamp(mn, mx)


def _sigma_tensor(
    sigma_vst,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator = None,
):
    """
    Returns sigma shaped [B,1,1,1] to broadcast over [B,C,H,W].
    sigma_vst can be:
      - float
      - (lo, hi) tuple/list
      - torch scalar tensor
      - torch tensor of shape [B] or [B,1,1,1]
    """
    if torch.is_tensor(sigma_vst):
        s = sigma_vst.to(device=device, dtype=dtype)
        if s.dim() == 0:
            s = s.view(1, 1, 1, 1)
        elif s.dim() == 1:
            if s.numel() != B:
                raise ValueError(f"sigma_vst has shape {tuple(s.shape)} but batch B={B}")
            s = s.view(B, 1, 1, 1)
        elif s.dim() == 4:
            if s.size(0) != B:
                raise ValueError(f"sigma_vst has shape {tuple(s.shape)} but batch B={B}")
        else:
            raise ValueError(f"Unsupported sigma_vst tensor shape: {tuple(s.shape)}")
        if s.size(0) == 1 and B > 1:
            s = s.expand(B, 1, 1, 1)
        return s

    if isinstance(sigma_vst, (tuple, list)):
        lo, hi = float(sigma_vst[0]), float(sigma_vst[1])
        r = torch.rand((B, 1, 1, 1), device=device, dtype=dtype, generator=generator)
        return lo + (hi - lo) * r

    return torch.full((B, 1, 1, 1), float(sigma_vst), device=device, dtype=dtype)


def _unwrap_batch(batch):
    """Dataloader may return (x, y) or x. We only need x."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def augmentation(x, k=0, inverse=False):
    """
    Applies a geometric transformation to the input tensor. The function supports 8 distinct
    transformations defined by `k` in the range [0, 7], combining 4 possible rotations
    (0°, 90°, 180°, 270°) with or without horizontal flip.

    Parameters:
        x (torch.Tensor): Input tensor of shape (..., H, W), where H and W are spatial dimensions.
        k (int, optional): Transformation index in the range [0, 7]. Defaults to 0.
                           - k // 2 determines the rotation: 0 (0°), 1 (90°), 2 (180°), 3 (270°).
                           - k % 2 determines whether to apply a horizontal flip.
        inverse (bool, optional): If True, applies the inverse of the transformation corresponding
                                  to `k`. Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as the input.

    Example:
        x_aug = augmentation(x, k=5)        # Apply rotation + flip
        x_orig = augmentation(x_aug, k=5, inverse=True)  # Recover original
    """
    k = k % 8
    if inverse:
        k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k=k // 2, dims=[-2, -1])


def _per_image_normalize(z4: torch.Tensor, eps: float = 1e-12):
    """
    Per-image min/max normalization:
      z_norm = (z - z_min) / lam, lam = max(z_max - z_min, eps)
    Returns (z_norm, z_min, lam) with z_min/lam shaped [B,1,1,1] for broadcasting.
    """
    z_min = z4.amin(dim=(1, 2, 3), keepdim=True)
    z_max = z4.amax(dim=(1, 2, 3), keepdim=True)
    lam = (z_max - z_min).clamp_min(eps)
    z_norm = (z4 - z_min) / lam
    return z_norm, z_min, lam


class Donut3x3(nn.Module):
    """
    Fixed 3x3 convolutional filter with a donut-shaped kernel, as recommended by Noise2Self.

    This module applies a non-trainable depthwise convolution with a predefined "donut"-shaped
    kernel, which averages the four direct neighbors (top, bottom, left, right) of each pixel,
    ignoring the center and diagonal pixels. It is commonly used for smoothing or extracting
    neighbor-based features while preserving the center pixel.

    The kernel has the following fixed structure:
            [[0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]]

    Args:
        channel (int): Number of input (and output) channels.
                       The filter is applied independently to each channel.

    Attributes:
        conv (nn.Conv2d): Depthwise convolutional layer with fixed weights and `reflect` padding.

    Example:
        donut = Donut3x3(channel=3)
        x = torch.randn(1, 3, 32, 32)  # Example input
        y = donut(x)                  # Apply donut filter
    """

    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(
            channel, channel, 3, padding=1, padding_mode="reflect", bias=False, groups=channel
        )
        self.conv.weight.data = torch.zeros_like(self.conv.weight.data)
        self.conv.weight.data[:, :, 0, 1] = 0.25
        self.conv.weight.data[:, :, 1, 0] = 0.25
        self.conv.weight.data[:, :, 1, 2] = 0.25
        self.conv.weight.data[:, :, 2, 1] = 0.25
        self.conv.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class Spline(nn.Module):
    """
    Piecewise linear spline transformation with optional monotonicity constraints.

    This module defines a learnable spline function with a fixed number of knots over a bounded
    input domain. It supports both forward and inverse transformations, with an option to enforce
    the function to be strictly increasing (monotonic).

    Args:
        nb_knots (int): Number of knots used to define the piecewise linear spline. Defaults to 128.
        x_min (float or None): Minimum value of the input domain. Defaults to 0 if None.
        x_max (float or None): Maximum value of the input domain. Defaults to 1 if None.
        is_strictly_increasing (bool): If True, constrains the spline to be strictly increasing.
                                       Defaults to True.

    Attributes:
        theta (nn.Parameter): Learnable parameters defining the spline values at knots.
        alpha (nn.Parameter): Linear bias coefficient used in inverse mode to correct
                              for invertibility shifts.
        beta (nn.Parameter): Linear bias offset used in inverse mode to correct
                             for invertibility shifts.

    Methods:
        forward(z, inverse=False): Applies the spline transformation (or its inverse
                                   if `inverse=True`) to the input tensor `z`.

    Example:
        spline = Spline(nb_knots=32, x_min=-3, x_max=3, is_strictly_increasing=True)
        z = torch.randn(100)
        y = spline(z)                   # Apply forward spline transformation
        z_recovered = spline(y, inverse=True)  # Apply inverse transformation

    Notes:
        - When `is_strictly_increasing=True`, the spline is parameterized via a differentiable
          mapping that ensures strict monotonicity, using exponential reparameterization.
        - Inverse transform requires monotonicity (`is_strictly_increasing=True`).
        - In inverse mode, the output includes a learnable affine bias (`alpha * z + beta`).
    """

    def __init__(self, nb_knots=128, x_min=None, x_max=None, is_strictly_increasing=True):
        super().__init__()
        self.nb_knots = nb_knots
        self.x_min = 0 if x_min is None else x_min
        self.x_max = 1 if x_max is None else x_max
        assert self.x_min < self.x_max

        self.alpha = nn.Parameter(torch.zeros(1))  # for inverse transform
        self.beta = nn.Parameter(torch.zeros(1))  # for inverse transform

        self.is_strictly_increasing = is_strictly_increasing
        self.eps = 1e-6  # guarantees strict monotonicity

        # Initialization of parameters theta
        if self.is_strictly_increasing:
            self.theta = nn.Parameter(self.y2theta(torch.linspace(self.x_min, self.x_max, self.nb_knots)))
        else:
            self.theta = nn.Parameter(torch.linspace(self.x_min, self.x_max, self.nb_knots))

    def theta2y(self, theta):
        if not self.is_strictly_increasing:
            return theta
        theta0, theta1 = torch.split(theta, [1, self.nb_knots - 1], dim=0)
        return torch.cumsum(torch.cat((theta0, theta1.exp() + self.eps), dim=0), dim=0)

    def y2theta(self, y):
        if not self.is_strictly_increasing:
            return y
        return torch.cat((y[:1], torch.log(y[1:] - y[:-1] - self.eps)), dim=0)

    def _forward_1dtheta(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        z_input_size = z.size()
        z_flat = z.flatten()
        y = self.theta2y(theta)

        z_norm = (self.nb_knots - 1) * (z_flat - self.x_min) / (self.x_max - self.x_min)
        with torch.no_grad():
            i = torch.floor(z_norm).clip(min=0, max=self.nb_knots - 2).long()

        y_left = torch.gather(y, dim=0, index=i)
        y_right = torch.gather(y, dim=0, index=i + 1)
        t = z_norm - i
        out = y_left * (1 - t) + y_right * t
        return out.view(z_input_size)

    def _inverse_1dtheta(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_strictly_increasing
        bias = alpha * z + beta

        z_input_size = z.size()
        z_flat = z.flatten()
        y = self.theta2y(theta)

        with torch.no_grad():
            i = torch.searchsorted(y, z_flat).clip(min=1, max=self.nb_knots - 1).long()

        y_left = torch.gather(y, dim=0, index=i - 1)
        y_right = torch.gather(y, dim=0, index=i)
        t = (z_flat - y_left) / (y_right - y_left)
        out_flat = (self.x_max - self.x_min) / (self.nb_knots - 1) * (i - 1 + t) + self.x_min
        out = out_flat.view(z_input_size)
        return out + bias

    def forward(
        self,
        z: torch.Tensor,
        inverse: bool = False,
        theta: torch.Tensor = None,
        alpha: torch.Tensor = None,
        beta: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Backwards-compatible forward, but supports:
          - theta shape [nb_knots] (global)
          - theta shape [B, nb_knots] (per-sample), where z is [B, ...]
        """
        if theta is None:
            theta = self.theta

        # -------- Forward transform --------
        if not inverse:
            if theta.dim() == 2:
                B = z.shape[0]
                if theta.shape[0] != B:
                    raise ValueError(f"theta batch {theta.shape[0]} != z batch {B}")
                outs = []
                for b in range(B):
                    outs.append(self._forward_1dtheta(z[b : b + 1], theta[b]))
                return torch.cat(outs, dim=0)
            return self._forward_1dtheta(z, theta)

        # -------- Inverse transform --------
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        if theta.dim() == 2:
            B = z.shape[0]
            if theta.shape[0] != B:
                raise ValueError(f"theta batch {theta.shape[0]} != z batch {B}")

            def _slice_param(p: torch.Tensor, b: int) -> torch.Tensor:
                # If p is per-sample (shape [B,...]) slice it; else use as-is
                if torch.is_tensor(p) and p.dim() >= 1 and p.shape[0] == B:
                    return p[b : b + 1]
                return p

            outs = []
            for b in range(B):
                a_b = _slice_param(alpha, b)
                b_b = _slice_param(beta, b)
                outs.append(self._inverse_1dtheta(z[b : b + 1], theta[b], a_b, b_b))
            return torch.cat(outs, dim=0)

        return self._inverse_1dtheta(z, theta, alpha, beta)


class CondParamNet(nn.Module):
    """
    g(c) -> (Δtheta, Δalpha, Δbeta)

    Uses LazyLinear so it works with cond vectors of size 2C (C can be 1,3,4,...).
    """

    def __init__(self, nb_knots: int, hidden_dim: int = 64, delta_scale: float = 0.1):
        super().__init__()
        self.nb_knots = nb_knots
        self.delta_scale = float(delta_scale)

        self.fc1 = nn.LazyLinear(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_theta = nn.Linear(hidden_dim, nb_knots)
        self.fc_alpha = nn.Linear(hidden_dim, 1)
        self.fc_beta = nn.Linear(hidden_dim, 1)

        # Start near "no conditioning" (all deltas ~ 0)
        nn.init.zeros_(self.fc_theta.weight)
        nn.init.zeros_(self.fc_theta.bias)
        nn.init.zeros_(self.fc_alpha.weight)
        nn.init.zeros_(self.fc_alpha.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, cond: torch.Tensor):
        h = self.act(self.fc1(cond))
        h = self.act(self.fc2(h))
        dtheta = self.fc_theta(h) * self.delta_scale
        dalpha = self.fc_alpha(h) * self.delta_scale
        dbeta = self.fc_beta(h) * self.delta_scale
        return dtheta, dalpha, dbeta


class GlobalNoise2VST(nn.Module):
    """
    Dataset-level data-driven VST.

    Learns a single spline-based VST (and inverse/bias) shared across a dataset by
    minimizing the same blind-spot self-supervised loss, but with patches sampled
    from many noisy images.

      - Uses a FIXED dataset-level normalization (data_min/data_range), not per-image min/max.
      - Trains once across a dataloader via fit_dataset(...).
    """

    def __init__(
        self,
        nb_knots: int = 128,
        inverse: bool = True,
        data_min: float = 0.0,
        data_max: float = 1.0,
        clamp: bool = True,
    ):
        super().__init__()
        self.inverse = inverse

        # Spline works on normalized domain [0, 1]
        self.spline1 = Spline(nb_knots, 0, 1)
        if not self.inverse:
            self.spline2 = Spline(nb_knots, 0, 1)

        # Dataset-level affine normalization: z_norm = (z - data_min) / data_range
        data_range = (data_max - data_min) if (data_max > data_min) else 1.0
        self.register_buffer("data_min", torch.tensor(float(data_min)))
        self.register_buffer("data_range", torch.tensor(float(data_range)))
        self.clamp = clamp

    @torch.no_grad()
    def set_data_range(
        self,
        data_min: Union[float, torch.Tensor],
        data_max: Union[float, torch.Tensor],
    ):
        """Manually set dataset normalization parameters."""
        if isinstance(data_min, torch.Tensor):
            data_min = float(data_min.detach().cpu().item())
        if isinstance(data_max, torch.Tensor):
            data_max = float(data_max.detach().cpu().item())
        data_range = (data_max - data_min) if (data_max > data_min) else 1.0

        device = self.data_min.device
        self.data_min.copy_(torch.tensor(data_min, device=device))
        self.data_range.copy_(torch.tensor(data_range, device=device))

    @torch.no_grad()
    def calibrate_from_dataloader(self, dataloader: Iterable, max_batches: int = 100):
        """
        Estimate global data_min/data_range by scanning up to max_batches batches.
        Assumes dataloader yields tensors shaped [B,C,H,W] or [C,H,W].
        """
        mn = float("inf")
        mx = float("-inf")
        nb = 0

        for batch in dataloader:
            x = _unwrap_batch(batch)
            if not torch.is_tensor(x):
                raise TypeError("dataloader must yield a Tensor or (Tensor, ...)")

            x = x.detach()
            if x.dim() == 3:
                # [C,H,W]
                mn = min(mn, float(x.min().cpu().item()))
                mx = max(mx, float(x.max().cpu().item()))
            elif x.dim() == 4:
                # [B,C,H,W]
                mn = min(mn, float(x.min().cpu().item()))
                mx = max(mx, float(x.max().cpu().item()))
            else:
                raise ValueError(f"Expected dim 3 or 4, got shape {tuple(x.shape)}")

            nb += 1
            if nb >= max_batches:
                break

        if not (mn < float("inf") and mx > float("-inf")):
            raise RuntimeError("Failed to calibrate data range: no batches read?")
        if mx <= mn:
            mx = mn + 1.0

        self.set_data_range(mn, mx)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        z = (z - self.data_min) / self.data_range
        return z.clamp(0.0, 1.0) if self.clamp else z

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.data_range + self.data_min

    def fit_dataset(
        self,
        dataloader: Iterable,
        denoiser,
        total_steps: int = 20000,
        patch_size: int = 64,
        patch_batch_size: int = 4,
        stride: int = 4,
        lr: float = 1e-2,
        progress_callback=None,
    ):
        """
        Train the shared/global VST parameters using patches sampled from a dataset.

        Args:
            dataloader: yields noisy images (Tensor [B,C,H,W] or [C,H,W]).
            denoiser: callable denoiser operating on [B,C,H,W] in the VST domain.
            total_steps: number of SGD steps.
            patch_size: spatial crop size.
            patch_batch_size: number of random patches per step (sampled from current loader batch).
            stride: blind-spot mask stride.
            lr: Adam LR.
        """
        if total_steps <= 0:
            return

        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters(), lr=lr)
        step_size = max(1, total_steps // 3)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

        data_iter = iter(dataloader)
        conv = None  # lazy init once we know C

        for k in range(1, total_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            z = _unwrap_batch(batch)
            if z.dim() == 3:
                z = z.unsqueeze(0)  # [1,C,H,W]
            if z.dim() != 4:
                raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(z.shape)}")

            z = z.to(device)
            z = self._normalize(z)

            B, C, H, W = z.shape
            if H < patch_size or W < patch_size:
                raise ValueError(f"patch_size={patch_size} too large for image {H}x{W}")

            if conv is None:
                conv = Donut3x3(C).to(device)

            # Sample random augmented patches
            patches = torch.empty(patch_batch_size, C, patch_size, patch_size, device=device)
            for b in range(patch_batch_size):
                n = randint(0, B - 1)
                i = randint(0, H - patch_size)
                j = randint(0, W - patch_size)
                patch = augmentation(z[n : n + 1, :, i : i + patch_size, j : j + patch_size], randint(0, 7))
                patches[b : b + 1] = patch

            # Blind-spot mask (same across patches, matching your per-image implementation)
            mask = torch.zeros_like(patches)
            s = stride
            ii, jj, cc = randint(0, s - 1), randint(0, s - 1), randint(0, C - 1)
            mask[:, cc : cc + 1, ii::s, jj::s] = 1.0

            optimizer.zero_grad(set_to_none=True)

            vst_patches = self.spline1(patches)
            bs_input = conv(vst_patches) * mask + vst_patches * (1.0 - mask)

            if self.inverse:
                den = self.spline1(denoiser(bs_input), inverse=True)
            else:
                den = self.spline2(denoiser(bs_input))

            loss = F.mse_loss(
                den[:, cc : cc + 1, ii::s, jj::s],
                patches[:, cc : cc + 1, ii::s, jj::s],
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if progress_callback is not None:
                progress_callback(k / total_steps * 100.0)

    def forward(self, z: torch.Tensor, denoiser):
        """
        Apply learned global VST -> Gaussian denoiser -> inverse VST.
        Input z must be in the same scaling used to calibrate data_min/data_range.
        """
        device = next(self.parameters()).device
        z = z.to(device)
        z_norm = self._normalize(z)

        if self.inverse:
            out_norm = self.spline1(denoiser(self.spline1(z_norm)), inverse=True)
        else:
            out_norm = self.spline2(denoiser(self.spline1(z_norm)))

        return self._denormalize(out_norm)

    @torch.no_grad()
    def augment(
        self,
        z: torch.Tensor,
        denoiser,
        sigma_vst=0.05,
        clamp: bool = True,
        generator: torch.Generator = None,
        amp: bool = False,
        per_image_norm: bool = False,
        per_image_eps: float = 1e-12,
    ):
        if not self.inverse:
            raise NotImplementedError("augment() requires inverse=True (invertible VST).")

        device = next(self.parameters()).device
        z4, squeeze = _as_4d(z.to(device))

        autocast_ctx = torch.cuda.amp.autocast(enabled=amp) if (amp and z4.is_cuda) else nullcontext()

        with autocast_ctx:
            if per_image_norm:
                z_norm, z_min, lam = _per_image_normalize(z4, eps=per_image_eps)
            else:
                z_norm = self._normalize(z4)
                z_min = lam = None

            t = self.spline1(z_norm)
            t_hat = denoiser(t)

            B = t_hat.size(0)
            sig = _sigma_tensor(sigma_vst, B, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            if generator is None:
                eps = torch.randn_like(t_hat)
            else:
                eps = torch.randn(t_hat.shape, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            t_aug = t_hat + sig * eps

            out_norm = self.spline1(t_aug, inverse=True)

            if per_image_norm:
                out = out_norm * lam + z_min
                if clamp:
                    out = out.clamp(z_min, z_min + lam)
            else:
                out = self._denormalize(out_norm)
                if clamp:
                    out = _clamp_to_data_range(out, self.data_min, self.data_range)

        return out.squeeze(0) if squeeze else out

    @torch.no_grad()
    def augment_pair(
        self,
        z: torch.Tensor,
        denoiser,
        sigma_vst=0.05,
        clamp: bool = True,
        generator: torch.Generator = None,
        amp: bool = False,
        per_image_norm: bool = False,
        per_image_eps: float = 1e-12,
    ):
        if not self.inverse:
            raise NotImplementedError("augment_pair() requires inverse=True (invertible VST).")

        device = next(self.parameters()).device
        z4, squeeze = _as_4d(z.to(device))

        autocast_ctx = torch.cuda.amp.autocast(enabled=amp) if (amp and z4.is_cuda) else nullcontext()

        with autocast_ctx:
            if per_image_norm:
                z_norm, z_min, lam = _per_image_normalize(z4, eps=per_image_eps)
            else:
                z_norm = self._normalize(z4)
                z_min = lam = None

            t = self.spline1(z_norm)
            t_hat = denoiser(t)

            B = t_hat.size(0)
            sig = _sigma_tensor(sigma_vst, B, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            if generator is None:
                eps1 = torch.randn_like(t_hat)
                eps2 = torch.randn_like(t_hat)
            else:
                eps1 = torch.randn(t_hat.shape, device=t_hat.device, dtype=t_hat.dtype, generator=generator)
                eps2 = torch.randn(t_hat.shape, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            t1 = t_hat + sig * eps1
            t2 = t_hat + sig * eps2

            out1_norm = self.spline1(t1, inverse=True)
            out2_norm = self.spline1(t2, inverse=True)

            if per_image_norm:
                out1 = out1_norm * lam + z_min
                out2 = out2_norm * lam + z_min
                if clamp:
                    out1 = out1.clamp(z_min, z_min + lam)
                    out2 = out2.clamp(z_min, z_min + lam)
            else:
                out1 = self._denormalize(out1_norm)
                out2 = self._denormalize(out2_norm)
                if clamp:
                    out1 = _clamp_to_data_range(out1, self.data_min, self.data_range)
                    out2 = _clamp_to_data_range(out2, self.data_min, self.data_range)

        if squeeze:
            return out1.squeeze(0), out2.squeeze(0)
        return out1, out2


class ConditionalGlobalNoise2VST(nn.Module):
    """
    Estimate condition from a "hidden" patch where supervised pixels are replaced
    by neighbor averages (same J/mask as the loss), then generate per-patch spline params.

    - Still dataset-level training (single shared networks), but per-sample VST parameters.
    - Keeps dataset-level normalization (data_min/data_range).
    """

    def __init__(
        self,
        nb_knots: int = 128,
        inverse: bool = True,
        data_min: float = 0.0,
        data_max: float = 1.0,
        clamp: bool = True,
        cond_hidden_dim: int = 64,
        cond_delta_scale: float = 0.1,
    ):
        super().__init__()
        self.inverse = inverse

        # Base splines in [0,1]
        self.spline1 = Spline(nb_knots, 0, 1)
        self.cond1 = CondParamNet(nb_knots, hidden_dim=cond_hidden_dim, delta_scale=cond_delta_scale)

        if not self.inverse:
            self.spline2 = Spline(nb_knots, 0, 1)
            self.cond2 = CondParamNet(nb_knots, hidden_dim=cond_hidden_dim, delta_scale=cond_delta_scale)

        data_range = (data_max - data_min) if (data_max > data_min) else 1.0
        self.register_buffer("data_min", torch.tensor(float(data_min)))
        self.register_buffer("data_range", torch.tensor(float(data_range)))
        self.clamp = bool(clamp)

    @torch.no_grad()
    def set_data_range(self, data_min: Union[float, torch.Tensor], data_max: Union[float, torch.Tensor]):
        if isinstance(data_min, torch.Tensor):
            data_min = float(data_min.detach().cpu().item())
        if isinstance(data_max, torch.Tensor):
            data_max = float(data_max.detach().cpu().item())
        data_range = (data_max - data_min) if (data_max > data_min) else 1.0

        device = self.data_min.device
        self.data_min.copy_(torch.tensor(data_min, device=device))
        self.data_range.copy_(torch.tensor(data_range, device=device))

    @torch.no_grad()
    def calibrate_from_dataloader(self, dataloader: Iterable, max_batches: int = 100):
        mn = float("inf")
        mx = float("-inf")
        nb = 0

        for batch in dataloader:
            x = _unwrap_batch(batch)
            if not torch.is_tensor(x):
                raise TypeError("dataloader must yield a Tensor or (Tensor, ...)")
            x = x.detach()

            mn = min(mn, float(x.min().cpu().item()))
            mx = max(mx, float(x.max().cpu().item()))

            nb += 1
            if nb >= max_batches:
                break

        if not (mn < float("inf") and mx > float("-inf")):
            raise RuntimeError("Failed to calibrate data range: no batches read?")
        if mx <= mn:
            mx = mn + 1.0
        self.set_data_range(mn, mx)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        z = (z - self.data_min) / self.data_range
        return z.clamp(0.0, 1.0) if self.clamp else z

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.data_range + self.data_min

    @staticmethod
    def _cond_features(hidden: torch.Tensor) -> torch.Tensor:
        """
        h(·): simple condition extractor.
        hidden: [B,C,H,W] -> cond: [B,2C] = [per-channel mean ; per-channel std]
        """
        mean = hidden.mean(dim=(2, 3))
        std = hidden.std(dim=(2, 3), unbiased=False)
        return torch.cat([mean, std], dim=1)

    def _params_from_hidden(self, hidden: torch.Tensor):
        """
        Given hidden patch/image (with supervised pixels removed), produce per-sample VST parameters.
        Returns:
          theta1: [B, nb_knots]
          alpha1: [B, 1] (only used if inverse=True)
          beta1 : [B, 1] (only used if inverse=True)
          theta2: [B, nb_knots] or None (only used if inverse=False)
        """
        cond = self._cond_features(hidden)

        dtheta1, dalpha1, dbeta1 = self.cond1(cond)
        theta1 = self.spline1.theta[None, :] + dtheta1

        if self.inverse:
            alpha1 = self.spline1.alpha[None, :] + dalpha1
            beta1 = self.spline1.beta[None, :] + dbeta1
            theta2 = None
        else:
            alpha1 = None
            beta1 = None
            dtheta2, _, _ = self.cond2(cond)
            theta2 = self.spline2.theta[None, :] + dtheta2

        return theta1, alpha1, beta1, theta2

    def fit_dataset(
        self,
        dataloader: Iterable,
        denoiser,
        total_steps: int = 20000,
        patch_size: int = 64,
        patch_batch_size: int = 4,
        stride: int = 4,
        lr: float = 1e-2,
        progress_callback=None,
    ):
        if total_steps <= 0:
            return

        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=max(1, total_steps // 3), gamma=0.1)

        data_iter = iter(dataloader)
        conv = None  # lazy init once we know C

        for k in range(1, total_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            z = _unwrap_batch(batch)
            if z.dim() == 3:
                z = z.unsqueeze(0)  # [1,C,H,W]
            if z.dim() != 4:
                raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(z.shape)}")

            z = z.to(device)
            z = self._normalize(z)

            B, C, H, W = z.shape
            if H < patch_size or W < patch_size:
                raise ValueError(f"patch_size={patch_size} too large for image {H}x{W}")

            if conv is None:
                conv = Donut3x3(C).to(device)

            # Sample random augmented patches
            patches = torch.empty(patch_batch_size, C, patch_size, patch_size, device=device)
            for b in range(patch_batch_size):
                n = randint(0, B - 1)
                i = randint(0, H - patch_size)
                j = randint(0, W - patch_size)
                patch = augmentation(z[n : n + 1, :, i : i + patch_size, j : j + patch_size], randint(0, 7))
                patches[b : b + 1] = patch

            # J mask (same as your current code)
            mask = torch.zeros_like(patches)
            s = stride
            ii, jj, cc = randint(0, s - 1), randint(0, s - 1), randint(0, C - 1)
            mask[:, cc : cc + 1, ii::s, jj::s] = 1.0

            # ---------------- Tier-C conditioning ----------------
            # Build "hidden" patch where supervised pixels are removed via η(·) before estimating condition:
            hidden = conv(patches) * mask + patches * (1.0 - mask)
            theta1, alpha1, beta1, theta2 = self._params_from_hidden(hidden)
            # -----------------------------------------------------

            optimizer.zero_grad(set_to_none=True)

            # VST with conditional parameters
            vst_patches = self.spline1(patches, theta=theta1)

            # Blind-spot input for denoiser in VST space
            bs_input = conv(vst_patches) * mask + vst_patches * (1.0 - mask)

            den_vst = denoiser(bs_input)

            if self.inverse:
                den = self.spline1(den_vst, inverse=True, theta=theta1, alpha=alpha1, beta=beta1)
            else:
                den = self.spline2(den_vst, theta=theta2)

            loss = F.mse_loss(
                den[:, cc : cc + 1, ii::s, jj::s],
                patches[:, cc : cc + 1, ii::s, jj::s],
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if progress_callback is not None:
                progress_callback(k / total_steps * 100.0)

    def forward(self, z: torch.Tensor, denoiser):
        """
        Inference:
          - normalize
          - estimate condition from the *full* image (no masking needed at inference)
          - apply conditional VST -> denoise -> inverse conditional VST
        """
        device = next(self.parameters()).device
        z_in = z
        z = z.to(device)

        if z.dim() == 3:
            z = z.unsqueeze(0)

        z_norm = self._normalize(z)

        # Condition from full image at inference
        theta1, alpha1, beta1, theta2 = self._params_from_hidden(z_norm)

        if self.inverse:
            out_norm = self.spline1(
                denoiser(self.spline1(z_norm, theta=theta1)),
                inverse=True,
                theta=theta1,
                alpha=alpha1,
                beta=beta1,
            )
        else:
            out_norm = self.spline2(
                denoiser(self.spline1(z_norm, theta=theta1)),
                theta=theta2,
            )

        out = self._denormalize(out_norm)
        return out if z_in.dim() == 4 else out.squeeze(0)

    @torch.no_grad()
    def augment(
        self,
        z: torch.Tensor,
        denoiser,
        sigma_vst=0.05,
        clamp: bool = True,
        generator: torch.Generator = None,
        amp: bool = False,
        per_image_norm: bool = False,
        per_image_eps: float = 1e-12,
    ):
        if not self.inverse:
            raise NotImplementedError("augment() requires inverse=True (invertible VST).")

        device = next(self.parameters()).device
        z4, squeeze = _as_4d(z.to(device))

        autocast_ctx = torch.cuda.amp.autocast(enabled=amp) if (amp and z4.is_cuda) else nullcontext()

        with autocast_ctx:
            if per_image_norm:
                z_norm, z_min, lam = _per_image_normalize(z4, eps=per_image_eps)
            else:
                z_norm = self._normalize(z4)
                z_min = lam = None

            # conditional params inferred in the same normalized domain used for VST
            theta1, alpha1, beta1, _ = self._params_from_hidden(z_norm)

            t = self.spline1(z_norm, theta=theta1)
            t_hat = denoiser(t)

            B = t_hat.size(0)
            sig = _sigma_tensor(sigma_vst, B, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            if generator is None:
                eps = torch.randn_like(t_hat)
            else:
                eps = torch.randn(t_hat.shape, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            t_aug = t_hat + sig * eps

            out_norm = self.spline1(t_aug, inverse=True, theta=theta1, alpha=alpha1, beta=beta1)

            if per_image_norm:
                out = out_norm * lam + z_min
                if clamp:
                    out = out.clamp(z_min, z_min + lam)
            else:
                out = self._denormalize(out_norm)
                if clamp:
                    out = _clamp_to_data_range(out, self.data_min, self.data_range)

        return out.squeeze(0) if squeeze else out

    @torch.no_grad()
    def augment_pair(
        self,
        z: torch.Tensor,
        denoiser,
        sigma_vst=0.05,
        clamp: bool = True,
        generator: torch.Generator = None,
        amp: bool = False,
        per_image_norm: bool = False,
        per_image_eps: float = 1e-12,
    ):
        if not self.inverse:
            raise NotImplementedError("augment_pair() requires inverse=True (invertible VST).")

        device = next(self.parameters()).device
        z4, squeeze = _as_4d(z.to(device))

        autocast_ctx = torch.cuda.amp.autocast(enabled=amp) if (amp and z4.is_cuda) else nullcontext()

        with autocast_ctx:
            if per_image_norm:
                z_norm, z_min, lam = _per_image_normalize(z4, eps=per_image_eps)
            else:
                z_norm = self._normalize(z4)
                z_min = lam = None

            theta1, alpha1, beta1, _ = self._params_from_hidden(z_norm)

            t = self.spline1(z_norm, theta=theta1)
            t_hat = denoiser(t)

            B = t_hat.size(0)
            sig = _sigma_tensor(sigma_vst, B, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            if generator is None:
                eps1 = torch.randn_like(t_hat)
                eps2 = torch.randn_like(t_hat)
            else:
                eps1 = torch.randn(t_hat.shape, device=t_hat.device, dtype=t_hat.dtype, generator=generator)
                eps2 = torch.randn(t_hat.shape, device=t_hat.device, dtype=t_hat.dtype, generator=generator)

            t1 = t_hat + sig * eps1
            t2 = t_hat + sig * eps2

            out1_norm = self.spline1(t1, inverse=True, theta=theta1, alpha=alpha1, beta=beta1)
            out2_norm = self.spline1(t2, inverse=True, theta=theta1, alpha=alpha1, beta=beta1)

            if per_image_norm:
                out1 = out1_norm * lam + z_min
                out2 = out2_norm * lam + z_min
                if clamp:
                    out1 = out1.clamp(z_min, z_min + lam)
                    out2 = out2.clamp(z_min, z_min + lam)
            else:
                out1 = self._denormalize(out1_norm)
                out2 = self._denormalize(out2_norm)
                if clamp:
                    out1 = _clamp_to_data_range(out1, self.data_min, self.data_range)
                    out2 = _clamp_to_data_range(out2, self.data_min, self.data_range)

        if squeeze:
            return out1.squeeze(0), out2.squeeze(0)
        return out1, out2


class DeepInvDRUNetWrapper(nn.Module):
    """
    Wrap deepinv.models.DRUNet so it matches:
      denoiser(x: [B,C,H,W]) -> [B,C,H,W]

    sigma_denoiser is the DRUNet noise-level input (not your augmentation sigma_vst).
    """

    def __init__(self, drunet: nn.Module, sigma_denoiser: float = 20 / 255, clip_input: bool = False):
        super().__init__()
        self.drunet = drunet.eval()
        for p in self.drunet.parameters():
            p.requires_grad_(False)

        self.sigma_denoiser = float(sigma_denoiser)
        self.clip_input = bool(clip_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

        x_in = x.clamp(0.0, 1.0) if self.clip_input else x
        return self.drunet(x_in, self.sigma_denoiser)
