"""Practitioner API for the research core in seana.py (PyTorch only)."""

from contextlib import contextmanager
from collections.abc import Iterator
import math
from numbers import Real
from pathlib import Path
import random
from typing import Callable, Iterable

import torch
from torch import nn

__version__ = "0.1.3"
_DEFAULT_SIGMA_D = 20 / 255


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _sigma(value, batch_size=None):
    if torch.is_tensor(value):
        if batch_size is None:
            raise ValueError("The default sigma_vst must be a number or (low, high) pair")
        if value.shape not in (torch.Size([]), torch.Size([batch_size]), torch.Size([batch_size, 1, 1, 1])):
            raise ValueError("sigma_vst must be scalar, [B], or [B,1,1,1]")
        if value.dtype == torch.bool or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("sigma_vst must be finite and nonnegative")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("A sigma_vst interval needs exactly (low, high)")
        lo, hi = (_number(v, "sigma_vst") for v in value)
        if lo > hi:
            raise ValueError("sigma_vst interval must satisfy low <= high")
        return lo, hi
    return _number(value, "sigma_vst")


def _build_drunet(*, sigma_d, channels, pretrained, device):
    from seana import DeepInvDRUNetWrapper

    if channels not in (1, 3):
        raise ValueError("Pretrained DRUNet supports one or three image channels")
    if pretrained is not None and pretrained != "download" and not Path(pretrained).is_file():
        raise FileNotFoundError(f"DRUNet checkpoint does not exist: {pretrained}")
    try:
        from deepinv.models import DRUNet
    except ImportError as error:
        raise ImportError(
            "The default DRUNet needs DeepInverse, included by uv pip install seana-ssl. "
            "Alternatively, supply your own denoiser as SEANA(my_denoiser)."
        ) from error
    return DeepInvDRUNetWrapper(
        DRUNet(
            in_channels=channels,
            out_channels=channels,
            pretrained=str(pretrained) if pretrained is not None else None,
            device=device,
        ),
        sigma_denoiser=sigma_d,
    )


@contextmanager
def _fit_rng(seed):
    if seed is None:
        yield
        return
    _positive_int(seed, "seed", minimum=0)
    python_state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)


class _TrainingImages:
    def __init__(self, source, owner, patch_size, batch_key):
        if isinstance(source, Iterator):
            raise ValueError("fit() needs a re-iterable loader, not a one-shot iterator/generator")
        self.source, self.owner, self.patch_size, self.batch_key = source, owner, patch_size, batch_key

    def __iter__(self):
        seen = False
        for batch in self.source:
            if self.batch_key is not None:
                if not isinstance(batch, dict) or self.batch_key not in batch:
                    raise ValueError(f"Expected batches with image key {self.batch_key!r}")
                image = batch[self.batch_key]
            else:
                image = batch[0] if isinstance(batch, (tuple, list)) and batch else batch
            image, _ = self.owner._image(image)
            if min(image.shape[-2:]) < self.patch_size:
                raise ValueError("Training images must be at least as large as patch_size")
            seen = True
            yield image
        if not seen:
            raise ValueError("Training loader is empty (check batch_size and drop_last)")


class SEANA(nn.Module):
    """Fit an invertible VST around a frozen denoiser, then make noise-aligned views.

    Args:
        denoiser: Defaults to pretrained DeepInverse DRUNet. Supply an nn.Module
            mapping floating [B,C,H,W] images to the same shape to use your own.
            It must be differentiable with respect to its input during fitting.
            Its weights and train/eval state are frozen by this object.
        sigma_d: Built-in DRUNet's noise-level input, distinct from sigma_vst.
        pretrained: Built-in DRUNet weights. 'download' loads cached pretrained
            weights or downloads them on first use. A path loads local weights.
            None explicitly initializes untrained weights, mainly for testing.
        sigma_vst: Nonnegative resampling standard deviation in VST space, or a
            (low, high) interval sampled once per image, shared across a view pair.
        variant: "global" for one dataset-level transform, or "conditional" for
            the captured per-image conditioning implementation.
        channels: Number of input channels (also used to initialize conditional layers).
        knots: Number of spline knots.
        input_range: Initial physical intensity range; fit(calibrate=True) estimates
            a dataset-level range from training batches instead.
        clamp: Clamp inputs to the fitted range and generated outputs to that range.
        device: Device for the VST and denoiser. Defaults to CPU for built-in DRUNet
            or the custom denoiser's device when one is supplied.

    Call fit() once on raw noisy training images before inference. forward()
    denoises the original images and returns two independent noise draws at the
    same spatial geometry; augment() makes one such source. Apply independent
    SSL crops, photometric transforms and encoder normalization afterward.
    Outputs are detached tensors that can still be inputs to an encoder trained
    with backpropagation. Other operation orders are distinct experiment protocols.
    """

    def __init__(
        self,
        denoiser: nn.Module | None = None,
        *,
        sigma_vst=0.05,
        sigma_d=_DEFAULT_SIGMA_D,
        pretrained="download",
        variant="global",
        channels=3,
        knots=128,
        input_range=(0.0, 1.0),
        clamp=True,
        device=None,
    ):
        super().__init__()
        # Imported lazily so the root research module can re-export this facade.
        from seana import GlobalNoise2VST, ConditionalGlobalNoise2VST

        if denoiser is not None and not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module or None for pretrained DRUNet")
        if variant not in ("global", "conditional"):
            raise ValueError("variant must be 'global' or 'conditional'")
        _positive_int(channels, "channels")
        _positive_int(knots, "knots", minimum=2)
        if (
            not isinstance(input_range, (tuple, list))
            or len(input_range) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in input_range
            )
            or input_range[0] >= input_range[1]
        ):
            raise ValueError("input_range must contain two finite values with low < high")
        if not isinstance(clamp, bool):
            raise ValueError("clamp must be boolean")
        self.sigma_vst = _sigma(sigma_vst)
        sigma_d = _number(sigma_d, "sigma_d")
        self._settings = dict(
            variant=variant,
            channels=channels,
            knots=knots,
            input_range=tuple(map(float, input_range)),
            clamp=clamp,
        )
        builtin_spec = None
        if denoiser is None:
            device = torch.device("cpu" if device is None else device)
            denoiser = _build_drunet(sigma_d=sigma_d, channels=channels, pretrained=pretrained, device=device)
            builtin_spec = dict(kind="drunet", channels=channels, sigma_d=sigma_d)
        elif sigma_d != _DEFAULT_SIGMA_D or pretrained != "download":
            raise ValueError(
                "sigma_d and pretrained configure built-in DRUNet only. Configure your custom denoiser directly."
            )
        self.denoiser = denoiser
        if device is None:
            reference = next(iter(denoiser.parameters()), None)
            if reference is None:
                reference = next(iter(denoiser.buffers()), torch.empty(0))
            device = reference.device
        constructor = GlobalNoise2VST if variant == "global" else ConditionalGlobalNoise2VST
        self.vst = constructor(
            nb_knots=knots, inverse=True, data_min=input_range[0], data_max=input_range[1], clamp=clamp
        ).to(device)
        self.denoiser.to(device)
        if variant == "conditional":
            # Materialize LazyLinear before freezing, saving, or nesting in another module.
            with torch.no_grad():
                self.vst._params_from_hidden(torch.zeros(1, channels, 2, 2, device=device))
        self.register_buffer("_ready", torch.tensor(False, device=device))
        self._denoiser_spec = builtin_spec or dict(
            kind="custom",
            type=f"{type(denoiser).__module__}.{type(denoiser).__qualname__}",
            representation=repr(denoiser),
        )
        self.fit_summary = None
        self.train(False)
        self.requires_grad_(False)

    @property
    def fitted(self):
        return bool(self._ready.item())

    @property
    def device(self):
        return self.vst.data_min.device

    def train(self, mode=True):
        """Keep the augmenter frozen even when an enclosing SSL model enters train mode."""
        super().train(False)
        return self

    def _image(self, image):
        if not torch.is_tensor(image) or image.ndim not in (3, 4) or not image.is_floating_point():
            raise TypeError("Expected floating images [C,H,W] or [B,C,H,W], before encoder normalization")
        squeeze = image.ndim == 3
        if squeeze:
            image = image.unsqueeze(0)
        if min(image.shape) < 1 or image.shape[1] != self._settings["channels"]:
            raise ValueError(f"Expected nonempty images with {self._settings['channels']} channels")
        if not torch.isfinite(image).all():
            raise ValueError("Images must be finite")
        return image.detach().to(device=self.device, dtype=self.vst.data_min.dtype), squeeze

    def _denoise_checked(self, image):
        estimate = self.denoiser(image)
        if (
            not torch.is_tensor(estimate)
            or estimate.shape != image.shape
            or estimate.device != image.device
            or not estimate.is_floating_point()
        ):
            raise ValueError("Denoiser must return a floating tensor with the same shape and device")
        if not torch.isfinite(estimate).all():
            raise ValueError("Denoiser produced nonfinite values")
        return estimate

    def fit(
        self,
        loader: Iterable,
        *,
        steps=5000,
        patch_size=64,
        patch_batch_size=4,
        stride=4,
        lr=1e-2,
        calibrate=True,
        calibration_batches=50,
        batch_key=None,
        seed=None,
        progress: Callable[[float], None] | None = None,
    ):
        """Learn the VST from training images, ignoring labels, and freeze it afterward.

        loader must be re-iterable and yield tensors or (images, labels). For dict
        batches, specify batch_key. Fitting needs gradients through the denoiser
        input, but never changes denoiser weights or BatchNorm statistics.
        An optional seed isolates Python/PyTorch RNG state; a loader's private
        generator and stochastic worker processes are still the caller's responsibility.
        Calling fit again continues from the current VST weights.
        progress receives percent completion in [0,100]. Returns self.
        """
        for key, value in dict(
            steps=steps,
            patch_size=patch_size,
            patch_batch_size=patch_batch_size,
            stride=stride,
            calibration_batches=calibration_batches,
        ).items():
            _positive_int(value, key)
        if patch_size < 2 or stride > patch_size:
            raise ValueError("Require patch_size >= 2 and stride <= patch_size")
        lr = _number(lr, "lr", positive=True)
        if not isinstance(calibrate, bool) or (batch_key is not None and not isinstance(batch_key, str)):
            raise ValueError("calibrate must be boolean; batch_key must be a string or None")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Call fit outside torch.inference_mode(); VST fitting needs autograd")
        images = _TrainingImages(loader, self, patch_size, batch_key)
        self._ready.fill_(False)
        self.fit_summary = None
        self.train(False)
        self.denoiser.requires_grad_(False)
        self.vst.train().requires_grad_(True)

        def checkpoint(percent):
            if any(not torch.isfinite(p).all() for p in self.vst.parameters()):
                raise RuntimeError("VST fit became nonfinite; check inputs, denoiser and learning rate")
            if progress is not None:
                progress(percent)

        try:
            with _fit_rng(seed), torch.enable_grad():
                if calibrate:
                    self.vst.calibrate_from_dataloader(images, max_batches=calibration_batches)
                self.vst.fit_dataset(
                    images,
                    self._denoise_checked,
                    total_steps=steps,
                    patch_size=patch_size,
                    patch_batch_size=patch_batch_size,
                    stride=stride,
                    lr=lr,
                    progress_callback=checkpoint,
                )
            self._ready.fill_(True)
            self.fit_summary = dict(
                steps=steps,
                patch_size=patch_size,
                patch_batch_size=patch_batch_size,
                stride=stride,
                lr=lr,
                calibrated=calibrate,
                calibration_batches=calibration_batches,
                seed=seed,
                data_min=float(self.vst.data_min),
                data_range=float(self.vst.data_range),
            )
        finally:
            self.train(False)
            self.requires_grad_(False)
            for parameter in self.parameters():
                parameter.grad = None
        return self

    def _require_fit(self):
        if not self.fitted:
            raise RuntimeError("Fit SEANA on training images first, or load a fitted checkpoint")
        self.train(False)

    @torch.no_grad()
    def denoise(self, images):
        """Return the VST+denoiser estimate without drawing noise or advancing RNG state."""
        self._require_fit()
        x, squeeze = self._image(images)
        estimate = self.vst(x, self._denoise_checked)
        if self._settings["clamp"]:
            estimate = estimate.clamp(self.vst.data_min, self.vst.data_min + self.vst.data_range)
        return estimate.squeeze(0) if squeeze else estimate

    @torch.no_grad()
    def augment(self, images, *, sigma_vst=None, generator=None):
        """Generate one noise-aligned source before downstream SSL transforms."""
        self._require_fit()
        x, squeeze = self._image(images)
        sigma = _sigma(self.sigma_vst if sigma_vst is None else sigma_vst, len(x))
        result = self.vst.augment(
            x, self._denoise_checked, sigma_vst=sigma, clamp=self._settings["clamp"], generator=generator
        )
        return result.squeeze(0) if squeeze else result

    @torch.no_grad()
    def forward(self, images, *, sigma_vst=None, generator=None):
        """Denoise once and return two independent noise draws at the input geometry."""
        self._require_fit()
        x, squeeze = self._image(images)
        sigma = _sigma(self.sigma_vst if sigma_vst is None else sigma_vst, len(x))
        a, b = self.vst.augment_pair(
            x, self._denoise_checked, sigma_vst=sigma, clamp=self._settings["clamp"], generator=generator
        )
        return (a.squeeze(0), b.squeeze(0)) if squeeze else (a, b)

    @classmethod
    def from_drunet(
        cls, *, sigma_d=_DEFAULT_SIGMA_D, channels=3, pretrained="download", device="cpu", **kwargs
    ):
        """Explicit factory for the same pretrained DRUNet used by SEANA() by default.

        pretrained may be 'download', a checkpoint path,
        or None for an explicitly untrained architecture (not a useful denoiser).
        sigma_d is DRUNet's input noise level, distinct from sigma_vst.
        """
        return cls(None, sigma_d=sigma_d, channels=channels, pretrained=pretrained, device=device, **kwargs)

    def save(self, path):
        """Save fitted VST, denoiser weights and settings; refuse to overwrite a file."""
        self._require_fit()
        config = {**self._settings, "sigma_vst": _sigma(self.sigma_vst)}
        payload = dict(
            format_version=1,
            kind="seana_augmenter",
            library_version=__version__,
            config=config,
            denoiser=self._denoiser_spec,
            fit_summary=self.fit_summary,
            state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()},
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(payload, stream)
        return path

    @classmethod
    def load(cls, path, *, denoiser=None, device="cpu"):
        """Load a fitted checkpoint with weights_only=True, without downloading weights.

        Custom denoisers must be supplied as the same class and configuration;
        tensor state is restored strictly. Built-in DRUNet is reconstructed from
        stored settings. Checkpoints are augmentation artifacts, not fit-optimizer resumes.
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "seana_augmenter"
            or payload.get("format_version") != 1
        ):
            raise ValueError("Expected a version-1 SEANA library checkpoint")
        config, spec = payload["config"], payload["denoiser"]
        if spec["kind"] == "drunet":
            if denoiser is not None or spec["channels"] != config["channels"]:
                raise ValueError("Built-in DRUNet is reconstructed from the checkpoint; omit denoiser")
            result = cls.from_drunet(**config, sigma_d=spec["sigma_d"], pretrained=None)
        elif spec["kind"] == "custom":
            if denoiser is None:
                raise ValueError("Supply the same configured custom denoiser when loading this checkpoint")
            result = cls(denoiser, **config, device="cpu")
            if result._denoiser_spec != spec:
                raise ValueError("Custom denoiser class/configuration differs from the saved checkpoint")
        else:
            raise ValueError("Unknown checkpoint denoiser")
        result.load_state_dict(payload["state_dict"], strict=True)
        if not result.fitted or any(not torch.isfinite(v).all() for v in result.state_dict().values()):
            raise ValueError("Checkpoint contains an unfitted or nonfinite model")
        result.fit_summary = payload["fit_summary"]
        return result.to(device).eval().requires_grad_(False)
