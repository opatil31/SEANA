# Self-Aligned Noise Augmentation (SEANA)

## Noise-aligned view generation for Noise-Robust Self-Supervised Learning (SSL)

SEANA is a drop-in view generation module for joint-embedding SSL on noisy images for noise-robust representation learning. It learns an invertible variance-stabilizing transform from noisy training data, estimates the clean signal with a frozen denoiser, and resamples independent noise to create two noise-aware SSL views. Your encoder, SSL objective, and remaining augmentation pipeline stay the same. You don't need to train a dataset-specific denoiser and then also you don't need to use a denoiser or SEANA module at inference. Further details are available below and in the [NeurIPS'26 publication](https://neurips.cc/virtual/2026/loc/sydney/poster/153415).

| Learn from noisy data | Use your SSL pipeline | Fit once, reuse |
| :--- | :--- | :--- |
| No clean training targets, noise-model specification, or training a dataset-specific denoiser required. | Generate independent noise-aware source views before your usual crops, color transforms and other augmentations. | Pretrained DRUNet denoiser is the calibration default used by SEANA. Save the fitted SEANA module. Feel free to supply your own custom denoiser for SEANA calibration. |

Author: **Oankar R. Patil**.

## Install with uv

Add SEANA to your uv project:

```bash
uv add seana-ssl
```

For an existing environment using uv's pip interface:

```bash
uv pip install seana-ssl
# Standard pip works too:
python -m pip install seana-ssl
```

For local development from a source checkout:

```bash
uv sync
uv run python -c "import seana"
# Or add the checkout to a different project:
uv add /path/to/SEANA
```

**Pretrained DRUNet is the default denoiser.** A standard install includes
PyTorch and DeepInverse, so `SEANA()` works without an extra dependency flag.
Constructing it loads cached pretrained weights or downloads them on first use.
Importing `seana` alone does not load DeepInverse or download weights.
Passing `SEANA(my_denoiser)` uses your module without loading DRUNet.

## Quick start

Install SEANA and LightlySSL into a uv project:

```bash
uv add seana-ssl lightly torchvision
```

With pip, use `python -m pip install seana-ssl lightly torchvision`. The SEANA
package is **`seana-ssl`**, and the Python import is **`seana`**. Python 3.10+
is supported.

This LightlySSL SimCLR example fits SEANA once, then runs one SSL training epoch.
Supply `noisy_train_loader` with batches of original noisy images as floating
tensors in [0, 1], shaped `[B, 3, H, W]`, followed by labels or other metadata.
Use at least two images per batch. Keep SSL augmentations and encoder
normalization out of this loader:

```python
import torch
from torch import nn
from torchvision.models import resnet18
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRViewTransform
from seana import SEANA

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fit once on the training split. Labels are ignored.
augmenter = SEANA(sigma_d=0.08, sigma_vst=0.1, device=device)
augmenter.fit(noisy_train_loader, steps=5000, patch_size=64, seed=0)

# Set up a standard LightlySSL SimCLR pipeline.
ssl_view = SimCLRViewTransform(input_size=224)
resnet = resnet18(weights=None)
model = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Flatten(1),
    SimCLRProjectionHead(512, 512, 128),
).to(device).train()
criterion = NTXentLoss(temperature=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for images, *_ in noisy_train_loader:
    # SEANA first, then independent Lightly augmentations for each source.
    source_a, source_b = augmenter(images.to(device))
    view_a = torch.stack([ssl_view(image) for image in source_a])
    view_b = torch.stack([ssl_view(image) for image in source_b])

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(view_a), model(view_b))
    loss.backward()
    optimizer.step()
```

Cropping, photometric augmentation and normalization happen **after SEANA**.
The fitted SEANA module stays frozen while the encoder and projection head train.

`SEANA()` uses frozen pretrained DRUNet, downloading its weights on first use.
`SEANA(my_denoiser)` uses your own module.

**The example hyperparameter values sigma_vst (SEANA resampling strength in VST domain) and sigma_d (sigma denoising value supplied to DRUNet) are starting
values, and need to be tuned for a given noisy dataset**.

The [library guide](https://github.com/opatil31/SEANA/blob/main/docs/library.md) covers
input scaling, fitting, custom denoisers, resampling strength and checkpoints.

The [runnable LightlySSL example](https://github.com/opatil31/SEANA/blob/main/examples/ssl_pipeline.py) also provides a toy
dataset and checkpoint saving. From a repository checkout, run
`uv run --python 3.12 --extra experiments examples/ssl_pipeline.py`.

## Try the Colab demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opatil31/SEANA/blob/6350a61a9b43ec0100496cc58637a158eb17cccb/SEANA_BBBC048_Demo.ipynb)

The [BBBC048 notebook](https://github.com/opatil31/SEANA/blob/main/SEANA_BBBC048_Demo.ipynb) brings the pipeline to real
fluorescence microscopy images. Explore the resampling-strength slider and a
baseline-versus-SEANA SSL comparison using the package from PyPI.

**Demo budget:** 70 SSL epochs and 50 linear-probe epochs per arm. These are
shorter demonstration runs than the paper's training recipes. Follow the
notebook's setup cells to load the repository helpers and prepared dataset.

## Supply your own denoiser

```python
seana = SEANA(my_denoiser, channels=1, sigma_vst=0.05, device="cuda")
seana.fit(train_loader, steps=5000, patch_size=64)
```

The denoiser must be an `nn.Module` mapping floating `[B,C,H,W]` tensors to the
same shape/device. Its forward pass must be differentiable with respect to its
**input** during VST fitting, although its own weights and BatchNorm statistics
stay frozen. A denoiser whose forward uses `torch.no_grad()` cannot train a VST
through that method.

The `sigma_d` and `pretrained` options configure the built-in DRUNet only.
Configure any noise level or weight loading directly on your custom denoiser.

SEANA owns the module by reference: it moves and freezes it. Pass a separate copy
if another part of your application trains the same denoiser. Inputs may be CHW
or BCHW for inference. Denoiser-specific image/patch-size constraints still apply.

## Fitting options

`fit()` accepts a re-iterable loader of tensors or `(images, labels)` batches.
For dictionaries, pass `batch_key="image"`. Other fields are ignored. One-shot
generators and empty loaders are rejected. Training images must have the configured
channel count and be at least as large as `patch_size`.

By default, the first 50 training batches calibrate one dataset-level intensity
range. To use a known range:

```python
seana = SEANA(my_denoiser, input_range=(0.0, 1.0), sigma_vst=0.1)
seana.fit(train_loader, calibrate=False, steps=5000, patch_size=64,
          patch_batch_size=4, stride=4, lr=0.01)
```

The default spline has 128 knots and uses the captured global fitting objective.
Use `variant="conditional"` for the conditional VST, with the correct
`channels` specified at construction. Existing low-level imports
`GlobalNoise2VST`, `ConditionalGlobalNoise2VST` and `DeepInvDRUNetWrapper` remain
available for research code.

An optional `seed` isolates the loop's Python/PyTorch RNG state. Separate loader
generators and stochastic worker transforms remain the caller's responsibility.
The `progress` callback receives percent completion. Repeated `fit()` calls
continue the current VST weights. Construct a new augmenter for a fresh fit.
A failed fit remains unusable until fitting succeeds. Fitting needs autograd
and must run outside `torch.inference_mode()`.

## Denoising and reproducible resampling

```python
estimate = seana.denoise(images)  # No noise draw and no RNG advance.
a, b = seana(images, sigma_vst=0.0)  # Equal views with resampling disabled.
a, b = seana(images, sigma_vst=(0.05, 0.15))  # One scale per image, shared by the pair.

generator = torch.Generator(device=device).manual_seed(123)
a, b = seana(images, sigma_vst=0.1, generator=generator)
```

Use a generator on the augmenter's device. Recreating it with the same seed
repeats the noise. Reusing it advances the stream. Each view receives independent
noise. Per-image strengths may be tensors shaped `[B]` or `[B,1,1,1]`. All
strengths must be finite and nonnegative. Outputs are clamped to the fitted
physical intensity range by default. `clamp=False` disables both input and output
clamps. `denoise()` is an estimate, not clean ground truth.

## Save and reload

```python
seana.save("weights/seana.pt")
restored = SEANA.load("weights/seana.pt", device=device)
```

For a custom denoiser, supply the same configured module:

```python
restored = SEANA.load("weights/seana.pt", denoiser=my_reconstructed_denoiser,
                      device=device)
```

Checkpoints store the VST, denoiser tensors, settings and fit summary. DRUNet
reloads with `pretrained=None`, without another download. Custom denoisers are
never dynamically imported or unpickled: the provided class/representation is
checked and tensor states load strictly. Preserve any custom behavior not encoded
in `state_dict()` or `extra_repr()` yourself. Loading uses a typed payload and
`torch.load(weights_only=True)`. Saving refuses an existing file. These are
inference artifacts, not optimizer-state resume checkpoints.

## Building the library and running research code

```bash
uv lock --check
uv build
```

For the complete research checkout:

```bash
uv sync --python 3.12 --extra experiments --extra imagenet100 --extra bbbc048 \
  --extra demo --group notebooks
```

Original SEANA code is MIT licensed. Dependencies, pretrained weights and the
repository's comparison methods/datasets retain their own terms. Please cite
*Breaking Noise Shortcuts in Self-Supervised Learning via Noise-Aligned View
Generation*, Oankar R. Patil. The repository's `CITATION.cff`
tracks publication details.
