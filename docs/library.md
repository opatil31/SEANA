# Self-Aligned Noise Augmentation (SEANA) for your SSL pipeline

SEANA (Self-Aligned Noise Augmentation) is a noise-aware view generation and SSL augmentation module that learns a variance-stabilizing transform for noisy image data and generates independently resampled (noise-aware) views for self-supervised learning. It is a plug-and-play module that works with any joint-embedding SSL approach and can be plugged directly into typical augmentation / view generation pipelines used in SSL. Further details are available below and in the [NeurIPS'26 publication](https://neurips.cc/virtual/2026/loc/sydney/poster/153415).

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
The old `seana-ssl[drunet]` install spelling remains supported for compatibility.

Choose the PyTorch CPU/CUDA build appropriate for your environment. The library
does not impose a CUDA index. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Python 3.10+ is required. Research dependencies retain their Python 3.12 recipe.

To reproduce this package release, pin `seana-ssl==0.1.3` in your environment.

## Fit once, generate views

```python
import torch
from seana import SEANA

device = "cuda" if torch.cuda.is_available() else "cpu"
seana = SEANA(
    sigma_d=0.08,       # Gaussian denoiser's noise-level input
    sigma_vst=0.1,      # Resampling standard deviation in VST space
    channels=3,
    device=device,
)

# train_loader yields raw floating [B,3,H,W] images, optionally followed by labels.
# Fit on training images only, before ImageNet/encoder mean-std normalization.
seana.fit(train_loader, steps=5000, patch_size=64, seed=0)

for images, _ in train_loader:
    source_a, source_b = seana(images.to(device))
    # Apply independent SSL transforms to each source, as shown below.
```

These noise levels demonstrate the API. Appropriate values depend on the dataset,
fitted transform and SSL objective. The library does not automatically tune them
against downstream labels. `sigma_d` and `sigma_vst` have different roles.

`SEANA()` defaults to RGB DRUNet on CPU with `sigma_d=20/255` and `sigma_vst=0.05`.
Set `channels=1` to use the pretrained grayscale model. For an offline machine,
pass `pretrained="/path/to/drunet_color.pth"` with compatible local weights.
`SEANA.from_drunet(...)` remains available as an explicit factory for the same
denoiser. Setting `pretrained=None` explicitly creates an untrained DRUNet for
testing and is not a substitute for pretrained denoising.

Calling `seana(images)` denoises once and returns two independent noise draws
**at the same spatial geometry**. It does not add an implicit crop or normalization
policy. For a dataset of already-noisy images, use this order:

**Original noisy image → SEANA denoising and independent resampling → independent
SSL crops/photometric transforms → encoder normalization → SSL model.**

The fitting loader and the SSL source loader should yield original noisy images,
converted to floating tensors at the intended intensity scale, before SSL
augmentations. Apply your single-view transforms to the two SEANA outputs:

```python
source_a, source_b = seana(images)
view_a = torch.stack([ssl_view_a(image) for image in source_a])
view_b = torch.stack([ssl_view_b(image) for image in source_b])
loss = ssl_loss(encoder(view_a), encoder(view_b))
loss.backward()
```

Here `ssl_view_a` and `ssl_view_b` are your single-view augmentation pipelines,
including normalization as their last step. They can share a stochastic transform
or use different branch settings, as in BYOL. Calling them per image gives
independent transform draws across examples as well as across branches.

This order lets SEANA denoise the original acquisition noise before resizing,
blur or color operations alter it. Those operations then act on the resampled
sources, so noise after the SSL transforms need not retain its pre-transform
statistics. A crop without interpolation or a flip preserves retained pixel
values, but a crop still changes the denoiser's available context.

The API also permits `.augment(crop)` for an explicitly chosen alternate protocol,
but SEANA should be integrated into augmentation pipelines in the order described
above.

## LightlySSL integration

Use a **single-view** Lightly transform on each independently resampled source.
Its normal final normalization can stay enabled because SEANA runs first:

```python
from lightly.transforms import SimCLRViewTransform
from seana import SEANA

device = "cuda" if torch.cuda.is_available() else "cpu"
seana = SEANA(
    sigma_d=0.08,       # Gaussian denoiser's noise-level input
    sigma_vst=0.1,      # Resampling standard deviation in VST space
    channels=3,
    device=device,
)

# train_loader yields raw floating [B,3,H,W] images, optionally followed by labels.
# Fit on training images only, before ImageNet/encoder mean-std normalization.
seana.fit(raw_train_loader, steps=5000, patch_size=64, seed=0)

ssl_view = SimCLRViewTransform(input_size=224)
for images, _ in raw_train_loader:
    source_a, source_b = seana(images.to(device))
    view_a = torch.stack([ssl_view(image) for image in source_a])
    view_b = torch.stack([ssl_view(image) for image in source_b])
    loss = ssl_loss(encoder(view_a), encoder(view_b))
    loss.backward()
```

`SimCLRViewTransform` includes crop/resize, flipping, color jitter, grayscale,
blur and normalization. Tensor inputs require torchvision transforms v2, available
in the tested torchvision 0.20.1 / Lightly 1.5.25 research environment.
See the [Lightly transform API](https://docs.lightly.ai/self-supervised-learning/lightly.transforms.html#lightly.transforms.simclr_transform.SimCLRViewTransform).
Keep the raw-image loader free of the usual two-view dataset transform and run
SEANA in the training process, after batching. Using one SEANA noise draw as the
shared input to a two-view `SimCLRTransform` would share that draw across both
views, so it is not the independent-resampling path shown above.

The augmenter stays in evaluation mode with frozen parameters, even when it is
a child of a module on which `.train()` is called. Its outputs are detached
regular tensors, so the encoder still receives parameter gradients. Do not put
the complete SSL encoder/loss computation inside `torch.no_grad()`.

A complete **LightlySSL SimCLR + SEANA** example is included in the repository:

```bash
uv run --python 3.12 --extra experiments examples/ssl_pipeline.py
```

It fits SEANA using pretrained DRUNet, generates two independent sources, then
applies Lightly's `SimCLRViewTransform` to each. A ResNet-18 backbone and
`SimCLRProjectionHead` train with Lightly's `NTXentLoss`. Only the SSL model is
optimized during this stage. SEANA stays frozen, and normalization happens after
SEANA. For GPU execution:

```bash
uv run --python 3.12 --extra experiments examples/ssl_pipeline.py --device cuda
```

The default uses 16 synthetic 32x32 tensors, two VST fitting steps, and one SSL
epoch to demonstrate integration. It is not a performance experiment. Replace
the loader with your noisy training images and set `--fit-steps` and `--epochs`
for your experiment. The raw loader should yield images before SSL transforms
or normalization. Labels and filenames, if present, are ignored.

For a download-free check with a toy averaging denoiser instead of DRUNet:

```bash
uv run --python 3.12 --extra experiments examples/ssl_pipeline.py --smoke \
  --checkpoint runs/library_example/smoke_seana.pt
```

The example saves and verifies a fitted SEANA checkpoint. Choose a new
`--checkpoint` path when rerunning it. Lightly is required by this example and
remains optional for the core library.

The root BBBC048 notebook provides a real-data demo in the full repository.

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
Use `variant="conditional"` for the captured conditional VST, with the correct
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

`uv.lock` records the checkout's dependency resolution. Consumer projects resolve the
package dependency bounds in their own environment, using their own lockfile.

For the complete research checkout:

```bash
uv sync --python 3.12 --extra experiments --extra imagenet100 --extra bbbc048 \
  --extra demo --group notebooks
```

Legacy `requirements*.txt` files retain the research/Colab installation paths.
Run experiment scripts from the full checkout. Installing the core into another
project does not install those scripts or datasets.

Original SEANA code is MIT licensed. Dependencies, pretrained weights and the
repository's comparison methods/datasets retain their own terms. Please cite
*Breaking Noise Shortcuts in Self-Supervised Learning via Noise-Aligned View
Generation*, Oankar R. Patil. The repository's `CITATION.cff`
tracks publication details.
