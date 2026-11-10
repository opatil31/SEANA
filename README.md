<div align="center">

<h1>SEANA</h1>
<p><strong>Self-Aligned Noise Augmentation (SEANA) </strong></p>
<p>Noise-aligned view generation for self-supervised learning.</p>

<p>
  <a href="https://pypi.org/project/seana-ssl/"><img src="https://img.shields.io/pypi/v/seana-ssl?color=2563eb&amp;label=PyPI" alt="PyPI version"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/SEANA_core-MIT-0f766e" alt="SEANA core: MIT license"></a>
  <a href="https://colab.research.google.com/github/opatil31/SEANA/blob/main/SEANA_BBBC048_Demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open demo in Colab"></a>
</p>

<p>
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/153415">Paper</a> ·
  <a href="docs/library.md">Library guide</a> ·
  <a href="examples/ssl_pipeline.py">LightlySSL example</a> ·
  <a href="#research-experiments">Experiments</a> ·
  <a href="#prepared-datasets">Datasets</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

SEANA is a drop-in view generation module for joint-embedding SSL on noisy images.
It learns an invertible variance-stabilizing transform from noisy training data,
estimates the clean signal with a frozen denoiser, and resamples independent noise to
create two SSL views. Your encoder, SSL objective and remaining augmentation pipeline stay the same.

| Learn from noisy data | Use your SSL pipeline | Fit once, reuse |
| :--- | :--- | :--- |
| No clean training targets or noise-model specification required. | Generate independent source views before your usual crops, color transforms and other augmentations. | Pretrained DRUNet denoiser is the calibration default. Save the fitted SEANA module, or supply your own denoiser. |

## Quick start

Install the reusable library into a uv project:

```bash
uv add seana-ssl
```

With pip, use `python -m pip install seana-ssl`. The package is **`seana-ssl`**,
and the Python import is **`seana`**. Python 3.10+ is supported.

Use a loader of original noisy training images, converted to floating tensors
before SSL augmentations or encoder normalization:

```python
import torch
from seana import SEANA

device = "cuda" if torch.cuda.is_available() else "cpu"
augmenter = SEANA(sigma_d=0.08, sigma_vst=0.1, device=device)

# noisy_train_loader yields (images, labels), with images shaped [B, 3, H, W].
# Fit once on the training split. Labels are ignored.
augmenter.fit(noisy_train_loader, steps=5000, patch_size=64, seed=0)

for images, _ in noisy_train_loader:
    source_a, source_b = augmenter(images.to(device))
    # Apply your usual SSL transforms independently to each source,
    # then pass the resulting views to your SSL model.
```

`SEANA()` uses frozen pretrained DRUNet, downloading its weights on first use.
`SEANA(my_denoiser)` uses your own module. The example strengths are starting
values, not a universal recipe. The [library guide](docs/library.md) covers
input scaling, fitting, custom denoisers, resampling strength and checkpoints.

<details>
<summary><strong>Integrate with LightlySSL</strong></summary>

Add Lightly to your environment with `uv add lightly torchvision`. After fitting
SEANA, use a single-view transform on each independently resampled source:

```python
from lightly.transforms import SimCLRViewTransform

ssl_view = SimCLRViewTransform(input_size=224)

# Inside your existing SSL training loop:
source_a, source_b = augmenter(images.to(device))
view_a = torch.stack([ssl_view(image) for image in source_a])
view_b = torch.stack([ssl_view(image) for image in source_b])
```

This keeps cropping, photometric augmentation and normalization **after SEANA**.
The [complete SimCLR example](examples/ssl_pipeline.py) includes a ResNet-18
backbone, Lightly's projection head and NT-Xent loss. From a repository checkout:

```bash
uv run --python 3.12 --extra experiments examples/ssl_pipeline.py
```

</details>

## Try the Colab demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opatil31/SEANA/blob/main/SEANA_BBBC048_Demo.ipynb)

The [BBBC048 notebook](SEANA_BBBC048_Demo.ipynb) brings the pipeline to real
fluorescence microscopy images. Explore the resampling-strength slider and a
baseline-versus-SEANA SSL comparison using the package from PyPI.

**Demo budget:** 50 SSL epochs and 20 linear-probe epochs per arm. These are
shorter demonstration runs than the paper's training recipes. Follow the
notebook's setup cells to load the repository helpers and prepared dataset.

## Research experiments

The full repository includes experiment code, configurations, analysis tools
and prepared imaging datasets. The library wheel contains the reusable SEANA
core and practitioner API.

**SSL models:** VICReg · SimCLR · BYOL · MoCoV2 · LeJEPA<br>
**Backbones:** ResNet-18 · ResNet-50

| Experiment | What it covers | Code |
| :--- | :--- | :--- |
| **CIFAR-10** | Gaussian, Poisson–Gaussian and speckle noise, with clean-test linear probing | [cifar10/](cifar10/) |
| **ImageNet-100** | Gaussian, Poisson shot and speckle noise, with clean-validation linear probing | [imagenet100/](imagenet100/) |
| **BBBC048** | Cell-cycle classification, denoiser comparisons, component ablations and resampling-strength sweeps | [fluorescence_imaging_bbbc048/](fluorescence_imaging_bbbc048/) |
| **Fetal ultrasound** | Anatomical-plane classification, patient splits, per-class and device metrics | [fetal_ultrasound/](fetal_ultrasound/) |
| **Denoiser/oracle isolation** | PSNR, denoising-only controls, known-noise resampling and oracle augmentation | [denoiser_oracle_isolation_experiments/](denoiser_oracle_isolation_experiments/) |
| **Noise invariance** | Frozen-backbone ANOVA, content re-identification and embedding dimensionality | [Analysis guide](noise_invariance_analysis/README.md) |

### Set up the research environment

```bash
git clone https://github.com/opatil31/SEANA.git
cd SEANA
uv sync --python 3.12 --extra experiments --extra imagenet100 --extra bbbc048 \
  --extra demo --group notebooks
uv run --no-sync python -m cifar10.train --dry-run
```

Research recipes use Python 3.12, PyTorch 2.5.1 and torchvision 0.20.1. For a
CUDA-specific build, choose the matching packages using the
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
The `requirements*.txt` files remain available for pip-based environments.

<details>
<summary><strong>CIFAR-10 — single run and full grid</strong></summary>

```bash
uv run --no-sync python -m cifar10.train \
  --config configs/cifar10/resnet18.json \
  --method vicreg --condition pg_high --treatment seana \
  --seed 0 --download --output-dir runs/cifar10_example
```

Use [resnet50.json](configs/cifar10/resnet50.json) for ResNet-50. Treatments are
`baseline`, `seana`, `nsp` and `n2n`. CLI options override the JSON recipe.

| Condition | Corruption on images in [0, 1] |
| :--- | :--- |
| `gauss_low` / `gauss_high` | Gaussian sigma = 0.1 / 0.5 |
| `pg_low` | Poisson peak = 30, Gaussian sigma = 0.01 |
| `pg_high` | Poisson peak = 5, Gaussian sigma = 0.2 |
| `speckle_low` / `speckle_high` | Multiplicative Gaussian sigma = 0.2 / 0.6 |

Corruptions are clipped to [0, 1]. Noise is redrawn by default. Set
`--noise-sampling fixed` to hold each image's corruption fixed across epochs.
SEANA fits a VST around frozen DRUNet, NSP fits on noisy SSL views, and N2N fits
on raw noisy training images using the official neighbor-pair objective.
For offline SEANA fitting, pass `--drunet-weights /path/to/drunet_color.pth`.

The default grid has **720 runs**: 5 methods × 2 backbones × 6 conditions ×
4 treatments × 3 seeds.

```bash
uv run --no-sync python -m scripts.prepare_data --data-root data/cifar10
uv run --no-sync python -m scripts.make_grid --output runs/cifar10_grid.jsonl
uv run --no-sync python -m scripts.run_grid --grid runs/cifar10_grid.jsonl --index 0 --dry-run
uv run --no-sync python -m scripts.run_grid --grid runs/cifar10_grid.jsonl --index 0
```

Run one zero-based index per process or device. For Slurm, pass the array task
ID as `--index` and supply your cluster's account, partition and resources.
Use `--methods`, `--backbones`, `--conditions`, `--treatments` or `--seeds` when
generating the grid to select a smaller set.

</details>

<details>
<summary><strong>ImageNet-100 — prepare, train and evaluate</strong></summary>

```bash
uv run --no-sync python -m imagenet100.prepare --output data/imagenet100
uv run --no-sync python -m imagenet100.train \
  --config configs/imagenet100/resnet50.json \
  --method byol --condition shot_high --treatment seana \
  --output-dir runs/imagenet100_example
uv run --no-sync python -m imagenet100.evaluate --run runs/imagenet100_example --device cuda
```

Use [resnet18.json](configs/imagenet100/resnet18.json) for ResNet-18.
`python -m imagenet100.grid generate` creates the default 720-run matrix.
ImageNet data are obtained separately and are not bundled in this repository.

</details>

<details>
<summary><strong>BBBC048 — fluorescence imaging and ablations</strong></summary>

```bash
uv run --no-sync python -m fluorescence_imaging_bbbc048.dataset_archive extract \
  --archive fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip \
  --output data/bbbc048
uv run --no-sync python -m fluorescence_imaging_bbbc048.train \
  --method vicreg --treatment seana --manifest data/bbbc048/manifest.json \
  --output-dir runs/bbbc048_example
```

The default grid has **90 runs** on ResNet-18: 5 methods × 6 treatments ×
3 seeds. Selecting both backbones gives 180 runs. The
[resampling-strength sweep](fluorescence_imaging_bbbc048/resampling_strength_sweep/README.md)
covers BYOL, VICReg and SimCLR.

BBBC048 has no paired clean test images. By default, N2N, NSP and VST-only
preprocess both probe-training and probe-test inputs. `--probe-inputs raw`
uses the acquired images directly across all arms.

</details>

<details>
<summary><strong>Fetal ultrasound — patient splits and anatomical metrics</strong></summary>

Retrieve the prepared archive with Git LFS first, as described below.

```bash
uv run --no-sync python -m fetal_ultrasound.dataset_archive extract \
  --archive fetal_ultrasound/dataset/fetal_planes_native192_v1.zip --output data/fetal
uv run --no-sync python -m fetal_ultrasound.train \
  --method simclr --treatment seana --cache-estimates \
  --manifest data/fetal/manifest.json --output-dir runs/fetal_example
uv run --no-sync python -m fetal_ultrasound.evaluate --run runs/fetal_example --device cuda
```

See [fetal_ultrasound/configs/](fetal_ultrasound/configs/) for backbone recipes,
patient-validation splits and resampling-strength selection.

</details>

## Prepared datasets

Both real-imaging datasets are included as **single-file archives** with
preparation metadata, split information and licensing notices.

| Dataset | Prepared archive | Size | Dataset license |
| :--- | :--- | ---: | :--- |
| **BBBC048** | [bbbc048_merged_v1.zip](fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip) | 54.3 MB | [CC BY-NC-SA 3.0](fluorescence_imaging_bbbc048/dataset/LICENSE.txt) |
| **FETAL_PLANES_DB** | [fetal_planes_native192_v1.zip](fetal_ultrasound/dataset/fetal_planes_native192_v1.zip) | 251.9 MB | [CC BY 4.0](fetal_ultrasound/dataset/LICENSE.txt) |

The fetal archive is stored with [Git LFS](https://git-lfs.com/). After installing
Git LFS and cloning the repository, retrieve it with:

```bash
git lfs pull --include="fetal_ultrasound/dataset/fetal_planes_native192_v1.zip"
```

## Citation

Code for **Breaking Noise Shortcuts in Self-Supervised Learning via Noise-Aligned
View Generation**, by **Oankar R. Patil**.

[Read the paper](https://neurips.cc/virtual/2026/loc/sydney/poster/153415) ·
[Machine-readable citation](CITATION.cff)

```bibtex
@misc{patil2026seana,
  title  = {Breaking Noise Shortcuts in Self-Supervised Learning via Noise-Aligned View Generation},
  author = {Patil, Oankar R.},
  year   = {2026},
  url    = {https://neurips.cc/virtual/2026/loc/sydney/poster/153415}
}
```

## Licensing

Original SEANA code is [MIT licensed](LICENSE). LeJEPA-derived code retains
[CC BY-NC 4.0](third_party/lejepa-LICENSE.txt), and Neighbor2Neighbor-derived code
retains [BSD-3-Clause](third_party/neighbor2neighbor-LICENSE.txt).
Dataset licenses are listed above. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution, component scope and the unresolved NSP redistribution terms.
