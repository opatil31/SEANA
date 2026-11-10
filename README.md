# SEANA: noise-aligned views for self-supervised learning

Self-Aligned Noise Augmentation (SEANA) is a noise-aware view generation module that is plug-and-play for joint-embedding / latent Self-Supervised Learning (SSL) pipelines.

**Use SEANA in your own pipeline: [Python library guide](docs/library.md)**

Published release: [seana-ssl 0.1.3 on PyPI](https://pypi.org/project/seana-ssl/0.1.3/).

Install into a uv project, including pretrained DRUNet support:

```bash
uv add seana-ssl
```

You can also use `python -m pip install seana-ssl` with pip.


```python
from lightly.transforms import SimCLRViewTransform
from seana import SEANA

device = "cuda" if torch.cuda.is_available() else "cpu"
augmenter_seana = SEANA(
    sigma_d=0.08,       # Gaussian denoiser's noise-level input
    sigma_vst=0.1,      # Resampling standard deviation in VST space
    channels=3,
    device=device,
)

# train_loader yields raw floating [B,3,H,W] images, optionally followed by labels.
# Fit on training images only, before ImageNet/encoder mean-std normalization.
augmenter_seana.fit(raw_train_loader, steps=5000, patch_size=64, seed=0)

ssl_view = SimCLRViewTransform(input_size=224)
for images, _ in raw_train_loader:
    source_a, source_b = augmenter_seana(images.to(device))
    view_a = torch.stack([ssl_view(image) for image in source_a])
    view_b = torch.stack([ssl_view(image) for image in source_b])
    loss = ssl_loss(encoder(view_a), encoder(view_b))
    loss.backward()
```

Generate the two SEANA sources first, then apply your usual SSL transforms
independently to each source. The guide covers LightlySSL integration as seen above, custom denoisers, resampling strengths, checkpoint loading and a complete SSL training example. The package installs PyTorch and DeepInverse. `SEANA()` uses pretrained DRUNet, while `SEANA(my_denoiser)` uses your own denoiser.

Run the [LightlySSL SimCLR + SEANA example](examples/ssl_pipeline.py), which uses
a ResNet-18 backbone, Lightly projection head and NT-Xent loss:

```bash
uv run --python 3.12 --extra experiments examples/ssl_pipeline.py
```

The distribution is named `seana-ssl`, imported as `seana`. Its core remains
in [seana.py](seana.py) at the repository root. See the guide for local-checkout
installation and integration examples.

**Try SEANA interactively: [BBBC048 demo notebook](SEANA_BBBC048_Demo.ipynb)**

[![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open Colab, choose **File → Upload notebook**, and upload the root notebook.
It includes a resampling-strength slider, real fluorescence images, captured
results, and a live baseline-versus-SEANA SSL comparison using `seana-ssl` from PyPI.
Both arms default to **50 SSL epochs and 20 linear-probe epochs**, which is less than main paper btw.

Code for **Breaking Noise Shortcuts in Self-Supervised Learning via Noise-Aligned
View Generation**, by Oankar R. Patil.


The method implementation is [seana.py](seana.py) at the repository root. This
release covers VICReg, SimCLR, BYOL, MoCoV2 and LeJEPA on noisy CIFAR-10 and ImageNet-100 with
ResNet-18 and ResNet-50. Each model supports four treatments: ordinary noisy
views (`baseline`), SEANA (`seana`), Next-Scale Prediction (`nsp`) SSL denoising, and Neighbor2Neighbor (`n2n`) SSL denoising, integrated with the project's dataset recipes.

The separate [BBBC048 fluorescence-imaging experiments](fluorescence_imaging_bbbc048/README.md)
add the same five SSL models, NSP/N2N, VST-only and resampling-only component
ablations.

The [fetal ultrasound experiments](fetal_ultrasound/README.md) add the five SSL
models on FETAL_PLANES_DB, baseline/NSP/SEANA comparisons, anatomical and device
metrics, and a separate patient-validation strength sweep. 

**Both real-imaging
datasets include a single-file prepared dataset in their experiment folder.**

**Synthetic-benchmark evaluation protocol:** pretrain the backbone on noisy images, freeze it, fit a
linear classifier using noisy training images, then evaluate on **clean test
images** (the validation split for ImageNet-100). For **N2N and NSP**, the fitted,
frozen denoiser preprocesses the noisy probe-training images. Clean test images
go directly to the frozen backbone, with the usual normalization and no denoising.
Baseline and SEANA use noisy probe-training images directly. No probe uses the
SSL projection head or noise resampling.

## Research installation

To run the experiments from the full checkout with uv:

```bash
uv sync --python 3.12 --extra experiments --extra imagenet100 --extra bbbc048 \
  --extra demo --group dev --group notebooks
uv run --no-sync python -m cifar10.train --dry-run
```

The library wheel contains the reusable SEANA implementation. Experiment scripts,
captured results and prepared datasets are available in the full repository.
The existing requirements-file workflow is also supported:

Use Python 3.12 for the research environment. Install a matching
PyTorch 2.5.1 / torchvision 0.20.1 build for your CPU or CUDA environment using
the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
Then, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you install a CUDA-specific PyTorch build, install it **inside this environment**
before the remaining requirements. The core `seana.py` needs only PyTorch,
DeepInverse is loaded only by a SEANA treatment.

## Run a CIFAR-10 experiment

```bash
python -m cifar10.train \
  --config configs/cifar10/resnet18.json \
  --method vicreg --condition pg_high --treatment seana \
  --seed 0 --download --output-dir runs/example
```

Use `configs/cifar10/resnet50.json` for ResNet-50. Method choices are `vicreg`,
`simclr`, `byol`, `mocov2`, and `lejepa`. Treatment choices are `baseline`,
`seana`, `nsp`, and `n2n`. CLI values override the JSON recipe.

| Condition | Corruption on images in [0, 1] |
| --- | --- |
| `gauss_low` | Gaussian, sigma = 0.1 |
| `gauss_high` | Gaussian, sigma = 0.5 |
| `pg_low` | Poisson peak = 30, Gaussian sigma = 0.01 |
| `pg_high` | Poisson peak = 5, Gaussian sigma = 0.2 |
| `speckle_low` | Multiplicative Gaussian, sigma = 0.2 |
| `speckle_high` | Multiplicative Gaussian, sigma = 0.6 |

All corruptions are clipped to [0, 1]. Recipes default to stochastic corruption,
matching the corrected captured launchers. Use `--noise-sampling fixed` for
per-image corruption held fixed across epochs.

SEANA first fits the VST on noisy training images, using a frozen pretrained
DRUNet. DeepInverse downloads its weights on first use. For an offline node,
pass `--drunet-weights /path/to/drunet_color.pth` with a compatible local
checkpoint. NSP fits on noisy SSL views. N2N fits on raw noisy training images
using the official neighbor-pair objective.

Preview all settings without downloading data or importing training libraries:

```bash
python -m cifar10.train --method mocov2 --backbone resnet50 --dry-run
```

## Full experiment grid

Prepare data once, then generate the matrix. The default is 5 methods x 2
backbones x 6 noise conditions x 4 treatments x 3 seeds = **720 runs**.

```bash
python -m scripts.prepare_data --data-root data/cifar10
python -m scripts.make_grid --output runs/grid.jsonl
python -m scripts.run_grid --grid runs/grid.jsonl --index 0 --dry-run
python -m scripts.run_grid --grid runs/grid.jsonl --index 0
```

Grid indices are zero-based. Run one row per process/device. For a Slurm array,
pass its task ID as `--index`. Supply the account, partition and resource
requests for your own cluster. The generator accepts `--methods`, `--backbones`,
`--conditions`, `--treatments` and `--seeds` to select a smaller matrix.

## Denoiser/oracle isolation

The separate [denoiser_oracle_isolation_experiments](denoiser_oracle_isolation_experiments/README.md)
folder adds PSNR comparisons, denoise-plus-known-noise-resampling arms, oracle augmentation, and a VST+DRUNet control that holds the clean estimate
fixed while changing resampling.

## ImageNet-100

The [ImageNet-100 guide](imagenet100/README.md) covers offline dataset preparation,
both backbones, all five SSL models and the four main treatments.

```bash
python -m pip install -r requirements-imagenet100.txt
python -m imagenet100.prepare --output data/imagenet100
python -m imagenet100.train --config configs/imagenet100/resnet50.json \
  --method byol --condition shot_high --treatment seana \
  --output-dir runs/imagenet100_example
python -m imagenet100.evaluate --run runs/imagenet100_example --device cuda
```

Use `configs/imagenet100/resnet18.json` for ResNet-18, or
`python -m imagenet100.grid generate` for the 720-run matrix.
## Noise-invariance analysis

The [noise-invariance analysis](noise_invariance_analysis/README.md) folder adds
frozen-backbone CIFAR measurements.

## BBBC048 fluorescence imaging

The [BBBC048 guide](fluorescence_imaging_bbbc048/README.md) covers dataset indexing,
cell-level splits, all five SSL models and six treatment arms, view-order controls,
and the ten-strength SEANA sweep for BYOL, VICReg and SimCLR.
The complete prepared dataset is a single repository file,
[bbbc048_merged_v1.zip](fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip)
(54.3 MB), containing original images, labels and explicit split assignments.

```bash
python -m pip install -r requirements-bbbc048.txt
python -m fluorescence_imaging_bbbc048.dataset_archive extract \
  --archive fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip \
  --output data/bbbc048
python -m fluorescence_imaging_bbbc048.train --method vicreg --treatment seana \
  --manifest data/bbbc048/manifest.json --output-dir runs/bbbc048_example
```

The default 90-run grid covers ResNet-18/ResNet-50, all six arms and three seeds. BBBC048 has no paired clean test images.

By default, NSP/N2N/VST-only denoise probe inputs at both training and test time.

`--probe-inputs raw` makes probe inputs
uniform across all arms.

## Fetal ultrasound

The prepared dataset file used can be found at:
[one dataset file](fetal_ultrasound/dataset/fetal_planes_native192_v1.zip)
(251.9 MB, stored with Git LFS). Install Git LFS before staging this repository;
after cloning, use `git lfs pull` to retrieve the dataset.

```bash
python -m fetal_ultrasound.dataset_archive extract \
  --archive fetal_ultrasound/dataset/fetal_planes_native192_v1.zip --output data/fetal
python -m fetal_ultrasound.train --method simclr --treatment seana --cache-estimates \
  --manifest data/fetal/manifest.json --output-dir runs/fetal_example
python -m fetal_ultrasound.evaluate --run runs/fetal_example --device cuda
```

## Layout and licensing

```text
seana.py                 Core VST fitting and noise-aligned view generation
SEANA_BBBC048_Demo.ipynb  Interactive Google Colab demo and matched SSL comparison
cifar10/                 Data, SSL models, training, probing and evaluation
imagenet100/             ImageNet data preparation, models, views, runs and grids
denoisers/               Official-method N2N and project NSP implementation
configs/cifar10/         ResNet-18 and ResNet-50 recipes
configs/imagenet100/     ImageNet ResNet-18 and ResNet-50 recipes
scripts/                 Dataset preparation, grid execution and summaries
denoiser_oracle_isolation_experiments/  Rebuttal code and captured results
noise_invariance_analysis/  Frozen-backbone ANOVA, retrieval, ranks and captured tables
fluorescence_imaging_bbbc048/  Microscopy SSL, denoisers, component ablations and captures
fetal_ultrasound/         Ultrasound SSL, patient splits, anatomy/device metrics and captures
docs/                    Library and Colab guides
third_party/             Retained upstream license text
```

Original SEANA code is [MIT licensed](LICENSE). LeJEPA-derived code retains
CC BY-NC 4.0. **NSP redistribution terms must be resolved before publication**;
its inspected upstream repository had no license file. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The bundled BBBC048 dataset
retains CC BY-NC-SA 3.0, the prepared FETAL_PLANES_DB regions retain CC BY 4.0.
Both archives include attribution and license text.
