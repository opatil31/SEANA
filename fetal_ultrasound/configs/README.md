# Fetal ultrasound training configurations

The method-named JSON files contain flat training flags accepted by
`python -m fetal_ultrasound.train --config`. They start a new experiment with
SEANA, ResNet-18 and seed 0, using 200 SSL epochs and 100 probe epochs.

| SSL method | Configuration | SSL batch size | `sigma_vst` |
| --- | --- | ---: | ---: |
| SimCLR | [simclr.json](simclr.json) | 128 | 0.16 |
| VICReg | [vicreg.json](vicreg.json) | 128 | 0.16 |
| BYOL | [byol.json](byol.json) | 128 | 0.12 |
| MoCoV2 | [mocov2.json](mocov2.json) | 128 | 0.04 |
| LeJEPA | [lejepa.json](lejepa.json) | 256 | 0.24 |

The recipes make the current training settings explicit, including augmentation,
projection heads, optimizer, probing, SEANA fitting and NSP/N2N fitting flags.
LeJEPA uses cosine scheduling with 10 warmup epochs. VICReg uses a 2048-dimensional
projection. All five use frozen DRUNet with `sigma_d=0.08` for SEANA.

## Prepare data and run

From the repository root, after installing the research dependencies and Git LFS:

```bash
git lfs pull --include="fetal_ultrasound/dataset/fetal_planes_native192_v1.zip"
uv run --no-sync python -m fetal_ultrasound.dataset_archive extract \
  --archive fetal_ultrasound/dataset/fetal_planes_native192_v1.zip \
  --output data/fetal
uv run --no-sync python -m fetal_ultrasound.train \
  --config fetal_ultrasound/configs/simclr.json \
  --output-dir runs/fetal_simclr_seana
```

The `manifest` flag points to the dataset index produced by extraction. It
contains image locations, labels and patient/split metadata. The configuration
files contain no previous-run identifiers, results or checkpoint paths. DRUNet
weights are downloaded on first use, and the treatment is fitted on training data.

## Change the experiment

Choose the JSON file for your SSL method. Command-line flags override its values:

- `--backbone resnet50` selects ResNet-50.
- `--treatment baseline --no-cache-estimates` selects the baseline.
- `--treatment nsp` or `--treatment n2n` selects a dataset-specific denoiser.
- `--seed 1` or `--seed 2` selects another training seed.
- `--manifest /path/to/data/manifest.json` uses a different dataset extraction.
- `--dry-run` prints the resolved flags without loading data or training.

The recipes use the official patient-disjoint training/test partition, a native
192-pixel region, 128-pixel SSL views, and shared cached estimates before spatial
augmentation. Color jitter and Gaussian blur are disabled. The final probe is
evaluated without selecting an epoch on test results. The `probe_inputs=historical`
value means NSP and N2N preprocess acquired probe-training and probe-test images.
Baseline and SEANA probes use acquired images directly.

For a separate patient-disjoint validation experiment, add:

```bash
--split-mode official_validation --validation-fraction 0.1 \
--probe-selection validation_macro_f1 --no-evaluate-test
```

For the training-data sigma rule, add `--sigma-mode unsupervised`; its controls
are `--sigma-multiplier` and `--sigma-rule-max-batches`. The smaller
`resnet18.json`, `resnet50.json`, `validation.json` and `unsupervised.json` files
remain available as generic presets.
