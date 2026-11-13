# BBBC048 training configurations

The method-named JSON files contain flat training flags accepted by
`python -m fluorescence_imaging_bbbc048.train --config`. They start a new
experiment with SEANA, ResNet-18 and seed 0, using 200 SSL epochs and 100 probe
epochs. These full experiment budgets are separate from the Colab demo.

| SSL method | Configuration | SSL batch size | `sigma_vst` |
| --- | --- | ---: | ---: |
| SimCLR | [simclr.json](simclr.json) | 128 | 0.8 |
| VICReg | [vicreg.json](vicreg.json) | 128 | 0.8 |
| BYOL | [byol.json](byol.json) | 128 | 0.8 |
| MoCoV2 | [mocov2.json](mocov2.json) | 128 | 0.6 |
| LeJEPA | [lejepa.json](lejepa.json) | 256 | 0.8 |

The recipes make the current training settings explicit, including augmentation,
projection heads, optimizer, probing, SEANA fitting and NSP/N2N fitting flags.
LeJEPA uses cosine scheduling with 10 warmup epochs. VICReg uses a 2048-dimensional
projection. All five use frozen DRUNet with `sigma_d=0.039` for SEANA.

## Prepare data and run

From the repository root, after installing the research dependencies:

```bash
uv run --no-sync python -m fluorescence_imaging_bbbc048.dataset_archive extract \
  --archive fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip \
  --output data/bbbc048
uv run --no-sync python -m fluorescence_imaging_bbbc048.train \
  --config fluorescence_imaging_bbbc048/configs/vicreg.json \
  --output-dir runs/bbbc048_vicreg_seana
```

DRUNet weights are downloaded on first use, and the treatment is fitted on the training split.

## Change the experiment

Choose the JSON file for your SSL method. Command-line flags override its values:

- `--backbone resnet50` selects ResNet-50.
- `--treatment baseline`, `--treatment nsp` or `--treatment n2n` selects a comparison.
- `--treatment vst_only` or `--treatment resample_only` selects a component ablation.
- `--seed 1` or `--seed 2` selects another seed.
- `--dry-run` prints the resolved flags without loading data or training.

The random split uses 80% of the cells for training. `split_seed=null` follows
the training seed, matching the existing experiment configuration. Set
`--split-seed 0` to hold the partition fixed while varying training seeds.

The default `view_order=paper` applies the experiment's spatial transforms before
SEANA. To cache estimates, select `--view-order shared --cache-estimates`. Baseline and SEANA probes use acquired images directly. BBBC048 has no paired clean test images.
