# BBBC048 resampling-strength sweep

This is the SEANA strength sweep for **BYOL, VICReg and SimCLR** in paper Figure 4
and Appendix G. The ten strengths are `0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
1.5, 2.0`; they change the Gaussian noise resampled in learned VST space.
DRUNet's denoising sigma stays at **0.039**. All three methods use SEANA's learned
VST, 200 SSL epochs, and a 100-epoch frozen-backbone linear probe.

From the repository root, after extracting the bundled dataset:

```bash
python -m fluorescence_imaging_bbbc048.resampling_sweep \
  --manifest data/bbbc048/manifest.json --output runs/bbbc048_sigma_grid.jsonl
python -m fluorescence_imaging_bbbc048.grid run \
  --grid runs/bbbc048_sigma_grid.jsonl --index 0 --dry-run
python -m fluorescence_imaging_bbbc048.grid run \
  --grid runs/bbbc048_sigma_grid.jsonl --index 0 --device cuda
```

Run indices 0–29 to cover the full grid. The default uses seed 0 and ResNet-18,
matching the captured microscopy runner's architecture. The paper describes
ResNet-50; use `--backbones resnet50`, or `--backbones resnet18 resnet50` to
generate both. Add `--seeds 0 1 2` for training replicates. The partition remains
fixed at split seed 0 across strengths and training seeds so changing the
strength does not change the cells. BYOL retains its standard 0.99→1 EMA schedule.

The portable protocol follows the captured microscopy runner, including
crop-then-SEANA view ordering and acquired-image probe inputs. Estimators are
fitted exclusively on training cells. They can be reused through the normal
`--treatment-checkpoint` option in an individual training command; the loader
checks fitting settings and membership. The root SEANA implementation is shared.

The explicit grid is in [resampling_sweep.json](../configs/resampling_sweep.json)
and the fixed settings in
[resampling_sweep_protocol.json](../configs/resampling_sweep_protocol.json).
Use the standard BBBC048 report command to summarize newly completed runs:

```bash
python -m fluorescence_imaging_bbbc048.report runs \
  --runs runs/bbbc048 --output runs/bbbc048_sigma_summary.csv
```

[paper_reference.csv](paper_reference.csv) contains 30 accuracy values copied
programmatically from `GRID_BBBC` and `GT_BBBC` in the captured `sigma_semantic.py`.
That source explicitly describes them as the seed-0 paper Figure 4 / Appendix G
values. [provenance.json](provenance.json) records the source and CSV hashes.
The captured `generate_bbbc048_extra.py` independently confirms the ten-point grid.

The underlying per-run sweep CSVs were not recovered. These arrays do not identify
whether accuracy is the final head or the maximum over test epochs, and do not
establish all original settings. They are preserved as source-backed paper
references, separately from verified per-run captures and new-run statistics.
The new recipe has not been rerun as a full GPU benchmark.
