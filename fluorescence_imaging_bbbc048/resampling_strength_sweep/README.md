# BBBC048 resampling-strength sweep

This is the SEANA strength sweep for **BYOL, VICReg and SimCLR** in paper Figure 4
and Appendix. The ten strengths are `0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
1.5, 2.0`. They change the Gaussian noise resampled in learned VST space.
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
