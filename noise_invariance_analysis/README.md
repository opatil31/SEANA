# Noise-invariance analysis

Frozen-backbone measurements for the six synthetic-noise CIFAR-10 conditions.
For each clean test image, draw K independent corruptions, extract raw backbone
features, and decompose their variation by **image identity**, not class label.
There is no training, linear probing, projection head, test-time denoising or
SEANA augmentation in these measurements. An arm such as `nsp` or `seana`
describes how its backbone was trained.

The implementation consolidates the author's `probe_noise_encoding.py` and
supports the release's five SSL models and both ResNet backbones. Captured
results concern BYOL, SimCLR and VICReg with the original ResNet-18 runner;
new model/backbone combinations have no captured analysis scores.

## Captured Tables 3 and 4

The three original CSVs are retained byte-for-byte under `captured_results/raw/`,
with hashes in the [manifest](captured_results/manifest.json). The primary file
contains **211 checkpoint measurements**. The Gaussian-high file repeats 21 of
those rows and is not counted twice. The sigma sweep contains 18 additional
measurements. [Source provenance](source_provenance.json) identifies the captured
script and accompanying writeup by hash.

- [Table 3: baseline versus SEANA](captured_results/tables/table3_captured.csv)
- [Table 4: Gaussian-high decomposition](captured_results/tables/table4_captured.csv)
- [Table 3 with matched model/seed membership](captured_results/tables/table3_matched.csv)
- [Table 4 with matched model/seed membership](captured_results/tables/table4_matched.csv)
- [All primary checkpoint measurements](captured_results/tables/per_checkpoint_captured.csv)
- [Captured sigma sweep](captured_results/tables/sigma_sweep_captured.csv)

The first two tables reproduce the pasted rebuttal values when rounded to its
precision, using **accurate metric names**. They average per-checkpoint metrics;
the mean fraction is not recomputed from mean S_W and mean S_B.

The captured averages do not always use matching seeds. For PG-high, Table 3's
baseline averages six checkpoints and SEANA three. In Table 4, baseline/SEANA
each average six checkpoints, while NSP averages three. The matched tables use
the intersection of `(SSL method, training seed, epoch suffix)` across the
compared arms. Both versions record counts and the exact members. These are
descriptive averages across methods/seeds, not confidence intervals or evidence
that every other training setting was matched. Historical checkpoint hashes and
complete launch arguments were not recorded in these CSVs.
The CSV `seed` identifies the training seed; the analysis seed and selected test
indices were not recorded. The captured source defaults to analysis seed 0,
but that does not establish the settings of every historical run.

Regenerate the captured tables without PyTorch or checkpoint files:

```bash
python -m noise_invariance_analysis.report captured \
  --output-dir runs/noise_invariance_captured_tables
```

The raw `oracle`, `oracle_postcrop`, `analyticalvst` and resampling labels remain
as recorded. Their presence in the CSV does not establish their exact training
protocols, and they are not pooled into the baseline/NSP/SEANA comparisons.

## Metric definitions and corrections

Let `f[i,k]` be the D-dimensional backbone embedding for image i under noise
draw k, `m[i] = mean_k f[i,k]`, and `m = mean_i m[i]`. For N images and K draws:

```text
S_W = (1/(N*K)) sum_i sum_k ||f[i,k] - m[i]||²
S_B = (1/N)     sum_i       ||m[i] - m||²
noise_variance_fraction = S_W / (S_W + S_B) = 1 - eta_squared
```

This exactly describes the captured `ndv` column. It is the complement of the
ANOVA explained-variation fraction, **not the conventional one-way ICC(1,1)**.
The identity follows from `SS_within = N*K*S_W`, `SS_between = N*K*S_B` and
`SS_total = SS_within + SS_between`; see the
[NIST one-way ANOVA definitions](https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm).
Squared Euclidean distances sum variation across feature coordinates. Features
are not L2-normalized for this decomposition.

For comparison, new runs also report:

```text
MS_between = N*K*S_B / (N-1)
MS_within  = K*S_W / (K-1)
icc_1_1_trace = (MS_between - MS_within) /
               (MS_between + (K-1)*MS_within)
```

This applies the conventional one-way variance-component estimator to the
**trace-pooled** sums of squares of the vector embeddings. It is not the
arithmetic mean of per-coordinate ICCs. The underlying scalar formula and
variance components are given in
[Atenafu et al., Eq. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC3554464/).
`one_minus_icc_1_1_trace` is separately named and can exceed one if the ICC estimate
is negative; estimates are not clipped. The per-checkpoint captured report can
derive this coefficient from the recorded S_W, S_B, N and K. It never substitutes
it for the captured `ndv` values in Tables 3 or 4.

At finite K, S_B is the variation of noisy sample means; it is not a pure latent
content variance estimate. The variance-component quantities are `MS_within`
and `(MS_between - MS_within)/K`. Their ratio estimate is itself not guaranteed
to be unbiased. No ANOVA significance test or multivariate p-value is computed.

Two different dimensionality measures are explicit:

```text
C = covariance across the N per-image means m[i]
p[j] = eigenvalue[j] / sum(eigenvalues)
effective_rank_entropy = exp(-sum_j p[j] log p[j])
participation_ratio    = 1 / sum_j p[j]²
```

The captured `eff_rank` is **entropy effective rank**, not participation ratio.
The latter cannot be reconstructed from the CSV's scalar rank, so its captured
cells remain blank. New analyses report both on the covariance of per-image
means. Completely constant embeddings yield rank zero and undefined variance
fractions/ICC (`null`), avoiding a false claim of perfect invariance.

## Retrieval measurements

**Content re-ID:** take M clean images, draw independent noise for query and
gallery, embed both, then find each query's highest-cosine-similarity gallery
item. The correct item has the same clean-image identity. The captured gallery
has **M=500**, with chance **0.002**, as its CSV records; it is not a 2,000-item
gallery. Cosine normalization is used only for retrieval.

**Noise re-ID:** use disjoint clean images for query and gallery while sharing
the Gaussian epsilon tensor for each paired item. This diagnostic is available
for Gaussian and speckle noise. It is omitted (`null`) for Poisson–Gaussian
noise because the implemented shared-epsilon construction does not share the
Poisson component. New runs also report one shuffled-identity control for each
retrieval task, with a separate RNG; this is not a p-value or uncertainty estimate.
The old script mentioned permutation controls but did not compute them.

Both tasks use `M = min(retrieval_images, floor(N/2))`, matching the captured
script. Similarity computation is chunked to bound memory; ties use the first
gallery index. `--prepool` retains the tensor before global average pooling,
but these ResNets already produce 1x1 final spatial maps for 32x32 CIFAR inputs.

## Analyze a release checkpoint

Use the root requirements and prepare CIFAR-10 as described in the main README.
The backbone may come from the main CIFAR recipes or the CIFAR isolation folder.

```bash
python -m noise_invariance_analysis.analyze \
  --run runs/cifar10/example \
  --config noise_invariance_analysis/configs/cifar10.json \
  --data-root data/cifar10 --device cuda \
  --output-dir runs/noise_invariance_analysis/example
```

`--run` loads `backbone.pt`. Alternatively, `--checkpoint` accepts a release
`backbone.pt` or `ssl.pt`. Only online `backbone.*` weights are extracted from
full SSL checkpoints; momentum encoders and heads are excluded. Checkpoint
loading is strict, uses `weights_only=True`, and checks CIFAR dataset identity.
ImageNet-100 checkpoints are rejected because this module measures CIFAR-10.

Defaults are N=2,000, K=8, requested M=500, analysis seed 0, inference batch size
512 and pooled features. Override `--n-images`, `--k-reps`, `--retrieval-images`,
`--seed`, `--batch-size`, `--retrieval-chunk-size` or `--prepool`. `--download`
permits an initial CIFAR download; it is off by default. No spatial/color
augmentation is applied during measurement.

Images use the captured NumPy permutation selection. Corruptions are generated
on CPU with a private seeded generator in the captured draw order, independently
of inference batching and model initialization. The same settings yield the
same input corruptions across backbones; changing N or K changes the protocol.
All models are in evaluation mode, all parameters are frozen and inference is
gradient-free. Variance and covariance calculations now use FP64 and the smaller
equivalent Gram matrix when useful; historical calculations used FP32 for some
intermediates. Last-digit agreement across implementations/devices is not promised.

Each output contains `config.json` and `analysis.json`, including checkpoint
hash, training metadata when available, selected image indices, clean-image and
ANOVA-corruption hashes, gallery size, metric definitions/version, source hashes
and package versions. `--save-features` additionally writes the `[N,K,D]` tensor
to `embeddings.pt` for auditing or recomputing metrics. Nonempty output directories
are protected. There is no optimizer, backbone fine-tuning or linear-probe fit.

## Analyze older checkpoints

The captured script read full SSL state dictionaries with `backbone.*` prefixes.
Both that format and plain backbone state dictionaries are supported with explicit
metadata, because old weights do not identify their architecture or noise regime:

```bash
python -m noise_invariance_analysis.analyze \
  --checkpoint weights/byol_gauss_high_baseline_s0.pt \
  --legacy-backbone resnet18 --method byol --condition gauss_high \
  --arm baseline --training-seed 0 --device cuda \
  --output-dir runs/noise_invariance_analysis/legacy_baseline
```

Training seed and analysis seed are distinct. Metadata disagreements with a
release-format checkpoint are errors. The analysis never loads a denoiser,
even when a legacy checkpoint is labeled `nsp` or `n2n`.

## Batch analysis and new reports

```bash
python -m noise_invariance_analysis.grid discover \
  --runs runs/cifar10 --output runs/noise_invariance_grid.jsonl
python -m noise_invariance_analysis.grid run \
  --grid runs/noise_invariance_grid.jsonl --index 0 --dry-run
python -m noise_invariance_analysis.grid run \
  --grid runs/noise_invariance_grid.jsonl --index 0 --device cuda
python -m noise_invariance_analysis.report runs \
  --runs runs/noise_invariance_analysis --output-dir runs/noise_invariance_new_tables
```

Discovery selects completed CIFAR release runs and verifies their backbone
hashes. `--include-smoke` is an explicit development option. For legacy filenames:

```bash
python -m noise_invariance_analysis.grid legacy \
  --models 'weights/*_gauss_high_*_s0.pt' 'weights/*_pg_high_*_s0.pt' \
  --backbone resnet18 --output runs/noise_invariance_legacy_grid.jsonl
```

The legacy importer parses the captured method/condition/arm/seed convention,
including sigma-sweep labels such as `sv0p185`; unknown names fail rather than
being silently skipped. Manifests use paths relative to the manifest and pin
checkpoint bytes by SHA-256. Rows are zero-based; no jobs are submitted.

New reports retain one row per checkpoint/evaluation protocol and reject duplicate
analyses. They do not automatically average different training recipes, models,
seeds or sample sets. Captured and newly computed outputs remain separate.
Fake-data or smoke-trained checkpoint results are excluded from benchmark reports.

## Interpretation and validation

For the camera-ready text, label the primary column **noise-driven variance
fraction (1−η²)**, and the captured rank column **entropy effective rank**.
This keeps the quoted numerical results intact while matching what was computed.

Lower noise fraction means less within-image movement relative to total variation
under this corruption protocol. Report it with S_W, S_B, rank and retrieval:
a collapsed representation can have a misleading ratio. Absolute S_W/S_B depend
on feature scale, so their magnitudes alone do not establish content loss across
separately trained networks. The joint rank/retrieval observations are informative
descriptions, but they do not by themselves isolate a causal training mechanism
or prove memorization of a noise field. The saved scalars do not provide uncertainty
over image selection/noise draws; stronger claims need repeated analyses or saved
embeddings and a stated uncertainty procedure.

Tests cover analytic ANOVA/ICC examples, rank definitions, collapse, retrieval,
checkpoint loading, frozen BatchNorm/weights, input-noise reproducibility, and
exact regeneration of the captured tables. For a small software check:

```bash
python -m noise_invariance_analysis.analyze --run runs/cifar10/example \
  --smoke --device cpu --output-dir runs/noise_invariance_smoke
```

Smoke mode uses eight fake images, K=2 and M=4. It does not train a replacement
backbone or reproduce scientific results. Full analysis on the historical
checkpoints has not been rerun during release preparation.
