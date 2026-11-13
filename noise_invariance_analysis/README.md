# Noise-invariance analysis

Frozen-backbone measurements for the six synthetic-noise CIFAR-10 conditions.
For each clean test image, draw K independent corruptions, extract raw backbone
features, and decompose their variation by **image identity**, not class label.
There is no training, linear probing, projection head, test-time denoising or
SEANA augmentation in these measurements. An arm such as `nsp` or `seana`
describes how its backbone was trained.

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
`SS_total = SS_within + SS_between`, see the
[NIST one-way ANOVA definitions](https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm).

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
