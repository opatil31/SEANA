"""Label-free VST-space signal/residual statistics on training images only."""

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path


class Moments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, values):
        values = values.detach().double()
        count = values.numel()
        mean = float(values.mean())
        m2 = float(((values - mean) ** 2).sum())
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def std(self):
        if not self.count:
            raise ValueError("No pixels for VST calibration")
        return math.sqrt(max(0.0, self.m2 / self.count))


def measure(treatment, loader, device, max_batches=50):
    import torch

    if treatment.vst is None or treatment.denoiser is None:
        raise ValueError("VST calibration requires a fitted VST and its Gaussian denoiser")
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    treatment.eval().requires_grad_(False)
    signal, residual = Moments(), Moments()
    n_images = 0
    with torch.inference_mode():
        for i, (image, _) in enumerate(loader):
            if i >= max_batches:
                break
            t = treatment.vst.spline1(treatment.vst._normalize(image.to(device)))
            estimate = treatment.denoiser(t)
            signal.update(estimate)
            residual.update(t - estimate)
            n_images += len(image)
    signal_std, residual_std = signal.std(), residual.std()
    ratio = signal_std / residual_std if residual_std else None
    return dict(
        n_images=n_images,
        n_values=signal.count,
        population_std=True,
        signal_std=signal_std,
        residual_std=residual_std,
        sigma_vst_initial=max(signal_std, residual_std),
        signal_to_residual_std_ratio=ratio,
        ratio_db=20 * math.log10(ratio) if ratio is not None and ratio > 0 else None,
        space="fitted VST after its calibrated input normalization",
        formula="max(std(D(T(x))), std(T(x)-D(T(x))))",
        interpretation="denoiser-dependent proxy; not ground-truth acquisition SNR",
    )


def measure_run(run, device="cpu", max_batches=50):
    from cifar10.train import sha256_file
    from torch.utils.data import DataLoader

    run = Path(run)
    result = json.loads((run / "metrics.json").read_text())
    if (
        result.get("status") not in ("complete", "smoke_not_benchmark")
        or result["config"]["treatment"] != "seana"
    ):
        raise ValueError("Calibration requires a completed SEANA run")
    if result["experiment"] == "fetal_ultrasound":
        from .config import Config, load_config
        from .data import build_datasets
        from .views import load_treatment

        cfg = load_config(**{f.name: result["config"][f.name] for f in fields(Config)})
        splits, dataset = build_datasets(cfg)
        train = splits["train"]
    elif result["experiment"] == "fluorescence_imaging_bbbc048":
        from fluorescence_imaging_bbbc048.config import Config, load_config
        from fluorescence_imaging_bbbc048.data import build_datasets
        from fluorescence_imaging_bbbc048.views import load_treatment

        cfg = load_config(**{f.name: result["config"][f.name] for f in fields(Config)})
        train, _, dataset = build_datasets(cfg)
    else:
        raise ValueError("Expected a completed fetal or BBBC048 run")
    if (
        dataset != result["dataset"]
        or sha256_file(run / "treatment.pt") != result["artifacts"]["treatment.pt"]
    ):
        raise ValueError("Dataset or treatment differs from the saved run")
    treatment, _ = load_treatment(run / "treatment.pt", cfg, dataset, device)
    stats = measure(
        treatment, DataLoader(train, batch_size=cfg.fit_batch_size, shuffle=False), device, max_batches
    )
    return dict(
        **stats,
        dataset=dataset["dataset"],
        dataset_fingerprint=dataset["fingerprint"],
        training_split_sha256=dataset["split_sha256"]["train"],
        treatment_sha256=result["artifacts"]["treatment.pt"],
        status=result["status"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()
    result = measure_run(args.run, args.device, args.max_batches)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
