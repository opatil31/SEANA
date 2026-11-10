"""Compare clean estimates on identical noisy CIFAR-10 test images (Table 1)."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import time

from cifar10.config import CONDITIONS, load_config
from cifar10.train import seed_all, write_json, sha256_file, runtime_metadata

HIGH_CONDITIONS = ("gauss_high", "pg_high", "speckle_high")
ESTIMATORS = ("noisy", "n2n", "nsp", "drunet_raw", "drunet_vst")


def evaluate_estimator(fn, clean, noisy, device, batch_size=200):
    """PSNR from pooled pixel MSE, not the mean of per-image dB values."""
    import torch

    if clean.shape != noisy.shape or clean.ndim != 4 or clean.numel() == 0:
        raise ValueError("Expected nonempty matching BCHW clean/noisy tensors")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sse = count = 0
    with torch.inference_mode():
        for start in range(0, len(clean), batch_size):
            target = clean[start : start + batch_size].cpu()
            prediction = fn(noisy[start : start + batch_size].to(device)).clamp(0, 1).cpu()
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Estimator returned invalid shape or nonfinite pixels")
            sse += float((prediction.double() - target.double()).square().sum())
            count += target.numel()
    mse = sse / count
    return {"mse": mse, "psnr_db": -10.0 * math.log10(max(mse, 1e-12)), "pixels": count}


def run_condition(cfg, output_dir, *, eval_images=2000, eval_batch_size=200, checkpoints=None):
    import torch
    from torchvision import transforms as T
    from cifar10.data import build_datasets, make_loader, SyntheticNoise
    from cifar10.views import ViewTreatment
    from .views import load_fitted_treatment

    cfg.validate()
    if cfg.method != "byol":
        raise ValueError(
            "This PSNR recipe fits NSP on BYOL-augmented views; official N2N uses raw noisy images"
        )
    if eval_images < 1 or eval_batch_size < 1:
        raise ValueError("Evaluation sizes must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use CPU or CUDA")
    if cfg.smoke:
        torch.set_num_threads(2)
    checkpoints = checkpoints or {}
    if set(checkpoints) - {"seana", "n2n", "nsp"}:
        raise ValueError("Checkpoint keys must be seana, n2n or nsp")
    seed_all(cfg.seed)
    datasets = build_datasets(cfg)
    test = datasets["probe_test"].base
    if eval_images > len(test):
        raise ValueError(f"eval_images={eval_images} exceeds test size={len(test)}")
    clean = torch.stack([T.ToTensor()(test[index][0]) for index in range(eval_images)])
    # Draw once on CPU with a private generator. Every estimator and every
    # evaluation batch size receives exactly these same noisy pixels.
    noisy = SyntheticNoise(cfg.condition, "fixed", cfg.seed)(clean, idx=0)
    import hashlib

    evaluation = {
        "split": "test",
        "first_index": 0,
        "images": eval_images,
        "batch_size": eval_batch_size,
        "data_range": 1.0,
        "aggregation": "PSNR_from_pooled_pixel_MSE",
        "mse_floor": 1e-12,
        "noise_draw_device": "cpu",
        "noise_seed": cfg.seed,
        "noisy_tensor_sha256": hashlib.sha256(noisy.numpy().tobytes()).hexdigest(),
    }
    write_json(output / "config.json", {"fit": cfg.resolved(), "evaluation": evaluation})
    scores = {"noisy": evaluate_estimator(lambda x: x, clean, noisy, device, eval_batch_size)}
    fit_info = {}
    for name in ("seana", "n2n", "nsp"):
        seed_all(cfg.seed)
        fit_cfg = replace(cfg, treatment=name)
        treatment = ViewTreatment(fit_cfg, device, initialize_from_checkpoint=name in checkpoints)
        if name in checkpoints:
            load_fitted_treatment(treatment, checkpoints[name], fit_cfg)
        else:
            ssl = make_loader(datasets["ssl"], cfg, shuffle=True, drop_last=True)
            raw = make_loader(datasets["fit"], cfg, shuffle=True, drop_last=True, seed_offset=1)
            treatment.fit(ssl, raw, device)
        checkpoint_path = output / f"{name}_treatment.pt"
        torch.save(
            {
                "format_version": 1,
                "kind": "view_treatment",
                "config": fit_cfg.resolved(),
                "state_dict": treatment.state_dict(),
            },
            checkpoint_path,
        )
        fit_info[name] = {
            "config": fit_cfg.resolved(),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        if name in checkpoints:
            fit_info[name]["input_checkpoint_sha256"] = treatment.input_checkpoint_sha256
        if name == "seana":
            scores["drunet_raw"] = evaluate_estimator(
                treatment.denoiser, clean, noisy, device, eval_batch_size
            )
            scores["drunet_vst"] = evaluate_estimator(
                lambda x, fitted=treatment: fitted.vst(x, fitted.denoiser),
                clean,
                noisy,
                device,
                eval_batch_size,
            )
        else:
            scores[name] = evaluate_estimator(treatment.denoiser, clean, noisy, device, eval_batch_size)
        del treatment
    result = {
        "schema_version": 1,
        "experiment": "denoiser_oracle_isolation_psnr",
        "status": "smoke_not_benchmark" if cfg.smoke else "complete",
        "config": cfg.resolved(),
        "evaluation": evaluation,
        "estimators": scores,
        "fits": fit_info,
        "seconds": time.time() - started,
        "runtime": runtime_metadata(("denoiser_oracle_isolation_experiments",)),
    }
    write_json(output / "psnr_metrics.json", result)
    with (output / "psnr.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["condition", "seed", "estimator", "implementation", "psnr_db", "mse"]
        )
        writer.writeheader()
        for name in ESTIMATORS:
            writer.writerow(
                {
                    "condition": cfg.condition,
                    "seed": cfg.seed,
                    "estimator": name,
                    "implementation": (
                        fit_info["n2n"]["config"]["n2n_implementation"] if name == "n2n" else ""
                    ),
                    "psnr_db": scores[name]["psnr_db"],
                    "mse": scores[name]["mse"],
                }
            )
    print(cfg.condition + ": " + ", ".join(f"{name}={scores[name]['psnr_db']:.2f} dB" for name in ESTIMATORS))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=HIGH_CONDITIONS)
    p.add_argument("--data-root", default="data/cifar10")
    p.add_argument("--output-root", type=Path, default=Path("runs/denoiser_oracle_psnr"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-images", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--download", action="store_true")
    p.add_argument("--drunet-weights", default="download")
    p.add_argument("--noise-sampling", choices=("stochastic", "fixed"), default="stochastic")
    for name in ("n2n", "nsp", "seana"):
        p.add_argument(f"--{name}-fit-steps", type=int)
        p.add_argument(
            f"--{name}-checkpoint", type=Path, help="Compatible treatment checkpoint (one condition only)"
        )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    checkpoints = {
        name: getattr(args, name + "_checkpoint")
        for name in ("n2n", "nsp", "seana")
        if getattr(args, name + "_checkpoint")
    }
    if checkpoints and len(args.conditions) != 1:
        p.error("Checkpoint reuse requires exactly one --conditions value")
    for condition in args.conditions:
        values = dict(
            method="byol",
            condition=condition,
            seed=args.seed,
            data_root=args.data_root,
            device=args.device,
            download=args.download,
            num_workers=args.num_workers,
            drunet_weights=args.drunet_weights,
            noise_sampling=args.noise_sampling,
        )
        values.update(
            {name + "_fit_steps": getattr(args, name + "_fit_steps") for name in ("n2n", "nsp", "seana")}
        )
        if args.smoke:
            values.update(
                smoke=True,
                batch_size=4,
                num_workers=0,
                seana_fit_steps=1,
                nsp_fit_steps=1,
                n2n_fit_steps=1,
                n2n_base_ch=8,
                nsp_base_ch=8,
                nsp_num_module=0,
                denoiser_patch_batch_size=2,
                seana_patch_batch_size=2,
            )
        cfg = load_config(**values)
        if args.dry_run:
            print(
                json.dumps(
                    {"fit": cfg.resolved(), "eval_images": 8 if args.smoke else args.eval_images}, indent=2
                )
            )
        else:
            run_condition(
                cfg,
                args.output_root / f"{condition}_s{cfg.seed}",
                eval_images=8 if args.smoke else args.eval_images,
                eval_batch_size=args.eval_batch_size,
                checkpoints=checkpoints,
            )


if __name__ == "__main__":
    main()
