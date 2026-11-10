"""Measure a frozen CIFAR-10 backbone under repeated synthetic corruption; no training."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from cifar10.config import CONDITIONS, METHODS, BACKBONES
from cifar10.train import runtime_metadata, write_json
from .config import Config, load_config

EXPERIMENT = "noise_invariance_analysis"


def apply_noise(x, condition, generator, eps=None):
    """CPU noise, matching the captured draw order; shared eps only for Gaussian/speckle."""
    import torch

    if x.device.type != "cpu":
        raise ValueError("Evaluation corruption is generated on CPU for every backbone")
    spec = CONDITIONS[condition]
    if spec["family"] == "poisson_gaussian":
        if eps is not None:
            raise ValueError("Shared-noise re-ID is not defined here for Poisson-Gaussian noise")
        y = torch.poisson(x * spec["peak"], generator=generator) / spec["peak"]
        eps = torch.randn(x.shape, dtype=x.dtype, generator=generator)
        return (y + spec["sigma"] * eps).clamp(0, 1)
    if eps is None:
        eps = torch.randn(x.shape, dtype=x.dtype, generator=generator)
    if eps.shape != x.shape:
        raise ValueError("Shared noise tensor must match the images")
    return (x + spec["sigma"] * eps if spec["family"] == "gaussian" else x * (1 + spec["sigma"] * eps)).clamp(
        0, 1
    )


def embed(model, images, device, *, batch_size=512, prepool=False):
    import torch
    from cifar10.data import normalize_imagenet

    model.eval().requires_grad_(False)
    feature_model = torch.nn.Sequential(*list(model.children())[:-1]) if prepool else model
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = normalize_imagenet(images[start : start + batch_size].to(device))
            features.append(feature_model(x).flatten(1).float().cpu())
    if not features:
        raise ValueError("Cannot embed an empty image set")
    return torch.cat(features)


def load_images(cfg):
    import numpy as np
    import torch
    import torchvision
    from torchvision.transforms import ToTensor

    if cfg.smoke:
        dataset = torchvision.datasets.FakeData(
            size=max(8, cfg.n_images),
            image_size=(3, 32, 32),
            num_classes=10,
            random_offset=100000,
            transform=ToTensor(),
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            cfg.data_root, train=False, download=cfg.download, transform=ToTensor()
        )
    if cfg.n_images > len(dataset):
        raise ValueError(f"Requested {cfg.n_images} images, but split contains {len(dataset)}")
    # Matches the captured script's image selection, independent of torch's RNG.
    indices = np.random.RandomState(cfg.seed).permutation(len(dataset))[: cfg.n_images].tolist()
    images = torch.stack([dataset[i][0] for i in indices]).clamp(0, 1)
    identity = {
        "dataset": "fake_data" if cfg.smoke else "cifar10",
        "split": "test",
        "indices": indices,
        "n_images": len(indices),
        "clean_images_sha256": hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
    }
    return images, identity


def measure(model, images, condition, cfg, device):
    import torch
    from .metrics import embedding_statistics, retrieval_scores

    if len(images) != cfg.n_images:
        raise ValueError("Image count must match the analysis recipe")
    generator = torch.Generator().manual_seed(cfg.seed)
    features = []
    noise_hash = hashlib.sha256()
    for _ in range(cfg.k_reps):
        noisy = apply_noise(images, condition, generator)
        noise_hash.update(noisy.contiguous().numpy().tobytes())
        features.append(embed(model, noisy, device, batch_size=cfg.batch_size, prepool=cfg.prepool))
    cube = torch.stack(features, dim=1)
    metrics = embedding_statistics(cube)
    m = min(cfg.retrieval_images, len(images) // 2)  # captured gallery construction
    retrieval_args = dict(chunk_size=cfg.retrieval_chunk_size, permutation_seed=cfg.seed + 10000)
    if CONDITIONS[condition]["family"] in ("gaussian", "speckle"):
        eps = torch.randn(images[:m].shape, generator=generator)
        queries = apply_noise(images[:m], condition, generator, eps)
        gallery = apply_noise(images[m : 2 * m], condition, generator, eps)
        score = retrieval_scores(
            embed(model, queries, device, batch_size=cfg.batch_size, prepool=cfg.prepool),
            embed(model, gallery, device, batch_size=cfg.batch_size, prepool=cfg.prepool),
            **retrieval_args,
        )
        metrics.update(noise_reid=score["top1"], noise_reid_permuted=score["permuted_top1"])
    else:
        metrics.update(noise_reid=None, noise_reid_permuted=None)
    queries = apply_noise(images[:m], condition, generator)
    gallery = apply_noise(images[:m], condition, generator)
    score = retrieval_scores(
        embed(model, queries, device, batch_size=cfg.batch_size, prepool=cfg.prepool),
        embed(model, gallery, device, batch_size=cfg.batch_size, prepool=cfg.prepool),
        **retrieval_args,
    )
    metrics.update(
        content_reid=score["top1"],
        content_reid_permuted=score["permuted_top1"],
        retrieval_chance=score["chance"],
    )
    evaluation = {
        "k_reps": cfg.k_reps,
        "retrieval_images": m,
        "feature_dim": cube.shape[-1],
        "anova_corruption_sha256": noise_hash.hexdigest(),
        "input_treatment": "none",
        "noise_spec": CONDITIONS[condition],
        "feature_space": "prepool" if cfg.prepool else "pooled",
        "rank_input": "per_image_mean_embeddings",
        "noise_reid_supported": metrics["noise_reid"] is not None,
        "metric_version": 1,
        "noise_rng": "private_cpu_generator_captured_draw_order",
    }
    return metrics, evaluation, cube


def run_analysis(checkpoint, output_dir, cfg=None, legacy_metadata=None, *, expected_checkpoint_sha256=None):
    import torch
    from .checkpoints import load_backbone

    cfg = (cfg or Config()).validate()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty analysis output: {output}")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if cfg.smoke:
        torch.set_num_threads(2)
    model, checkpoint_info = load_backbone(checkpoint, device, legacy_metadata)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_info["checkpoint_sha256"] != expected_checkpoint_sha256
    ):
        raise ValueError("Checkpoint bytes differ from the analysis manifest")
    images, data_info = load_images(cfg)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(cfg))
    metrics, evaluation, features = measure(model, images, checkpoint_info["condition"], cfg, device)
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "smoke_not_benchmark" if cfg.smoke or checkpoint_info["checkpoint_smoke"] else "complete",
        "config": asdict(cfg),
        "checkpoint": checkpoint_info,
        "dataset": data_info,
        "evaluation": evaluation,
        "metrics": metrics,
        "device": str(device),
        "runtime": runtime_metadata(("noise_invariance_analysis",)),
    }
    if cfg.save_features:
        feature_path = output / "embeddings.pt"
        torch.save(
            {
                "format_version": 1,
                "kind": "noise_invariance_embeddings",
                "features": features,
                "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                "dataset": data_info,
                "evaluation": evaluation,
            },
            feature_path,
        )
        from cifar10.train import sha256_file

        result["embeddings_sha256"] = sha256_file(feature_path)
    write_json(output / "analysis.json", result)
    print(json.dumps({"status": result["status"], "output": str(output), "metrics": metrics}, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run", type=Path, help="Release run directory containing backbone.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    for name in ("n_images", "k_reps", "retrieval_images", "batch_size", "retrieval_chunk_size", "seed"):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("download", "prepool", "save_features"):
        p.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--legacy-backbone", choices=BACKBONES)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--condition", choices=CONDITIONS)
    p.add_argument("--arm")
    p.add_argument("--training-seed", type=int)
    args = vars(p.parse_args(argv))
    checkpoint, run = args.pop("checkpoint"), args.pop("run")
    output, config_path = args.pop("output_dir"), args.pop("config")
    legacy = {k: args.pop(k) for k in ("method", "condition", "arm", "training_seed")}
    legacy["backbone"] = args.pop("legacy_backbone")
    legacy = {k: v for k, v in legacy.items() if v is not None}
    smoke = args.pop("smoke")
    if smoke:
        args.update(smoke=True, n_images=8, k_reps=2, retrieval_images=4, batch_size=4)
    cfg = load_config(config_path, **args)
    run_analysis(checkpoint if checkpoint else run / "backbone.pt", output, cfg, legacy)


if __name__ == "__main__":
    main()
