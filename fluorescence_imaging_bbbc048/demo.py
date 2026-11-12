"""Notebook helpers for a matched, explicitly non-benchmark BBBC048 comparison."""

from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from seana import SEANA
from cifar10.data import make_loader
from cifar10.train import seed_all, sha256_file, write_json, runtime_metadata
from .config import PHASES, load_config
from .data import build_datasets, TransformedDataset, eval_transform
from .models import build_ssl
from .probe import train_probe
from .train import pretrain
from .views import ViewTreatment, cache_estimates, fit_identity


def recipe(
    manifest,
    *,
    preset="quick",
    method="simclr",
    seed=0,
    sigma=0.8,
    device="cuda",
    ssl_epochs=50,
    probe_epochs=20,
):
    """Presets select training cells and VST steps; epoch controls apply to both arms."""
    if preset not in ("quick", "long"):
        raise ValueError("Preset must be quick or long")
    if not 0 <= sigma <= 2:
        raise ValueError("Demo strength must be in [0,2]")
    cfg = load_config(
        manifest=str(manifest),
        method=method,
        treatment="seana",
        backbone="resnet18",
        split_seed=0,
        seed=seed,
        device=device,
        num_workers=0,
        view_order="shared",
        cache_estimates=True,
        probe_inputs="raw",
        sigma_vst=sigma,
        fit_batch_size=16,
        seana_fit_steps=500 if preset == "quick" else 5000,
        epochs=ssl_epochs,
        probe_epochs=probe_epochs,
    )
    return cfg, 8192 if preset == "quick" else None


def _subsample(subset, limit, seed):
    if limit is None or limit >= len(subset):
        return subset
    groups = {}
    for i in subset.indices:
        groups.setdefault(subset.dataset.records[i]["label"], []).append(i)
    if type(limit) is not int or limit < len(groups):
        raise ValueError("Training limit must keep at least one cell per phase")
    # Proportional allocation, with at least one cell per class present in the parent split.
    quotas = {k: 1 for k in groups}
    while sum(quotas.values()) < limit:
        k = max(
            (k for k in groups if quotas[k] < len(groups[k])),
            key=lambda k: (limit * len(groups[k]) / len(subset) - quotas[k], -k),
        )
        quotas[k] += 1
    rng, selected = random.Random(seed), set()
    for k in sorted(groups):
        rng.shuffle(groups[k])
        selected.update(groups[k][: quotas[k]])
    return Subset(subset.dataset, [i for i in subset.indices if i in selected])


def prepare_data(cfg, train_limit=None):
    train, test, parent = build_datasets(cfg)
    train = _subsample(train, train_limit, seed=cfg.split_seed + 4000)
    identities = {
        name: [ds.dataset.records[i]["object_id"] for i in ds.indices]
        for name, ds in (("train", train), ("test", test))
    }
    if set(identities["train"]) & set(identities["test"]):
        raise ValueError("Demo train/test cells overlap")
    metadata = {
        **parent,
        "identities": identities,
        "split_sha256": {
            k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in identities.items()
        },
        "demo_train_limit": train_limit,
        "sampling": "proportional by phase inside fixed parent training split; test split unchanged",
        "class_counts": {
            name: {
                phase: sum(ds.dataset.records[i]["label"] == k for i in ds.indices)
                for k, phase in enumerate(PHASES)
            }
            for name, ds in (("train", train), ("test", test))
        },
    }
    return train, test, metadata


class LibraryViews(nn.Module):
    """Use a fitted library augmenter with the demo's cached/shared view protocol.

    Fitting, denoising and serialization use SEANA's public API. The existing
    experiment view builder resamples cached estimates through its fitted VST,
    avoiding a DRUNet forward pass for every crop in every SSL epoch.
    """

    def __init__(self, cfg, augmenter):
        super().__init__()
        self.cfg, self.augmenter = cfg, augmenter
        standard = ViewTreatment(replace(cfg, treatment="baseline"), augmenter.device)
        self.spatial, self.photometric = standard.spatial, standard.photometric

    @property
    def vst(self):
        return self.augmenter.vst

    @property
    def denoiser(self):
        return self.augmenter.denoiser

    def clean_estimate(self, image):
        return self.augmenter.denoise(image)

    def renoise(self, estimate, sigma):
        return ViewTreatment.renoise(self, estimate, sigma)

    def forward(self, image, *, input_is_estimate=False):
        return ViewTreatment.forward(self, image, input_is_estimate=input_is_estimate)


def fit_seana(cfg, train, dataset, output, device, progress=None):
    """Use only training cells for VST range calibration and self-supervised fitting."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new demo directory before fitting again")
    if cfg.treatment != "seana" or cfg.view_order != "shared" or cfg.probe_report_best:
        raise ValueError("Demo requires shared SEANA and final probes")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    denoiser_options = (
        dict(denoiser=nn.AvgPool2d(3, stride=1, padding=1))
        if cfg.smoke
        else dict(sigma_d=cfg.resolved()["sigma_d"], pretrained=cfg.drunet_weights)
    )
    augmenter = SEANA(**denoiser_options, sigma_vst=cfg.sigma_vst, knots=cfg.seana_knots, device=device)
    loader = make_loader(train, replace(cfg, batch_size=cfg.fit_batch_size), shuffle=True, seed_offset=1)
    started = time.time()
    augmenter.fit(
        loader,
        steps=cfg.seana_fit_steps,
        patch_size=cfg.seana_patch_size,
        patch_batch_size=cfg.seana_patch_batch_size,
        stride=cfg.seana_stride,
        lr=cfg.seana_fit_lr,
        calibration_batches=cfg.seana_calibration_batches,
        seed=cfg.seed,
        progress=progress,
    )
    treatment = LibraryViews(cfg, augmenter)
    treatment.eval().requires_grad_(False)
    runtime = runtime_metadata(("fluorescence_imaging_bbbc048",))
    augmenter.save(output / "treatment.pt")
    write_json(output / "dataset.json", dataset)
    write_json(
        output / "fit.json",
        dict(
            status="demo_not_benchmark",
            config=cfg.resolved(),
            runtime=runtime,
            fit_identity=fit_identity(cfg, dataset),
            fit_summary=augmenter.fit_summary,
            seconds=time.time() - started,
            treatment_sha256=sha256_file(output / "treatment.pt"),
        ),
    )
    return treatment


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def preview(treatment, image, sigma, seed=0):
    """Fixed geometry: only the injected noise changes. Preserve training RNG state."""
    if not 0 <= sigma <= 2:
        raise ValueError("Preview strength must be in [0,2]")
    device = next(treatment.vst.parameters()).device
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        x = image.unsqueeze(0).to(device)
        estimate = treatment.clean_estimate(x)
        outputs = dict(
            acquired=x,
            drunet=treatment.denoiser(x).clamp(0, 1),
            estimate=estimate,
            view_a=treatment.renoise(estimate, sigma),
            view_b=treatment.renoise(estimate, sigma),
        )
    return {k: v[0].cpu() for k, v in outputs.items()}


@torch.inference_mode()
def _features(backbone, data, cfg, device):
    backbone.eval().requires_grad_(False)
    features, labels = [], []
    loader = make_loader(
        TransformedDataset(data, eval_transform(cfg)),
        replace(cfg, batch_size=cfg.probe_batch_size),
        shuffle=False,
    )
    for x, y in loader:
        features.append(backbone(x.to(device)).flatten(1).cpu())
        labels.append(y)
    return TensorDataset(torch.cat(features), torch.cat(labels))


def compare(cfg, train, test, dataset, treatment, output, device):
    """Train both backbones afresh; final probes see the same acquired train/test cells."""
    output, device = Path(output), torch.device(device)
    fitted = json.loads((output / "fit.json").read_text())
    if fitted["config"] != cfg.resolved() or json.loads((output / "dataset.json").read_text()) != dataset:
        raise ValueError("Demo data/config changed after VST fitting; start a new experiment")
    if sha256_file(output / "treatment.pt") != fitted["treatment_sha256"]:
        raise ValueError("Fitted treatment checkpoint changed")
    payload = torch.load(output / "treatment.pt", map_location="cpu", weights_only=True)
    for name, value in treatment.augmenter.state_dict().items():
        if not torch.equal(value.cpu(), payload["state_dict"][name]):
            raise ValueError("In-memory treatment changed after fitting")
    del payload
    if any((output / arm).exists() for arm in ("baseline", "seana")):
        raise FileExistsError("Comparison already started here; use a new experiment directory")
    rows, initial = [], None
    for arm in ("baseline", "seana"):
        print(
            f"\n--- {arm.upper()}: {cfg.epochs} SSL epochs, {cfg.probe_epochs} probe epochs ---", flush=True
        )
        started = time.time()
        current = replace(cfg, treatment=arm, cache_estimates=arm == "seana")
        directory = output / arm
        directory.mkdir()
        view = treatment if arm == "seana" else ViewTreatment(current, device)
        ssl_data = train
        if arm == "seana":
            print("Caching frozen SEANA estimates for training cells...", flush=True)
            ssl_data = cache_estimates(
                train,
                view,
                device,
                directory / "estimates.npy",
                dict(
                    dataset_fingerprint=dataset["fingerprint"],
                    training_split_sha256=dataset["split_sha256"]["train"],
                    treatment_sha256=fitted["treatment_sha256"],
                ),
                cfg.fit_batch_size,
            )
        seed_all(cfg.seed)
        model, criterion, dim = build_ssl(current)
        model, criterion = model.to(device), criterion.to(device)
        backbone_initial = state_digest(model.backbone)
        if initial is None:
            initial = backbone_initial
        elif initial != backbone_initial:
            raise RuntimeError("Comparison must start from identical backbone weights")
        history = pretrain(
            model,
            criterion,
            view,
            make_loader(ssl_data, current, shuffle=True, drop_last=True),
            current,
            device,
        )
        write_json(directory / "ssl_history.json", history)
        # Deterministic acquired-image features are computed once, then reused by every probe epoch.
        # No test scores or labels enter backbone training or head selection.
        frozen = state_digest(model.backbone)
        train_features, test_features = (
            _features(model.backbone, ds, current, device) for ds in (train, test)
        )
        probe_cfg = replace(current, batch_size=current.probe_batch_size, probe_report_best=False)
        seed_all(cfg.seed + 10000)
        head, metrics, history = train_probe(
            nn.Identity().to(device),
            dim,
            make_loader(train_features, probe_cfg, shuffle=True, seed_offset=10000),
            make_loader(test_features, probe_cfg, shuffle=False, seed_offset=20000),
            current,
            device,
        )
        if state_digest(model.backbone) != frozen:
            raise RuntimeError("Probe changed the frozen backbone")
        write_json(directory / "probe_history.json", history)
        torch.save(
            dict(kind="demo_backbone", state_dict=model.backbone.state_dict(), config=current.resolved()),
            directory / "backbone.pt",
        )
        torch.save(
            dict(
                kind="demo_linear_probe",
                state_dict=head.state_dict(),
                backbone_sha256=sha256_file(directory / "backbone.pt"),
            ),
            directory / "probe.pt",
        )
        row = dict(
            treatment=arm,
            initial_backbone_sha256=backbone_initial,
            final=metrics["final"],
            seconds=time.time() - started,
            config=current.resolved(),
        )
        write_json(directory / "demo_metrics.json", row)
        rows.append(row)
        del model, criterion, head, train_features, test_features, ssl_data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(
        schema_version=1,
        experiment="bbbc048_colab_demo",
        status="demo_not_benchmark",
        runs=rows,
        dataset=dataset,
        runtime=runtime_metadata(("fluorescence_imaging_bbbc048",)),
        probe="final head on cached, frozen acquired-image features; no test-epoch selection",
        treatment_sha256=fitted["treatment_sha256"],
        delta_percentage_points={
            key: 100 * (rows[1]["final"][key] - rows[0]["final"][key])
            for key in ("accuracy", "macro_f1")
        },
    )
    write_json(output / "demo_results.json", result)
    return result


def plot_results(result):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = result["runs"]
    metrics = ("accuracy", "macro_f1")
    positions = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for offset, row in zip((-0.18, 0.18), rows):
        ax.bar(
            positions + offset,
            [100 * row["final"][k] for k in metrics],
            width=0.36,
            label=row["treatment"].upper(),
        )
    ax.set(
        xticks=positions,
        xticklabels=["Top-1 accuracy", "Macro-F1"],
        ylim=(0, 100),
        ylabel="Held-out score (%)",
        title="This run: final frozen probes",
    )
    ax.legend()
    return fig


def export_results(output):
    """Small download with metrics, memberships and figures; omit weights/cached images."""
    output = Path(output)
    if not (output / "demo_results.json").is_file():
        raise ValueError("Complete the comparison before exporting results")
    target = output.with_suffix(".zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".png", ".csv"):
                archive.write(path, str(path.relative_to(output)))
    return target
