"""Pretrain fetal SSL and fit a patient-separated frozen six-plane probe."""

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import time

from cifar10.train import seed_all, sha256_file, write_json, runtime_metadata
from fluorescence_imaging_bbbc048.train import pretrain
from .config import Config, load_config, smoke_overrides, METHODS, BACKBONES, TREATMENTS, SPLITS


def run(cfg, output_dir):
    import torch
    from cifar10.data import make_loader
    from fluorescence_imaging_bbbc048.models import build_ssl
    from .data import build_datasets, TransformedDataset, eval_transform
    from .probe import train_probe, evaluate
    from .views import ViewTreatment, load_treatment, fit_identity, cache_estimates
    from .calibration import measure

    cfg.validate()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process per experiment")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is nonempty: {output}")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if cfg.smoke:
        torch.set_num_threads(2)
    started = time.time()
    seed_all(cfg.seed)
    splits, dataset = build_datasets(cfg)
    train = splits["train"]
    output.mkdir(parents=True, exist_ok=True)
    requested = cfg.resolved()
    runtime = runtime_metadata(("fluorescence_imaging_bbbc048", "fetal_ultrasound"))
    input_treatment = None
    if cfg.treatment_checkpoint:
        treatment, input_treatment = load_treatment(cfg.treatment_checkpoint, cfg, dataset, device)
    else:
        treatment = ViewTreatment(cfg, device)
        fit_loader = make_loader(
            train, replace(cfg, batch_size=cfg.fit_batch_size), shuffle=True, seed_offset=1
        )
        treatment.fit(fit_loader, device)
    calibration = None
    if cfg.treatment == "seana":
        calibration = measure(
            treatment,
            make_loader(train, replace(cfg, batch_size=cfg.fit_batch_size), shuffle=False),
            device,
            cfg.sigma_rule_max_batches,
        )
        calibration.update(
            training_split_sha256=dataset["split_sha256"]["train"],
            sigma_d=cfg.sigma_d,
            dataset_fingerprint=dataset["fingerprint"],
        )
        if cfg.sigma_mode == "unsupervised":
            cfg = replace(cfg, sigma_vst=calibration["sigma_vst_initial"] * cfg.sigma_multiplier).validate()
            treatment.cfg = cfg
        calibration["effective_sigma_vst"] = cfg.sigma_vst
        write_json(output / "calibration.json", calibration)
    write_json(output / "config.json", cfg.resolved())
    write_json(output / "dataset.json", dataset)
    treatment_hash = None
    if cfg.treatment != "baseline":
        torch.save(
            dict(
                format_version=1,
                kind="fetal_treatment",
                config=cfg.resolved(),
                fit_identity=fit_identity(cfg, dataset),
                state_dict=treatment.state_dict(),
                fit_provenance=getattr(treatment, "fit_provenance", dict(seed=cfg.seed, runtime=runtime)),
            ),
            output / "treatment.pt",
        )
        treatment_hash = sha256_file(output / "treatment.pt")
    estimate_cache = {}

    def estimates(split):
        if split not in estimate_cache:
            identity = dict(
                dataset_fingerprint=dataset["fingerprint"],
                split_sha256=dataset["split_sha256"][split],
                treatment_sha256=treatment_hash,
            )
            estimate_cache[split] = cache_estimates(
                splits[split],
                treatment,
                device,
                output / "cache" / f"{split}.npy",
                identity,
                cfg.fit_batch_size,
            )
        return estimate_cache[split]

    ssl_data = estimates("train") if cfg.cache_estimates else train
    seed_all(cfg.seed)
    ssl_loader = make_loader(ssl_data, cfg, shuffle=True, drop_last=True)
    model, criterion, dim = build_ssl(cfg)
    model, criterion = model.to(device), criterion.to(device)
    history = pretrain(model, criterion, treatment, ssl_loader, cfg, device)
    write_json(output / "ssl_history.json", history)
    common = dict(format_version=1, config=cfg.resolved(), dataset=dataset, feature_dim=dim)
    torch.save(
        dict(**common, kind="backbone", state_dict=model.backbone.state_dict()), output / "backbone.pt"
    )
    torch.save(
        dict(**common, kind="ssl", state_dict=model.state_dict(), loss_state_dict=criterion.state_dict()),
        output / "ssl.pt",
    )

    transform = eval_transform(cfg)
    probe_loaders = {}

    def probe_loader(split):
        if split not in probe_loaders:
            data = estimates(split) if cfg.probe_preprocessing() != "none" else splits[split]
            probe_loaders[split] = make_loader(
                TransformedDataset(data, transform),
                replace(cfg, batch_size=cfg.probe_batch_size),
                shuffle=split == "train",
                seed_offset=10000 if split == "train" else 20000,
            )
        return probe_loaders[split]

    val_loader = probe_loader("validation") if len(splits["validation"]) else None
    seed_all(cfg.seed + 10000)
    head, history, selected_epoch = train_probe(
        model.backbone, dim, probe_loader("train"), val_loader, splits["validation"].records, cfg, device
    )
    torch.save(
        dict(
            **common,
            kind="linear_probe",
            backbone_sha256=sha256_file(output / "backbone.pt"),
            treatment_sha256=treatment_hash if cfg.probe_preprocessing() != "none" else None,
            selected_epoch=selected_epoch,
            state_dict=head.state_dict(),
        ),
        output / "probe.pt",
    )
    write_json(output / "probe_history.json", history)
    metrics = dict(validation=None, test=None)
    for split in ("validation", "test"):
        if not len(splits[split]) or (split == "test" and not cfg.evaluate_test):
            continue
        metrics[split], predictions = evaluate(
            model.backbone, head, probe_loader(split), splits[split].records, device
        )
        write_json(output / f"{split}_predictions.json", predictions)
    result = dict(
        schema_version=1,
        experiment="fetal_ultrasound",
        status="smoke_not_benchmark" if cfg.smoke else "complete",
        config=cfg.resolved(),
        requested_config=requested,
        dataset=dataset,
        metrics=metrics,
        calibration=calibration,
        runtime=runtime,
        device=str(device),
        seconds=time.time() - started,
        probe_checkpoint_selection=cfg.probe_selection,
        selected_probe_epoch=selected_epoch,
        test_evaluated=cfg.evaluate_test,
        input_treatment_sha256=input_treatment,
        artifacts={p.name: sha256_file(p) for p in sorted(output.glob("*.pt"))},
        prediction_sha256={p.name: sha256_file(p) for p in sorted(output.glob("*_predictions.json"))},
    )
    write_json(output / "metrics.json", result)
    print(json.dumps(dict(output=str(output), status=result["status"], metrics=metrics), indent=2))
    return result


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    choices = dict(
        method=METHODS,
        backbone=BACKBONES,
        treatment=TREATMENTS,
        split_mode=SPLITS,
        device=("auto", "cpu", "cuda"),
        sigma_mode=("fixed", "unsupervised"),
        probe_selection=("final", "validation_macro_f1"),
        probe_inputs=("historical", "raw"),
    )
    for field in fields(Config):
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(
                name,
                type=int if field.name == "split_seed" else type(field.default),
                choices=choices.get(field.name),
            )
    return parser


def main(argv=None):
    p = parser()
    args = vars(p.parse_args(argv))
    path, output, dry_run = args.pop("config"), args.pop("output_dir"), args.pop("dry_run")
    if args.get("smoke"):
        args.update(smoke_overrides())
    try:
        cfg = load_config(path, **args)
    except (ValueError, TypeError) as error:
        p.error(str(error))
    if dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
    else:
        run(cfg, output or Path("runs/fetal") / cfg.run_name())


if __name__ == "__main__":
    main()
