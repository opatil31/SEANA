"""Run one rebuttal SSL isolation arm, sharing the main CIFAR training engine."""

import json
from pathlib import Path

from cifar10.train import parser as base_parser, run_experiment
from .config import load_config, ARM_TREATMENTS


def run(cfg, output_dir):
    from .views import IsolationTreatment, build_datasets

    result = run_experiment(
        cfg,
        output_dir,
        dataset_builder=build_datasets,
        treatment_builder=IsolationTreatment,
        extra_source_roots=("denoiser_oracle_isolation_experiments",),
    )
    return result


def parser():
    p = base_parser()
    p.description = __doc__
    p.add_argument("--arm", choices=tuple(ARM_TREATMENTS))
    p.add_argument("--treatment-checkpoint", help="Reuse a compatible fitted treatment.pt")
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    path, output = args.pop("config"), args.pop("output_dir")
    dry_run, smoke = args.pop("dry_run"), args.pop("smoke")
    if args.pop("treatment") is not None:
        parser().error("Use --arm instead of --treatment")
    if smoke:
        args.update(
            smoke=True,
            epochs=1,
            probe_epochs=1,
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
            lejepa_num_slices=8,
            moco_queue_size=32,
            warmup_epochs=0,
        )
    try:
        cfg = load_config(path, **args)
    except (ValueError, TypeError) as error:
        parser().error(str(error))
    if dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
        return
    run(cfg, output or Path("runs/denoiser_oracle_isolation") / cfg.run_name())


if __name__ == "__main__":
    main()
