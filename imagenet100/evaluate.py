"""Evaluate a saved ImageNet-100 final probe on clean validation images."""

import argparse
import json
from pathlib import Path

from .config import Config, load_config


def evaluate_run(run, device="cpu", data_root=None):
    from cifar10.evaluate import evaluate_run as evaluate_shared
    from .data import build_datasets

    return evaluate_shared(
        run, device, data_root, config_type=Config, config_loader=load_config, dataset_builder=build_datasets
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--data-root")
    args = p.parse_args()
    print(json.dumps(evaluate_run(args.run, args.device, args.data_root), indent=2))


if __name__ == "__main__":
    main()
