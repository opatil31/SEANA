"""BBBC048 SEANA strength sweep for BYOL, VICReg and SimCLR (paper Fig. 4 / App. G)."""

import argparse
import json
from pathlib import Path

from .config import BACKBONES
from .grid import make_grid


def make_sweep(*, manifest=None, backbones=("resnet18",), seeds=(0,)):
    recipe = json.loads((Path(__file__).parent / "configs/resampling_sweep.json").read_text())
    rows = make_grid(
        config=Path(__file__).parent / "configs/resampling_sweep_protocol.json",
        methods=recipe["methods"],
        backbones=backbones,
        treatments=("seana",),
        seeds=seeds,
        sigmas=recipe["sigma_vst"],
        manifest=manifest,
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=("resnet18",))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    args = vars(parser.parse_args())
    output = args.pop("output")
    rows = make_sweep(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} SEANA resampling-strength runs; execute with the BBBC048 grid runner.")


if __name__ == "__main__":
    main()
