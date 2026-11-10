"""Download once on a connected machine and save ImageNet-100 for offline training."""

import argparse
from pathlib import Path

from .config import HF_DATASET, HF_REVISION


def save_dataset(splits, output, *, source):
    from datasets import load_from_disk
    from .data import split_metadata
    from cifar10.train import write_json

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {output}")
    split_metadata(splits)
    output.parent.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output))
    # save_to_disk/load_from_disk can change fingerprints; record what runs read.
    manifest = {
        "schema_version": 1,
        "dataset": "imagenet100",
        "source": source,
        "splits": split_metadata(load_from_disk(str(output))),
    }
    write_json(output / "seana_dataset.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/imagenet100"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", default=HF_REVISION, help="Dataset commit SHA")
    parser.add_argument("--from-disk", type=Path, help="Import an existing HF save_to_disk DatasetDict")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    from datasets import load_dataset, load_from_disk

    if args.from_disk:
        splits = load_from_disk(str(args.from_disk))
        source = {"kind": "user_supplied_hf_disk", "revision": "unverified"}
    else:
        if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
            parser.error("--revision must be a full commit SHA")
        splits = load_dataset(
            HF_DATASET, revision=args.revision, cache_dir=str(args.cache_dir) if args.cache_dir else None
        )
        source = {"kind": "huggingface", "dataset": HF_DATASET, "revision": args.revision}
    manifest = save_dataset(splits, args.output, source=source)
    print(
        f"Prepared {manifest['splits']['train']['images']} train and "
        f"{manifest['splits']['validation']['images']} validation images at {args.output}"
    )


if __name__ == "__main__":
    main()
