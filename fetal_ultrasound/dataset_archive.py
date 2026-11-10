"""Package or extract the prepared ultrasound dataset as one lossless ZIP file."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile

import numpy as np

from cifar10.train import sha256_file
from .prepare import validate_manifest
from .splits import split_metadata

ASSETS = Path(__file__).parent / "dataset"
MEMBERS = {
    "images.npy",
    "manifest.json",
    "FETAL_PLANES_DB_data.csv",
    "UPSTREAM_README.md",
    "NOTICE.md",
    "LICENSE.txt",
}


def _entry(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def pack(directory, output):
    directory, output = Path(directory), Path(output)
    payload = json.loads((directory / "manifest.json").read_text())
    validate_manifest(payload)
    if sha256_file(directory / "images.npy") != payload["images_sha256"]:
        raise ValueError("Prepared array hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(MEMBERS):
            path = (ASSETS if name in ("NOTICE.md", "LICENSE.txt") else directory) / name
            entries[name] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            with path.open("rb") as source, archive.open(_entry(name), "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
        meta = dict(
            schema_version=1,
            format="seana_fetal_native_regions",
            license="CC-BY-4.0",
            fingerprint=payload["fingerprint"],
            images=len(payload["records"]),
            shape=payload["shape"],
            members=entries,
        )
        archive.writestr(_entry("archive.json"), json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return verify(output)


def _verify_open(archive):
    names = archive.namelist()
    if set(names) != MEMBERS | {"archive.json"} or len(names) != len(set(names)):
        raise ValueError("Unexpected, missing or duplicate archive member")
    if any(stat.S_ISLNK(i.external_attr >> 16) for i in archive.infolist()):
        raise ValueError("Archive symlinks are not supported")
    meta = json.loads(archive.read("archive.json"))
    if (
        meta.get("schema_version") != 1
        or meta.get("format") != "seana_fetal_native_regions"
        or set(meta["members"]) != MEMBERS
    ):
        raise ValueError("Invalid archive metadata")
    for name, expected in meta["members"].items():
        digest, size = hashlib.sha256(), 0
        with archive.open(name) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != expected["sha256"] or size != expected["bytes"]:
            raise ValueError(f"Archive hash/size mismatch: {name}")
    payload = json.loads(archive.read("manifest.json"))
    validate_manifest(payload)
    if (
        payload["images_sha256"] != meta["members"]["images.npy"]["sha256"]
        or payload["source_metadata_sha256"] != meta["members"]["FETAL_PLANES_DB_data.csv"]["sha256"]
        or payload["fingerprint"] != meta["fingerprint"]
        or len(payload["records"]) != meta["images"]
    ):
        raise ValueError("Archive and manifest disagree")
    with archive.open("images.npy") as stream:
        images = np.load(stream, allow_pickle=False)
    if list(images.shape) != payload["shape"] or images.dtype != np.uint8:
        raise ValueError("Invalid image array")
    for image, row in zip(images, payload["records"]):
        if hashlib.sha256(image.tobytes()).hexdigest() != row["crop_sha256"]:
            raise ValueError("Prepared crop hash mismatch")
    splits = []
    for plan in payload["prepared_splits"]:
        info = split_metadata(payload["records"], plan["indices"])
        splits.append(
            dict(
                mode=plan["mode"],
                seed=plan["seed"],
                images={k: len(v) for k, v in info["identities"].items()},
                patients={k: len(v) for k, v in info["patients"].items()},
            )
        )
    return dict(
        images=meta["images"],
        shape=payload["shape"],
        fingerprint=payload["fingerprint"],
        license=meta["license"],
        splits=splits,
    )


def _check_download(path):
    with Path(path).open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(
                "This is a Git LFS pointer. Run git lfs pull, or download the full dataset file from GitHub."
            )


def verify(path, expected_sha256=None):
    path = Path(path)
    _check_download(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        metadata = _verify_open(archive)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=digest, **metadata)


def extract(path, output, expected_sha256=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Extraction destination exists: {output}")
    _check_download(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        _verify_open(archive)
        output.mkdir(parents=True)
        for name in archive.namelist():
            with archive.open(name) as source, (output / name).open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("pack")
    build.add_argument("--directory", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    for name in ("verify", "extract"):
        action = sub.add_parser(name)
        action.add_argument("--archive", type=Path, required=True)
        action.add_argument("--sha256")
        if name == "extract":
            action.add_argument("--output", type=Path, default=Path("data/fetal"))
    args = parser.parse_args()
    if args.command == "pack":
        print(json.dumps(pack(args.directory, args.output), indent=2))
    elif args.command == "verify":
        print(json.dumps(verify(args.archive, args.sha256), indent=2))
    else:
        print(extract(args.archive, args.output, args.sha256))


if __name__ == "__main__":
    main()
