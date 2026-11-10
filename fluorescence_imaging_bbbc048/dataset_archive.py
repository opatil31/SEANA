"""Build, verify or extract the single-file BBBC048 release dataset; no ML dependencies needed."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from cifar10.train import sha256_file
from .config import PHASES
from .prepare import fingerprint
from .splits import prepared_splits, validate_splits

ASSETS = Path(__file__).parent / "dataset"
METADATA_NAMES = {"manifest.json", "archive.json", "NOTICE.md", "LICENSE.txt", "Ground_truth.lst"}


def _safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or path.as_posix() != name
        or name.endswith("/")
    ):
        raise ValueError(f"Unsafe archive member: {name}")
    return name


def _validate_manifest(payload):
    if (
        payload.get("schema_version") != 1
        or payload.get("dataset") != "bbbc048"
        or payload.get("phases") != list(PHASES)
        or payload.get("layout") not in ("merged", "channels")
    ):
        raise ValueError("Invalid dataset manifest")
    records, layout = payload["records"], payload["layout"]
    if not records or fingerprint(records, layout) != payload["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch")
    ids, paths = set(), set()
    for row in records:
        identity = row["object_id"]
        if not isinstance(identity, str) or not identity.isdigit() or identity in ids:
            raise ValueError("Invalid or duplicate cell identity")
        ids.add(identity)
        if type(row["label"]) is not int or row["label"] not in range(7):
            raise ValueError("Invalid phase label")
        if len(row["files"]) != (1 if layout == "merged" else 3):
            raise ValueError("Incorrect channel count")
        for file in row["files"]:
            name = _safe_name(file["path"])
            if name in paths or name in METADATA_NAMES:
                raise ValueError("Repeated or reserved image path")
            paths.add(name)
    validate_splits(records, payload.get("prepared_splits", []))
    return paths


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_member(archive, name, data):
    info = zipfile.ZipInfo(_safe_name(name), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, data, compresslevel=9)


def pack(manifest, output, *, ground_truth=None, seeds=(0, 1, 2)):
    manifest, output = Path(manifest), Path(output)
    payload = json.loads(manifest.read_text())
    _validate_manifest(payload)
    root = (manifest.parent / payload["source_root"]).resolve()
    payload["source_root"] = "."
    payload["prepared_splits"] = prepared_splits(payload["records"], seeds)
    payload["split_provenance"] = (
        "Reconstructed from the captured cell ordering and split algorithm; "
        "historical run-specific membership files were not available."
    )
    members = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing release artifact.
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w") as archive:

            def add(name, data):
                _write_member(archive, name, data)
                members[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

            for row in payload["records"]:
                for file in row["files"]:
                    path = (root / file["path"]).resolve()
                    if root not in path.parents:
                        raise ValueError("Image must stay inside the dataset root")
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != file["sha256"]:
                        raise ValueError(f"Source image hash mismatch: {file['path']}")
                    add(file["path"], data)
            add("manifest.json", _json_bytes(payload))
            for name in ("NOTICE.md", "LICENSE.txt"):
                add(name, (ASSETS / name).read_bytes())
            if ground_truth is not None:
                add("Ground_truth.lst", Path(ground_truth).read_bytes())
            metadata = dict(
                schema_version=1,
                format="seana_bbbc048_archive",
                dataset="BBBC048 v1",
                fingerprint=payload["fingerprint"],
                cells=len(payload["records"]),
                class_counts={
                    phase: sum(r["label"] == i for r in payload["records"]) for i, phase in enumerate(PHASES)
                },
                layout=payload["layout"],
                image_processing="Original image bytes, without re-encoding or resizing",
                label_source="phase_directory",
                license="CC-BY-NC-SA-3.0",
                split_provenance=payload["split_provenance"],
                splits=[
                    dict(
                        mode=p["mode"],
                        seed=p["seed"],
                        train_fraction=p["train_fraction"],
                        train=len(p["train"]),
                        test=len(p["test"]),
                    )
                    for p in payload["prepared_splits"]
                ],
                members=members,
            )
            _write_member(archive, "archive.json", _json_bytes(metadata))
    return verify(output)


def _verify_open(archive):
    names = archive.namelist()
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive member")
    for info in archive.infolist():
        _safe_name(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Archive symlinks are not supported")
    metadata = json.loads(archive.read("archive.json"))
    if metadata.get("schema_version") != 1 or metadata.get("format") != "seana_bbbc048_archive":
        raise ValueError("Invalid archive metadata")
    if set(names) != set(metadata["members"]) | {"archive.json"}:
        raise ValueError("Unexpected or missing archive members")
    for name, expected in metadata["members"].items():
        data = archive.read(name)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Archive member hash/size mismatch: {name}")
    payload = json.loads(archive.read("manifest.json"))
    image_names = _validate_manifest(payload)
    if payload["source_root"] != "." or not payload.get("prepared_splits"):
        raise ValueError("Archive must be self-contained and include prepared splits")
    if set(names) - image_names not in (METADATA_NAMES, METADATA_NAMES - {"Ground_truth.lst"}):
        raise ValueError("Unexpected or missing dataset files")
    for row in payload["records"]:
        for file in row["files"]:
            if file["sha256"] != metadata["members"][file["path"]]["sha256"]:
                raise ValueError("Image hashes disagree with manifest")
    counts = Counter(r["label"] for r in payload["records"])
    if (
        metadata["fingerprint"] != payload["fingerprint"]
        or metadata["cells"] != len(payload["records"])
        or metadata["class_counts"] != {phase: counts[i] for i, phase in enumerate(PHASES)}
    ):
        raise ValueError("Archive summary disagrees with manifest")
    return {k: v for k, v in metadata.items() if k != "members"}


def verify(path, expected_sha256=None):
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        metadata = _verify_open(archive)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=digest, **metadata)


def extract(path, output, expected_sha256=None):
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(f"Extraction destination already exists: {output}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        _verify_open(archive)
        output.mkdir(parents=True)
        # Every member was checked above; only regular files with safe relative names remain.
        for name in archive.namelist():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.read(name))
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("pack")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--ground-truth", type=Path)
    build.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    for command in ("verify", "extract"):
        action = sub.add_parser(command)
        action.add_argument("--archive", type=Path, required=True)
        action.add_argument("--sha256")
        if command == "extract":
            action.add_argument("--output", type=Path, default=Path("data/bbbc048"))
    args = parser.parse_args()
    if args.command == "pack":
        print(
            json.dumps(
                pack(args.manifest, args.output, ground_truth=args.ground_truth, seeds=args.seeds), indent=2
            )
        )
    elif args.command == "verify":
        print(json.dumps(verify(args.archive, args.sha256), indent=2))
    else:
        print(extract(args.archive, args.output, args.sha256))


if __name__ == "__main__":
    main()
