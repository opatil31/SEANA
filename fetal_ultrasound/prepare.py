"""Prepare lossless native ultrasound regions and explicit patient partitions."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from cifar10.train import sha256_file
from .config import CLASSES
from .splits import partitions


def native_region(array, size=192):
    mask = array > 4
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if rows.any() and cols.any():
        r, c = np.where(rows)[0], np.where(cols)[0]
        box = [int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1]
    else:
        box = [0, array.shape[0], 0, array.shape[1]]
    r0, r1, c0, c1 = box
    cropped = array[r0:r1, c0:c1]
    h, w = cropped.shape
    padding = [max(0, size - h), max(0, size - w)]
    if any(padding):
        cropped = np.pad(cropped, ((0, padding[0]), (0, padding[1])), mode="reflect")
    h, w = cropped.shape
    top, left = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(cropped[top : top + size, left : left + size]), box, padding


def fingerprint(payload):
    identity = {k: payload[k] for k in ("classes", "shape", "dtype", "records", "images_sha256")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def finalize(directory, records, region_size=192):
    directory = Path(directory)
    for row in records:
        row["label"] = CLASSES.index(row["plane"])
    payload = dict(
        schema_version=1,
        dataset="fetal_planes_db",
        classes=list(CLASSES),
        dtype="uint8",
        shape=[len(records), region_size, region_size],
        images_file="images.npy",
        images_sha256=sha256_file(directory / "images.npy"),
        records=records,
        preprocessing="grayscale; intensity>4 bounding box; reflect padding; centered native crop without interpolation",
        source_metadata_sha256=sha256_file(directory / "FETAL_PLANES_DB_data.csv"),
        prepared_splits=[
            dict(mode=mode, validation_fraction=0.1, seed=0, indices=partitions(records, mode))
            for mode in ("official", "official_validation")
        ],
    )
    payload["fingerprint"] = fingerprint(payload)
    validate_manifest(payload)
    with (directory / "manifest.json").open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


def validate_manifest(payload):
    if (
        payload.get("schema_version") != 1
        or payload.get("dataset") != "fetal_planes_db"
        or payload.get("classes") != list(CLASSES)
        or payload.get("dtype") != "uint8"
        or payload.get("images_file") != "images.npy"
    ):
        raise ValueError("Invalid fetal manifest")
    records = payload["records"]
    if len(payload["shape"]) != 3 or payload["shape"][0] != len(records) or not records:
        raise ValueError("Invalid prepared image shape")
    seen = set()
    for row in records:
        if not row["image_id"] or row["image_id"] in seen or not row["patient_id"]:
            raise ValueError("Missing/duplicate image identity or patient")
        seen.add(row["image_id"])
        if (
            type(row["label"]) is not int
            or row["label"] not in range(6)
            or CLASSES[row["label"]] != row["plane"]
        ):
            raise ValueError("Invalid plane label")
    partitions(records)
    if fingerprint(payload) != payload["fingerprint"]:
        raise ValueError("Dataset fingerprint mismatch")
    seen = set()
    for split in payload["prepared_splits"]:
        key = (split["mode"], split["validation_fraction"], split["seed"])
        if key in seen or split["indices"] != partitions(records, *key):
            raise ValueError("Prepared patient split mismatch")
        seen.add(key)


def prepare(source, output, region_size=192):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Prepared directory exists: {output}")
    raw_csv = (source / "FETAL_PLANES_DB_data.csv").read_bytes()
    rows = [
        {k.strip(): v.strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(raw_csv.decode()), delimiter=";")
    ]
    output.mkdir(parents=True)
    images = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.uint8, shape=(len(rows), region_size, region_size)
    )
    records = []
    for i, row in enumerate(rows):
        name = row["Image_name"]
        if Path(name).name != name or not name or row["Train"] not in ("0", "1"):
            raise ValueError("Invalid source image name or official split flag")
        raw = (source / "Images" / (name + ".png")).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("L"))
        crop, box, pad = native_region(array, region_size)
        images[i] = crop
        records.append(
            dict(
                image_id=name,
                patient_id=row["Patient_num"],
                plane=row["Plane"],
                brain_plane=row["Brain_plane"],
                machine=row["US_Machine"],
                operator=row["Operator"],
                official_split="train" if row["Train"] == "1" else "test",
                source_png_sha256=hashlib.sha256(raw).hexdigest(),
                source_shape=list(array.shape),
                fan_bbox=box,
                reflect_pad_bottom_right=pad,
                crop_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
            )
        )
    images.flush()
    del images
    (output / "FETAL_PLANES_DB_data.csv").write_bytes(raw_csv)
    if (source / "README.md").is_file():
        shutil.copyfile(source / "README.md", output / "UPSTREAM_README.md")
    return finalize(output, records, region_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/fetal"))
    parser.add_argument("--region-size", type=int, default=192)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.region_size)
    print(f"Prepared {len(result['records'])} native regions; fingerprint {result['fingerprint']}")


if __name__ == "__main__":
    main()
