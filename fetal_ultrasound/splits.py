"""Official patient split plus an explicitly separate release validation recipe."""

import hashlib
import json
import random


def partitions(records, mode="official", validation_fraction=0.1, seed=0):
    if mode not in ("official", "official_validation") or not 0 < validation_fraction < 1:
        raise ValueError("Invalid patient split recipe")
    by_patient = {}
    for row in records:
        split = row["official_split"]
        if split not in ("train", "test"):
            raise ValueError("Invalid official split flag")
        previous = by_patient.setdefault(row["patient_id"], split)
        if previous != split:
            raise ValueError("Patient occurs in both official training and test data")
    validation = set()
    if mode == "official_validation":
        patients = sorted(p for p, split in by_patient.items() if split == "train")
        if len(patients) < 2:
            raise ValueError("Need at least two training patients to create validation")
        random.Random(seed).shuffle(patients)
        count = min(max(1, round(len(patients) * validation_fraction)), len(patients) - 1)
        validation = set(patients[:count])
    result = dict(train=[], validation=[], test=[])
    for i, row in enumerate(records):
        split = "validation" if row["patient_id"] in validation else row["official_split"]
        result[split].append(i)
    if not result["train"] or not result["test"]:
        raise ValueError("Training and test partitions must be nonempty")
    return result


def split_metadata(records, indices):
    identities = {split: [records[i]["image_id"] for i in ids] for split, ids in indices.items()}
    patients = {split: sorted({records[i]["patient_id"] for i in ids}) for split, ids in indices.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(patients[a]) & set(patients[b]):
            raise ValueError("Patient overlap across partitions")
    return dict(
        identities=identities,
        patients=patients,
        split_sha256={k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in identities.items()},
    )
