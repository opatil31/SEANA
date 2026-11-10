"""Cell identity partitions, shared by the dataset packager and experiment loader."""

import random

from .config import SPLITS


def split_indices(records, mode, train_fraction, seed):
    if mode not in SPLITS or not 0 < train_fraction < 1:
        raise ValueError("Invalid split policy")
    rng = random.Random(seed)
    by_class = {label: [i for i, r in enumerate(records) if r["label"] == label] for label in range(7)}
    if mode == "acquisition_order_global":
        groups = [list(range(len(records)))]
    else:
        groups = list(by_class.values())
    train, test = [], []
    for indices in groups:
        if not indices:
            continue
        if mode == "random":
            rng.shuffle(indices)
        else:
            indices.sort(key=lambda i: int(records[i]["object_id"]))
        count = round(train_fraction * len(indices))
        count = min(max(1, count), len(indices) - 1) if len(indices) > 1 else 1
        train.extend(indices[:count])
        test.extend(indices[count:])
    if mode == "random":
        rng.shuffle(train)
        rng.shuffle(test)
    if not train or not test or set(train) & set(test):
        raise ValueError("Split must have disjoint nonempty train/test sets")
    return train, test


def prepared_splits(records, seeds=(0, 1, 2), train_fraction=0.8):
    plans = []
    for mode, seed in [("random", s) for s in seeds] + [
        ("acquisition_order", 0),
        ("acquisition_order_global", 0),
    ]:
        train, test = split_indices(records, mode, train_fraction, seed)
        plans.append(dict(mode=mode, seed=seed, train_fraction=train_fraction, train=train, test=test))
    validate_splits(records, plans)
    return plans


def validate_splits(records, plans):
    seen = set()
    for plan in plans:
        key = (plan["mode"], plan["train_fraction"], plan["seed"])
        if key in seen or type(plan["seed"]) is not int or plan["seed"] < 0:
            raise ValueError("Invalid or duplicate prepared split")
        seen.add(key)
        if any(type(i) is not int for name in ("train", "test") for i in plan[name]):
            raise ValueError("Prepared split indices must be integers")
        expected = split_indices(records, *key)
        if (plan["train"], plan["test"]) != expected:
            raise ValueError("Prepared split disagrees with the recorded split policy")


def select_split(records, mode, train_fraction, seed, plans=()):
    for plan in plans:
        if (plan["mode"], plan["train_fraction"], plan["seed"]) == (mode, train_fraction, seed):
            return plan["train"], plan["test"]
    return split_indices(records, mode, train_fraction, seed)
