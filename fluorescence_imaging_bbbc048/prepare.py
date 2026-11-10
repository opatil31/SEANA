"""Index an extracted CellCycle/<phase> tree; do not copy or download image data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from cifar10.train import sha256_file
from .config import PHASES

EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def fingerprint(records, layout):
    return hashlib.sha256(
        json.dumps({"records": records, "layout": layout, "phases": PHASES}, sort_keys=True).encode()
    ).hexdigest()


def prepare(source, output, layout="merged"):
    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}")
    if layout not in ("merged", "channels"):
        raise ValueError("layout must be merged or channels")
    candidates = (
        [source]
        if source.name.lower() == "cellcycle"
        else sorted(p for p in source.rglob("CellCycle") if p.is_dir())
    )
    if len(candidates) != 1:
        raise ValueError("Supply the single extracted CellCycle directory or its parent")
    cellcycle = candidates[0]
    folders = {p.name.lower(): p for p in cellcycle.iterdir() if p.is_dir()}
    records, seen = [], set()
    for label, phase in enumerate(PHASES):
        if phase.lower() not in folders:
            raise ValueError(f"Missing phase directory: {phase}")
        groups = {}
        for path in sorted(folders[phase.lower()].rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if layout == "merged":
                if "merged" not in path.name.lower():
                    continue
                match = re.match(r"^(\d+)_", path.name)
                channel = 0
            else:
                match = re.match(r"^(\d+)_Ch(3|4|6)(?:\.ome)?\.(?:tiff?|jpe?g|png|bmp)$", path.name, re.I)
                channel = int(match[2]) if match else None
            if match is None:
                raise ValueError(f"Unrecognized cell filename: {path.name}")
            identity = str(int(match[1]))
            group = groups.setdefault(identity, {})
            if channel in group:
                raise ValueError(f"Duplicate image/channel for cell {identity}")
            group[channel] = path
        if not groups:
            raise ValueError(f"No {layout} cells found in phase {phase}")
        # Insertion order follows the captured merged loader's lexicographic file sort.
        for identity, group in groups.items():
            if identity in seen:
                raise ValueError(f"Cell identity occurs under multiple labels: {identity}")
            seen.add(identity)
            expected = {0} if layout == "merged" else {3, 4, 6}
            if set(group) != expected:
                raise ValueError(f"Incomplete channel set for cell {identity}")
            paths = [group[ch] for ch in sorted(group)]
            for path in paths:
                if source not in path.resolve().parents:
                    raise ValueError("Image symlinks must stay within the source tree")
            records.append(
                dict(
                    object_id=identity,
                    label=label,
                    files=[dict(path=p.relative_to(source).as_posix(), sha256=sha256_file(p)) for p in paths],
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        schema_version=1,
        dataset="bbbc048",
        phases=list(PHASES),
        layout=layout,
        source_root=os.path.relpath(source, output.parent.resolve()),
        records=records,
        fingerprint=fingerprint(records, layout),
        label_source="phase_directory",
    )
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/bbbc048/manifest.json"))
    p.add_argument("--layout", choices=("merged", "channels"), default="merged")
    args = p.parse_args()
    result = prepare(args.source, args.output, args.layout)
    print(f"Indexed {len(result['records'])} distinct cells; fingerprint {result['fingerprint']}")


if __name__ == "__main__":
    main()
