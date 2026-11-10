"""Build a BBBC048 Colab upload ZIP from an explicit public-file allowlist."""

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile


def public_files(root):
    root = Path(root)
    names = {
        "SEANA_BBBC048_Demo.ipynb",
        "requirements-colab.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "denoisers/__init__.py",
        "fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip",
    }
    for directory in ("cifar10",):
        names.update(str(p.relative_to(root)) for p in (root / directory).glob("*.py"))
    names.update(str(p.relative_to(root)) for p in (root / "third_party").glob("*.txt"))
    for p in (root / "fluorescence_imaging_bbbc048").rglob("*"):
        if (
            p.is_file()
            and p.suffix in (".py", ".md", ".json", ".csv", ".txt")
            and "__pycache__" not in p.parts
        ):
            names.add(str(p.relative_to(root)))
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or nonregular bundle input: {name}")
    return sorted(names)


def build(root, output):
    root, output = Path(root), Path(output)
    names = public_files(root)
    dataset = root / "fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip"
    expected = json.loads(dataset.with_name("metadata.json").read_text())["sha256"]
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != expected:
        raise ValueError("Bundled BBBC048 dataset checksum mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    with zipfile.ZipFile(output, "x") as archive:

        def add(name, data):
            info = zipfile.ZipInfo("SEANA_BBBC048/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED if name.endswith(".zip") else zipfile.ZIP_DEFLATED
            archive.writestr(info, data)

        for name in names:
            data = (root / name).read_bytes()
            entries[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            add(name, data)
        add(
            "colab_bundle.json", (json.dumps(dict(schema_version=1, files=entries), indent=2) + "\n").encode()
        )
    return dict(
        file=str(output),
        files=len(entries),
        bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parent.parent, args.output), indent=2))


if __name__ == "__main__":
    main()
