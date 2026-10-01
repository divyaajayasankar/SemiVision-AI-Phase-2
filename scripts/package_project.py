from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "Risk_Calibrated_Semiconductor_Visual_Screening.zip"

RAW_DATA = ROOT / "data" / "raw" / "carinthia_s" / "data"

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".idea",
    ".vscode",
    "dist",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
}

DATASET_SETUP = ROOT / "DATASET_SETUP.md"

dataset_text = """# Carinthia-S Dataset Setup

The raw Carinthia-S dataset is intentionally excluded from the final project ZIP
to keep the research package manageable.

Place the dataset under:

data/raw/carinthia_s/data/

Expected structure:

data/
└── raw/
    └── carinthia_s/
        └── data/
            ├── images/
            ├── masks/
            └── carinthia-s.csv

Current experiment verification:

- Images: 4591
- Masks: 4591
- Image dimensions: 480 x 480
- Image mode: grayscale (L)
- CSV separator: semicolon (;)
- Empty masks: 226
- Non-empty masks: 4365

Ground-truth rule:

- Empty mask = REFERENCE / NORMAL
- Non-empty mask = ANOMALY / DEFECT

After placing the dataset, run:

python experiments/01_verify_dataset.py
"""

DATASET_SETUP.write_text(dataset_text, encoding="utf-8")

DIST.mkdir(parents=True, exist_ok=True)

# Remove any partially generated archive from an interrupted earlier run.
if OUT.exists():
    OUT.unlink()

files_to_package = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    rel = p.relative_to(ROOT)

    # Exclude environments, caches and output ZIP directory.
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        continue

    if p.suffix.lower() in EXCLUDE_SUFFIXES:
        continue

    # Raw Carinthia-S is deliberately excluded from the distributable ZIP.
    try:
        p.relative_to(RAW_DATA)
        continue
    except ValueError:
        pass

    files_to_package.append(p)

print("=" * 78)
print("SEMIVISION AI - FINAL PROJECT PACKAGING")
print("=" * 78)
print(f"Project root       : {ROOT}")
print(f"Files to package   : {len(files_to_package)}")
print("Raw Carinthia-S    : EXCLUDED")
print("Dataset setup file : INCLUDED")
print()

stored_extensions = {
    ".pt", ".pth", ".npy", ".npz",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".pdf", ".docx", ".zip",
}

with zipfile.ZipFile(
    OUT,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as z:
    total = len(files_to_package)

    for idx, p in enumerate(files_to_package, start=1):
        rel = p.relative_to(ROOT)
        arcname = Path(ROOT.name) / rel

        # Already-compressed/binary files do not benefit much from recompression.
        compression = (
            zipfile.ZIP_STORED
            if p.suffix.lower() in stored_extensions
            else zipfile.ZIP_DEFLATED
        )

        z.write(
            p,
            arcname=arcname,
            compress_type=compression,
        )

        if idx == 1 or idx % 25 == 0 or idx == total:
            print(
                f"[PACKAGE] {idx}/{total} files added...",
                flush=True,
            )

size_mb = OUT.stat().st_size / (1024 * 1024)

print()
print("=" * 78)
print("FINAL PROJECT PACKAGE CREATED")
print("=" * 78)
print(f"ZIP : {OUT}")
print(f"Size: {size_mb:.2f} MB")
print()
print("Raw dataset is not included.")
print("Use DATASET_SETUP.md to restore Carinthia-S before rerunning experiments.")
