from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from .preprocessing import load_gray, load_mask, apply_perturbation


def _resolve_path(root: Path, raw, fallback_dir: str, filename: str | None = None) -> Path:
    candidates = []
    if raw is not None and str(raw).strip() and str(raw).lower() != "nan":
        q = Path(str(raw))
        candidates += [q, root / q, root / fallback_dir / q.name]
    if filename:
        candidates += [root / fallback_dir / Path(filename).name]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[-1] if candidates else root / fallback_dir / (filename or "")


def is_empty_mask(mask_path: str | Path) -> bool:
    arr = np.asarray(Image.open(mask_path).convert("L"))
    return not bool(np.any(arr > 0))


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(root: Path, csv_name: str = "carinthia-s.csv") -> tuple[pd.DataFrame, dict]:
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    required = {"image_path", "mask_path", "filename", "label"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_cols)}")

    rows = []
    dims, modes = set(), set()
    for i, r in df.iterrows():
        if i % 100 == 0:
            print(f"[VERIFY] Checked {i}/{len(df)} image-mask pairs...", flush=True)
        img_p = _resolve_path(root, r.get("image_path"), "images", str(r.get("filename", "")))
        mask_p = _resolve_path(root, r.get("mask_path"), "masks", str(r.get("filename", "")))
        if not img_p.exists():
            raise FileNotFoundError(f"Missing image at row {i}: {img_p}")
        if not mask_p.exists():
            # common mask naming variants
            stem = Path(str(r.get("filename", img_p.name))).stem
            alts = list((root / "masks").glob(stem + ".*"))
            if alts:
                mask_p = alts[0]
            else:
                raise FileNotFoundError(f"Missing mask at row {i}: {mask_p}")
        with Image.open(img_p) as im:
            dims.add(im.size); modes.add(im.mode)
            iw, ih = im.size
        with Image.open(mask_p) as mm:
            if mm.size != (iw, ih):
                raise ValueError(f"Image/mask size mismatch: {img_p.name}")
            marr = np.asarray(mm.convert("L"))
        empty = not bool(np.any(marr > 0))
        coverage = float(np.mean(marr > 0))
        rows.append({
            "filename": str(r["filename"]),
            "image_path": str(img_p),
            "mask_path": str(mask_p),
            "label": r["label"],
            "is_reference": int(empty),
            "is_anomaly": int(not empty),
            "mask_coverage": coverage,
            "width": iw,
            "height": ih,
        })
    meta = pd.DataFrame(rows)
    report = {
        "csv_rows": int(len(df)),
        "image_count": int(len(meta)),
        "mask_count": int(len(meta)),
        "unique_dimensions": [list(x) for x in sorted(dims)],
        "image_modes": sorted(modes),
        "empty_masks": int(meta["is_reference"].sum()),
        "non_empty_masks": int(meta["is_anomaly"].sum()),
        "avg_mask_coverage": float(meta["mask_coverage"].mean()),
        "median_mask_coverage": float(meta["mask_coverage"].median()),
        "max_mask_coverage": float(meta["mask_coverage"].max()),
        "missing_csv_values": int(df.isna().sum().sum()),
        "duplicate_filenames": int(meta["filename"].duplicated().sum()),
        "label_distribution": {str(k): int(v) for k, v in meta["label"].value_counts().sort_index().items()},
        "csv_sha256": compute_sha256(csv_path),
    }
    return meta, report


def create_split_manifest(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)
    refs = meta[meta.is_reference == 1].sample(frac=1, random_state=seed).copy()
    anom = meta[meta.is_anomaly == 1].sample(frac=1, random_state=seed + 1).copy()

    sr = cfg["split"]
    n = len(refs)
    n_train = round(n * float(sr["reference_train"]))
    n_val = round(n * float(sr["reference_validation"]))
    n_cal = round(n * float(sr["reference_calibration"]))
    n_pol = round(n * float(sr["reference_policy_validation"]))
    # ensure no overflow from rounding
    cuts = [n_train, n_train+n_val, n_train+n_val+n_cal, n_train+n_val+n_cal+n_pol]
    refs["split"] = "test"
    refs.iloc[:cuts[0], refs.columns.get_loc("split")] = "train"
    refs.iloc[cuts[0]:cuts[1], refs.columns.get_loc("split")] = "validation"
    refs.iloc[cuts[1]:cuts[2], refs.columns.get_loc("split")] = "calibration"
    refs.iloc[cuts[2]:min(cuts[3], n), refs.columns.get_loc("split")] = "policy_validation"

    na = len(anom)
    n_va = round(na * float(sr["anomaly_validation"]))
    n_pa = round(na * float(sr["anomaly_policy_validation"]))
    anom["split"] = "test"
    anom.iloc[:n_va, anom.columns.get_loc("split")] = "validation"
    anom.iloc[n_va:n_va+n_pa, anom.columns.get_loc("split")] = "policy_validation"

    out = pd.concat([refs, anom], ignore_index=True)
    # stable order independent from source CSV
    out = out.sort_values(["split", "filename"]).reset_index(drop=True)
    return out


def leakage_report(manifest: pd.DataFrame) -> dict:
    duplicates = manifest.groupby("filename")["split"].nunique()
    cross_split_dups = duplicates[duplicates > 1]
    ref_counts = {}
    anom_counts = {}
    if "is_reference" in manifest.columns:
        ref_counts = {str(k): int(v) for k, v in manifest[manifest["is_reference"] == 1]["split"].value_counts().items()}
    if "is_anomaly" in manifest.columns:
        anom_counts = {str(k): int(v) for k, v in manifest[manifest["is_anomaly"] == 1]["split"].value_counts().items()}
    return {
        "rows": int(len(manifest)),
        "cross_split_duplicate_filenames": int(len(cross_split_dups)),
        "cross_split_duplicates": cross_split_dups.index.tolist()[:100],
        "split_counts": {str(k): int(v) for k, v in manifest["split"].value_counts().items()},
        "reference_by_split": ref_counts,
        "anomaly_by_split": anom_counts,
        "passed": len(cross_split_dups) == 0,
    }


class SemDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, size: int, condition: str = "clean", include_mask: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.size = int(size)
        self.condition = condition
        self.include_mask = include_mask

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        r = self.frame.iloc[idx]
        arr = load_gray(r.image_path, self.size)
        if self.condition != "clean":
            arr = apply_perturbation(arr, self.condition, np.random.default_rng(42 + idx))
        x = torch.from_numpy(arr[None, ...].astype(np.float32))
        item = {
            "image": x,
            "label": torch.tensor(int(r.is_anomaly), dtype=torch.long),
            "filename": str(r.filename),
            "image_path": str(r.image_path),
            "mask_path": str(r.mask_path),
        }
        if self.include_mask:
            item["mask"] = torch.from_numpy(load_mask(r.mask_path, self.size).astype(np.uint8))
        return item



