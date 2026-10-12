"""Exploratory localization threshold sweep; calibration and validation only."""
from pathlib import Path, PureWindowsPath
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import SemDataset
from src.models_convae import ConvAE

QUANTILES = (0.970, 0.980, 0.990, 0.995, 0.997, 0.999)


def local_paths(frame):
    frame = frame.copy().reset_index(drop=True)
    base = ROOT / "data" / "raw" / "carinthia_s" / "data"

    for field, folder in (
        ("image_path", "images"),
        ("mask_path", "masks")
    ):
        frame[field] = frame[field].map(
            lambda p: str(
                base / folder / PureWindowsPath(str(p)).name
            )
        )
        missing = [
            p for p in frame[field] if not Path(p).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{field}: {missing[0]}"
            )
    return frame


@torch.no_grad()
def calibration_errors(model, dataset, device):
    chunks = []
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0
    )

    for i, batch in enumerate(loader, 1):
        x = batch["image"].to(device)
        error = (x - model(x)).square().cpu().numpy()
        chunks.append(error.reshape(-1))

        if i % 10 == 0:
            print(
                f"Calibration: {i}/{len(loader)}",
                flush=True
            )

    return np.concatenate(chunks)


def mask_group(percentage):
    if percentage < 0.5:
        return "tiny"
    if percentage < 2:
        return "small"
    if percentage < 10:
        return "medium"
    return "large"


@torch.no_grad()
def evaluate(model, dataset, device, thresholds):
    rows = []
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for i, batch in enumerate(loader, 1):
        x = batch["image"].to(device)
        error = (x - model(x)).square()[0, 0].cpu().numpy()
        truth = batch["mask"][0].cpu().numpy() > 0

        gt_pixels = int(truth.sum())
        group = (
            "reference" if gt_pixels == 0
            else mask_group(100 * gt_pixels / truth.size)
        )

        for quantile, threshold in thresholds.items():
            predicted = error >= threshold
            predicted_pixels = int(predicted.sum())
            intersection = int(
                np.logical_and(predicted, truth).sum()
            )
            union = predicted_pixels + gt_pixels - intersection

            rows.append({
                "filename": str(batch["filename"][0]),
                "quantile": quantile,
                "group": group,
                "gt_pixels": gt_pixels,
                "dice": (
                    2 * intersection / (predicted_pixels + gt_pixels)
                    if gt_pixels else np.nan
                ),
                "iou": (
                    intersection / union
                    if gt_pixels and union else np.nan
                ),
                "recall": (
                    intersection / gt_pixels
                    if gt_pixels else np.nan
                ),
                "reference_flagged_pixel_fraction": (
                    predicted_pixels / truth.size
                    if not gt_pixels else np.nan
                )
            })

        if i % 100 == 0:
            print(
                f"Validation: {i}/{len(loader)}",
                flush=True
            )

    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    frame = pd.read_csv(
        ROOT / "results" / "dataset" / "split_manifest.csv"
    )

    calibration = local_paths(
        frame[
            (frame.split == "calibration") &
            (frame.is_reference == 1)
        ]
    )
    validation = local_paths(
        frame[frame.split == "validation"]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvAE(
        int(cfg["phase1"]["base_channels"])
    ).to(device)

    checkpoint = torch.load(
        ROOT / "models" / "convae_best.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint.get("model_state", checkpoint)
    )
    model.eval()

    size = int(cfg["data"]["input_size"])

    print(
        "LOCALIZATION THRESHOLD SWEEP - DEVELOPMENT ONLY",
        flush=True
    )
    print(
        f"Calibration normal images: {len(calibration)} "
        f"| Validation images: {len(validation)}",
        flush=True
    )
    print(
        "Test split: NOT USED; policies: NOT MODIFIED",
        flush=True
    )

    reference_errors = calibration_errors(
        model,
        SemDataset(calibration, size),
        device
    )

    thresholds = {
        q: float(np.quantile(reference_errors, q))
        for q in QUANTILES
    }

    print("Reference thresholds:", thresholds, flush=True)

    details = evaluate(
        model,
        SemDataset(
            validation, size, include_mask=True
        ),
        device,
        thresholds
    )

    abnormal = details[details.group != "reference"]
    normal = details[details.group == "reference"]

    summary = abnormal.groupby(
        "quantile", as_index=False
    ).agg(
        anomaly_images=("dice", "count"),
        mean_anomaly_dice=("dice", "mean"),
        mean_anomaly_iou=("iou", "mean"),
        mean_anomaly_recall=("recall", "mean")
    )

    normal_summary = normal.groupby(
        "quantile", as_index=False
    ).agg(
        reference_images=("filename", "count"),
        mean_reference_flagged_pixel_fraction=(
            "reference_flagged_pixel_fraction", "mean"
        )
    )

    summary = summary.merge(
        normal_summary, on="quantile"
    )

    by_size = abnormal.groupby(
        ["quantile", "group"], as_index=False
    ).agg(
        images=("dice", "count"),
        mean_dice=("dice", "mean"),
        mean_iou=("iou", "mean"),
        mean_recall=("recall", "mean")
    )

    output = (
        ROOT / "reports" /
        "localization_threshold_development"
    )
    output.mkdir(parents=True, exist_ok=True)

    details.to_csv(
        output / "per_image.csv", index=False
    )
    summary.to_csv(
        output / "summary.csv", index=False
    )
    by_size.to_csv(
        output / "by_defect_size.csv", index=False
    )

    print("\nTHRESHOLD COMPARISON (no test data):")
    print(summary.round(5).to_string(index=False))

    print("\nDEFECT-SIZE COMPARISON:")
    print(by_size.round(5).to_string(index=False))

    print("\nReports:", output)
    print("EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
