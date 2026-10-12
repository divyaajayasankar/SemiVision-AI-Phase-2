from pathlib import Path, PureWindowsPath
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import SemDataset
from src.models_convae import ConvAE
from src.metrics import dice_iou


def fix_paths(frame):
    frame = frame.copy().reset_index(drop=True)
    base = ROOT / "data/raw/carinthia_s/data"

    for column, folder in [
        ("image_path", "images"),
        ("mask_path", "masks")
    ]:
        frame[column] = [
            str(base / folder / PureWindowsPath(str(p)).name)
            for p in frame[column]
        ]

        missing = [
            p for p in frame[column]
            if not Path(p).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing {column}: {missing[0]}"
            )

    return frame


def heatmaps(error):
    return {
        "raw": error,
        "smooth_3x3": F.avg_pool2d(
            error, kernel_size=3, stride=1, padding=1
        ),
        "smooth_5x5": F.avg_pool2d(
            error, kernel_size=5, stride=1, padding=2
        )
    }


@torch.no_grad()
def calibration_thresholds(model, dataset, device, quantile):
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0
    )

    values = {
        "raw": [],
        "smooth_3x3": [],
        "smooth_5x5": []
    }

    for index, batch in enumerate(loader, 1):
        x = batch["image"].to(device)
        reconstruction = model(x)
        error = (x - reconstruction).square()

        for name, heatmap in heatmaps(error).items():
            values[name].append(
                heatmap.cpu().numpy().reshape(-1)
            )

        if index % 10 == 0:
            print("Calibration batches:", index, "/", len(loader),
                  flush=True)

    return {
        name: float(
            np.quantile(np.concatenate(parts), quantile)
        )
        for name, parts in values.items()
    }


@torch.no_grad()
def evaluate(model, dataset, device, thresholds, max_images=0):
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    rows = []

    for index, batch in enumerate(loader, 1):
        if max_images and index > max_images:
            break

        x = batch["image"].to(device)
        reconstruction = model(x)
        error = (x - reconstruction).square()

        gt = batch["mask"][0].cpu().numpy().astype(np.uint8)

        for name, heatmap in heatmaps(error).items():
            scores = heatmap[0, 0].cpu().numpy()

            if scores.shape != gt.shape:
                raise ValueError(
                    f"Mask shape mismatch: {scores.shape} vs {gt.shape}"
                )

            prediction = (
                scores >= thresholds[name]
            ).astype(np.uint8)

            dice, iou = dice_iou(gt, prediction)

            rows.append({
                "filename": batch["filename"][0],
                "method": name,
                "dice": float(dice),
                "iou": float(iou),
                "mask_pixels": int((gt > 0).sum()),
                "predicted_pixels": int(prediction.sum())
            })

        if index % 100 == 0:
            print("Validation images:", index, "/", len(loader),
                  flush=True)

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    manifest = pd.read_csv(
        ROOT / "results/dataset/split_manifest.csv"
    )

    calibration = manifest[
        (manifest["split"] == "calibration") &
        (manifest["is_reference"] == 1)
    ]

    validation = manifest[
        manifest["split"] == "validation"
    ]

    calibration = fix_paths(calibration)
    validation = fix_paths(validation)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvAE(
        int(cfg["phase1"]["base_channels"])
    ).to(device)

    checkpoint = torch.load(
        ROOT / "models/convae_best.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint.get("model_state", checkpoint)
    )
    model.eval()

    size = int(cfg["data"]["input_size"])
    quantile = float(
        cfg["phase2"]["localization_reference_quantile"]
    )

    print("=" * 65, flush=True)
    print("RCC-CONVAE LOCALIZATION | DEVELOPMENT ONLY", flush=True)
    print("=" * 65, flush=True)
    print("Device:", device, flush=True)
    print("Calibration references:", len(calibration), flush=True)
    print("Validation images:", len(validation), flush=True)
    print("Threshold quantile:", quantile, flush=True)
    print("Final test split: NOT USED", flush=True)

    thresholds = calibration_thresholds(
        model,
        SemDataset(calibration, size, include_mask=True),
        device,
        quantile
    )

    print("Calibration thresholds:", thresholds, flush=True)

    max_images = 20 if args.smoke else 0

    predictions = evaluate(
        model,
        SemDataset(validation, size, include_mask=True),
        device,
        thresholds,
        max_images=max_images
    )

    if predictions.empty:
        raise RuntimeError("No validation results produced")

    summary = (
        predictions.groupby("method", as_index=False)
        .agg(
            mean_dice=("dice", "mean"),
            mean_iou=("iou", "mean"),
            images=("filename", "count")
        )
        .sort_values("mean_dice", ascending=False)
    )

    folder = (
        "localization_smoke"
        if args.smoke
        else "localization_development"
    )

    output = ROOT / "reports" / folder
    output.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        output / "per_image_metrics.csv", index=False
    )
    summary.to_csv(
        output / "method_comparison.csv", index=False
    )

    metadata = {
        "calibration_split": "calibration",
        "evaluation_split": "validation",
        "final_test_used": False,
        "quantile": quantile,
        "thresholds": thresholds,
        "methods": list(thresholds),
        "smoke": args.smoke,
        "model_retrained": False,
        "checkpoint_modified": False
    }

    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nLOCALIZATION COMPARISON")
    print(summary.to_string(index=False))
    print("\nSaved reports:", output)
    print("Original model and Phase II results: UNCHANGED")
    print("LOCALIZATION EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
