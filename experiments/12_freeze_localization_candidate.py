from pathlib import Path, PureWindowsPath
import sys
import json
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import SemDataset
from src.models_convae import ConvAE

BASELINE_Q = 0.995
CANDIDATE_Q = 0.999


def get_row(frame, quantile):
    matches = frame[
        np.isclose(frame["quantile"].astype(float), quantile)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one result for quantile {quantile}"
        )
    return matches.iloc[0]


def main():
    cfg = load_config()

    checkpoint_path = ROOT / "models" / "convae_best.pt"
    policy_path = ROOT / "models" / "phase2" / "rcc_policy.json"
    manifest_path = ROOT / "results" / "dataset" / "split_manifest.csv"

    report_dir = ROOT / "reports" / "localization_threshold_development"

    summary = pd.read_csv(report_dir / "summary.csv")
    by_size = pd.read_csv(report_dir / "by_defect_size.csv")
    manifest = pd.read_csv(manifest_path)

    calibration = manifest[
        (manifest["split"] == "calibration") &
        (manifest["is_reference"] == 1)
    ].copy().reset_index(drop=True)

    base = ROOT / "data" / "raw" / "carinthia_s" / "data"

    for column, folder in (
        ("image_path", "images"),
        ("mask_path", "masks")
    ):
        calibration[column] = [
            str(base / folder / PureWindowsPath(str(p)).name)
            for p in calibration[column]
        ]

        for path in calibration[column]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvAE(
        int(cfg["phase1"]["base_channels"])
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path, map_location=device
    )

    model.load_state_dict(
        checkpoint.get("model_state", checkpoint)
    )
    model.eval()

    size = int(cfg["data"]["input_size"])

    loader = DataLoader(
        SemDataset(calibration, size),
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    errors = []

    print("LOCALIZATION CANDIDATE FINALIZATION", flush=True)
    print("Calibration reference images:", len(calibration), flush=True)
    print("Final test images: NOT USED", flush=True)

    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            x = batch["image"].to(device)
            reconstruction = model(x)
            error = (x - reconstruction).square()

            errors.append(
                error.cpu().numpy().reshape(-1)
            )

            if index % 5 == 0:
                print(
                    f"Calibration batch {index}/{len(loader)}",
                    flush=True
                )

    values = np.concatenate(errors)

    baseline_threshold = float(
        np.quantile(values, BASELINE_Q)
    )
    candidate_threshold = float(
        np.quantile(values, CANDIDATE_Q)
    )

    original_policy = json.loads(
        policy_path.read_text(encoding="utf-8")
    )

    original_threshold = float(
        original_policy["localization_threshold"]
    )

    if not np.isclose(
        baseline_threshold,
        original_threshold,
        rtol=1e-4,
        atol=1e-9
    ):
        raise RuntimeError(
            "Baseline threshold mismatch. "
            "Do not finalize candidate."
        )

    baseline = get_row(summary, BASELINE_Q)
    candidate = get_row(summary, CANDIDATE_Q)

    tiny = by_size[by_size["group"] == "tiny"]
    baseline_tiny = get_row(tiny, BASELINE_Q)
    candidate_tiny = get_row(tiny, CANDIDATE_Q)

    def metrics(row):
        return {
            "mean_anomaly_dice": float(row["mean_anomaly_dice"]),
            "mean_anomaly_iou": float(row["mean_anomaly_iou"]),
            "mean_anomaly_recall": float(row["mean_anomaly_recall"]),
            "mean_reference_flagged_pixel_fraction": float(
                row["mean_reference_flagged_pixel_fraction"]
            )
        }

    policy = {
        "status": "EXPERIMENTAL_NOT_DEPLOYED",
        "method": "raw_reconstruction_error",
        "threshold_source": "normal_calibration_images",
        "parameter_selection_split": "validation",
        "final_test_evaluated": False,
        "calibration_reference_count": len(calibration),
        "baseline_quantile": BASELINE_Q,
        "baseline_threshold": baseline_threshold,
        "candidate_quantile": CANDIDATE_Q,
        "candidate_threshold": candidate_threshold,
        "baseline_metrics": metrics(baseline),
        "candidate_metrics": metrics(candidate),
        "baseline_tiny_dice": float(
            baseline_tiny["mean_dice"]
        ),
        "candidate_tiny_dice": float(
            candidate_tiny["mean_dice"]
        ),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "limitations": [
            "Selected using development validation data",
            "Candidate has reduced pixel-level recall",
            "Not independently confirmed",
            "Existing RCC policy remains unchanged"
        ]
    }

    output = report_dir / "candidate_policy_0999.json"

    output.write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8"
    )

    print("\nBaseline threshold:", baseline_threshold)
    print("Candidate threshold:", candidate_threshold)
    print("Baseline parity: PASS")
    print("Saved candidate:", output)
    print("Original RCC policy: UNCHANGED")
    print("CANDIDATE FINALIZATION COMPLETE")


if __name__ == "__main__":
    main()
