"""Development-only RCC-ConvAE drift disagreement review.
Uses policy-validation images, never the final test split.
Human-review outcomes are assumed, not experimentally verified.
"""
import argparse
import json
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import SemDataset
from src.models_convae import ConvAE
from src.scoring import (
    RobustStats, fit_robust, robust_z,
    score_dataset_multi_topk
)
from src.conformal import conformal_pvalues
from src.utils import read_json


def resolve_paths(frame):
    frame = frame.copy().reset_index(drop=True)
    base = ROOT / "data" / "raw" / "carinthia_s" / "data"

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
                f"{len(missing)} missing {column}; example: {missing[0]}"
            )

    return frame


def score(model, frame, device, size, topk, condition):
    result = score_dataset_multi_topk(
        model,
        SemDataset(frame, size, condition=condition),
        device,
        [topk],
        batch_size=2
    )

    result = result.rename(
        columns={f"local_{topk:g}": "local_score"}
    )

    if len(result) != len(frame):
        raise RuntimeError("Scoring row count mismatch")

    return result


def summarize(y, actions, condition, method, review_cost,
              fn_cost, fp_cost):

    y = np.asarray(y, dtype=int)
    actions = np.asarray(actions, dtype=int)

    if not np.isin(actions, [0, 1, 2]).all():
        raise ValueError("Invalid actions")

    n = len(y)

    fp = int(((y == 0) & (actions == 1)).sum())
    fn = int(((y == 1) & (actions == 0)).sum())
    review = int((actions == 2).sum())

    return {
        "condition": condition,
        "method": method,
        "images": n,
        "normal_images": int((y == 0).sum()),
        "defect_images": int((y == 1).sum()),
        "auto_tn": int(((y == 0) & (actions == 0)).sum()),
        "auto_tp": int(((y == 1) & (actions == 1)).sum()),
        "auto_fp": fp,
        "auto_fn": fn,
        "review_count": review,
        "reviewed_normal": int(
            ((y == 0) & (actions == 2)).sum()
        ),
        "reviewed_defect": int(
            ((y == 1) & (actions == 2)).sum()
        ),
        "review_rate": review / max(1, n),
        "automatic_coverage": (n - review) / max(1, n),
        "ideal_review_cost_per_image": (
            fn_cost * fn +
            fp_cost * fp +
            review_cost * review
        ) / max(1, n)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--first-n-conditions", type=int, default=0
    )
    parser.add_argument(
        "--review-cost", type=float, default=0.25
    )
    args = parser.parse_args()

    if args.first_n_conditions < 0:
        parser.error("Invalid condition count")

    if not np.isfinite(args.review_cost) or args.review_cost < 0:
        parser.error("Invalid review cost")

    cfg = load_config()

    manifest = pd.read_csv(
        ROOT / "results" / "dataset" / "split_manifest.csv"
    )

    calibration = manifest[
        (manifest.split == "calibration") &
        (manifest.is_reference == 1)
    ]

    development = manifest[
        manifest.split == "policy_validation"
    ]

    if args.smoke:
        normal = development[
            development.is_reference == 1
        ].head(5)

        anomaly = development[
            development.is_reference != 1
        ].head(10)

        development = pd.concat(
            [normal, anomaly], ignore_index=True
        )

    if calibration.empty or development.empty:
        raise ValueError("Empty calibration/development split")

    calibration = resolve_paths(calibration)
    development = resolve_paths(development)

    policy = read_json(
        ROOT / "models" / "phase2" / "rcc_policy.json"
    )

    reference_scores = np.load(
        ROOT / "models" / "phase2" / "calibration_scores.npy"
    )

    topk = float(policy["topk"])
    lam = float(policy["lambda"])
    alpha = float(policy["alpha"])

    old_g = RobustStats(**policy["global_stats"])
    old_l = RobustStats(**policy["local_stats"])

    fn_cost = float(policy.get("fn_cost", 10))
    fp_cost = float(policy.get("fp_cost", 1))

    conditions = list(cfg["drift"]["conditions"])

    if args.smoke:
        if "clean" not in conditions:
            raise ValueError("Clean condition not configured")
        conditions = ["clean"]
    elif args.first_n_conditions:
        conditions = conditions[:args.first_n_conditions]

    if not conditions:
        raise ValueError("No drift conditions configured")

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

    folder = (
        "drift_review_dev_smoke"
        if args.smoke
        else "drift_review_development"
    )

    output_dir = ROOT / "reports" / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65, flush=True)
    print(
        "RCC-CONVAE DRIFT REVIEW POLICY | DEVELOPMENT-ONLY",
        flush=True
    )
    print("=" * 65, flush=True)
    print(
        "Input split: policy_validation ONLY (never test)",
        flush=True
    )
    print("Calibration references:", len(calibration), flush=True)
    print("Development images:", len(development), flush=True)
    print("Conditions:", conditions, flush=True)

    rows = []

    for condition in conditions:
        print("Scoring condition:", condition, flush=True)

        cal_sc = score(
            model, calibration, device, size, topk, condition
        )

        dev_sc = score(
            model, development, device, size, topk, condition
        )

        y = dev_sc["label"].to_numpy(dtype=int)

        frozen_scores = (
            robust_z(dev_sc.global_score, old_g) +
            lam * robust_z(dev_sc.local_score, old_l)
        )

        frozen_pred = (
            conformal_pvalues(
                reference_scores, frozen_scores
            ) <= alpha
        ).astype(int)

        new_g = fit_robust(cal_sc.global_score)
        new_l = fit_robust(cal_sc.local_score)

        new_reference = (
            robust_z(cal_sc.global_score, new_g) +
            lam * robust_z(cal_sc.local_score, new_l)
        )

        new_development = (
            robust_z(dev_sc.global_score, new_g) +
            lam * robust_z(dev_sc.local_score, new_l)
        )

        recal_pred = (
            conformal_pvalues(
                new_reference, new_development
            ) <= alpha
        ).astype(int)

        # 0 = normal, 1 = defect, 2 = human review
        selective = np.where(
            frozen_pred == recal_pred,
            frozen_pred,
            2
        )

        methods = [
            ("frozen_phase2", frozen_pred),
            ("condition_recalibrated_phase2", recal_pred),
            ("disagreement_review", selective)
        ]

        for name, predictions in methods:
            rows.append(
                summarize(
                    y, predictions, str(condition), name,
                    args.review_cost, fn_cost, fp_cost
                )
            )

        disagreements = int(
            (frozen_pred != recal_pred).sum()
        )

        print(
            "Disagreements sent to review:",
            disagreements, "/", len(y),
            flush=True
        )

    report = pd.DataFrame(rows)

    report.to_csv(
        output_dir / "policy_comparison.csv",
        index=False
    )

    metadata = {
        "test_split_accessed": False,
        "split_used": "policy_validation",
        "development_count": len(development),
        "calibration_count": len(calibration),
        "conditions": [str(c) for c in conditions],
        "alpha": alpha,
        "lambda": lam,
        "assumed_review_cost": args.review_cost,
        "false_negative_cost": fn_cost,
        "false_positive_cost": fp_cost,
        "limitations": [
            "Exploratory policy-validation experiment",
            "Known synthetic drift is used",
            "Perfect human-review resolution is assumed",
            "No independent test performance is claimed"
        ]
    }

    (
        output_dir / "study_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    columns = [
        "condition", "method", "auto_fp", "auto_fn",
        "review_count", "review_rate",
        "ideal_review_cost_per_image"
    ]

    print(report[columns].to_string(index=False))

    print("Saved:", output_dir / "policy_comparison.csv")
    print("Saved:", output_dir / "study_metadata.json")
    print(
        "Existing model, policy and final-test results: UNCHANGED"
    )
    print("DEVELOPMENT EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
