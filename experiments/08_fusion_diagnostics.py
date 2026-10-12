"""Read-only diagnostic of saved Phase II RCC-ConvAE score fusion.

Reads calibration and policy-validation predictions. Does NOT load any image,
retrain the ConvAE, change the frozen policy, or inspect the final test set.

CAUTION: Policy validation was already used to select alpha in the original
experiment. These alternative-lambda results are exploratory diagnostics,
not an unbiased estimate or a basis for reporting improved generalization.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.conformal import conformal_pvalues


def require_columns(df, required, filename):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


def main():
    saved = ROOT / "results" / "phase2"
    model_dir = ROOT / "models" / "phase2"
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cal = pd.read_csv(saved / "calibration_details.csv")
    pol = pd.read_csv(saved / "policy_validation_predictions.csv")
    choices = pd.read_csv(saved / "lambda_selection.csv")
    policy = json.loads((model_dir / "rcc_policy.json").read_text(encoding="utf-8"))

    cols = ["global_z", "local_z"]
    require_columns(cal, cols, "calibration_details.csv")
    require_columns(pol, cols + ["label", "p_value"], "policy_validation_predictions.csv")
    require_columns(choices, ["lambda"], "lambda_selection.csv")

    for frame, name in [(cal, "calibration"), (pol, "policy validation")]:
        for col in cols:
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {name} {col}")

    labels = pol["label"].astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Policy-validation labels must include both reference (0) and anomaly (1).")

    weights = sorted({float(v) for v in choices["lambda"]})
    old_lambda = float(policy["lambda"])
    if old_lambda not in weights:
        weights.append(old_lambda)
        weights.sort()

    alpha = float(policy["alpha"])
    fn_cost = float(policy["fn_cost"])
    fp_cost = float(policy["fp_cost"])
    n_cal = len(cal)
    old_predictions = None
    diagnostics = []

    for weight in weights:
        cal_fused = (
            cal["global_z"].to_numpy(dtype=float)
            + weight * cal["local_z"].to_numpy(dtype=float)
        )
        pol_fused = (
            pol["global_z"].to_numpy(dtype=float)
            + weight * pol["local_z"].to_numpy(dtype=float)
        )
        p_values = np.asarray(conformal_pvalues(cal_fused, pol_fused), dtype=float)
        pred = (p_values <= alpha).astype(int)

        if np.isclose(weight, old_lambda, atol=1e-12, rtol=0):
            saved_p = pol["p_value"].to_numpy(dtype=float)
            discrepancy = float(np.max(np.abs(p_values - saved_p)))
            if discrepancy > 1e-10:
                raise RuntimeError(
                    f"Saved and recomputed p-values disagree (maximum = {discrepancy}). "
                    "Check that the saved files and policy correspond to one run."
                )
            old_predictions = pred.copy()

        tp = int(((labels == 1) & (pred == 1)).sum())
        tn = int(((labels == 0) & (pred == 0)).sum())
        fp = int(((labels == 0) & (pred == 1)).sum())
        fn = int(((labels == 1) & (pred == 0)).sum())
        diagnostics.append({
            "lambda": weight,
            "roc_auc": float(roc_auc_score(labels, pol_fused)),
            "pr_auc": float(average_precision_score(labels, pol_fused)),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "fpr": float(fp / (fp + tn)),
            "fnr": float(fn / (fn + tp)),
            "balanced_accuracy": float(0.5 * (tp / (tp + fn) + tn / (tn + fp))),
            "cost_per_image": float((fn_cost * fn + fp_cost * fp) / len(labels)),
            "p_value_unique_count": int(np.unique(p_values).size),
            "decisions_changed_vs_saved_lambda": None,
        })

    if old_predictions is None:
        raise RuntimeError("Could not find the original selected lambda in the candidate grid.")

    for row in diagnostics:
        weight = row["lambda"]
        cal_fused = cal["global_z"].to_numpy(dtype=float) + weight * cal["local_z"].to_numpy(dtype=float)
        pol_fused = pol["global_z"].to_numpy(dtype=float) + weight * pol["local_z"].to_numpy(dtype=float)
        pvals = np.asarray(conformal_pvalues(cal_fused, pol_fused), dtype=float)
        row["decisions_changed_vs_saved_lambda"] = int(
            ((pvals <= alpha).astype(int) != old_predictions).sum()
        )

    table = pd.DataFrame(diagnostics)
    csv_path = report_dir / "fusion_diagnostics_policy_validation.csv"
    table.to_csv(csv_path, index=False)

    summary = {
        "dataset": "saved policy-validation predictions only; final test untouched",
        "calibration_reference_count": n_cal,
        "policy_validation_count": len(pol),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "original_lambda": old_lambda,
        "fixed_alpha": alpha,
        "minimum_possible_conformal_pvalue": 1.0 / (n_cal + 1),
        "global_local_pearson_correlation": float(pol["global_z"].corr(pol["local_z"])),
        "global_local_spearman_correlation": float(
            pol["global_z"].corr(pol["local_z"], method="spearman")
        ),
        "warning": (
            "Exploratory only: alpha was previously chosen on this policy-validation split; "
            "do not choose an improved lambda here and claim unbiased gains."
        ),
    }
    json_path = report_dir / "fusion_diagnostics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 76)
    print("RCC-CONVAE FUSION DIAGNOSTICS — NO TEST DATA USED")
    print("=" * 76)
    print("Calibration references:", n_cal)
    print("Policy-validation samples:", len(pol))
    print("Original lambda:", old_lambda, "| Fixed alpha:", alpha)
    print("Smallest attainable conformal p-value:", f"{1.0 / (n_cal + 1):.6f}")
    print("Global/local Pearson correlation:", f"{summary['global_local_pearson_correlation']:.6f}")
    print("Global/local Spearman correlation:", f"{summary['global_local_spearman_correlation']:.6f}")
    print("\nExploratory fixed-alpha comparisons (NOT an unbiased test):")
    print(table[[
        "lambda", "roc_auc", "pr_auc", "FP", "FN", "cost_per_image",
        "decisions_changed_vs_saved_lambda",
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print("Existing checkpoint, final-test predictions, results and RCC policy: UNCHANGED")


if __name__ == "__main__":
    main()
