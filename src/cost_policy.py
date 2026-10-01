from __future__ import annotations
import numpy as np
import pandas as pd


def confusion(y_true, y_pred):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    return tn, fp, fn, tp


def select_alpha(y_true, pvalues, candidates, fn_cost=10.0, fp_cost=1.0):
    rows = []
    for alpha in candidates:
        pred = (np.asarray(pvalues) <= float(alpha)).astype(int)
        tn, fp, fn, tp = confusion(y_true, pred)
        cost = fn_cost * fn + fp_cost * fp
        rows.append({"alpha": float(alpha), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total_cost": float(cost), "cost_per_image": float(cost / max(1, len(pred)))})
    frame = pd.DataFrame(rows).sort_values(["total_cost", "alpha"], ascending=[True, True]).reset_index(drop=True)
    return float(frame.iloc[0]["alpha"]), frame
