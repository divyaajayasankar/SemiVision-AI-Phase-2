from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)


def binary_metrics(y_true, scores, y_pred, fn_cost=10.0, fp_cost=1.0):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    try:
        roc = float(roc_auc_score(y, s))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y, s))
    except Exception:
        pr = float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    fpr = float(fp / (tn + fp)) if tn + fp else float("nan")
    fnr = float(fn / (fn + tp)) if fn + tp else float("nan")
    cost = float(fn_cost * fn + fp_cost * fp)
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y, p, zero_division=0)),
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_cost": cost,
        "cost_per_image": float(cost / max(1, len(y))),
    }


def dice_iou(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    d = (2 * inter) / (gt.sum() + pred.sum() + 1e-12)
    i = inter / (np.logical_or(gt, pred).sum() + 1e-12)
    return float(d), float(i)
