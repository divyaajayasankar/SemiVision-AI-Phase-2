from __future__ import annotations
import numpy as np


def conformal_pvalues(calibration_scores, new_scores):
    cal = np.asarray(calibration_scores, dtype=float).reshape(-1)
    new = np.asarray(new_scores, dtype=float).reshape(-1)
    # Larger score = more anomalous. Small p => unusual relative to reference calibration.
    return np.asarray([(1.0 + np.sum(cal >= s)) / (len(cal) + 1.0) for s in new], dtype=float)
