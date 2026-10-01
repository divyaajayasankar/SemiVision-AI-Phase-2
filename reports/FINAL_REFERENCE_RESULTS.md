# Final Archived Reference Results

These are the previously verified results preserved for research continuity. They are not recomputed inside the packaging environment.

## Phase 1 - 2-D ConvAE
- ROC-AUC: 0.9971
- PR-AUC: 0.9998
- F1: 0.9989
- Balanced Accuracy: 0.9886
- FPR: 0.0217
- FNR: 0.0011
- Exploratory localization: Dice 0.4142, IoU 0.2704, Pixel AUROC 0.9450

## Phase 2 - RCC-ConvAE
- ROC-AUC: 0.9971
- PR-AUC: 0.9998
- F1: 0.9989
- Balanced Accuracy: 0.9886
- FPR: 0.0217
- FNR: 0.0011
- Cost/Image: 0.0119
- Selected top-k: 0.001
- Selected lambda: 1.0
- Selected alpha*: 0.075
- Localization: Dice 0.5381, IoU 0.3795, Pixel AUROC 0.9860, Disk Dice 0.3751

## Distribution Drift
- Frozen-policy mean balanced accuracy: 0.680495
- Recalibrated mean balanced accuracy: 0.933084
- Frozen-policy mean FPR: 0.633152
- Recalibrated mean FPR: 0.103261
- Frozen-policy worst balanced accuracy: 0.500000
- Recalibrated worst balanced accuracy: 0.878211
