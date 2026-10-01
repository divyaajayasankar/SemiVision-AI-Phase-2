# Model Artifacts

A trained checkpoint is intentionally not fabricated in this package.

After the Carinthia-S dataset is placed in the documented folder, run:

```powershell
python experiments\01_verify_dataset.py
python experiments\02_train_phase1.py
python experiments\03_evaluate_phase1.py
python experiments\04_phase2_rcc.py
```

This creates:
- `convae_best.pt`
- `phase1_policy.json`
- `phase2/rcc_policy.json`
- `phase2/calibration_scores.npy`
- `phase2/normalization.json`
- `phase2/localization_threshold.json`
