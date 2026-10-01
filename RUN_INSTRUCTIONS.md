# Windows PowerShell Run Instructions

Run every command from the project root.

## 1. Create environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**Expected:** installation completes without dependency errors.

## 2. Place Carinthia-S
Expected location:
```text
data\raw\carinthia_s\data\
    images\
    masks\
    carinthia-s.csv
```

## 3. Verify dataset and split integrity
```powershell
python experiments\01_verify_dataset.py
```
**Expected:** JSON dataset summary followed by `Leakage check: PASS`. In the verified Carinthia-S release, counts should be approximately 4,591 images/masks, 224 empty masks, and 4,367 non-empty masks. If actual official files differ, investigate rather than hard-code the old counts.

Generated files:
```text
results\dataset\verification.json
results\dataset\image_metadata.csv
results\dataset\split_manifest.csv
results\dataset\leakage_report.json
```

## 4. Train Phase 1
```powershell
python experiments\02_train_phase1.py
```
**Expected:** epoch-by-epoch training/validation MSE and a final best epoch/validation loss. Creates `models\convae_best.pt`.

If a valid checkpoint already exists, `python run_all.py` skips retraining by default.

## 5. Evaluate Phase 1
```powershell
python experiments\03_evaluate_phase1.py
```
**Expected:** JSON containing ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, FPR, FNR, confusion counts, cost/image, threshold, and exploratory localization metrics.

## 6. Run Phase 2 RCC-ConvAE
```powershell
python experiments\04_phase2_rcc.py
```
**Expected:** selected top-k, selected lambda, selected alpha, then Phase-2 metrics and localization results. Final test data is evaluated only after selections are frozen.

## 7. Distribution drift
```powershell
python experiments\05_distribution_drift.py
```
**Expected:** per-condition metrics and an aggregate summary comparing Phase 1/Phase 2 with and without reference-only recalibration.

## 8. Ablation
```powershell
python experiments\06_ablation.py
```
**Expected:** table comparing global baseline, global+local, fixed-alpha conformal, cost-aware alpha, and local-only evidence.

## 9. Reports
```powershell
python experiments\07_generate_reports.py
```
**Expected:** `reports\final_metrics.csv`, localization/ablation/drift summaries, `experiment_manifest.json`, and `experiment_summary.md`.

## 10. Full pipeline
```powershell
python run_all.py
```
Resume from Phase 2 if Phase 1 is already frozen:
```powershell
python run_all.py --from 4
```

## 11. Tests
```powershell
pytest -q
```
**Expected:** all synthetic/unit tests pass. Dataset-dependent full experiments are not silently mocked.

## 12. Dashboard
```powershell
streamlit run app\streamlit_app.py
```
**Expected:** browser opens the SemiVision AI dashboard. Research pages can display the historical verified reference metrics immediately; live single/batch screening becomes available after trained model/policy artifacts exist.

## 13. Final ZIP
```powershell
python scripts\package_project.py
```
**Expected:** `dist\Risk_Calibrated_Semiconductor_Visual_Screening.zip` is created.
