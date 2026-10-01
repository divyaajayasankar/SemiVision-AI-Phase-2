# ================================================================
# SemiVision AI - Complete Windows PowerShell Run Script
# Project: Risk-Calibrated and Cost-Aware Anomaly Detection Visual
# Screening for Semiconductor Manufacturing Under Distribution Drift
# ================================================================

# STEP 1 - Create virtual environment (run once)
python -m venv .venv

# STEP 2 - Activate environment
.\.venv\Scripts\Activate.ps1

# STEP 3 - Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# STEP 4 - Confirm Carinthia-S has been placed under:
# data\raw\carinthia_s\data\images\
# data\raw\carinthia_s\data\masks\
# data\raw\carinthia_s\data\carinthia-s.csv

# STEP 5 - Verify dataset and leakage-free split
python experiments\01_verify_dataset.py
# Expected final message: Leakage check: PASS

# STEP 6 - Run the complete research pipeline
python run_all.py
# If models already exist and you only want to continue from Phase 2:
# python run_all.py --from 4

# STEP 7 - Run tests
pytest -q
# Expected: all tests passed

# STEP 8 - Launch dashboard
streamlit run app\streamlit_app.py
# Browser should open the SemiVision AI dashboard.

# STEP 9 - Create the final ZIP package
# Stop Streamlit with Ctrl+C first, then run:
# python scripts\package_project.py
