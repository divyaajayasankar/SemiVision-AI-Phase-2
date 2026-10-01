from __future__ import annotations
from pathlib import Path
import argparse, subprocess, sys

ROOT=Path(__file__).resolve().parent
STAGES=[
    (1,"Dataset verification",[sys.executable,str(ROOT/"experiments"/"01_verify_dataset.py")]),
    (2,"Phase 1 training",[sys.executable,str(ROOT/"experiments"/"02_train_phase1.py")]),
    (3,"Phase 1 evaluation",[sys.executable,str(ROOT/"experiments"/"03_evaluate_phase1.py")]),
    (4,"Phase 2 RCC-ConvAE",[sys.executable,str(ROOT/"experiments"/"04_phase2_rcc.py")]),
    (5,"Distribution drift",[sys.executable,str(ROOT/"experiments"/"05_distribution_drift.py")]),
    (6,"Ablation",[sys.executable,str(ROOT/"experiments"/"06_ablation.py")]),
    (7,"Reports",[sys.executable,str(ROOT/"experiments"/"07_generate_reports.py")]),
    (8,"Tests",[sys.executable,"-m","pytest","-q"]),
]

def run(cmd):
    print("$"," ".join(map(str,cmd))); subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--from",dest="start",type=int,default=1); ap.add_argument("--skip-train-if-present",action="store_true",default=True); args=ap.parse_args()
    for n,name,cmd in STAGES:
        if n<args.start: continue
        if n==2 and (ROOT/"models"/"convae_best.pt").exists() and args.skip_train_if_present:
            print("[02] Valid checkpoint file exists; skipping training. Delete it or run experiments/02_train_phase1.py explicitly to retrain."); continue
        print("\n"+"="*78); print(f"[{n:02d}] {name}"); print("="*78); run(cmd)
    print("\nCore research pipeline complete. Launch dashboard with:")
    print("streamlit run app\\streamlit_app.py")
    print("Package ZIP with: python scripts\\package_project.py")
if __name__=="__main__": main()
