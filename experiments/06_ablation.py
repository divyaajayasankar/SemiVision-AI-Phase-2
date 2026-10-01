from pathlib import Path
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ROOT
from src.metrics import binary_metrics
from src.utils import read_json

cfg=load_config(); p1=pd.read_csv(ROOT/"results"/"phase1"/"predictions.csv"); p2=pd.read_csv(ROOT/"results"/"phase2"/"predictions.csv"); cal=pd.read_csv(ROOT/"results"/"phase2"/"calibration_details.csv"); policy=read_json(ROOT/"models"/"phase2"/"rcc_policy.json")
fnc=float(cfg["phase2"]["fn_cost"]); fpc=float(cfg["phase2"]["fp_cost"]); rows=[]
def add(name,y,score,pred): rows.append({"experiment":name,**binary_metrics(y,score,pred,fnc,fpc)})
add("A: Global reconstruction (Phase 1)",p1.label,p1.global_score,p1.prediction)
fth=float(np.quantile(cal.fused_score,float(cfg["phase1"]["threshold_quantile"]))); add("B: Global + local fusion",p2.label,p2.fused_score,(p2.fused_score>fth).astype(int))
add("C: Fusion + conformal (alpha=0.05)",p2.label,p2.fused_score,(p2.p_value<=0.05).astype(int))
alpha=float(policy["alpha"]); add(f"D: Fusion + conformal + cost-aware alpha={alpha:g}",p2.label,p2.fused_score,(p2.p_value<=alpha).astype(int))
lth=float(np.quantile(cal.local_score,float(cfg["phase1"]["threshold_quantile"]))); add("E: Local evidence only",p2.label,p2.local_score,(p2.local_score>lth).astype(int))
out=pd.DataFrame(rows); d=ROOT/"results"/"ablation"; d.mkdir(parents=True,exist_ok=True); out.to_csv(d/"ablation.csv",index=False); print(out.to_string(index=False))
