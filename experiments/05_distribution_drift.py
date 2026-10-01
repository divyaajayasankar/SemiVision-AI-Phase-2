from pathlib import Path
import sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ROOT
from src.data import SemDataset
from src.models_convae import ConvAE
from src.scoring import score_dataset_multi_topk, RobustStats, fit_robust, robust_z
from src.conformal import conformal_pvalues
from src.metrics import binary_metrics
from src.utils import read_json, write_json

cfg=load_config(); size=int(cfg["data"]["input_size"]); df=pd.read_csv(ROOT/"results"/"dataset"/"split_manifest.csv")
cal=df[(df.split=="calibration")&(df.is_reference==1)]; test=df[df.split=="test"]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ConvAE(int(cfg["phase1"]["base_channels"])).to(device); ckpt=torch.load(ROOT/"models"/"convae_best.pt",map_location=device); model.load_state_dict(ckpt.get("model_state",ckpt)); model.eval()
p1=read_json(ROOT/"models"/"phase1_policy.json"); p2=read_json(ROOT/"models"/"phase2"/"rcc_policy.json"); orig_cal=np.load(ROOT/"models"/"phase2"/"calibration_scores.npy")
topk=float(p2["topk"]); lam=float(p2["lambda"]); alpha=float(p2["alpha"]); G0=RobustStats(**p2["global_stats"]); L0=RobustStats(**p2["local_stats"])
rows=[]
for cond in cfg["drift"]["conditions"]:
    print("Drift condition:",cond)
    cal_sc=score_dataset_multi_topk(model,SemDataset(cal,size,condition=cond),device,[topk],batch_size=2).rename(columns={f"local_{topk:g}":"local_score"})
    test_sc=score_dataset_multi_topk(model,SemDataset(test,size,condition=cond),device,[topk],batch_size=2).rename(columns={f"local_{topk:g}":"local_score"})
    # Phase 1 frozen and recalibrated
    for name,th in [("Phase 1",float(p1["threshold"])),("Phase 1 + recalibration",float(np.quantile(cal_sc.global_score,float(cfg["phase1"]["threshold_quantile"]))))]:
        pred=(test_sc.global_score>th).astype(int); m=binary_metrics(test_sc.label,test_sc.global_score,pred,cfg["phase2"]["fn_cost"],cfg["phase2"]["fp_cost"]); rows.append({"condition":cond,"method":name,**m})
    # Phase 2 frozen calibration
    fused0=robust_z(test_sc.global_score,G0)+lam*robust_z(test_sc.local_score,L0); pval0=conformal_pvalues(orig_cal,fused0); pred0=(pval0<=alpha).astype(int); m=binary_metrics(test_sc.label,fused0,pred0,cfg["phase2"]["fn_cost"],cfg["phase2"]["fp_cost"]); rows.append({"condition":cond,"method":"Phase 2",**m})
    # Phase 2 reference-only recalibration: refit normalizers and conformal reference scores on shifted calibration refs, keep cost-selected alpha frozen.
    G=fit_robust(cal_sc.global_score); L=fit_robust(cal_sc.local_score); cal_f=robust_z(cal_sc.global_score,G)+lam*robust_z(cal_sc.local_score,L); test_f=robust_z(test_sc.global_score,G)+lam*robust_z(test_sc.local_score,L); pv=conformal_pvalues(cal_f,test_f); pred=(pv<=alpha).astype(int); m=binary_metrics(test_sc.label,test_f,pred,cfg["phase2"]["fn_cost"],cfg["phase2"]["fp_cost"]); rows.append({"condition":cond,"method":"Phase 2 + recalibration",**m})
out=pd.DataFrame(rows); outdir=ROOT/"results"/"drift"; outdir.mkdir(parents=True,exist_ok=True); out.to_csv(outdir/"drift_metrics.csv",index=False)
clean=out[out.condition=="clean"].set_index("method")
summary=[]
for method,g in out.groupby("method"):
    base=float(clean.loc[method,"roc_auc"]) if method in clean.index else np.nan
    summary.append({"method":method,"mean_roc_auc":g.roc_auc.mean(),"mean_balanced_accuracy":g.balanced_accuracy.mean(),"mean_fpr":g.fpr.mean(),"mean_fnr":g.fnr.mean(),"worst_balanced_accuracy":g.balanced_accuracy.min(),"mean_delta_roc_auc":(g.roc_auc-base).mean()})
s=pd.DataFrame(summary); s.to_csv(outdir/"drift_summary.csv",index=False); print(s.to_string(index=False))
