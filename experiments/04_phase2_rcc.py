from pathlib import Path
import sys, json
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ROOT
from src.data import SemDataset
from src.models_convae import ConvAE
from src.scoring import score_dataset_multi_topk, fit_robust, robust_z
from src.conformal import conformal_pvalues
from src.cost_policy import select_alpha
from src.metrics import binary_metrics
from src.localization import fit_localization_threshold, evaluate_localization
from src.utils import write_json, sha256

cfg=load_config(); size=int(cfg["data"]["input_size"]); df=pd.read_csv(ROOT/"results"/"dataset"/"split_manifest.csv")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ConvAE(int(cfg["phase1"]["base_channels"])).to(device); ckpt=torch.load(ROOT/"models"/"convae_best.pt",map_location=device); model.load_state_dict(ckpt.get("model_state",ckpt)); model.eval()
val=df[df.split=="validation"]; cal=df[(df.split=="calibration")&(df.is_reference==1)]; pol=df[df.split=="policy_validation"]; test=df[df.split=="test"]
topks=[float(x) for x in cfg["phase2"]["topk_candidates"]]
val_all=score_dataset_multi_topk(model,SemDataset(val,size),device,topks,batch_size=2)
sel=[]
for k in topks:
    col=f"local_{k:g}"; sel.append({"topk":k,"val_roc_auc":float(roc_auc_score(val_all.label,val_all[col])),"val_pr_auc":float(average_precision_score(val_all.label,val_all[col]))})
sel_df=pd.DataFrame(sel).sort_values(["val_roc_auc","val_pr_auc","topk"],ascending=[False,False,True]); topk=float(sel_df.iloc[0].topk); print("Selected top-k",topk)
# Score once at selected top-k for allowed calibration/policy/final test splits.
def selected(frame):
    x=score_dataset_multi_topk(model,SemDataset(frame,size),device,[topk],batch_size=2); return x.rename(columns={f"local_{topk:g}":"local_score"})
cal_sc=selected(cal); pol_sc=selected(pol); test_sc=selected(test)
# robust stats fit ONLY on reference calibration
G=fit_robust(cal_sc.global_score); L=fit_robust(cal_sc.local_score)
for x in (val_all,):
    x["global_z"]=robust_z(x.global_score,G)
# selected local validation column
val_all["local_z"]=robust_z(val_all[f"local_{topk:g}"],L)
lrows=[]
for lam in [float(x) for x in cfg["phase2"]["lambda_candidates"]]:
    fused=val_all.global_z + lam*val_all.local_z
    lrows.append({"lambda":lam,"val_roc_auc":float(roc_auc_score(val_all.label,fused)),"val_pr_auc":float(average_precision_score(val_all.label,fused))})
lambda_df=pd.DataFrame(lrows).sort_values(["val_roc_auc","val_pr_auc","lambda"],ascending=[False,False,True]); lam=float(lambda_df.iloc[0]["lambda"]); print("Selected lambda",lam)
for x in (cal_sc,pol_sc,test_sc):
    x["global_z"]=robust_z(x.global_score,G); x["local_z"]=robust_z(x.local_score,L); x["fused_score"]=x.global_z+lam*x.local_z
cal_fused=cal_sc.fused_score.to_numpy(); pol_sc["p_value"]=conformal_pvalues(cal_fused,pol_sc.fused_score); alpha,alpha_table=select_alpha(pol_sc.label,pol_sc.p_value,cfg["phase2"]["alpha_candidates"],cfg["phase2"]["fn_cost"],cfg["phase2"]["fp_cost"]); print("Selected alpha",alpha)
test_sc["p_value"]=conformal_pvalues(cal_fused,test_sc.fused_score); test_sc["prediction"]=(test_sc.p_value<=alpha).astype(int)
metrics=binary_metrics(test_sc.label,test_sc.fused_score,test_sc.prediction,cfg["phase2"]["fn_cost"],cfg["phase2"]["fp_cost"])
loc_th=fit_localization_threshold(model,SemDataset(cal,size,include_mask=True),device,float(cfg["phase2"]["localization_reference_quantile"]),batch_size=1)
loc=evaluate_localization(model,SemDataset(test,size,include_mask=True),device,loc_th,int(cfg["phase2"]["pixel_auc_samples_per_image"]),batch_size=1)
metrics.update({"topk":topk,"lambda":lam,"alpha":alpha,"calibration_reference_count":len(cal_sc),"policy_validation_count":len(pol_sc),"test_count":len(test_sc),"localization":loc,"localization_threshold":loc_th})
out=ROOT/"results"/"phase2"; out.mkdir(parents=True,exist_ok=True); test_sc.to_csv(out/"predictions.csv",index=False); cal_sc.to_csv(out/"calibration_details.csv",index=False); pol_sc.to_csv(out/"policy_validation_predictions.csv",index=False); sel_df.to_csv(out/"topk_selection.csv",index=False); lambda_df.to_csv(out/"lambda_selection.csv",index=False); alpha_table.to_csv(out/"alpha_selection.csv",index=False); write_json(out/"metrics.json",metrics)
modeldir=ROOT/"models"/"phase2"; modeldir.mkdir(parents=True,exist_ok=True); np.save(modeldir/"calibration_scores.npy",cal_fused)
policy={"topk":topk,"lambda":lam,"alpha":alpha,"global_stats":G.to_dict(),"local_stats":L.to_dict(),"localization_threshold":loc_th,"fn_cost":float(cfg["phase2"]["fn_cost"]),"fp_cost":float(cfg["phase2"]["fp_cost"]),"calibration_reference_count":len(cal_sc),"checkpoint_sha256":sha256(ROOT/"models"/"convae_best.pt")}; write_json(modeldir/"rcc_policy.json",policy); write_json(modeldir/"normalization.json",{"global":G.to_dict(),"local":L.to_dict()}); write_json(modeldir/"localization_threshold.json",{"threshold":loc_th,"source":"reference calibration","quantile":float(cfg["phase2"]["localization_reference_quantile"])})
figdir=ROOT/"results"/"figures"; RocCurveDisplay.from_predictions(test_sc.label,test_sc.fused_score); plt.tight_layout(); plt.savefig(figdir/"phase2_roc.png",dpi=180); plt.close(); PrecisionRecallDisplay.from_predictions(test_sc.label,test_sc.fused_score); plt.tight_layout(); plt.savefig(figdir/"phase2_pr.png",dpi=180); plt.close(); ConfusionMatrixDisplay.from_predictions(test_sc.label,test_sc.prediction); plt.tight_layout(); plt.savefig(figdir/"phase2_confusion.png",dpi=180); plt.close()
print(json.dumps(metrics,indent=2))
