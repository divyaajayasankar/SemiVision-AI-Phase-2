from pathlib import Path
import sys, json, platform, subprocess
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ROOT
from src.utils import read_json, write_json, environment_info, sha256

cfg=load_config(); rep=ROOT/"reports"; rep.mkdir(parents=True,exist_ok=True)
p1=read_json(ROOT/"results"/"phase1"/"metrics.json"); p2=read_json(ROOT/"results"/"phase2"/"metrics.json")
rows=[]
for phase,m in [("Phase 1 - 2D ConvAE",p1),("Phase 2 - RCC-ConvAE",p2)]:
    loc=m.get("localization",m.get("localization_exploratory",{})); rows.append({"phase":phase,"roc_auc":m.get("roc_auc"),"pr_auc":m.get("pr_auc"),"accuracy":m.get("accuracy"),"balanced_accuracy":m.get("balanced_accuracy"),"precision":m.get("precision"),"recall":m.get("recall"),"f1":m.get("f1"),"fpr":m.get("fpr"),"fnr":m.get("fnr"),"cost_per_image":m.get("cost_per_image"),"dice":loc.get("dice"),"iou":loc.get("iou"),"pixel_auroc":loc.get("pixel_auroc")})
pd.DataFrame(rows).to_csv(rep/"final_metrics.csv",index=False)
pd.DataFrame([{"phase":"Phase 1",**p1.get("localization_exploratory",{})},{"phase":"Phase 2",**p2.get("localization",{})}]).to_csv(rep/"localization_summary.csv",index=False)
abl=pd.read_csv(ROOT/"results"/"ablation"/"ablation.csv"); abl.to_csv(rep/"ablation_table.csv",index=False)
drift=pd.read_csv(ROOT/"results"/"drift"/"drift_summary.csv"); drift.to_csv(rep/"drift_summary.csv",index=False)
env=environment_info(ROOT); env.update({"seed":cfg["project"]["seed"],"checkpoint_sha256":sha256(ROOT/"models"/"convae_best.pt"),"split_manifest_sha256":sha256(ROOT/"results"/"dataset"/"split_manifest.csv"),"phase1_metrics":"results/phase1/metrics.json","phase2_metrics":"results/phase2/metrics.json"}); write_json(rep/"experiment_manifest.json",env)
text=f"""# Experiment Summary\n\nProject: {cfg['project']['title']}\n\n## Phase 1\nROC-AUC: {p1['roc_auc']:.6f}\nPR-AUC: {p1['pr_auc']:.6f}\nF1: {p1['f1']:.6f}\nBalanced accuracy: {p1['balanced_accuracy']:.6f}\n\n## Phase 2\nROC-AUC: {p2['roc_auc']:.6f}\nPR-AUC: {p2['pr_auc']:.6f}\nF1: {p2['f1']:.6f}\nBalanced accuracy: {p2['balanced_accuracy']:.6f}\nSelected top-k: {p2['topk']}\nSelected lambda: {p2['lambda']}\nSelected alpha: {p2['alpha']}\n\n## Research-integrity note\nAll Phase-2 choices are made from validation/calibration/policy-validation data before final test evaluation.\n"""; (rep/"experiment_summary.md").write_text(text,encoding="utf-8")
print("Reports generated in",rep)
