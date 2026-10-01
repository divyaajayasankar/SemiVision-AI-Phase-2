from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from .models_convae import ConvAE
from .preprocessing import load_gray
from .scoring import local_topk_score, RobustStats, robust_z
from .conformal import conformal_pvalues


class ScreeningEngine:
    def __init__(self, root: Path, cfg: dict, phase: str = "phase2"):
        self.root=root; self.cfg=cfg; self.phase=phase
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=ConvAE(int(cfg["phase1"]["base_channels"])).to(self.device)
        ckpt=torch.load(root/"models"/"convae_best.pt",map_location=self.device)
        self.model.load_state_dict(ckpt.get("model_state",ckpt)); self.model.eval()
        self.p1=json.loads((root/"models"/"phase1_policy.json").read_text())
        self.p2=None; self.cal_scores=None
        if phase=="phase2":
            self.p2=json.loads((root/"models"/"phase2"/"rcc_policy.json").read_text())
            self.cal_scores=np.load(root/"models"/"phase2"/"calibration_scores.npy")

    @torch.no_grad()
    def predict_array(self, arr: np.ndarray):
        size=int(self.cfg["data"]["input_size"])
        im=Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8)).resize((size,size),Image.Resampling.LANCZOS)
        x=torch.from_numpy(np.asarray(im,dtype=np.float32)[None,None]/255.0).to(self.device)
        r=self.model(x); e=(x-r).pow(2)[0,0].cpu().numpy(); g=float(e.mean())
        out={"global_score":g,"reconstruction":r[0,0].cpu().numpy(),"error_map":e}
        if self.phase=="phase1":
            th=float(self.p1["threshold"]); out.update({"threshold":th,"decision":"ANOMALY" if g>th else "REFERENCE"}); return out
        topk=float(self.p2["topk"]); lam=float(self.p2["lambda"]); alpha=float(self.p2["alpha"])
        l=local_topk_score(e,topk)
        gs=RobustStats(**self.p2["global_stats"]); ls=RobustStats(**self.p2["local_stats"])
        fused=float(robust_z([g],gs)[0] + lam*robust_z([l],ls)[0]); pv=float(conformal_pvalues(self.cal_scores,[fused])[0])
        loc_th=float(self.p2["localization_threshold"])
        out.update({"local_score":l,"fused_score":fused,"p_value":pv,"alpha":alpha,"threshold":alpha,"decision":"ANOMALY" if pv<=alpha else "REFERENCE","predicted_mask":(e>=loc_th).astype(np.uint8)})
        return out

    def predict_path(self, path: str | Path):
        arr=load_gray(path,int(self.cfg["data"]["input_size"]))
        return self.predict_array(arr)
