from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .metrics import dice_iou


@torch.no_grad()
def fit_localization_threshold(model, dataset, device, quantile=0.995, batch_size=2):
    model.eval(); vals = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_idx, b in enumerate(loader):
        if batch_idx % 100 == 0:
            print(f"[LOCALIZE] batch {batch_idx+1}/{len(loader)}", flush=True)
        x = b["image"].to(device); r = model(x); e = (x-r).pow(2).cpu().numpy()
        vals.append(e.reshape(-1))
    if not vals:
        raise ValueError("No calibration images for localization threshold")
    allv = np.concatenate(vals)
    return float(np.quantile(allv, quantile))


@torch.no_grad()
def evaluate_localization(model, dataset, device, threshold, samples_per_image=2000, batch_size=1):
    model.eval(); dices=[]; ious=[]; ys=[]; ss=[]
    rng = np.random.default_rng(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_idx, b in enumerate(loader):
        if batch_idx % 100 == 0:
            print(f"[LOCALIZE] batch {batch_idx+1}/{len(loader)}", flush=True)
        x = b["image"].to(device); r = model(x); e = (x-r).pow(2)[0,0].cpu().numpy()
        gt = b["mask"][0].numpy().astype(np.uint8)
        pred = (e >= threshold).astype(np.uint8)
        d, i = dice_iou(gt, pred); dices.append(d); ious.append(i)
        n = min(int(samples_per_image), gt.size)
        idx = rng.choice(gt.size, size=n, replace=False)
        ys.extend(gt.reshape(-1)[idx].tolist()); ss.extend(e.reshape(-1)[idx].tolist())
    try: pix_auc=float(roc_auc_score(ys, ss))
    except Exception: pix_auc=float("nan")
    try: pix_ap=float(average_precision_score(ys, ss))
    except Exception: pix_ap=float("nan")
    return {"dice": float(np.mean(dices)), "iou": float(np.mean(ious)), "pixel_auroc": pix_auc, "pixel_average_precision": pix_ap, "images": len(dices)}

