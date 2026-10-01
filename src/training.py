from __future__ import annotations
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_convae(model, train_ds, val_ds, device, output_path: Path, epochs=150, lr=1e-3, patience=25, batch_size=8):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    best = float("inf"); best_epoch=0; wait=0; history=[]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs)+1):
        model.train(); total=0.0; count=0
        for b in train_loader:
            x=b["image"].to(device); opt.zero_grad(); r=model(x); loss=loss_fn(r,x); loss.backward(); opt.step()
            total += float(loss.item())*len(x); count += len(x)
        train_loss=total/max(1,count)
        model.eval(); vtotal=0.0; vcount=0
        with torch.no_grad():
            for b in val_loader:
                x=b["image"].to(device); r=model(x); loss=loss_fn(r,x)
                vtotal += float(loss.item())*len(x); vcount += len(x)
        val_loss=vtotal/max(1,vcount); history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss})
        print(f"epoch {epoch:3d} train {train_loss:.6f} val {val_loss:.6f}")
        if val_loss < best - 1e-9:
            best=val_loss; best_epoch=epoch; wait=0
            torch.save({"model_state":model.state_dict(),"epoch":epoch,"val_loss":val_loss}, output_path)
        else:
            wait += 1
            if wait >= int(patience):
                print(f"early stop at epoch {epoch}"); break
    hist_path=output_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps({"best_epoch":best_epoch,"best_val_loss":best,"history":history},indent=2),encoding="utf-8")
    return best_epoch, best
