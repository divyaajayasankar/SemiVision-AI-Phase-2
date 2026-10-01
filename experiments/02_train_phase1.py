from pathlib import Path
import sys, torch, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ROOT
from src.seed import set_seed
from src.data import SemDataset
from src.models_convae import ConvAE, count_parameters
from src.training import train_convae

cfg=load_config(); set_seed(int(cfg["project"]["seed"])); manifest=ROOT/"results"/"dataset"/"split_manifest.csv"
if not manifest.exists(): raise SystemExit("Run experiments/01_verify_dataset.py first")
df=pd.read_csv(manifest); train=df[(df.split=="train")&(df.is_reference==1)]; val=df[(df.split=="validation")&(df.is_reference==1)]
size=int(cfg["data"]["input_size"]); model=ConvAE(int(cfg["phase1"]["base_channels"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {len(train)} reference images; validating on {len(val)} reference images; device={device}; params={count_parameters(model)}")
train_ds=SemDataset(train,size); val_ds=SemDataset(val,size)
best_epoch,best_loss=train_convae(model,train_ds,val_ds,device,ROOT/"models"/"convae_best.pt",epochs=cfg["phase1"]["epochs"],lr=cfg["phase1"]["learning_rate"],patience=cfg["phase1"]["patience"],batch_size=cfg["phase1"]["batch_size"])
print(f"Best epoch={best_epoch}; validation loss={best_loss:.8f}")
