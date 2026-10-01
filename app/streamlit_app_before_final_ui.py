from __future__ import annotations
from pathlib import Path
import sys, io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.inference import ScreeningEngine
from app.dashboard_utils import load_json, load_csv, source_badge

cfg=load_config(); st.set_page_config(page_title="SemiVision AI",page_icon="🔬",layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#f8fafc 0%,#eef2f7 100%);}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.6rem 1.8rem; border-radius: 22px; background: white; box-shadow: 0 10px 30px rgba(15,23,42,.08); border:1px solid #e2e8f0; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.15rem; color:#0f172a;}
.hero p {color:#475569; margin:.5rem 0 0 0;}
.metric-card {background:white; border:1px solid #e2e8f0; border-radius:18px; padding:1rem 1.1rem; box-shadow:0 6px 18px rgba(15,23,42,.05);}
.badge {display:inline-block;padding:.26rem .55rem;border-radius:999px;background:#e2e8f0;color:#334155;font-size:.78rem;font-weight:600;}
.result-ref {padding:1rem;border-radius:14px;background:#ecfdf5;border:1px solid #a7f3d0;font-size:1.25rem;font-weight:700;color:#065f46;text-align:center;}
.result-anom {padding:1rem;border-radius:14px;background:#fff1f2;border:1px solid #fecdd3;font-size:1.25rem;font-weight:700;color:#9f1239;text-align:center;}
[data-testid="stSidebar"] {background:#0f172a;}
[data-testid="stSidebar"] * {color:#e2e8f0;}
</style>
""",unsafe_allow_html=True)

st.sidebar.markdown("## 🔬 SemiVision AI")
st.sidebar.caption("Semiconductor Defect Intelligence Platform")
page=st.sidebar.radio("Navigate",["Home / Project Overview","Single SEM Screening","Batch Screening","Research Metrics","Distribution Drift","Ablation Study","Model & Policy Information","About / Methodology"])

p1,p1path=load_json(ROOT/"results"/"phase1"/"metrics.json",ROOT/"results"/"reference_run"/"phase1_metrics.json")
p2,p2path=load_json(ROOT/"results"/"phase2"/"metrics.json",ROOT/"results"/"reference_run"/"phase2_metrics.json")

def hero(title,subtitle): st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)
def metric_row(items):
    cols=st.columns(len(items))
    for c,(label,val) in zip(cols,items):
        with c: st.metric(label,val)

def engine_ready(): return (ROOT/"models"/"convae_best.pt").exists() and (ROOT/"models"/"phase1_policy.json").exists()

if page=="Home / Project Overview":
    hero("SemiVision AI","Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for Semiconductor Manufacturing Under Distribution Drift")
    c1,c2,c3=st.columns([1.1,1.1,.8])
    with c1:
        st.subheader("Phase 1 — 2-D ConvAE")
        st.markdown("**Reference SEM images → 2-D convolutional autoencoder → reconstruction → global reconstruction error → calibrated threshold → REFERENCE/ANOMALY.**")
    with c2:
        st.subheader("Phase 2 — RCC-ConvAE")
        st.markdown("**Frozen ConvAE → global + localized top-k evidence → robust fusion → conformal risk calibration → cost-aware alpha → localization → drift evaluation.**")
    with c3:
        st.subheader("Dataset")
        st.markdown("**Carinthia-S**  \n4,591 paired SEM images and masks in the verified release. Ground truth is derived from **mask content**, not a presumed class ID.")
    if p1 and p2:
        st.markdown("### Research snapshot")
        metric_row([("Phase-1 ROC-AUC",f"{p1.get('roc_auc',float('nan')):.4f}"),("Phase-2 ROC-AUC",f"{p2.get('roc_auc',float('nan')):.4f}"),("Phase-2 Dice",f"{p2.get('localization',{}).get('dice',float('nan')):.4f}"),("Phase-2 Pixel AUROC",f"{p2.get('localization',{}).get('pixel_auroc',float('nan')):.4f}")])
        st.caption(f"Metrics source: {source_badge(p2path)}")

elif page=="Single SEM Screening":
    hero("Single SEM Screening","Upload one grayscale SEM image and inspect reconstruction, anomaly evidence, calibrated decision, and localization.")
    phase=st.segmented_control("Screening policy",["Phase 1","Phase 2"],default="Phase 2")
    up=st.file_uploader("Upload SEM image",type=["png","jpg","jpeg","tif","tiff","bmp"])
    if up:
        img=Image.open(up).convert("L"); arr=np.asarray(img,dtype=np.float32)/255.0
        st.image(img,caption="Input SEM image",width=380)
        try:
            eng=ScreeningEngine(ROOT,cfg,"phase1" if phase=="Phase 1" else "phase2"); res=eng.predict_array(arr)
            klass="result-anom" if res["decision"]=="ANOMALY" else "result-ref"; st.markdown(f'<div class="{klass}">{res["decision"]}</div>',unsafe_allow_html=True)
            metrics=[("Global score",f"{res['global_score']:.6g}")]
            if phase=="Phase 2": metrics += [("Local score",f"{res['local_score']:.6g}"),("Fused score",f"{res['fused_score']:.4f}"),("Conformal p",f"{res['p_value']:.4f}")]
            metric_row(metrics)
            a,b,c=st.columns(3); a.image(res["reconstruction"],clamp=True,caption="Reconstruction",use_container_width=True); b.image(res["error_map"],clamp=True,caption="Reconstruction-error map",use_container_width=True)
            if phase=="Phase 2": c.image(res["predicted_mask"]*255,clamp=True,caption="Predicted anomaly mask",use_container_width=True)
        except Exception as e:
            st.error("A trained checkpoint/policy is required for live screening. Run the training/evaluation pipeline first.")
            st.code("python run_all.py\nstreamlit run app\\streamlit_app.py",language="powershell")
            st.caption(str(e))

elif page=="Batch Screening":
    hero("Batch Screening","Screen multiple SEM images and export a compact decision table.")
    phase=st.segmented_control("Policy",["Phase 1","Phase 2"],default="Phase 2",key="batchphase")
    files=st.file_uploader("Upload multiple SEM images",type=["png","jpg","jpeg","tif","tiff","bmp"],accept_multiple_files=True)
    if files:
        try:
            eng=ScreeningEngine(ROOT,cfg,"phase1" if phase=="Phase 1" else "phase2"); rows=[]
            for f in files:
                arr=np.asarray(Image.open(f).convert("L"),dtype=np.float32)/255.0; r=eng.predict_array(arr); row={"filename":f.name,"decision":r["decision"],"global_score":r["global_score"]}
                for k in ["local_score","fused_score","p_value","threshold"]:
                    if k in r: row[k]=r[k]
                rows.append(row)
            df=pd.DataFrame(rows); metric_row([("Screened",len(df)),("Reference",int((df.decision=="REFERENCE").sum())),("Anomaly",int((df.decision=="ANOMALY").sum())),("Anomaly rate",f"{100*(df.decision=='ANOMALY').mean():.1f}%")]); st.dataframe(df,use_container_width=True,hide_index=True); st.download_button("Download CSV",df.to_csv(index=False),"batch_screening.csv","text/csv")
        except Exception as e: st.error(f"Live model artifacts are not ready: {e}")

elif page=="Research Metrics":
    hero("Research Metrics","Phase-1 baseline and Phase-2 RCC-ConvAE metrics with explicit result provenance.")
    if not (p1 and p2): st.warning("No result files found.")
    else:
        loc1=p1.get("localization_exploratory",{}); loc2=p2.get("localization",{})
        table=pd.DataFrame([
            {"Method":"Phase 1 - 2D ConvAE","ROC-AUC":p1.get("roc_auc"),"PR-AUC":p1.get("pr_auc"),"F1":p1.get("f1"),"Balanced Accuracy":p1.get("balanced_accuracy"),"FPR":p1.get("fpr"),"FNR":p1.get("fnr"),"Dice":loc1.get("dice"),"IoU":loc1.get("iou"),"Pixel AUROC":loc1.get("pixel_auroc")},
            {"Method":"Phase 2 - RCC-ConvAE","ROC-AUC":p2.get("roc_auc"),"PR-AUC":p2.get("pr_auc"),"F1":p2.get("f1"),"Balanced Accuracy":p2.get("balanced_accuracy"),"FPR":p2.get("fpr"),"FNR":p2.get("fnr"),"Dice":loc2.get("dice"),"IoU":loc2.get("iou"),"Pixel AUROC":loc2.get("pixel_auroc")},
        ]); st.dataframe(table.style.format(precision=4),use_container_width=True,hide_index=True)
        st.caption(f"Phase 1: {source_badge(p1path)} | Phase 2: {source_badge(p2path)}")
        long=table.melt(id_vars="Method",value_vars=["Dice","IoU","Pixel AUROC"],var_name="Localization metric",value_name="Score"); st.plotly_chart(px.bar(long,x="Localization metric",y="Score",color="Method",barmode="group",range_y=[0,1]),use_container_width=True)
        st.info("The reference run shows nearly unchanged image-level ranking metrics but materially stronger Phase-2 localization. The package therefore does not claim a fabricated image-level AUC gain.")

elif page=="Distribution Drift":
    hero("Distribution Drift","Compare frozen operating points with reference-only recalibration under controlled image perturbations.")
    d,path=load_csv(ROOT/"results"/"drift"/"drift_metrics.csv",ROOT/"results"/"reference_run"/"drift_conditions.csv")
    if d is None: st.warning("No drift results found.")
    elif "method" in d.columns:
        fig=px.line(d,x="condition",y="balanced_accuracy",color="method",markers=True); fig.update_yaxes(range=[0,1]); st.plotly_chart(fig,use_container_width=True); st.dataframe(d,use_container_width=True,hide_index=True)
    else:
        long=d.melt(id_vars="condition",var_name="Method",value_name="Balanced Accuracy"); fig=px.line(long,x="condition",y="Balanced Accuracy",color="Method",markers=True); fig.update_yaxes(range=[0,1]); st.plotly_chart(fig,use_container_width=True); st.dataframe(d,use_container_width=True,hide_index=True)
    st.caption(source_badge(path))

elif page=="Ablation Study":
    hero("Ablation Study","Isolate the contributions of global evidence, localized evidence, conformal calibration, and cost-aware policy selection.")
    d,path=load_csv(ROOT/"results"/"ablation"/"ablation.csv",ROOT/"results"/"reference_run"/"ablation.csv")
    if d is None: st.warning("No ablation results found.")
    else:
        st.dataframe(d,use_container_width=True,hide_index=True); cols=[c for c in ["f1","recall","fpr","fnr","cost_per_image"] if c in d.columns]; long=d.melt(id_vars=d.columns[0],value_vars=cols,var_name="Metric",value_name="Value"); st.plotly_chart(px.bar(long,x=d.columns[0],y="Value",color="Metric",barmode="group"),use_container_width=True); st.caption(source_badge(path))

elif page=="Model & Policy Information":
    hero("Model & Policy Information","Reproducibility metadata and frozen decision-policy settings.")
    st.write("Checkpoint present:", (ROOT/"models"/"convae_best.pt").exists())
    st.write("Seed:",cfg["project"]["seed"]," | Input size:",cfg["data"]["input_size"]," | FN cost:",cfg["phase2"]["fn_cost"]," | FP cost:",cfg["phase2"]["fp_cost"])
    if p2: st.json({k:p2.get(k) for k in ["topk","lambda","alpha","cost_per_image"] if k in p2})
    if (ROOT/"models"/"phase2"/"rcc_policy.json").exists(): st.json(__import__("json").loads((ROOT/"models"/"phase2"/"rcc_policy.json").read_text()))

else:
    hero("About / Methodology","Research design, ground truth, integrity rules, and interpretation.")
    st.markdown("""
### Research objective
To evaluate a reference-only 2-D convolutional autoencoder for SEM visual anomaly screening and extend it with localized reconstruction evidence, conformal risk calibration, asymmetric cost-aware decisions, pixel-level localization, and distribution-drift evaluation.

### Ground truth
An **empty segmentation mask** defines a reference/normal candidate; a **non-empty mask** defines an anomaly/defect. Masks are also used for localization metrics.

### Experimental integrity
The final test split is excluded from training, early stopping, top-k/lambda selection, normalization fitting, conformal calibration, alpha selection, localization threshold selection, and all other operating-point decisions.

### Interpretation
The methodology does not claim that RCC-ConvAE is a completely new neural architecture. It is a calibrated decision framework built around the frozen Phase-1 ConvAE backbone.
""")
