from __future__ import annotations

from datetime import datetime
from pathlib import Path
import io
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import ScreeningEngine
from app.dashboard_utils import load_csv, load_json

cfg = load_config()

st.set_page_config(
    page_title="SemiVision AI",
    page_icon="ðŸ”¬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
    }
    .block-container {
        max-width: 1320px;
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071827 0%, #0b2237 100%);
        border-right: 1px solid #17364e;
    }
    [data-testid="stSidebar"] * {
        color: #e8f1f8;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .45rem .55rem;
        border-radius: 8px;
        margin-bottom: .18rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #0f7aa8;
    }
    .brand {
        padding: .15rem .1rem .8rem .1rem;
    }
    .brand-title {
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: .2px;
    }
    .brand-sub {
        color: #9eb5c6 !important;
        font-size: .78rem;
        line-height: 1.3;
        margin-top: .18rem;
    }
    .page-title {
        text-align: center;
        font-size: 1.62rem;
        font-weight: 800;
        color: #18334a;
        margin-top: .2rem;
        margin-bottom: .12rem;
    }
    .page-subtitle {
        text-align: center;
        color: #718096;
        font-size: .9rem;
        margin-bottom: 1rem;
    }
    .panel {
        background: white;
        border: 1px solid #dce5ee;
        border-radius: 14px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 4px 14px rgba(23, 51, 74, .06);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.04rem;
        font-weight: 750;
        color: #1c445f;
        margin-bottom: .55rem;
    }
    .result-anomaly {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #a51632;
        border-radius: 11px;
        padding: .9rem 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 800;
    }
    .result-reference {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #087251;
        border-radius: 11px;
        padding: .9rem 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 800;
    }
    .small-note {
        color: #718096;
        font-size: .82rem;
    }
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)


def final_engine() -> ScreeningEngine:
    # The deployed application uses the complete proposed screening policy.
    return ScreeningEngine(ROOT, cfg, "phase2")


def clean_method_name(value: object) -> object:
    if not isinstance(value, str):
        return value
    replacements = {
        "Phase 1 + recalibration": "Baseline + recalibration",
        "Phase 2 + recalibration": "Proposed framework + recalibration",
        "Phase 1": "Reconstruction baseline",
        "Phase 2": "Proposed framework",
        "RCC-ConvAE": "Proposed framework",
    }
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def add_history(filename: str, result: dict) -> None:
    history = st.session_state.setdefault("inspection_history", [])
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "decision": result.get("decision", "-"),
            "global_score": result.get("global_score"),
            "local_score": result.get("local_score"),
            "p_value": result.get("p_value"),
        },
    )
    del history[50:]


p_baseline, baseline_path = load_json(
    ROOT / "results" / "phase1" / "metrics.json",
    ROOT / "results" / "reference_run" / "phase1_metrics.json",
)
p_proposed, proposed_path = load_json(
    ROOT / "results" / "phase2" / "metrics.json",
    ROOT / "results" / "reference_run" / "phase2_metrics.json",
)

st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-title">ðŸ”¬ SemiVision AI</div>
      <div class="brand-sub">Semiconductor Visual Anomaly Screening<br>Risk-calibrated â€¢ Cost-aware â€¢ Drift-evaluated</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Workspace",
    [
        "ðŸ”Ž SEM Inspection",
        "ðŸ—‚ Batch Screening",
        "ðŸ“Š Research Metrics",
        "ðŸŒ Distribution Robustness",
        "ðŸ§ª Evaluation Study",
        "âš™ï¸ Model & Policy",
        "ðŸ•˜ History",
        "ðŸ“ Reports",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Final unified research application")
st.sidebar.caption("Carinthia-S SEM visual screening")


if page == "ðŸ”Ž SEM Inspection":
    page_header(
        "SEM Inspection",
        "Upload a semiconductor SEM image for automated anomaly screening and localized visual evidence.",
    )

    with st.container(border=True):
        st.markdown('<div class="section-title">Image Screening</div>', unsafe_allow_html=True)
        upload = st.file_uploader(
            "Drag and drop a SEM image or browse from your computer",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="single_upload",
        )
        left, middle, right = st.columns([1, 1, 3])
        with left:
            analyze = st.button("Analyze Image", type="primary", use_container_width=True)
        with middle:
            clear = st.button("Clear", use_container_width=True)
        with right:
            st.markdown(
                '<div class="small-note">The deployed workflow automatically applies the complete proposed anomaly-screening policy.</div>',
                unsafe_allow_html=True,
            )

    if clear:
        for key in ["single_result", "single_image", "single_filename"]:
            st.session_state.pop(key, None)
        st.rerun()

    if analyze:
        if upload is None:
            st.warning("Please upload a SEM image before selecting Analyze Image.")
        else:
            try:
                image = Image.open(upload).convert("L")
                arr = np.asarray(image, dtype=np.float32) / 255.0
                with st.spinner("Analyzing SEM structure and anomaly evidence..."):
                    result = final_engine().predict_array(arr)
                st.session_state["single_result"] = result
                st.session_state["single_image"] = np.asarray(image)
                st.session_state["single_filename"] = upload.name
                add_history(upload.name, result)
            except Exception as exc:
                st.error(f"Screening could not be completed: {exc}")

    result = st.session_state.get("single_result")
    input_image = st.session_state.get("single_image")
    if result is not None and input_image is not None:
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
        if result["decision"] == "ANOMALY":
            st.markdown('<div class="result-anomaly">Prediction: ANOMALY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-reference">Prediction: REFERENCE</div>', unsafe_allow_html=True)

        metric_row(
            [
                ("Global anomaly score", f"{result['global_score']:.6g}"),
                ("Localized evidence", f"{result.get('local_score', float('nan')):.6g}"),
                ("Conformal p-value", f"{result.get('p_value', float('nan')):.4f}"),
                ("Risk threshold", f"{result.get('alpha', result.get('threshold', float('nan'))):.4f}"),
            ]
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.image(input_image, caption="Input SEM image", use_container_width=True, clamp=True)
        c2.image(result["reconstruction"], caption="Reconstruction", use_container_width=True, clamp=True)
        c3.image(result["error_map"], caption="Anomaly evidence map", use_container_width=True, clamp=True)
        c4.image(result["predicted_mask"] * 255, caption="Localized anomaly mask", use_container_width=True, clamp=True)

        with st.expander("Advanced screening evidence"):
            st.write(f"Fused evidence score: {result.get('fused_score', float('nan')):.6f}")
            st.write(
                "Decision rule: a sample is flagged as ANOMALY when its conformal p-value is at or below the selected risk threshold."
            )


elif page == "ðŸ—‚ Batch Screening":
    page_header(
        "Batch Screening",
        "Screen multiple SEM images using the same deployed decision policy and export the results.",
    )
    with st.container(border=True):
        files = st.file_uploader(
            "Upload SEM images",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
            key="batch_upload",
        )
        run_batch = st.button("Run Batch Screening", type="primary", disabled=not files)

    if run_batch and files:
        rows = []
        engine = final_engine()
        progress = st.progress(0)
        status = st.empty()
        for idx, file in enumerate(files, start=1):
            status.write(f"Screening {idx}/{len(files)}: {file.name}")
            image = Image.open(file).convert("L")
            arr = np.asarray(image, dtype=np.float32) / 255.0
            result = engine.predict_array(arr)
            rows.append(
                {
                    "filename": file.name,
                    "decision": result["decision"],
                    "global_score": result["global_score"],
                    "local_score": result.get("local_score"),
                    "fused_score": result.get("fused_score"),
                    "conformal_p_value": result.get("p_value"),
                }
            )
            add_history(file.name, result)
            progress.progress(idx / len(files))
        status.empty()
        st.session_state["batch_results"] = pd.DataFrame(rows)

    batch_df = st.session_state.get("batch_results")
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
        metric_row(
            [
                ("Images screened", str(len(batch_df))),
                ("Reference", str(int((batch_df["decision"] == "REFERENCE").sum()))),
                ("Anomaly", str(int((batch_df["decision"] == "ANOMALY").sum()))),
                ("Anomaly rate", f"{100 * (batch_df['decision'] == 'ANOMALY').mean():.1f}%"),
            ]
        )
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Batch Results",
            batch_df.to_csv(index=False),
            file_name="semivision_batch_screening.csv",
            mime="text/csv",
        )


elif page == "ðŸ“Š Research Metrics":
    page_header(
        "Research Metrics",
        "Final test-set performance of the reconstruction baseline and the proposed screening framework.",
    )
    if not p_baseline or not p_proposed:
        st.warning("Research metric files are not available yet.")
    else:
        base_loc = p_baseline.get("localization_exploratory", {})
        prop_loc = p_proposed.get("localization", {})
        comparison = pd.DataFrame(
            [
                {
                    "Method": "Reconstruction baseline",
                    "ROC-AUC": p_baseline.get("roc_auc"),
                    "PR-AUC": p_baseline.get("pr_auc"),
                    "Accuracy": p_baseline.get("accuracy"),
                    "Balanced Accuracy": p_baseline.get("balanced_accuracy"),
                    "F1": p_baseline.get("f1"),
                    "FPR": p_baseline.get("fpr"),
                    "FNR": p_baseline.get("fnr"),
                    "Cost / image": p_baseline.get("cost_per_image"),
                    "Pixel AUROC": base_loc.get("pixel_auroc"),
                    "Dice": base_loc.get("dice"),
                },
                {
                    "Method": "Proposed framework",
                    "ROC-AUC": p_proposed.get("roc_auc"),
                    "PR-AUC": p_proposed.get("pr_auc"),
                    "Accuracy": p_proposed.get("accuracy"),
                    "Balanced Accuracy": p_proposed.get("balanced_accuracy"),
                    "F1": p_proposed.get("f1"),
                    "FPR": p_proposed.get("fpr"),
                    "FNR": p_proposed.get("fnr"),
                    "Cost / image": p_proposed.get("cost_per_image"),
                    "Pixel AUROC": prop_loc.get("pixel_auroc"),
                    "Dice": prop_loc.get("dice"),
                },
            ]
        )
        metric_row(
            [
                ("ROC-AUC", f"{p_proposed.get('roc_auc', float('nan')):.4f}"),
                ("F1", f"{p_proposed.get('f1', float('nan')):.4f}"),
                ("Balanced accuracy", f"{p_proposed.get('balanced_accuracy', float('nan')):.4f}"),
                ("FPR", f"{p_proposed.get('fpr', float('nan')):.4f}"),
                ("Cost / image", f"{p_proposed.get('cost_per_image', float('nan')):.6f}"),
            ]
        )
        st.dataframe(comparison.style.format(precision=4), use_container_width=True, hide_index=True)
        chart_df = comparison.melt(
            id_vars="Method",
            value_vars=["ROC-AUC", "PR-AUC", "Balanced Accuracy", "F1"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(chart_df, x="Metric", y="Score", color="Method", barmode="group", range_y=[0, 1.02])
        st.plotly_chart(fig, use_container_width=True)


elif page == "ðŸŒ Distribution Robustness":
    page_header(
        "Distribution Robustness",
        "Evaluate stability under controlled brightness, contrast, gamma, noise, and blur shifts.",
    )
    drift_df, _ = load_csv(
        ROOT / "results" / "drift" / "drift_metrics.csv",
        ROOT / "results" / "reference_run" / "drift_conditions.csv",
    )
    if drift_df is None:
        st.warning("Distribution-shift results are not available yet.")
    else:
        d = drift_df.copy()
        if "method" in d.columns:
            d["method"] = d["method"].map(clean_method_name)
            fig = px.line(d, x="condition", y="balanced_accuracy", color="method", markers=True)
            fig.update_yaxes(range=[0, 1.02])
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(d, use_container_width=True, hide_index=True)


elif page == "ðŸ§ª Evaluation Study":
    page_header(
        "Evaluation Study",
        "Ablation analysis of reconstruction evidence, localized evidence, risk calibration, and cost-aware decision selection.",
    )
    ablation_df, _ = load_csv(
        ROOT / "results" / "ablation" / "ablation.csv",
        ROOT / "results" / "reference_run" / "ablation.csv",
    )
    if ablation_df is None:
        st.warning("Ablation results are not available yet.")
    else:
        a = ablation_df.copy()
        first_col = a.columns[0]
        a[first_col] = a[first_col].map(clean_method_name)
        st.dataframe(a, use_container_width=True, hide_index=True)
        metric_cols = [c for c in ["balanced_accuracy", "f1", "fpr", "fnr", "cost_per_image"] if c in a.columns]
        if metric_cols:
            long_df = a.melt(id_vars=first_col, value_vars=metric_cols, var_name="Metric", value_name="Value")
            fig = px.bar(long_df, x=first_col, y="Value", color="Metric", barmode="group")
            st.plotly_chart(fig, use_container_width=True)


elif page == "âš™ï¸ Model & Policy":
    page_header(
        "Model & Policy",
        "Configuration of the final deployed anomaly-screening framework.",
    )
    model_ok = (ROOT / "models" / "convae_best.pt").exists()
    policy_ok = (ROOT / "models" / "phase2" / "rcc_policy.json").exists()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model checkpoint", "Ready" if model_ok else "Missing")
    c2.metric("Decision policy", "Ready" if policy_ok else "Missing")
    c3.metric("Input resolution", f"{cfg['data']['input_size']} Ã— {cfg['data']['input_size']}")
    c4.metric("Reference seed", str(cfg["project"]["seed"]))

    with st.container(border=True):
        st.markdown("#### Final framework flow")
        st.markdown(
            "**SEM image â†’ 2-D convolutional autoencoder â†’ reconstruction-error evidence â†’ global/local scoring â†’ "
            "risk calibration â†’ cost-aware decision â†’ REFERENCE / ANOMALY â†’ localized visual evidence**"
        )

    if p_proposed:
        metric_row(
            [
                ("Selected top-k", str(p_proposed.get("topk", "-"))),
                ("Fusion weight", str(p_proposed.get("lambda", "-"))),
                ("Risk level", str(p_proposed.get("alpha", "-"))),
                ("FN : FP cost", f"{cfg['phase2']['fn_cost']:.0f} : {cfg['phase2']['fp_cost']:.0f}"),
            ]
        )

    st.markdown("#### Ground-truth definition")
    st.info(
        "Reference/normal candidates are defined by empty segmentation masks. Non-empty masks define anomaly/defect samples and provide pixel-level localization ground truth."
    )


elif page == "ðŸ•˜ History":
    page_header("Inspection History", "Recent screening decisions from the current dashboard session.")
    history = st.session_state.get("inspection_history", [])
    if not history:
        st.info("No images have been screened in this session yet.")
    else:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download History",
            history_df.to_csv(index=False),
            file_name="semivision_history.csv",
            mime="text/csv",
        )
        if st.button("Clear History"):
            st.session_state["inspection_history"] = []
            st.rerun()


else:
    page_header("Reports", "Open or export the final research result tables produced by the completed pipeline.")
    report_items = [
        ("Research metrics", ROOT / "reports" / "research_metrics.csv"),
        ("Ablation study", ROOT / "results" / "ablation" / "ablation.csv"),
        ("Distribution robustness", ROOT / "results" / "drift" / "drift_metrics.csv"),
        ("Final baseline metrics", ROOT / "results" / "phase1" / "metrics.json"),
        ("Final proposed metrics", ROOT / "results" / "phase2" / "metrics.json"),
    ]
    available = 0
    for label, path in report_items:
        if path.exists():
            available += 1
            data = path.read_bytes()
            st.download_button(
                f"Download {label}",
                data,
                file_name=path.name,
                mime="application/octet-stream",
                key=str(path),
            )
    if available == 0:
        st.info("No generated report files were found in the current project directory.")

