from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import ScreeningEngine
from app.dashboard_utils import load_json

logger = logging.getLogger(__name__)
cfg = load_config()

st.set_page_config(
    page_title="SemiVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #f4f7fb !important;
        color: #243b53;
    }
    header[data-testid="stHeader"] {
        background: #f4f7fb !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label {
        color: #243b53;
    }
    .block-container {
        max-width: 1380px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 14rem !important;
        min-width: 14rem !important;
        max-width: 14rem !important;
        background: #091b2b;
        border-right: 1px solid #17364e;
    }
    [data-testid="stSidebar"] * {
        color: #e8f1f8;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [role="radiogroup"] label span {
        color: #b3c5d2 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span {
        color: #f1f8fc !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 1.25rem .8rem 1rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .55rem .65rem;
        border-radius: 9px;
        margin-bottom: .25rem;
        font-size: .84rem;
        line-height: 1.3;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #123e59;
        box-shadow: inset 3px 0 #48c6d9;
    }
    .brand {
        padding: .15rem .1rem 1rem .1rem;
        margin-bottom: .8rem;
        border-bottom: 1px solid #1b3446;
    }
    .brand-mark {
        display: inline-flex;
        width: 2rem;
        height: 2rem;
        align-items: center;
        justify-content: center;
        margin-bottom: .65rem;
        border: 1px solid #28627b;
        border-radius: 10px;
        background: #10354a;
        color: #75e0e9;
        font-size: 1rem;
    }
    .brand-title {
        font-size: 1.08rem;
        font-weight: 800;
        letter-spacing: .2px;
    }
    .brand-sub {
        color: #b3c5d2 !important;
        font-size: .7rem;
        line-height: 1.4;
        margin-top: .3rem;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: .74rem;
        line-height: 1.35;
    }
    @media (max-width: 640px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: min(14rem, 85vw) !important;
            min-width: min(14rem, 85vw) !important;
            max-width: min(14rem, 85vw) !important;
        }
        .block-container {
            padding: 1rem 1rem 2rem;
        }
        .page-title {
            font-size: 1.65rem;
        }
    }
    .page-title {
        text-align: left;
        font-size: 1.9rem;
        font-weight: 800;
        color: #18334a;
        letter-spacing: -.035em;
        margin: .1rem 0 .2rem;
    }
    .page-subtitle {
        color: #718096;
        font-size: .92rem;
        margin-bottom: 1.25rem;
    }
    .page-eyebrow {
        color: #16829a;
        font-size: .69rem;
        font-weight: 800;
        letter-spacing: .13em;
        margin-bottom: .35rem;
        text-transform: uppercase;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e1e8ef !important;
        border-radius: 14px !important;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(25, 50, 72, .035);
    }
    [data-testid="stMetric"] {
        padding: .9rem 1rem;
        border: 1px solid #e1e8ef;
        border-radius: 12px;
        background: #ffffff;
    }
    [data-testid="stMetricLabel"] {
        color: #718096 !important;
        font-size: .75rem;
    }
    [data-testid="stMetricValue"] {
        color: #18334a !important;
        font-size: 1.45rem;
        font-weight: 750;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #9ab5c6;
        border-radius: 12px;
        background: #f8fbfd;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #16829a;
        background: #f2fafb;
    }
    div.stButton > button[kind="primary"] {
        border: 1px solid #087e8b;
        background: #087e8b;
        color: #ffffff !important;
        box-shadow: 0 3px 9px rgba(8, 126, 139, .16);
    }
    div.stButton > button p {
        color: inherit !important;
    }
    div.stButton > button[kind="primary"]:hover {
        border-color: #066873;
        background: #066873;
        color: #ffffff !important;
    }
    div.stButton > button:not([kind="primary"]),
    [data-testid="stFileUploader"] button {
        border: 1px solid #d7e1e8;
        background: #ffffff;
        color: #344b5e !important;
    }
    div.stButton > button:not([kind="primary"]):hover,
    [data-testid="stFileUploader"] button:hover {
        border-color: #9ab5c6;
        background: #f4f8fa;
        color: #18334a !important;
    }
    .result-anomaly, .result-reference {
        border-radius: 12px;
        padding: .8rem 1rem;
        text-align: left;
        font-size: 1rem;
        font-weight: 800;
        margin: .5rem 0 .85rem;
    }
    .result-anomaly {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #a51632;
    }
    .result-reference {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #087251;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden;
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
    st.markdown('<div class="page-eyebrow">SEM defect intelligence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


@st.cache_resource
def final_engine() -> ScreeningEngine:
    return ScreeningEngine(ROOT, cfg, "phase2")


def clear_inspection() -> None:
    for key in ("single_result", "single_image", "single_filename"):
        st.session_state.pop(key, None)
    st.session_state["single_uploader_version"] = (
        st.session_state.get("single_uploader_version", 0) + 1
    )


def load_uploaded_image(upload) -> tuple[Image.Image, np.ndarray]:
    with Image.open(upload) as source:
        image = source.convert("L")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return image.copy(), pixels


def format_metric(metrics: dict, key: str, digits: int = 4) -> str:
    value = metrics.get(key)
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}"


phase1_metrics, _ = load_json(ROOT / "results" / "phase1" / "metrics.json")
phase2_metrics, _ = load_json(ROOT / "results" / "phase2" / "metrics.json")

st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-mark">✳</div>
      <div class="brand-title">SemiVision AI</div>
      <div class="brand-sub">
        Semiconductor visual screening<br>
        Risk-calibrated · Cost-aware
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Workspace",
    [
        "SEM Inspection",
        "Batch Screening",
        "Performance Summary",
        "About the Framework",
    ],
    index=0,
    key="final_panel_navigation",
    label_visibility="collapsed",
)


if page == "SEM Inspection":
    page_header(
        "SEM Inspection",
        "Upload a semiconductor SEM image for automated anomaly screening.",
    )

    with st.container(border=True):
        st.subheader("Image Screening")
        uploader_key = f"single_upload_{st.session_state.get('single_uploader_version', 0)}"
        upload = st.file_uploader(
            "Upload an SEM image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key=uploader_key,
            help="Choose a file or drag it into the upload area. Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP (maximum 200 MB).",
        )
        st.caption(
            "Click Upload or drag an image into this area · PNG, JPG, TIFF, or BMP · "
            "Maximum 200 MB"
        )
        analyze_col, clear_col, _ = st.columns([1.15, 0.8, 4])
        analyze = analyze_col.button(
            "Analyze Image",
            type="primary",
            use_container_width=True,
        )
        clear_col.button(
            "Clear",
            on_click=clear_inspection,
            use_container_width=True,
        )

    if analyze:
        for key in ("single_result", "single_image", "single_filename"):
            st.session_state.pop(key, None)
        if upload is None:
            st.warning("Upload an SEM image before selecting Analyze Image.")
        else:
            try:
                image, pixels = load_uploaded_image(upload)
                with st.spinner("Analyzing image with the final screening framework..."):
                    result = final_engine().predict_array(pixels)
                st.session_state["single_result"] = result
                st.session_state["single_image"] = np.asarray(image)
                st.session_state["single_filename"] = upload.name
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                logger.exception("Single-image screening failed for %s", upload.name)
                st.error(
                    "The image could not be screened. Check that the file is a valid, "
                    "readable image and try again."
                )

    result = st.session_state.get("single_result")
    input_image = st.session_state.get("single_image")
    if result is not None and input_image is not None:
        st.subheader("Analysis Result")
        decision = result.get("decision")
        if decision == "ANOMALY":
            st.markdown(
                '<div class="result-anomaly">⚠ &nbsp; Anomaly detected <span style="font-weight:500">· Review recommended</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-reference">✓ &nbsp; Reference structure detected <span style="font-weight:500">· No anomaly flagged</span></div>',
                unsafe_allow_html=True,
            )

        alpha = result.get("alpha", result.get("threshold"))
        metric_row(
            [
                ("Fused anomaly score", format_metric(result, "fused_score", 5)),
                ("Conformal p-value", format_metric(result, "p_value", 5)),
                ("Decision threshold", format_metric({"threshold": alpha}, "threshold", 5)),
            ]
        )

        image_tab, evidence_tab = st.tabs(["Image comparison", "Anomaly evidence"])
        with image_tab:
            input_col, reconstruction_col = st.columns(2)
            input_col.image(input_image, caption="Uploaded SEM image", use_container_width=True)
            reconstruction_col.image(
                result["reconstruction"],
                caption="Model reconstruction",
                use_container_width=True,
                clamp=True,
            )
        with evidence_tab:
            evidence_col, mask_col = st.columns(2)
            evidence_col.image(
                result["error_map"],
                caption="Reconstruction error",
                use_container_width=True,
                clamp=True,
            )
            mask_col.image(
                result["predicted_mask"] * 255,
                caption="Localized anomaly mask",
                use_container_width=True,
                clamp=True,
            )

        with st.expander("Technical Details"):
            st.write(f"Global anomaly score: {float(result['global_score']):.6g}")
            st.write(f"Local anomaly evidence: {float(result.get('local_score', float('nan'))):.6g}")
            st.write(f"Fused anomaly score: {float(result.get('fused_score', float('nan'))):.6g}")
            st.write(f"Conformal p-value: {float(result.get('p_value', float('nan'))):.6g}")
            st.write(f"Decision threshold / alpha: {float(alpha):.6g}")
            st.write(f"Final decision: {decision}")


elif page == "Batch Screening":
    page_header(
        "Batch Screening",
        "Screen multiple SEM images with the final framework and export a results file.",
    )
    with st.container(border=True):
        files = st.file_uploader(
            "Upload SEM images",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
            key="batch_upload",
            help="Choose one or more files or drag them into the upload area. Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP (maximum 200 MB per file).",
        )
        st.caption(
            "Select multiple images or drag them here · PNG, JPG, TIFF, or BMP · "
            "Maximum 200 MB per file"
        )
        run_batch = st.button(
            "Run Batch Screening",
            type="primary",
            disabled=not files,
        )

    if run_batch and files:
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_failures", None)
        rows = []
        failures = []
        progress = st.progress(0)
        status = st.empty()
        try:
            engine = final_engine()
            for index, file in enumerate(files, start=1):
                status.write(f"Screening image {index} of {len(files)}")
                try:
                    _, pixels = load_uploaded_image(file)
                    result = engine.predict_array(pixels)
                    alpha = float(result.get("alpha", result.get("threshold", float("nan"))))
                    p_value = float(result.get("p_value", float("nan")))
                    rows.append(
                        {
                            "Filename": file.name,
                            "Prediction": result["decision"],
                            "Anomaly Score": result["global_score"],
                            "Risk / p-value": f"{alpha:.4g} / {p_value:.4g}",
                        }
                    )
                except (UnidentifiedImageError, OSError, ValueError, RuntimeError, KeyError) as exc:
                    logger.exception("Batch screening failed for %s", file.name)
                    failures.append(file.name)
                    rows.append(
                        {
                            "Filename": file.name,
                            "Prediction": "ERROR",
                            "Anomaly Score": None,
                            "Risk / p-value": "N/A",
                        }
                    )
                progress.progress(index / len(files))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Unable to initialize batch screening")
            st.error("Batch screening could not start. Please try again.")
        else:
            st.session_state["batch_results"] = pd.DataFrame(
                rows,
                columns=["Filename", "Prediction", "Anomaly Score", "Risk / p-value"],
            )
            st.session_state["batch_failures"] = failures
        finally:
            status.empty()

    batch_df = st.session_state.get("batch_results")
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
        prediction = batch_df["Prediction"]
        reference_count = int((prediction == "REFERENCE").sum())
        anomaly_count = int((prediction == "ANOMALY").sum())
        metric_row(
            [
                ("Total Images", str(len(batch_df))),
                ("Reference", str(reference_count)),
                ("Anomaly", str(anomaly_count)),
            ]
        )
        failures = st.session_state.get("batch_failures", [])
        if failures:
            st.warning(
                "Some files could not be screened: " + ", ".join(failures)
            )
        filter_col, search_col = st.columns([1, 2])
        decision_filter = filter_col.selectbox(
            "Filter results",
            ["All results", "ANOMALY", "REFERENCE", "ERROR"],
            key="batch_decision_filter",
        )
        filename_query = search_col.text_input(
            "Find a filename",
            placeholder="Search uploaded images",
            key="batch_filename_search",
        )
        visible_results = batch_df
        if decision_filter != "All results":
            visible_results = visible_results[
                visible_results["Prediction"] == decision_filter
            ]
        if filename_query.strip():
            visible_results = visible_results[
                visible_results["Filename"].str.contains(
                    filename_query.strip(), case=False, na=False, regex=False
                )
            ]
        st.caption(f"Showing {len(visible_results)} of {len(batch_df)} screened images")
        st.dataframe(visible_results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Batch Results",
            batch_df.to_csv(index=False),
            file_name="semivision_batch_screening.csv",
            mime="text/csv",
        )


elif page == "Performance Summary":
    page_header(
        "Performance Summary",
        "Reported test-set metrics from the saved experimental results.",
    )
    if phase2_metrics is None:
        st.warning("Final framework metric results are not available.")
    else:
        metric_row(
            [
                ("ROC-AUC", format_metric(phase2_metrics, "roc_auc")),
                ("PR-AUC", format_metric(phase2_metrics, "pr_auc")),
                ("F1 Score", format_metric(phase2_metrics, "f1")),
                ("Balanced Accuracy", format_metric(phase2_metrics, "balanced_accuracy")),
            ]
        )
        metric_row(
            [
                ("False Positive Rate", format_metric(phase2_metrics, "fpr")),
                ("False Negative Rate", format_metric(phase2_metrics, "fnr")),
                ("Cost per Image", format_metric(phase2_metrics, "cost_per_image", 6)),
            ]
        )

        if phase1_metrics is not None:
            comparison = pd.DataFrame(
                [
                    {
                        "Method": "Reconstruction Baseline",
                        "Balanced Accuracy": format_metric(phase1_metrics, "balanced_accuracy"),
                        "FPR": format_metric(phase1_metrics, "fpr"),
                        "FNR": format_metric(phase1_metrics, "fnr"),
                        "Cost / Image": format_metric(phase1_metrics, "cost_per_image", 6),
                    },
                    {
                        "Method": "Proposed Framework",
                        "Balanced Accuracy": format_metric(phase2_metrics, "balanced_accuracy"),
                        "FPR": format_metric(phase2_metrics, "fpr"),
                        "FNR": format_metric(phase2_metrics, "fnr"),
                        "Cost / Image": format_metric(phase2_metrics, "cost_per_image", 6),
                    },
                ]
            )
            st.caption("Optional comparison with the reconstruction baseline.")
            st.dataframe(comparison, use_container_width=True, hide_index=True)


else:
    page_header(
        "About the Framework",
        "A unified visual screening workflow for semiconductor manufacturing.",
    )
    st.markdown(
        """
        SemiVision AI is an unsupervised semiconductor SEM visual anomaly-screening
        framework that learns reference semiconductor structure using a 2-D
        convolutional autoencoder.

        The proposed RCC-ConvAE is a risk-calibrated and cost-aware decision
        framework built around a 2-D ConvAE backbone.
        """
    )
    st.subheader("Screening architecture")
    flow_steps = [
        "SEM Image",
        "2-D ConvAE",
        "Reconstruction",
        "Reconstruction Error",
        "Global + Local Evidence",
        "Risk Calibration",
        "Cost-Aware Decision",
        "REFERENCE / ANOMALY",
        "Localized Defect Evidence",
    ]
    for offset, width in ((0, 4), (4, 4)):
        step_columns = st.columns(4)
        for column, step in zip(step_columns, flow_steps[offset : offset + width]):
            column.info(step)
    st.info(flow_steps[-1])
    st.markdown(
        """
        The framework combines reconstruction-based anomaly detection, global
        and localized evidence, conformal risk calibration, cost-aware decision
        making, localized defect visualization, and robustness evaluation under
        distribution drift.
        """
    )
    with st.expander("Methodology and dataset"):
        st.write(
            "The experiments use the Carinthia-S semiconductor image dataset. "
            "The autoencoder learns reference structure; anomaly decisions and "
            "visual evidence are produced by the calibrated screening workflow."
        )
