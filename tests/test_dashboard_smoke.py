from pathlib import Path


def test_dashboard_exists():
    root = Path(__file__).resolve().parents[1]
    dashboard = root / "app" / "streamlit_app.py"

    assert dashboard.exists(), "Streamlit dashboard file is missing."

    text = dashboard.read_text(encoding="utf-8-sig")

    for page in (
        "SemiVision AI",
        "SEM Inspection",
        "Batch Screening",
        "Performance Summary",
        "About the Framework",
    ):
        assert page in text

    assert "Reconstruction Baseline" in text
    assert "Proposed Framework" in text
