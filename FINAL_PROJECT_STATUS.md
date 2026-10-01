# Packaged Project Status

## Included and verified in this ZIP
- Complete Phase-1 2-D ConvAE source pipeline
- Complete Phase-2 RCC-ConvAE source pipeline
- No-test-leakage experiment design
- Evaluation metric implementation
- Localization evaluation implementation
- Ablation and drift scripts
- Archived reference-run metrics with explicit provenance
- Streamlit dashboard source with research/results pages and live inference hooks
- Windows PowerShell run instructions
- Unit/integrity tests: 7 passed in the packaging environment
- IEEE-style paper in DOCX, PDF, and Markdown
- Architecture and result figures
- Reproducible ZIP packaging script

## Not fabricated
The raw Carinthia-S dataset and the trained `.pt` checkpoint were not attached to the packaging conversation. They are therefore not fabricated. Place the official dataset in the documented directory and run the pipeline to generate fresh model/policy/result artifacts.

## Dashboard QA note
The dashboard Python source passes compilation/static checks. The artifact-building container did not have Streamlit installed and had no network access to install it, so a real server-start smoke test could not be performed here. `requirements.txt` includes Streamlit and the Windows run instructions install it before launch.
