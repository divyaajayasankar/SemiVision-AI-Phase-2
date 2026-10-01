from pathlib import Path
import re

p = Path(r".\app\streamlit_app.py")
s = p.read_text(encoding="utf-8-sig", errors="replace")

# Replace the visible sidebar menu with only 4 final-project pages.
s = re.sub(
    r'page\s*=\s*st\.sidebar\.radio\(\s*"Workspace",\s*\[.*?\],\s*.*?label_visibility="collapsed",\s*\)',
    '''page = st.sidebar.radio(
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
)''',
    s,
    count=1,
    flags=re.S
)

# Rename Research Metrics page to Performance Summary.
s = s.replace(
    'elif page == "Research Metrics":',
    'elif page == "Performance Summary":'
)

s = s.replace(
    '"Research Metrics",',
    '"Performance Summary",',
)

# Reuse Model & Policy as the final framework explanation page.
s = s.replace(
    'elif page == "Model & Policy":',
    'elif page == "About the Framework":'
)

s = s.replace(
    'page_header("Model & Policy"',
    'page_header("About the Framework"'
)

p.write_text(s, encoding="utf-8", newline="\n")

print("FINAL FRONTEND MENU UPDATED")
