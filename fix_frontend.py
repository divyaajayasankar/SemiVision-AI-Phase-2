from pathlib import Path
import re

p = Path(r".\app\streamlit_app.py")
s = p.read_text(encoding="utf-8-sig", errors="replace")

# Keep UI source ASCII-safe to prevent Windows encoding corruption.
s = s.encode("ascii", "ignore").decode("ascii")

# Remove problematic page icon.
s = re.sub(
    r'page_icon="[^"]*"',
    'page_icon=None',
    s,
    count=1
)

# Give the page more top spacing.
s = s.replace(
    'padding-top: 1.35rem;',
    'padding-top: 2.6rem;',
    1
)

# Hide Streamlit deployment/menu controls for the final demo.
old_css = '''    div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
    }
'''

new_css = '''    div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden;
    }
'''

if old_css in s:
    s = s.replace(old_css, new_css, 1)

# Replace sidebar branding with clean final-project wording.
s = re.sub(
    r'st\.sidebar\.markdown\(\s*"""\s*<div class="brand">.*?</div>\s*""",\s*unsafe_allow_html=True,\s*\)',
    '''st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-title">SemiVision AI</div>
      <div class="brand-sub">
        Semiconductor Visual Anomaly Screening<br>
        Risk-Calibrated | Cost-Aware | Drift-Evaluated
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)''',
    s,
    count=1,
    flags=re.S,
)

# Replace the full sidebar navigation.
s = re.sub(
    r'page = st\.sidebar\.radio\(\s*"Workspace",\s*\[.*?\],\s*label_visibility="collapsed",\s*\)',
    '''page = st.sidebar.radio(
    "Workspace",
    [
        "SEM Inspection",
        "Batch Screening",
        "Research Metrics",
        "Distribution Robustness",
        "Evaluation Study",
        "Model & Policy",
        "History",
        "Reports",
    ],
    index=0,
    key="final_workspace_navigation_v2",
    label_visibility="collapsed",
)''',
    s,
    count=1,
    flags=re.S,
)

# Fix all page comparisons.
fixes = [
    (r'if page == "[^"]*SEM Inspection":',
     'if page == "SEM Inspection":'),

    (r'elif page == "[^"]*Batch Screening":',
     'elif page == "Batch Screening":'),

    (r'elif page == "[^"]*Research Metrics":',
     'elif page == "Research Metrics":'),

    (r'elif page == "[^"]*Distribution Robustness":',
     'elif page == "Distribution Robustness":'),

    (r'elif page == "[^"]*Evaluation Study":',
     'elif page == "Evaluation Study":'),

    (r'elif page == "[^"]*Model & Policy":',
     'elif page == "Model & Policy":'),

    (r'elif page == "[^"]*History":',
     'elif page == "History":'),
]

for pattern, replacement in fixes:
    s = re.sub(pattern, replacement, s)

# Replace special arrows with safe text where needed.
s = s.replace(
    '2-D convolutional autoencoder  reconstruction-error evidence  global/local scoring  risk calibration  cost-aware decision  REFERENCE / ANOMALY  localized visual evidence',
    '2-D convolutional autoencoder -> reconstruction-error evidence -> global/local scoring -> risk calibration -> cost-aware decision -> REFERENCE / ANOMALY -> localized visual evidence'
)

p.write_text(s, encoding="utf-8", newline="\n")

print("Frontend cleaned successfully.")
print("Updated:", p)
