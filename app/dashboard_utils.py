from pathlib import Path
import json
import pandas as pd


def load_json(primary: Path, fallback: Path | None = None):
    path=primary if primary.exists() else fallback
    if path and path.exists(): return json.loads(path.read_text(encoding="utf-8")), path
    return None, None


def load_csv(primary: Path, fallback: Path | None = None):
    path=primary if primary.exists() else fallback
    if path and path.exists(): return pd.read_csv(path), path
    return None, None


def source_badge(path: Path | None) -> str:
    if path is None: return "No results available"
    return "Recomputed by this package" if "reference_run" not in str(path) else "Historical verified reference run"
