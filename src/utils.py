from __future__ import annotations
from pathlib import Path
import json
import hashlib
import platform
import subprocess
import sys
import torch


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    if not path.exists(): return None
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path):
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_info(root: Path):
    return {"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"device":"cuda" if torch.cuda.is_available() else "cpu","git_commit":git_commit(root)}
