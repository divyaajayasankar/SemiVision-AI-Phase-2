from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ROOT
from src.data import verify_dataset, create_split_manifest, leakage_report
from src.utils import write_json

cfg=load_config(); data_root=ROOT/cfg["data"]["root"]
out=ROOT/"results"/"dataset"; out.mkdir(parents=True,exist_ok=True)
meta, report=verify_dataset(data_root,cfg["data"]["csv"])
manifest=create_split_manifest(meta,cfg); leak=leakage_report(manifest)
meta.to_csv(out/"image_metadata.csv",index=False); manifest.to_csv(out/"split_manifest.csv",index=False)
write_json(out/"verification.json",report); write_json(out/"leakage_report.json",leak)
print(json.dumps(report,indent=2)); print("Leakage check:","PASS" if leak["passed"] else "FAIL")
if not leak["passed"]: raise SystemExit(2)
