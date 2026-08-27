#!/usr/bin/env python3
from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parents[1]
m=json.loads((root/"reviewer_evidence/SPLIT_FILE_MANIFEST.json").read_text(encoding="utf-8"))
for e in m["files"]:
    out=root/e["original_path"]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("wb") as f:
        for part in e["parts"]:
            f.write((root/part).read_bytes())
    h=hashlib.sha256(out.read_bytes()).hexdigest()
    ok=(h==e["sha256"])
    print(("OK " if ok else "FAIL ")+e["original_path"]+"  "+h)
    if not ok: raise SystemExit(1)
