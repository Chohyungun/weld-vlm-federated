import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("outputs/probe_c/extrapolation.json", encoding="utf-8"))
for k in ("출처", "판정11", "통합형_크기곡선", "통합형"):
    if k in d:
        print(k, json.dumps(d[k], ensure_ascii=False, indent=1))
print("검출", json.dumps(d["검출"], ensure_ascii=False, indent=1))
