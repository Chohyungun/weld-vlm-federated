import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
for p in sorted(Path("outputs/probe_c").glob("_leg_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    keep = {k: d.get(k) for k in ("leg", "profile", "wall_s", "step_s_p50", "step_s_mean",
                                  "wait_ratio", "peak_vram_gb_torch", "n_steps_total")}
    print(json.dumps(keep, ensure_ascii=False))
    print("  gpu", json.dumps(d.get("gpu", {}), ensure_ascii=False))
