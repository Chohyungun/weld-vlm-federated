"""승격 어블레이션 — 검출 3칸을 두 표본에 각각 돌린다 (과제 3).

## 무엇을 가르려는 것인가

함정 #11(결함/정상 이미지 규격 지름길)의 대응으로 **크롭 한정본**이 제안됐다. 그런데
크롭 한정은 출처 축을 없애는 동시에 **표본을 39.5% 줄인다.** 지표가 움직였을 때 그것이
지름길이 사라진 효과인지 데이터가 줄어든 효과인지 구분되지 않는다.

그래서 팔이 둘이다.

| 팔 | 스냅샷 | 성격 |
|---|---|---|
| 크롭 한정 | `..._crop_only` | 출처를 N-crop 으로 한정 |
| 규모 대조 | `..._scale_control` | **같은 크기**로 줄이되 출처 구성은 원본 비율 유지 |

두 팔의 델타를 원본 파일럿과 각각 비교해야 두 효과가 분리된다. 규모 대조 팔이 없으면
"줄였더니 떨어졌다"를 "지름길을 없앴더니 떨어졌다"로 오독한다.

## 어떻게 돌리는가

**`scripts/pilot_c.py` 를 그대로 부른다.** 어블레이션 전용 학습 경로를 따로 만들면 두 팔의
차이가 표본 때문인지 코드 때문인지 다시 알 수 없게 된다. 스냅샷과 산출 경로만 바꾼다.

채점은 평가 담당 몫이다. 여기서는 각 팔의 **가중치와 예측 raw** 까지 낸다 — 팔마다
평가셋이 다르므로(431 / 418장) 예측도 팔별 평가셋에서 따로 만들어야 한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = {
    "crop_only": "data/processed/aihub71761_rt_v1_pilot3000_crop_only",
    "scale_control": "data/processed/aihub71761_rt_v1_pilot3000_scale_control",
}
OUT_ROOT = Path("outputs/ablation_c").resolve()
STEPS = ["views", "init", "cell2", "cell3", "cell4"]
#: 팔마다 평가셋이 다르므로 예측도 팔별로 만든다. 채점은 평가 담당 몫이다.
EXPORT = "scripts/pilot_export_det.py"


def run_arm(arm: str, snapshot: str) -> dict:
    out = OUT_ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    timings = {}
    for step in STEPS:
        t0 = time.perf_counter()
        rc = subprocess.run(
            [py, "scripts/pilot_c.py", step, "--snapshot", snapshot, "--out", str(out)]
        ).returncode
        timings[step] = round(time.perf_counter() - t0, 1)
        print(f"[{arm}] {step}: rc={rc} {timings[step]}s", flush=True)
        if rc != 0:
            return {"arm": arm, "snapshot": snapshot, "failed_at": step,
                    "returncode": rc, "timings": timings}

    t0 = time.perf_counter()
    rc = subprocess.run([py, EXPORT, "--snapshot", snapshot, "--out", str(out)]).returncode
    timings["export"] = round(time.perf_counter() - t0, 1)
    print(f"[{arm}] export: rc={rc} {timings['export']}s", flush=True)

    audit_path = out / "sep_fed" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    return {"arm": arm, "snapshot": snapshot, "timings": timings,
            "total_s": round(sum(timings.values()), 1),
            "fed_audit_ok": (audit or {}).get("ok"),
            "fed_total_steps": (audit or {}).get("total_optimizer_steps")}


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for arm, snap in ARMS.items():
        if only and arm != only:
            continue
        print(f"\n=== {arm} ({snap}) ===", flush=True)
        results.append(run_arm(arm, snap))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "ablation_runs.json"
    path.write_text(json.dumps({"arms": results}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n→ {path}", flush=True)


if __name__ == "__main__":
    main()
