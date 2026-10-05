"""검토 띠의 표본을 사람이 볼 수 있게 뽑는다 — 경계에서 무엇이 걸리는지 눈으로 판정하기 위한 것이다.

자동 제외(X)와 검토 띠(R)를 `definitions.json` 에서 그대로 읽어, 정합 봉우리와 이동 크기로 층을 나눠 뽑는다.
뽑은 쌍을 `review_sample.csv` 에 쓰고, 그 파일을 `nd_montage.py … --align` 에 넣으면 이동을 보정한 몽타주가 나온다.

사용: python nd_review_sample.py [층당 장수]     기본 6
출력: _workspace/.../review_sample.csv (미추적) — 몽타주에는 행 번호만 찍힌다.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage5_lists as S

W = S.W
#: (층 이름, 그 층을 고르는 조건). 경계가 어디인지 사람이 보게 나눈다.
BANDS = [
    ("R_봉우리_0.10-0.20", lambda r: 0.10 <= f(r, "rd_sharp") < 0.20),
    ("R_봉우리_0.20-0.25", lambda r: 0.20 <= f(r, "rd_sharp") < 0.25),
    ("R_블록_0.30-0.50", lambda r: f(r, "blk_frac") < 0.50 and f(r, "rd_sharp") >= 0.25),
    ("R_밝기상관만", lambda r: f(r, "rd_sharp") < 0.10 and f(r, "reg_corr") >= 0.85),
    ("R_큰이동", lambda r: max(abs(f(r, "rd_dx")), abs(f(r, "rd_dy")))> 48),
    ("X_경계_봉우리0.25-0.30", None),
    ("X_경계_블록0.50-0.55", None),
    ("X_고에너지블록_최저", None),
]


def f(r: dict, k: str) -> float:
    v = r.get(k, "")
    return float(v) if v != "" else 0.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    defs = {d["key"]: d["rule"] for d in json.loads((Path(__file__).resolve().parent / "definitions.json").read_text(encoding="utf-8"))["definitions"]}
    x_rule, _ = S.compile_rule(defs["X"])
    r_rule, _ = S.compile_rule(defs["R"])
    rows = S.load_cross()
    X = [r for r in rows if x_rule(r)]
    R = [r for r in rows if r_rule(r)]
    print(f"자동 제외 {len(X):,}쌍 · 검토 띠 {len(R):,}쌍", flush=True)
    rng = np.random.default_rng(20260921)
    picked, seen = [], set()

    def take(name, pool):
        pool = [r for r in pool if (r["image_id_a"], r["image_id_b"]) not in seen]
        if not pool:
            print(f"  {name}: 없음", flush=True)
            return
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        for i in idx:
            r = dict(pool[i])
            r["band"] = name
            picked.append(r)
            seen.add((r["image_id_a"], r["image_id_b"]))
        print(f"  {name}: {len(pool):,}쌍 중 {min(n, len(pool))}쌍", flush=True)

    for name, cond in BANDS[:5]:
        take(name, [r for r in R if cond(r)])
    take("X_경계_봉우리0.25-0.30", [r for r in X if 0.25 <= f(r, "rd_sharp") < 0.30])
    take("X_경계_블록0.50-0.55", [r for r in X if 0.50 <= f(r, "blk_frac") < 0.55])
    lo = sorted([r for r in X if r.get("hi_min")], key=lambda r: f(r, "hi_min"))[:200]
    take("X_고에너지블록_최저", lo)

    cols = ["band", "image_id_a", "image_id_b", "generator", "rd_sharp", "rd_dx", "rd_dy", "ov", "reg_corr", "reg_mid",
            "blk_frac", "blk_med", "hi_min", "hi_min_energy", "mad", "corr", "v1_reg_mad", "hamming"]
    out = W / "review_sample.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in picked:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\n{out.name}: {len(picked)}쌍. 몽타주는 다음으로 만든다(이동 보정 포함):", flush=True)
    print(f"  python scripts/neardup/nd_montage.py {out} <출력 폴더> review {','.join(str(i) for i in range(len(picked)))} --align", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
