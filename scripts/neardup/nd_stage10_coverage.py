"""근사 중복 10단계 — **옛 라벨이 덮지 못한 구간을 잰다.** 표본을 새로 뽑기 전에 해야 할 일이다.

무엇이 문제인가. 8단계가 모은 310쌍은 적대 검증 라운드에서 몽타주로 본 것이고,
그 라운드는 **기각된 네 규칙이 띄운 후보**를 봤다. 그러면 라벨 집합이 옛 규칙이 고른 쌍으로 치우친다.
**옛 규칙이 아예 띄우지 않은 계열의 "다름" 은 이 표에 없다.** 그 표에서 새 규칙을 채점하면
미탐률이 낙관적으로 나온다 — 옛 규칙이 놓친 것을 새 규칙도 놓치는지 이 표로는 알 수 없기 때문이다.

그래서 여기서 둘을 센다.
  ① 라벨 310쌍이 옛 규칙 안팎 어디에 있는가 — 밖이 얼마나 적은지가 편향의 크기다
  ② 후보 모집단에서 옛 규칙이 하나도 띄우지 않은 구간이 얼마나 큰가 — 새로 뽑을 곳이다

**무작위 대조만으로는 이 구간이 안 잡힌다.** 모집단의 대부분은 명백한 음성이라
무작위로 뽑으면 어려운 음성이 거의 안 걸린다. 구간을 지목해 뽑아야 한다.

사용: python nd_stage10_coverage.py     → stage10_coverage.json (화면에도 같은 내용)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nd_stage5_lists import compile_rule  # noqa: E402
from nd_stage9_devfit import load_features, _rows  # noqa: E402

DEFS = json.loads((Path(__file__).resolve().parent / "definitions.json").read_text(encoding="utf-8"))
OLD = [d for d in DEFS["definitions"] if d.get("기각됨")]     # 라벨을 띄운 네 규칙
csv.field_size_limit(10_000_000)


def applies(fn, names, f: dict):
    """열이 없어 적용할 수 없으면 None. 있으면 통과 여부."""
    if any(k not in f or f[k] in ("", None) for k in names):
        return None
    return bool(fn(f))


def old_cover(f: dict) -> tuple[bool, bool]:
    """(옛 규칙 하나라도 띄웠나, 적용 자체가 불가능했나)."""
    seen_any = False
    all_unknown = True
    for d in OLD:
        fn, names = compile_rule(d["rule"])
        v = applies(fn, names, f)
        if v is None:
            continue
        all_unknown = False
        seen_any = seen_any or v
    return seen_any, all_unknown


def main() -> int:
    feat = load_features()
    labels = _rows(W / "labels_prior.csv")

    # ① 라벨이 옛 규칙 안팎 어디에 있나
    inside = {"같음": 0, "다름": 0, "불확실": 0, "충돌": 0}
    outside = dict(inside)
    unknown = dict(inside)
    for r in labels:
        f = feat.get((r["image_id_a"], r["image_id_b"]))
        if f is None:
            continue
        hit, unk = old_cover(f)
        (unknown if unk else inside if hit else outside)[r["판정"]] += 1

    # ② 후보 모집단에서 옛 규칙이 하나도 띄우지 않은 구간의 크기
    labeled = {(r["image_id_a"], r["image_id_b"]) for r in labels}
    pop = {"전체": 0, "옛_규칙_안": 0, "옛_규칙_밖": 0, "적용_불가": 0,
           "옛_규칙_밖_라벨없음": 0, "옛_규칙_밖_라벨있음": 0}
    band = {"큰_이동_48px_초과": 0, "문자_띠_의심": 0, "블록_중간대": 0}
    for key, f in feat.items():
        if f.get("_src") not in ("cross", "rest", "new"):
            continue
        pop["전체"] += 1
        hit, unk = old_cover(f)
        if unk:
            pop["적용_불가"] += 1
            continue
        pop["옛_규칙_안" if hit else "옛_규칙_밖"] += 1
        if hit:
            continue
        pop["옛_규칙_밖_라벨있음" if key in labeled else "옛_규칙_밖_라벨없음"] += 1
        # 밖에서 어떤 계열이 있는가 — 새로 뽑을 구간의 후보
        def num(k):
            v = f.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        dx, dy, gap, en, frac = num("rd_dx"), num("rd_dy"), num("txt_gap"), num("txt_energy"), num("blk_frac")
        if dx is not None and dy is not None and max(abs(dx), abs(dy)) > 48:
            band["큰_이동_48px_초과"] += 1
        if gap is not None and en is not None and gap >= 0.3 and en >= 1.0:
            band["문자_띠_의심"] += 1
        if frac is not None and 0.30 <= frac < 0.50:
            band["블록_중간대"] += 1

    rep = {
        "①_라벨_310쌍이_옛_규칙_안팎_어디에": {
            "옛_규칙이_띄운_쌍": inside, "옛_규칙_밖의_쌍": outside, "적용_불가": unknown,
            "읽는_법": "밖의 '다름' 이 적을수록 편향이 크다 — 옛 규칙이 안 띄운 계열의 오판을 이 표가 못 본다는 뜻이다.",
        },
        "②_후보_모집단_가운데_옛_규칙_밖": pop,
        "③_옛_규칙_밖의_계열별_크기": band,
        "주의": "②·③ 은 이미 뽑아 둔 후보 쌍 안에서의 수다. 어떤 생성기도 못 본 쌍은 여기에도 없다.",
    }
    (W / "stage10_coverage.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
