"""근사 중복 9단계 — **모은 라벨 위에서 지금 규칙을 채점한다.** 여태 한 번도 하지 않은 일이다.

앞선 네 정의는 자기가 본 표본에서 문턱을 정하고 같은 표본에서 확인했다. 그래서 무너졌다.
여기서는 8단계가 모은 판정(눈으로 본 것)을 정답으로 두고 **규칙을 채점한다** — 오탐과 미탐을 따로 낸다.

**이 표본은 개발용이다.** 전부 앞선 규칙 설계에 이미 쓰였으므로 여기서 나온 수는
규칙이 자기가 배운 자리에서 얼마나 맞는지이지 새 자료에서의 성능이 아니다. 보류 표본은 따로 만든다.

함께 보는 것: 새로 계산한 두 열이 **실제 라벨을 가르는가.**
  hi_min · hi_min_energy  — 구조가 강한 블록의 최악 상관
  txt_gap · txt_energy    — 하단 필름 문자 띠와 본문의 상관 차이

사용: python nd_stage9_devfit.py     → stage9_devfit.json (화면에도 같은 내용)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nd_stage5_lists import compile_rule  # noqa: E402  규칙 식은 한 곳에서만 해석한다

csv.field_size_limit(10_000_000)

TAIL = {"cross": "cross_eval_vs_trainval", "rest": "rest_cross_eval_vs_trainval", "new": "new_cross"}
STAGES = {"feat3": "feat3_{}.csv", "feat5": "feat5_{}.csv", "feat6": "feat6_{}.csv", "feat7": "feat7_{}.csv"}
# 볼 열. 판정별로 분포를 낸다.
COLS = ["rd_sharp", "ov", "reg_corr", "reg_mid", "blk_frac", "blk_med", "blk_min",
        "hi_n", "hi_min", "hi_min_energy", "hi_badfrac", "txt_corr", "body_corr", "txt_gap", "txt_energy", "shift"]


def _rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    def q(f):
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 4)
    return {"n": len(xs), "최소": round(xs[0], 4), "10%": q(0.10), "중앙": q(0.50), "90%": q(0.90), "최대": round(xs[-1], 4)}


def load_features() -> dict:
    """쌍 → 모든 단계의 열을 합친 하나의 dict."""
    feat: dict[tuple, dict] = {}
    for src, tail in TAIL.items():
        for stage, pat in STAGES.items():
            p = W / pat.format(tail)
            if not p.exists():
                continue
            for r in _rows(p):
                key = (r["image_id_a"], r["image_id_b"])
                d = feat.setdefault(key, {"image_id_a": key[0], "image_id_b": key[1], "_src": src})
                for k, v in r.items():
                    if k not in ("image_id_a", "image_id_b"):
                        d[k] = v
    for d in feat.values():
        dx, dy = _num(d.get("rd_dx")), _num(d.get("rd_dy"))
        d["shift"] = max(abs(dx), abs(dy)) if dx is not None and dy is not None else None
    return feat


def score(rule: str, rows: list[dict]) -> dict:
    """규칙을 라벨에 대고 채점한다. **값이 없어 판정할 수 없는 쌍은 통과로 세지 않고 따로 센다.**

    `compile_rule` 의 함수는 열이 비면 False 를 돌려준다 — 그 False 는 "규칙이 떨어뜨렸다" 가 아니라
    "판정할 수 없다" 이다. 둘을 합치면 미탐률이 조용히 부풀거나 줄어든다. 그래서 여기서 먼저 가른다.
    """
    fn, names = compile_rule(rule)
    out = {"같음_통과": 0, "같음_탈락": 0, "다름_통과": 0, "다름_탈락": 0, "값_없음": 0}
    for r in rows:
        f = r["_f"]
        if any(k not in f or f[k] in ("", None) for k in names):
            out["값_없음"] += 1
            continue
        out[f"{r['판정']}_{'통과' if bool(fn(f)) else '탈락'}"] += 1
    tp, fn_, fp, tn = out["같음_통과"], out["같음_탈락"], out["다름_통과"], out["다름_탈락"]
    out["미탐률"] = round(fn_ / (tp + fn_), 4) if tp + fn_ else None
    out["오탐률"] = round(fp / (fp + tn), 4) if fp + tn else None
    return out


def main() -> int:
    feat = load_features()
    labels = _rows(W / "labels_prior.csv")
    rows, missing = [], 0
    for r in labels:
        f = feat.get((r["image_id_a"], r["image_id_b"]))
        if f is None:
            missing += 1
            continue
        rows.append({**r, "_f": f})
    graded = [r for r in rows if r["판정"] in ("같음", "다름")]

    rep = {
        "라벨_쌍": len(labels), "특징_붙은_쌍": len(rows), "특징_없음": missing,
        "판정별": {v: sum(1 for r in rows if r["판정"] == v) for v in ("같음", "다름", "불확실", "충돌")},
        "주의": "이 표본은 전부 앞선 규칙 설계에 쓰였다. 여기 수치는 개발 표본의 값이지 새 자료의 성능이 아니다.",
    }
    rep["열별_분포"] = {}
    for col in COLS:
        rep["열별_분포"][col] = {
            v: quantiles([x for r in rows if r["판정"] == v
                          for x in [_num(r["_f"].get(col))] if x is not None])
            for v in ("같음", "다름", "불확실")
        }
    defs = json.loads((Path(__file__).resolve().parent / "definitions.json").read_text(encoding="utf-8"))
    rep["규칙_채점"] = {}
    for d in defs["definitions"]:
        if d["key"] in ("A0", "A1"):
            continue
        rep["규칙_채점"][d["key"]] = score(d["rule"], graded)
    # 가드만 따로 — 같은 필름 다른 프레임을 정말 거르는가
    rep["가드_단독"] = {
        "hi 가드 (hi_min<=-0.2 and hi_min_energy>=2.5)":
            score("hi_min <= -0.2 and hi_min_energy >= 2.5", graded),
        "txt 가드 후보 (txt_gap>=0.3 and txt_energy>=1.0)":
            score("txt_gap >= 0.3 and txt_energy >= 1.0", graded),
    }
    (W / "stage9_devfit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
