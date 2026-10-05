"""근사 중복 12단계 — **개발 180쌍에서만 문턱을 정하고 얼린다.**

앞선 네 정의가 무너진 자리는 하나다 — 자기가 본 표본에서 문턱을 정하고 같은 표본에서 확인했다.
여기서는 순서를 뒤집었다. **적대적 음성을 먼저 모으고 반례를 먼저 찾은 뒤에** 문턱을 정한다.
보류 120쌍은 아직 열지 않았고 몽타주도 만들지 않았다.

**후보 규칙 꼴을 미리 적어 두고 그 안에서만 고른다.** 무제한 탐색은 180쌍에 과적합된다.
고르는 기준도 미리 적는다 — **오탐률 상한을 걸고 그 아래에서 미탐률이 가장 낮은 것**을 고른다.
동점이면 더 단순한 것(조건 수가 적은 것)을 고른다.

판독에서 나온 반례를 꼴에 박아 넣는다.
  · 문자 띠 가드는 `txt_gap` 과 `txt_energy` 를 **함께** 쓴다. 어느 하나로는 양방향으로 틀린다.
  · 가드에 **블록 일치가 강하면 발동하지 않는 예외**를 둔다(가드가 진짜 중복을 떨어뜨린 사례).
  · `txt_gap` 은 **부호를 살려 한쪽만** 본다(음수인데 같은 그림인 사례 셋).
  · 잴 것이 없는 쌍(양쪽 안쪽 상자가 평탄)은 자동 제외로도 자동 통과로도 보내지 않는다.

사용: python nd_stage12_fit.py     → stage12_frozen_rule.json (화면에도 같은 내용)
      고른 식은 사람이 `definitions.json` 에 판본과 함께 옮겨 적는다. 스크립트가 정의 파일을 고치지 않는다 —
      얼리는 일은 한 번뿐이고, 다시 돌렸을 때 조용히 덮어써지면 얼린 것이 아니기 때문이다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nd_stage5_lists import compile_rule  # noqa: E402
from nd_stage9_devfit import load_features, _rows  # noqa: E402

csv.field_size_limit(10_000_000)
OK_FP = 0.05          # 오탐률 상한. 먼저 정하고 시작한다.

#: 자동 제외 후보 꼴. {이름: (식, 조건 수)}. 이 목록 밖은 보지 않는다.
#: 공통 뼈대 — 이동이 작고, 블록이 맞고, 문자 띠 가드에 걸리지 않는다.
GUARD = ("not (txt_gap >= {g} and txt_energy >= {e} and blk_med < {m})")
BODY = [
    ("블록중앙", "blk_med >= {b}"),
    ("블록중앙+본문", "blk_med >= {b} and body_corr >= {c}"),
    ("블록중앙+비율", "blk_med >= {b} and blk_frac >= {f}"),
]
SHIFTS = [32, 48]
BMED = [0.15, 0.20, 0.25, 0.30]
BFRAC = [0.30, 0.50]
BODYC = [0.10, 0.20]
GAPS = [0.30, 0.40]
ENS = [1.0]
GMED = [0.20, 0.30]      # 블록 일치가 이만큼이면 가드를 발동하지 않는다


def load_dev() -> list[dict]:
    """개발 180쌍 — 표본 목록 + 판독 판정 + 특징."""
    feat = load_features()
    sample = _rows(W / "sample_v1.csv")
    verdict = {}
    for p in sorted(W.glob("dev_verdicts*.csv")):     # 첫 묶음은 part 가 없는 이름이다
        for r in _rows(p):
            verdict[int(r["행"])] = r["판정"]
    out = []
    for i, s in enumerate(sample):
        if s["쓰임"] != "개발":
            continue
        v = verdict.get(i)
        if v is None:
            raise SystemExit(f"개발 {i}번의 판정이 없다 — 판독을 마치고 다시 돌린다")
        f = feat.get((s["image_id_a"], s["image_id_b"]))
        out.append({"행": i, "층": s["층"], "구간": s["구간"], "판정": v, "_f": f})
    return out


def flat(f: dict) -> bool:
    """양쪽 안쪽 상자에 잴 구조가 없는가. 무늬 없는 면은 무엇과도 맞는다."""
    try:
        return float(f["blk_n"]) > 0 and abs(float(f["rd_sharp"])) < 0.08 and float(f["txt_energy"]) < 1.2 \
            and abs(float(f["body_corr"])) < 0.05
    except (TypeError, ValueError, KeyError):
        return False


def score(expr: str, rows: list[dict]) -> dict:
    fn, names = compile_rule(expr)
    o = {"같음_통과": 0, "같음_탈락": 0, "다름_통과": 0, "다름_탈락": 0, "값없음": 0}
    for r in rows:
        f = r["_f"]
        if f is None or any(k not in f or f[k] in ("", None) for k in names):
            o["값없음"] += 1
            continue
        o[f"{r['판정']}_{'통과' if bool(fn(f)) else '탈락'}"] += 1
    tp, fn_, fp, tn = o["같음_통과"], o["같음_탈락"], o["다름_통과"], o["다름_탈락"]
    o["미탐률"] = round(fn_ / (tp + fn_), 4) if tp + fn_ else None
    o["오탐률"] = round(fp / (fp + tn), 4) if fp + tn else None
    return o


def candidates():
    for sh in SHIFTS:
        for name, body in BODY:
            for b in BMED:
                for f in (BFRAC if "비율" in name else [None]):
                    for c in (BODYC if "본문" in name else [None]):
                        for g in GAPS:
                            for e in ENS:
                                for m in GMED:
                                    core = body.format(b=b, f=f, c=c)
                                    expr = (f"max(abs(rd_dx), abs(rd_dy)) <= {sh} and {core} and "
                                            + GUARD.format(g=g, e=e, m=m))
                                    n_cond = 2 + core.count("and") + 1 + 3
                                    yield {"이름": name, "이동": sh, "blk_med": b, "blk_frac": f,
                                           "body_corr": c, "가드": [g, e, m], "조건수": n_cond, "식": expr}


def main() -> int:
    dev = load_dev()
    graded = [r for r in dev if r["판정"] in ("같음", "다름")]
    rep = {
        "개발_쌍": len(dev),
        "판정별": {v: sum(1 for r in dev if r["판정"] == v) for v in ("같음", "다름", "불확실")},
        "층별": {},
        "고르는_기준": f"오탐률 ≤ {OK_FP} 인 것 가운데 미탐률 최소. 동점이면 조건 수가 적은 것.",
        "후보_꼴_수": sum(1 for _ in candidates()),
    }
    for s in ("확정양성", "적대적음성", "큰이동", "경계"):
        sub = [r for r in dev if r["층"] == s]
        rep["층별"][s] = {v: sum(1 for r in sub if r["판정"] == v) for v in ("같음", "다름", "불확실")}
    # 같음의 이동 상한 — 이동 문턱의 근거로 함께 적는다
    sh = [max(abs(float(r["_f"]["rd_dx"])), abs(float(r["_f"]["rd_dy"]))) for r in graded
          if r["판정"] == "같음" and r["_f"]]
    rep["같음의_이동"] = {"최대": max(sh), "90분위": sorted(sh)[int(0.9 * len(sh))]}

    best = None
    for c in candidates():
        s = score(c["식"], graded)
        if s["오탐률"] is None or s["오탐률"] > OK_FP:
            continue
        key = (s["미탐률"], c["조건수"], c["식"])
        if best is None or key < best[0]:
            best = (key, c, s)
    if best is None:
        rep["결과"] = f"오탐률 {OK_FP} 이하인 후보가 없다 — 상한을 바꾸려면 그 사실을 먼저 적는다"
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1
    _key, c, s = best
    rep["고른_자동제외"] = {**c, "개발_성능": s}
    # 현행 규칙을 같은 표본에 대고 나란히 둔다. 다른 표본의 수와 견주면 안 된다.
    old = json.loads((HERE / "definitions.json").read_text(encoding="utf-8"))
    rep["같은_표본의_현행_규칙"] = {d["key"]: score(d["rule"], graded)
                             for d in old["definitions"] if d["key"] in ("X", "X_narrow")}
    # 검토 띠 — 자동 제외 밖이면서 (블록이 어느 정도 맞거나 잴 것이 없는) 쌍.
    # **검토 띠에는 오탐률이라는 것이 없다.** 사람이 보기 전까지 어느 쪽도 아니기 때문이다.
    # 재야 할 것은 하나다 — 자동 제외가 놓친 같은 그림을 이 띠가 담아 주는가.
    band = f"not ({c['식']}) and (blk_med >= 0.05 or txt_energy < 1.2)"
    bfn, bnames = compile_rule(band)
    afn, anames = compile_rule(c["식"])
    missed = [r for r in graded if r["판정"] == "같음" and r["_f"] and not afn(r["_f"])]
    in_band = [r for r in missed if all(k in r["_f"] for k in bnames) and bfn(r["_f"])]
    rep["검토_띠"] = {
        "식": band,
        "띠의_크기(개발)": sum(1 for r in dev if r["_f"] and all(k in r["_f"] for k in bnames) and bfn(r["_f"])),
        "자동제외가_놓친_같음": len(missed),
        "그중_띠가_담은_것": len(in_band),
        "어디에도_없는_같음": len(missed) - len(in_band),
        "읽는_법": "이 띠에는 오탐률이 없다 — 사람이 보기 전까지 어느 쪽도 아니다. 재는 것은 놓친 같음을 담는지다.",
    }
    rep["놓친_같음의_성격"] = [
        {"행": r["행"], "층": r["층"],
         **{k: r["_f"].get(k) for k in ("rd_sharp", "blk_med", "blk_frac", "body_corr", "txt_gap", "txt_energy")}}
        for r in missed]
    rep["잴_것이_없는_쌍"] = sum(1 for r in dev if r["_f"] and flat(r["_f"]))
    rep["반례_확인"] = {}
    fn, names = compile_rule(c["식"])
    for row in (124, 136, 250, 260, 265, 267):
        hit = next((r for r in dev if r["행"] == row), None)
        if hit and hit["_f"]:
            ok = all(k in hit["_f"] and hit["_f"][k] not in ("", None) for k in names)
            rep["반례_확인"][row] = {"판정": hit["판정"], "자동제외_통과": bool(fn(hit["_f"])) if ok else None}
    out = W / "stage12_frozen_rule.json"
    rep["얼린_시각_기준"] = "이 파일이 쓰인 시점. 이후 보류를 열기 전까지 식을 고치지 않는다."
    rep["식의_sha256"] = hashlib.sha256(c["식"].encode()).hexdigest()
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
