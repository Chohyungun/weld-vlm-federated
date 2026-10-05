"""근사 중복 8단계 — **앞서 눈으로 본 판정을 한 표로 모은다.** 규칙 재설계의 출발점이다.

왜 필요한가. 적대 검증 라운드에서 쌍을 몽타주로 보고 판정한 기록이 여섯 군데에 흩어져 있다.
형식도 좌표계도 제각각이라 그대로는 쓸 수 없다. 이것을 **쌍 식별자로 바꿔** 한 표에 모은다.

**좌표계를 믿지 않는다.** 판정 파일은 쌍을 `(출처, 행 번호)` 로 적었는데 그 행 번호가 어느 CSV 의
몇 번째 줄인지는 적혀 있지 않다. 그래서 규약을 가정하지 않고 **되찾은 행의 특징값이 판정 파일에
적힌 특징값과 맞는지 대조한다.** 맞는 후보가 정확히 하나일 때만 받는다. 어긋나면 버리고 수를 적는다.

이 표는 라벨이지 규칙이 아니다. **개발·보류 분할은 여기서 하지 않는다** — 다음 단계가 한다.
여기 있는 쌍은 전부 앞선 규칙 설계에 이미 쓰였으므로 **보류 표본이 될 수 없다.**

사용: python nd_stage8_labelset.py     → labels_prior.csv · stage8_labelset.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
A = W / "agents"
csv.field_size_limit(10_000_000)

# 행 번호가 가리킬 수 있는 후보 표. 판정 파일이 출처를 적었어도 대조로 확인한다.
FEAT3 = {
    "cross": "feat3_cross_eval_vs_trainval.csv",
    "rest": "feat3_rest_cross_eval_vs_trainval.csv",
    "new": "feat3_new_cross.csv",
}
# 판정 파일에 함께 적힌 특징 열 ↔ 특징 CSV 의 열. 이 가운데 있는 것으로 대조한다.
CHECK = {
    "rd_sharp": ("feat3", "rd_sharp", 0.005),
    "ov": ("feat3", "ov", 0.005),
    "reg_corr": ("feat3", "reg_corr", 0.005),
    "reg_mid": ("feat3", "reg_mid", 0.005),
    "reg_coarse": ("feat3", "reg_coarse", 0.005),
    "reg_mad_tone": ("feat3", "reg_mad_tone", 0.01),
    "rd_peak": ("feat3", "rd_peak", 0.005),
    "rd_side": ("feat3", "rd_side", 0.005),
    "corr": ("feat3", "corr", 0.005),
    "blk_frac": ("feat5", "blk_frac", 0.005),
    "blk_med": ("feat5", "blk_med", 0.005),
    "blk_min": ("feat5", "blk_min", 0.005),
    "blk_n": ("feat5", "blk_n", 0.5),
}
# 판정 문자열을 셋으로 모은다. 그 밖의 값은 받지 않는다.
VERDICT = {
    "same": "같음", "s": "같음", "sa": "같음",
    "different": "다름", "diff": "다름", "d": "다름",
    "unsure": "불확실", "u": "불확실", "ambig": "불확실",
}
# (판정 파일, 메모 열). 출처 열은 머리글에서 찾는다 — `src` · `file` 중 있는 것, 둘 다 없으면 대조로 정한다.
# 나중 파일이 앞 파일과 같은 쌍을 다시 판정하면 앞의 것을 남긴다. 그래서 **더 나중에 본 판(c07)을 먼저 둔다.**
SOURCES = [
    ("attack_over_merge/c07_eye_all2.csv", None),
    ("attack_over_merge/c03_eye_all.csv", None),
    ("attack_over_merge/b06_eye.csv", "note"),
    ("attack_over_merge/eye_verdicts.csv", "note"),
    ("attack_over_split/eye_verdicts.csv", None),
    ("measurement/eye_verdicts.csv", None),
    ("conservative/inspected.csv", None),
]


def _rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_tables() -> dict:
    """출처마다 feat3·feat5 를 행 번호로 찾을 수 있게 붙여 둔다."""
    out = {}
    for src, name in FEAT3.items():
        f3 = _rows(W / name)
        f5name = name.replace("feat3", "feat5", 1)
        f5 = {(r["image_id_a"], r["image_id_b"]): r for r in _rows(W / f5name)}
        out[src] = [(r, f5.get((r["image_id_a"], r["image_id_b"]), {})) for r in f3]
    return out


def agrees(rec: dict, f3: dict, f5: dict) -> tuple[int, int]:
    """판정 파일에 함께 적힌 특징값이 되찾은 행과 맞는가. (맞은 수, 견준 수)."""
    ok = n = 0
    for col, (tbl, key, tol) in CHECK.items():
        want = _num(rec.get(col))
        if want is None:
            continue
        got = _num((f3 if tbl == "feat3" else f5).get(key))
        if got is None:
            continue
        n += 1
        ok += abs(got - want) <= tol
    return ok, n


def resolve(rec: dict, tables: dict, declared: str | None) -> tuple[str | None, dict, dict, str]:
    """행 번호를 쌍으로 바꾼다. 특징값이 맞는 출처가 정확히 하나일 때만 받는다."""
    row = _num(rec.get("row"))
    if row is None or row != int(row):
        return None, {}, {}, "행_번호_없음"
    row = int(row)
    hits = []
    for src, tab in tables.items():
        if declared and declared not in ("", src) and src not in declared:
            continue
        if not 0 <= row < len(tab):
            continue
        f3, f5 = tab[row]
        ok, n = agrees(rec, f3, f5)
        if n == 0:
            hits.append((src, f3, f5, "대조_불가"))
        elif ok == n:
            hits.append((src, f3, f5, "대조_일치"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None, {}, {}, "대조_불일치"
    return None, {}, {}, "출처_중의"


def main() -> int:
    tables = load_tables()
    seen: dict[tuple, dict] = {}
    tally = {"판정_행": 0, "받음": 0, "버림": {}, "충돌": 0, "출처별": {}}
    for rel, notecol in SOURCES:
        p = A / rel
        if not p.exists():
            tally["버림"]["파일_없음:" + rel] = 1
            continue
        recs = _rows(p)
        srccol = next((c for c in ("src", "file") if recs and c in recs[0]), None)
        got = dup = 0
        for rec in recs:
            tally["판정_행"] += 1
            v = VERDICT.get(str(rec.get("verdict", "")).strip().lower())
            if v is None:
                tally["버림"]["판정값_모름"] = tally["버림"].get("판정값_모름", 0) + 1
                continue
            src, f3, f5, why = resolve(rec, tables, rec.get(srccol) if srccol else None)
            if src is None:
                tally["버림"][why] = tally["버림"].get(why, 0) + 1
                continue
            key = (f3["image_id_a"], f3["image_id_b"])
            row = {
                "image_id_a": key[0], "image_id_b": key[1], "판정": v,
                "출처": rel, "출처_표": src, "출처_행": rec.get("row"),
                "층": rec.get("stratum", ""), "대조": why,
                "메모": (rec.get(notecol) or "") if notecol else "",
            }
            if key in seen:
                dup += 1
                if seen[key]["판정"] != v and seen[key]["판정"] != "충돌":
                    tally["충돌"] += 1
                    seen[key]["판정"] = "충돌"
                seen[key]["출처"] += " · " + rel
                continue
            seen[key] = row
            got += 1
            tally["받음"] += 1
        tally["출처별"][rel] = {"새_쌍": got, "앞에서_이미_본_쌍": dup}
    rows = list(seen.values())
    tally["고유_쌍"] = len(rows)
    tally["판정별"] = {}
    for r in rows:
        tally["판정별"][r["판정"]] = tally["판정별"].get(r["판정"], 0) + 1
    out = W / "labels_prior.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (W / "stage8_labelset.json").write_text(
        json.dumps(tally, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(tally, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
