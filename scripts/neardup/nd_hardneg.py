"""적대적 음성 집합 — **같은 필름의 다른 프레임**을 모은다.

왜 필요한가. 지금까지의 대조는 둘뿐이다. 걸러진 후보 600쌍(중간 난도, 양성이 섞여 있다)과 완전 무작위 300쌍(너무 쉽다).
둘 다 "다른 필름" 이라 규칙이 통과시켜도 표시가 나지 않는다. 규칙이 실제로 틀리는 자리는 **같은 필름을 다시 찍은 쌍**이고,
그 축에서는 어떤 문턱도 아직 검증되지 않았다. 문턱을 자기가 본 표본에서 정하고 같은 표본에서 확인하면 09-08 지각 해시 때와 같은 일이 난다.

어떻게 모으나. 씨앗(사람이 확인한 쌍)을 주고, 그와 **같은 성질**을 가진 쌍을 전 후보에서 찾아 올린다.
성질은 하단 문자 띠와 본문의 상관 차(`txt_gap`, 7단계)와 고에너지 블록 최솟값(`hi_min`, 2h단계)이다 — 둘 다 화소에서만 나온다.
뽑은 쌍은 자동으로 음성이라고 단정하지 않는다. **사람이 몽타주를 보고 판정한 것만 집합에 들어간다.**

사용:
  python nd_hardneg.py propose [장수]   후보를 뽑아 hardneg_candidates.csv 와 몽타주를 만든다
  python nd_hardneg.py check            hardneg_labels.json(사람이 적은 판정)으로 규칙의 문턱을 재검증한다

파일 (전부 _workspace, 미추적)
  hardneg_seeds.json       씨앗 — 확인된 쌍의 (파일, 행 번호)와 판정
  hardneg_candidates.csv   뽑힌 후보
  hardneg_labels.json      사람이 적은 판정 {"<파일>#<행>": "same"|"diff"|"unsure"}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage5_lists as S

W = S.W
SEEDS = W / "hardneg_seeds.json"
CANDS = W / "hardneg_candidates.csv"
LABELS = W / "hardneg_labels.json"

#: 씨앗 — 적대 검증에서 **사람이 확대해 보고** 다른 프레임으로 확정한 쌍들. (파일, 행 번호).
#: 행 번호는 그 파일의 0-기준이며 식별자가 아니다. 이 목록은 늘어난다.
SEED_ROWS: dict[str, list[int]] = {
    "feat3_cross_eval_vs_trainval.csv": [181, 1435, 2135, 2215, 2380],
    "feat3_new_cross.csv": [489, 460, 645, 1170, 1506, 1519, 1931],
}


def _key(r: dict) -> str:
    return f"{r['src']}#{r['row']}"


def load_all() -> list[dict]:
    rows = S.load_cross()
    for name in ("feat7_cross_eval_vs_trainval.csv", "feat7_rest_cross_eval_vs_trainval.csv", "feat7_new_cross.csv"):
        S._join(rows, name)
    return rows


def _num(r: dict, k: str, default: float = 0.0) -> float:
    v = r.get(k, "")
    return float(v) if v not in ("", None) else default


def propose(n: int) -> int:
    """씨앗과 같은 성질을 가진 쌍을 올린다. 자동 판정이 아니라 **사람이 볼 목록**을 만드는 것이다."""
    src_rows = {}
    for src in SEED_ROWS:
        src_rows[src] = list(csv.DictReader((W / src).open(encoding="utf-8")))
    seeds = []
    for src, idxs in SEED_ROWS.items():
        for i in idxs:
            r = dict(src_rows[src][i])
            r["src"], r["row"] = src, i
            seeds.append(r)
    SEEDS.write_text(json.dumps([{"src": s["src"], "row": s["row"], "verdict": "diff"} for s in seeds],
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"씨앗 {len(seeds)}쌍 (확인된 '같은 필름 다른 프레임')", flush=True)

    rows = load_all()
    have_txt = sum(r.get("txt_gap") not in ("", None) for r in rows)
    if not have_txt:
        print("!! 문자 띠 열(txt_gap)이 없다 — nd_stage7_textband.py 를 먼저 돌려라. 지금은 hi_min 만으로 뽑는다", flush=True)

    # 후보의 성질: 블록은 잘 맞는데(겉보기 같은 사진) 고에너지 블록이 어긋나거나 문자 띠만 맞는다
    def score(r: dict) -> float:
        s = 0.0
        if _num(r, "blk_frac") >= 0.5:
            s += 1.0
        s += max(0.0, -_num(r, "hi_min")) * 2.0
        if r.get("txt_gap") not in ("", None):
            s += max(0.0, _num(r, "txt_gap")) * 3.0
            if _num(r, "txt_energy") < 0.3:
                s -= 2.0                      # 문자 띠에 아무것도 없으면 그 상관은 뜻이 없다
        return s

    pool = [r for r in rows if _num(r, "blk_frac") >= 0.3 and _num(r, "rd_sharp") >= 0.15]
    pool.sort(key=score, reverse=True)
    picked = pool[:n]
    cols = ["src", "row", "image_id_a", "image_id_b", "generator", "rd_sharp", "rd_dx", "rd_dy", "blk_frac", "blk_med",
            "hi_min", "hi_min_energy", "txt_corr", "body_corr", "txt_gap", "txt_energy", "reg_corr", "mad"]
    for i, r in enumerate(picked):
        r.setdefault("src", "combined")
        r.setdefault("row", i)
    with CANDS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in picked:
            w.writerow([r.get(c, "") for c in cols])
    print(f"후보 {len(picked)}쌍 → {CANDS.name}", flush=True)
    print(f"  몽타주: python scripts/neardup/nd_montage.py {CANDS} <출력 폴더> hardneg "
          f"{','.join(str(i) for i in range(len(picked)))} --align", flush=True)
    print(f"  판정은 {LABELS.name} 에 {{\"combined#<행>\": \"same|diff|unsure\"}} 로 적는다", flush=True)
    return 0


def check() -> int:
    """사람이 판정한 집합으로 규칙의 문턱을 재검증한다. **자동 제외가 다른 프레임을 몇 개나 받는가.**"""
    if not LABELS.exists():
        print(f"!! {LABELS.name} 이 없다 — 몽타주를 보고 판정을 적은 뒤 다시 돌려라", flush=True)
        return 1
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    cands = list(csv.DictReader(CANDS.open(encoding="utf-8")))
    defs = json.loads((Path(__file__).resolve().parent / "definitions.json").read_text(encoding="utf-8"))["definitions"]
    tally = Counter(labels.values())
    print(f"사람 판정 {len(labels)}건: " + " · ".join(f"{k} {v}" for k, v in sorted(tally.items())), flush=True)
    for d in defs:
        if d["key"] in ("A0", "A1"):
            continue
        rule, _ = S.compile_rule(d["rule"])
        hit = Counter()
        for i, r in enumerate(cands):
            v = labels.get(f"{r.get('src', 'combined')}#{r.get('row', i)}")
            if v:
                hit[(v, rule(r))] += 1
        diff_pass = hit[("diff", True)]
        same_pass = hit[("same", True)]
        n_diff = diff_pass + hit[("diff", False)]
        n_same = same_pass + hit[("same", False)]
        print(f"  [{d['key']:9s}] 다른 프레임 통과 {diff_pass}/{n_diff} · 같은 사진 통과 {same_pass}/{n_same}"
              f" · 모름 통과 {hit[('unsure', True)]}/{hit[('unsure', True)] + hit[('unsure', False)]}", flush=True)
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "propose"
    if mode == "propose":
        return propose(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
