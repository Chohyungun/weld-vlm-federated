"""같은 그림인데 라벨이 다른 쌍 — **유형을 가르기 위한 표본**을 뽑는다.

어느 쪽 라벨이 맞는지 판정하려는 것이 아니다. 무엇이 어긋났는지 유형을 나누고 그 비율을 재는 것이다.
유형은 넷으로 본다.

  누락      한쪽에만 결함 표시가 있고, 그림에는 양쪽 다 같은 결함이 보인다 (한쪽 어노테이션이 빠졌다)
  경계      결함이 희미하거나 작아 "있다/없다" 가 갈릴 만하다 (판단 차이)
  재질표기  같은 필름인데 재질이 강재/알루미늄으로 갈린다 (표기 오류)
  불가      그림으로는 어느 쪽인지 말할 수 없다

사용: python nd_labelconflict.py [층당 장수]    기본 13 (네 층 = 약 50쌍)
출력: labelconflict_sample.csv + 몽타주용 인자. 라벨은 `note` 열에 적어 그림 위에 찍힌다(식별자는 찍지 않는다).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage5_lists as S

W = S.W
ROOT = S.ROOT
RULE = "rd_sharp >= 0.35 and blk_frac >= 0.5"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    man = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline=""))}
    rule, _ = S.compile_rule(RULE)
    cross = [r for r in S.load_cross() if rule(r)]
    within = [r for r in S.load_within_trainval() if rule(r)]
    for r in cross:
        r["where"] = "교차"
    for r in within:
        r["where"] = "학습풀안"

    def dfn(r):     # 결함/정상이 갈린다
        return man[r["image_id_a"]]["has_defect"] != man[r["image_id_b"]]["has_defect"]

    def mat(r):     # 재질이 갈린다
        return man[r["image_id_a"]]["material"] != man[r["image_id_b"]]["material"]

    strata = [
        ("교차·결함정상", [r for r in cross if dfn(r)]),
        ("교차·재질", [r for r in cross if mat(r)]),
        ("학습풀안·결함정상", [r for r in within if dfn(r)]),
        ("학습풀안·재질", [r for r in within if mat(r)]),
    ]
    rng = np.random.default_rng(20260921)
    picked = []
    counts = {}
    for name, pool in strata:
        counts[name] = len(pool)
        if not pool:
            print(f"  {name}: 없음", flush=True)
            continue
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        for i in idx:
            r = dict(pool[i])
            a, b = man[r["image_id_a"]], man[r["image_id_b"]]
            r["stratum"] = name
            r["note"] = (f"{name} | A 결함={a['has_defect']}({a['n_defects']}·{a['defect_types'] or '-'}) 재질={a['material']}"
                         f" / B 결함={b['has_defect']}({b['n_defects']}·{b['defect_types'] or '-'}) 재질={b['material']}")
            picked.append(r)
        print(f"  {name}: {len(pool):,}쌍 중 {min(n, len(pool))}쌍", flush=True)

    cols = ["stratum", "note", "image_id_a", "image_id_b", "where", "rd_sharp", "rd_dx", "rd_dy", "blk_frac", "blk_med",
            "hi_min", "txt_gap", "mad", "corr", "hamming"]
    out = W / "labelconflict_sample.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in picked:
            w.writerow([r.get(c, "") for c in cols])
    (W / "labelconflict_counts.json").write_text(json.dumps(
        {"rule": RULE, "모집단": counts, "표본": dict(Counter(r["stratum"] for r in picked))},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{out.name}: {len(picked)}쌍")
    print(f"  몽타주: python scripts/neardup/nd_montage.py {out} <출력 폴더> labelconf "
          f"{','.join(str(i) for i in range(len(picked)))} --align")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
