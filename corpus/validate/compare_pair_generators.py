"""두 D4 페어 생성기의 판정 논리 대조 — 74번 감사 P4 소급 검토의 측정 스크립트.

지시서는 `corpus/rules/skeleton_gen.py` 사용을 명시했는데 `corpus/generate/make_pairs_pilot.py`
는 그것을 import 하지 않고 신규 생성기를 썼다. 이 스크립트가 그 둘을 실물 표본에서
대조한다. **수치를 보고서에만 적고 산출 스크립트를 커밋하지 않으면 나중에 재현이 안 된다**
(감사가 D 에게 같은 지적을 했다).

세 가지를 잰다.

1. `skeleton_from_label` 의 D4 경로가 파일럿 표본에서 무엇을 내는가 — 격리 사유 분포.
   두께·스케일이 전량 결측이라(함정 #10) 전건 격리될 것으로 예상되며, 그것이 신규
   생성기를 쓴 실제 이유인지 판정한다.
2. 스케일·두께를 **가정 주입**해 강제로 통과시켰을 때 두 생성기의 **조항 선택이 같은가.**
   가정값은 논리 대조를 위한 것이고 산출물에 들어가지 않는다.
3. 어긋남이 동결본 `pairs_pilot_v1/` 몇 건에 실렸는가.

실행: uv run python -m corpus.validate.compare_pair_generators
"""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "data/processed/aihub71761_rt_v1_pilot3000"
PAIRS = REPO / "data/processed/pairs_pilot_v1/pairs.jsonl"
LIMITS_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"

# 논리 대조 전용 가정값. 표본에는 두께·스케일이 없어 정본 경로가 아무것도 만들지 못한다.
# 조항 선택만 보려고 강제로 통과시키는 값이며 산출물에 들어가지 않는다.
ASSUMED_T = Decimal("12")
ASSUMED_SCALE = Decimal("0.1")


def _dec(v):
    import pandas as pd

    return None if v is None or pd.isna(v) else Decimal(str(v))


def main() -> int:
    from corpus.generate.make_pairs_pilot import clause_basis
    from corpus.rules import limits_loader, skeleton_gen as sg
    from corpus.rules.limit_eval import applicable_row
    from corpus.rules.schema import InspectionMethod
    from data.manifest_io import load_snapshot

    table = limits_loader.load_limits(str(LIMITS_CSV), pilot=True)
    snap = load_snapshot(SNAP)
    m = snap.manifest
    tv_ids = set(m[m["split"].isin(["train", "val"])]["image_id"])
    material = dict(zip(m["image_id"], m["material"]))

    print("== 1. 정본 D4 경로(skeleton_from_label)를 표본에 그대로 걸었을 때 ==")
    print(f"manifest thickness_mm 비결측 {int(m['thickness_mm'].notna().sum())} 건 /"
          f" px_per_mm 비결측 {int(m['px_per_mm'].notna().sum())} 건")
    labels = []
    for _, a in snap.annotations.iterrows():
        if a["image_id"] not in tv_ids:
            continue
        labels.append(sg.D4Label(
            image_id=a["image_id"], defect_instance_id=a["ann_id"],
            defect_type=str(a["defect_type"]), material=str(material[a["image_id"]]),
            size_px=_dec(a["major_axis_px"]), thickness_mm=None))
    oks, bad = sg.skeletons_from_labels(table, labels, None, sg.D4Assumptions())
    print(f"어노테이션 {len(labels)} 건 → 골격 {len(oks)} · 격리 {len(bad)}")
    print("격리 사유:", dict(Counter(b.reason for b in bad).most_common()))

    print()
    print("== 2. 가정 두께·스케일을 주입해 강제 통과시켰을 때의 조항 선택 대조 ==")
    basis = clause_basis(table)
    print("make_pairs_pilot.clause_basis:",
          {k: v["clause_id"] for k, v in basis.items()})
    combos = Counter((str(a["iso_code"]), str(material[a["image_id"]]))
                     for _, a in snap.annotations.iterrows()
                     if a["image_id"] in tv_ids)
    mismatch = 0
    for (code, mat), n in combos.most_common():
        mine = basis.get(code, {}).get("clause_id", "(없음)")
        try:
            # 파일럿 CSV 의 품질 축은 none/ALL 이다. D4Assumptions 기본값(iso5817/C)을
            # 그대로 쓰면 품질 축에서 먼저 걸려 재질 축의 갈림이 보이지 않는다.
            theirs = applicable_row(table, code, mat, InspectionMethod("RT"),
                                    "none", "ALL", ASSUMED_T).clause_id
        except LookupError as e:
            theirs = f"거절 — {e}"
        same = mine == theirs
        mismatch += 0 if same else n
        print(f"[{'일치' if same else '불일치'}] 코드 {code} 재질 {mat} n={n}"
              f" | make_pairs={mine} | skeleton_gen={theirs}")
    print(f"어긋난 어노테이션 {mismatch} 건")

    print()
    print("== 3. 동결본에 실린 영향 ==")
    if not PAIRS.exists():
        print("pairs.jsonl 이 없다 — 드라이브 보관분")
        return 0
    covered = {str(x) for x in (
        r.material.value if hasattr(r.material, "value") else r.material
        for r in table.rows
        if getattr(r, "scope", "active") == "active")}
    hit = Counter()
    for line in PAIRS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["skeleton"]["defects"] and material[r["image_id"]] not in covered:
            hit[(material[r["image_id"]], r["split"])] += 1
    print(f"허용치 표가 덮지 않는 재질에 조항이 붙은 결함 페어: {sum(hit.values())} 건")
    print("  내역:", dict(sorted(hit.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
