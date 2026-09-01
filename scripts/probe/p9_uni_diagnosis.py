"""P9 통합형 경보의 원인 분해 — 출처(provenance) 축이 무엇과 교락돼 있는지 본다.

`uni_central` 이 N-crop 정상 43장 중 41장에 오탐하고 N-tile 221장에는 0건을 낸다.
그 차이가 **출처 신호** 때문인지, 출처와 함께 움직이는 다른 축(규격·재질·묶음) 때문인지
분리하지 않으면 경보의 뜻이 정해지지 않는다. 여기서 재는 것은 원인 후보의 분포뿐이다.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.adapters import read_records
from evaluation.eval_set import eval_rows as select_eval
from evaluation.eval_set import read_manifest

SNAP = Path("data/processed/aihub71761_rt_v1_pilot3000")
OUT = Path("outputs/pilot_d")
SEED = 20260828


def main() -> int:
    rows = read_manifest(SNAP)
    ev = select_eval(rows)
    by_id = {r["image_id"]: r for r in ev}
    with (SNAP / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}

    report: dict[str, object] = {}

    # 1. 출처 어휘와 결함 유무의 교차 — 결함 이미지가 어느 출처에 몰려 있는가
    cross: dict[tuple[str, str], int] = Counter()
    for r in ev:
        cross[(prov.get(r["image_id"], "(미상)"), r["has_defect"])] += 1
    report["출처×결함유무"] = {f"{k[0]}|has_defect={k[1]}": v for k, v in sorted(cross.items())}

    # 2. 규격(W×H)이 출처를 가려내는가 — 함정 #11 이 출처 축으로 옮겨왔는지
    size_by_prov: dict[str, Counter] = defaultdict(Counter)
    for r in ev:
        size_by_prov[prov.get(r["image_id"], "(미상)")][f"{r['width_px']}x{r['height_px']}"] += 1
    report["출처별 규격"] = {k: dict(v) for k, v in sorted(size_by_prov.items())}

    # 3. 재질 교락 (함정 #12)
    mat_by_prov: dict[str, Counter] = defaultdict(Counter)
    for r in ev:
        mat_by_prov[prov.get(r["image_id"], "(미상)")][r["material"]] += 1
    report["출처별 재질"] = {k: dict(v) for k, v in sorted(mat_by_prov.items())}

    # 4. 통합형 오탐을 결함 이미지의 예측률과 나란히 둔다 —
    #    "정상 N-crop 에만 오탐"인지 "N-crop 이면 무조건 결함이라 답"하는지 가른다
    for cell in ("uni_central", "uni_fed", "sep_central"):
        path = OUT / f"{cell}_s{SEED}.jsonl"
        if not path.exists():
            continue
        recs = read_records(path.read_text(encoding="utf-8").splitlines())
        fires: dict[tuple[str, str], list[int]] = defaultdict(list)
        for rec in recs:
            row = by_id[rec.image_id]
            key = (prov.get(rec.image_id, "(미상)"), row["has_defect"])
            fires[key].append(1 if (rec.parse_ok and rec.iso_codes) else 0)
        report[f"{cell} 발화율"] = {
            f"{k[0]}|has_defect={k[1]}": {
                "n": len(v), "발화": sum(v), "발화율": round(sum(v) / len(v), 4),
            }
            for k, v in sorted(fires.items())
        }

    # 4-b. N-crop 안에서 **예측 박스 수가 GT 결함 수를 따라가는가.**
    #      발화율이 같아도 개수가 따라가면 "약하게는 본다"가 되므로, 이 표가 그 여지를 가른다.
    for cell in ("uni_central", "uni_fed", "sep_central"):
        path = OUT / f"{cell}_s{SEED}.jsonl"
        if not path.exists():
            continue
        recs = read_records(path.read_text(encoding="utf-8").splitlines())
        buckets: dict[int, list[int]] = defaultdict(list)
        classsets: Counter = Counter()
        for rec in recs:
            classsets[tuple(sorted(rec.iso_codes))] += 1
            if prov.get(rec.image_id) == "N-crop":
                buckets[int(by_id[rec.image_id]["n_defects"] or 0)].append(len(rec.defects))
        report[f"{cell} 예측 클래스집합 분포"] = {
            ("정상(빈집합)" if not k else "+".join(k)): v for k, v in classsets.most_common()
        }
        report[f"{cell} N-crop GT결함수별 평균 예측박스"] = {
            str(k): {"n_images": len(v), "평균 예측박스": round(sum(v) / len(v), 3)}
            for k, v in sorted(buckets.items()) if k <= 8
        }

    # 5. "출처만 보고 결함 유무를 맞히는" 자명 규칙의 정확도 — 분할별.
    #    함정 #11(규격 지름길)이 타일링으로 규격 축에서는 닫혔는데(전 이미지 1280×720)
    #    출처 축으로 남아 있는지를 한 숫자로 본다.
    shortcut: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "eval"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        hit = sum(
            1 for r in sub
            if (prov.get(r["image_id"]) == "N-crop") == (r["has_defect"] == "True")
        )
        shortcut[split] = {"n": len(sub), "적중": hit, "정확도": round(hit / len(sub), 4)}
    report["출처 지름길 자명 정확도(N-crop=결함 규칙)"] = shortcut

    # 6. 지름길 규칙 자체를 **같은 단일 채점기**로 채점한다.
    #    "N-crop 이면 기공(2011) 하나, 아니면 아무것도 없음" — 이미지를 보지 않는 규칙이다.
    #    통합형 지표가 이 값과 붙으면, 통합형이 낸 것은 학습이 아니라 이 규칙이다.
    from data.label_map import load_label_map
    from evaluation.eval_set import read_gold
    from evaluation.schema import PredictionRecord
    from evaluation.score import score_records

    lm = load_label_map()
    classes = [lm.iso_code(n) for n in
               ("crack", "porosity", "lack_of_fusion", "slag_inclusion")]
    eval_ids = {r["image_id"] for r in ev}
    gold_codes, gold_boxes = read_gold(SNAP, eval_ids)
    for iid in eval_ids:
        gold_codes.setdefault(iid, set())
    shortcut_recs = [
        PredictionRecord(
            schema_version="1.3", image_id=r["image_id"], cell="uni_central",
            seed=SEED,
            defects=([{"iso_code": "2011"}] if prov.get(r["image_id"]) == "N-crop" else []),
            verdict="판정불가", cited_clauses=[], parse_ok=True,
        )
        for r in ev
    ]
    sc = score_records(shortcut_recs, gold_codes, gold_boxes, classes)
    report["지름길 규칙 채점(N-crop→2011, 이미지 미열람)"] = {
        "macro_f1": sc["macro_f1"], "miss_rate": sc["miss_rate"],
        "class_jaccard": sc["class_jaccard"], "defect_recall": sc["defect_recall"],
    }

    dest = OUT / "p9_uni_diagnosis.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
