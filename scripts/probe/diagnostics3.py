"""GPU 0 진단 3종 — 부분군 델타표 · 출처 고정 판별력 · P9 본실험 검정력. 71번 과제 1~3.

    uv run python scripts/probe/diagnostics3.py

세 과제가 같은 레코드·같은 채점기를 공유하므로 한 스크립트에 둔다. **재추론하지 않는다** —
65·66번이 저장한 계약 #4 레코드를 되읽어 모집단만 바꿔 채점한다.

과제 1은 68번 §2-3-(a)의 "승격의 채점 순효과 0" 주장을 저장소 채점기로 재현해 정본으로
만든다. 과제 2는 68번 §5-(나)가 지목한 지표를 구현·산출한다. 과제 3은 68번 §3 셋째의
"본실험 묶음 248개" 주장을 **동결 매니페스트**로 검산한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from evaluation.adapters import read_records
from evaluation.discrimination import CROP, score_discrimination_all_cells
from evaluation.eval_set import eval_rows as select_eval
from evaluation.eval_set import read_gold, read_manifest
from evaluation.schema import PredictionRecord
from evaluation.score import score_records

PILOT = Path("data/processed/aihub71761_rt_v1_pilot3000")
FROZEN = Path("data/interim/manifest_v1")
OUT = Path("outputs/pilot_d")
SEED = 20260828
CELLS = ("uni_central", "uni_fed", "sep_central",
         "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
DELTA_KEYS = ("macro_f1", "miss_rate", "defect_recall", "class_jaccard")
P9_MIN_CLUSTERS = 20


def provenance_map(snapshot: Path) -> dict[str, str]:
    with (snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        return {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}


def shortcut_records(rows, prov, cell="uni_central") -> list[PredictionRecord]:
    """"N-crop 이면 기공 하나, 아니면 없음" — 이미지를 열지 않는 규칙(66번 §0)."""
    return [
        PredictionRecord(
            schema_version="1.3", image_id=r["image_id"], cell=cell, seed=SEED,
            defects=([{"iso_code": "2011"}] if prov.get(r["image_id"]) == CROP else []),
            verdict="판정불가", cited_clauses=[], parse_ok=True,
        )
        for r in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lm = load_label_map()
    classes = [lm.iso_code(n) for n in
               ("crack", "porosity", "lack_of_fusion", "slag_inclusion")]

    rows = read_manifest(PILOT)
    ev = select_eval(rows)
    prov = provenance_map(PILOT)
    eval_ids = {r["image_id"] for r in ev}
    crop_ids = {r["image_id"] for r in ev if prov.get(r["image_id"]) == CROP}
    gold_codes, gold_boxes = read_gold(PILOT, eval_ids)
    for iid in eval_ids:
        gold_codes.setdefault(iid, set())

    def sub(ids: set[str]):
        gc = {k: v for k, v in gold_codes.items() if k in ids}
        gb = {k: v for k, v in gold_boxes.items() if k in ids}
        return gc, gb

    pops = {
        "전량": (eval_ids, *sub(eval_ids)),
        "N-crop": (crop_ids, *sub(crop_ids)),
    }
    print(f"모집단 — 전량 {len(eval_ids)}장 · N-crop {len(crop_ids)}장 "
          f"(결함 {sum(1 for r in ev if r['has_defect'] == 'True' and r['image_id'] in crop_ids)})")

    # --- 과제 1: 부분군 재채점 델타표 -----------------------------------------------
    all_records: list[PredictionRecord] = []
    per_cell: dict[str, list[PredictionRecord]] = {}
    for cell in CELLS:
        path = out / f"{cell}_s{SEED}.jsonl"
        if not path.exists():
            print(f"레코드 없음: {path}"); return 1
        recs = read_records(path.read_text(encoding="utf-8").splitlines())
        per_cell[cell] = recs
        all_records.extend(recs)
    per_cell["지름길 규칙"] = shortcut_records(ev, prov)

    delta_table: dict[str, dict] = {}
    for name, recs in per_cell.items():
        entry: dict[str, dict] = {}
        scored = {}
        for pop, (ids, gc, gb) in pops.items():
            scored[pop] = score_records(
                [r for r in recs if r.image_id in ids], gc, gb, classes
            )
        for k in DELTA_KEYS:
            a, b = float(scored["전량"][k]), float(scored["N-crop"][k])
            entry[k] = {"전량": a, "N-crop": b, "델타": round(b - a, 6)}
        delta_table[name] = entry
        print(f"[{name}] ΔmacroF1 {entry['macro_f1']['델타']:+.4f} "
              f"· Δ놓침 {entry['miss_rate']['델타']:+.4f} "
              f"· ΔJaccard {entry['class_jaccard']['델타']:+.4f}")

    # 68번이 "규칙과 최고 칸의 격차가 0.043 → 0.066 으로 벌어진다"고 적은 항목 검산
    gaps = {}
    for pop in pops:
        rule = delta_table["지름길 규칙"]["class_jaccard"][pop]
        best_cell = max(CELLS, key=lambda c: delta_table[c]["class_jaccard"][pop])
        gaps[pop] = {
            "규칙": rule,
            "최고 칸": best_cell,
            "최고 칸 값": delta_table[best_cell]["class_jaccard"][pop],
            "격차": round(rule - delta_table[best_cell]["class_jaccard"][pop], 6),
        }

    # --- 과제 2: 출처 고정 판별력 ----------------------------------------------------
    contexts = {
        r["image_id"]: (r["group_id"], r["has_defect"] == "True",
                        prov.get(r["image_id"], "(미상)"))
        for r in ev
    }
    disc = score_discrimination_all_cells(all_records, contexts, provenance=CROP)
    rule_disc = score_discrimination_all_cells(
        per_cell["지름길 규칙"], contexts, provenance=CROP
    )[0]
    for d in disc:
        tag = d.cell if not d.client else f"{d.cell}_{d.client}"
        print(f"[{tag}] Δ {d.delta.point:+.4f} [{d.delta.lo:+.4f}, {d.delta.hi:+.4f}] "
              f"· 묶음 {d.n_groups}")
    print(f"[지름길 규칙] Δ {rule_disc.delta.point:+.4f} "
          f"[{rule_disc.delta.lo:+.4f}, {rule_disc.delta.hi:+.4f}] — 0 이어야 한다")

    # --- 과제 3: P9 본실험 검정력 (동결 매니페스트) ----------------------------------
    power = p9_power(FROZEN)
    print(f"동결 평가셋 N-crop 정상 {power['n_images']}장 · 묶음 {power['n_groups']}개 "
          f"→ {power['verdict']}")

    payload = {
        "populations": {k: len(v[0]) for k, v in pops.items()},
        "task1_subgroup_delta": delta_table,
        "task1_class_jaccard_gap": gaps,
        "task2_discrimination": {
            (d.cell if not d.client else f"{d.cell}_{d.client}"): d.as_dict()
            for d in disc
        },
        "task2_shortcut_rule": rule_disc.as_dict(),
        "task3_p9_power": power,
    }
    dest = out / "diagnostics3_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {dest}")
    return 0


def p9_power(frozen: Path) -> dict:
    """동결 평가셋의 N-crop 정상 장수·묶음 수 → P9 형식 판정 가능 여부.

    68번 §3 셋째가 "묶음 248개"라 적고 검산을 요구했다. 파일럿의 18개는 3,279장 표본의
    산물이지 본실험의 성질이 아니라는 주장이 여기 걸려 있다.
    """
    prov = provenance_map(frozen)
    rows = read_manifest(frozen)
    normals = [r for r in rows
               if r["split"] == "eval" and r["has_defect"] == "False"]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in normals:
        by_src[prov.get(r["image_id"], "(미상)")].append(r)
    crop = by_src.get(CROP, [])
    groups = {r["group_id"] for r in crop}
    by_material = Counter(r["material"] for r in crop)
    group_by_material = {
        m: len({r["group_id"] for r in crop if r["material"] == m})
        for m in sorted(by_material)
    }
    ok = len(groups) >= P9_MIN_CLUSTERS
    return {
        "snapshot": str(frozen),
        "n_eval_images": sum(1 for r in rows if r["split"] == "eval"),
        "n_eval_normal": len(normals),
        "n_images": len(crop),
        "n_groups": len(groups),
        "images_by_material": dict(by_material),
        "groups_by_material": group_by_material,
        "by_source": {k: {"n_images": len(v),
                          "n_groups": len({r["group_id"] for r in v})}
                      for k, v in sorted(by_src.items())},
        "min_clusters": P9_MIN_CLUSTERS,
        "formal_verdict_possible": ok,
        "verdict": (
            f"형식 판정 성립 (묶음 {len(groups)} ≥ {P9_MIN_CLUSTERS}). "
            "파일럿의 '판정 불가'는 표본 크기의 산물이었다"
            if ok else
            f"형식 판정 불가 (묶음 {len(groups)} < {P9_MIN_CLUSTERS})"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
