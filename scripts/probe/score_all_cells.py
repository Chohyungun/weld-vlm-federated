"""다섯 칸 통합 채점 — 검출 3칸(65번 산출물 재채점) + 통합형 2칸(신규). 66번 지시.

    uv run python scripts/probe/score_all_cells.py

**단일 채점기 원칙**(불변조건 3-7)을 구조로 강제한다:
- 검출 3칸은 65번이 저장한 계약 #4 레코드를 **되읽어** 채점한다. 재추론하지 않는다.
- 통합형 2칸은 어댑터(`evaluation/adapters.py`)로 계약 #4 로 옮긴 뒤 **같은 함수**에 넣는다.
- 65번 저장 지표와 재채점 지표가 비트 단위로 같은지 검사한다. 다르면 즉시 실패한다 —
  채점기를 옮기면서 값이 바뀌었다면 65번 보고서가 무효가 되기 때문이다.

GPU 를 쓰지 않는다. 추론은 전부 끝나 있고 여기는 채점만 한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from evaluation.adapters import adapt_unified_generations, read_records
from evaluation.eval_set import eval_rows as select_eval
from evaluation.eval_set import image_sizes, read_gold, read_manifest
from evaluation.probes.metadata_probe import MetaSample, trivial_bound
from evaluation.probes.p9_runner import contexts_from_snapshot, p9_all_cells
from evaluation.score import coord_health, failure_breakdown, score_records

SEED = 20260828        # C 의 파일럿 base_seed (meta.json 실측). 65번과 동일

DET_CELLS = {
    "sep_central": "sep_central",
    "sep_local_C1": "sep_local_C1",
    "sep_local_C2": "sep_local_C2",
    "sep_local_C3": "sep_local_C3",
    "sep_fed": "sep_fed",
}
UNI_CELLS = ("uni_central", "uni_fed")

# 채점기가 옮겨가도 값이 변하지 않아야 하는 키. 부동소수 비교는 하지 않고 완전 일치를 본다.
REGRESSION_KEYS = (
    "macro_f1", "defect_recall", "miss_rate", "class_jaccard",
    "bbox_iou", "bbox_iou_matched_only", "n_gold", "n_matched",
    "coord_suspect", "map_50", "map_50_95",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data/processed/aihub71761_rt_v1_pilot3000")
    ap.add_argument("--pilot", default="outputs/pilot_c")
    ap.add_argument("--out", default="outputs/pilot_d")
    args = ap.parse_args()

    snapshot, pilot, out = Path(args.snapshot), Path(args.pilot), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lm = load_label_map()
    class_names = ["crack", "porosity", "lack_of_fusion", "slag_inclusion"]
    classes = [lm.iso_code(n) for n in class_names]
    known_codes = set(lm.iso_codes())

    rows = read_manifest(snapshot)
    ev = select_eval(rows)
    eval_ids = {r["image_id"] for r in ev}
    sizes = image_sizes(ev)
    gold_codes, gold_boxes = read_gold(snapshot, eval_ids)
    for iid in eval_ids:
        gold_codes.setdefault(iid, set())
    print(f"평가셋 {len(ev)}장 (정상 {sum(1 for r in ev if r['has_defect'] == 'False')})")

    all_records = []
    metrics: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    adapters: dict[str, dict] = {}
    citations: dict[str, dict[str, list[str]]] = {}

    # --- 검출 3칸: 65번 산출물 되읽기 (재추론 금지) --------------------------------
    for tag in DET_CELLS:
        path = out / f"{tag}_s{SEED}.jsonl"
        if not path.exists():
            print(f"65번 산출물 없음: {path} — 먼저 score_detection_cells.py 를 돌린다")
            return 1
        recs = read_records(path.read_text(encoding="utf-8").splitlines())
        all_records.extend(recs)
        metrics[tag] = score_records(recs, gold_codes, gold_boxes, classes)
        failures[tag] = failure_breakdown(recs)
        print(f"[{tag}] 재채점 {len(recs)}건 · macro_f1 {metrics[tag]['macro_f1']:.4f}")

    # --- 65번 저장값과 대조. 다르면 멈춘다 -----------------------------------------
    prev_path = out / "score_detection_v1.json"
    regression: dict[str, object] = {"checked": False}
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))["metrics"]
        diffs = []
        for tag in DET_CELLS:
            for k in REGRESSION_KEYS:
                a, b = prev[tag].get(k), metrics[tag].get(k)
                if a != b:
                    diffs.append({"cell": tag, "key": k, "65번": a, "재채점": b})
        regression = {"checked": True, "n_keys": len(DET_CELLS) * len(REGRESSION_KEYS),
                      "diffs": diffs}
        if diffs:
            print(f"채점기 이관으로 값이 바뀌었다 {diffs[:3]} — 중단")
            return 1
        print(f"65번 대조: {len(DET_CELLS) * len(REGRESSION_KEYS)}개 키 완전 일치")

    # --- 통합형 2칸: 어댑터 → 같은 채점기 ------------------------------------------
    for cell in UNI_CELLS:
        src = pilot / "predictions" / f"{cell}.generations.jsonl"
        if not src.exists():
            print(f"통합형 원시 출력 없음: {src}")
            return 1
        rep = adapt_unified_generations(
            src.read_text(encoding="utf-8").splitlines(),
            cell=cell, seed=SEED, known_iso_codes=known_codes, image_size=sizes,
        )
        missing = eval_ids - {r.image_id for r in rep.records}
        extra = {r.image_id for r in rep.records} - eval_ids
        if missing or extra:
            print(f"{cell}: 평가셋 불일치 결측 {len(missing)} 초과 {len(extra)} — 중단")
            return 1
        all_records.extend(rep.records)
        with (out / f"{cell}_s{SEED}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rep.records:
                fh.write(r.model_dump_json() + "\n")
        metrics[cell] = score_records(rep.records, gold_codes, gold_boxes, classes)
        failures[cell] = failure_breakdown(rep.records)
        adapters[cell] = rep.as_dict()
        citations[cell] = rep.citations
        print(f"[{cell}] {len(rep.records)}건 · macro_f1 {metrics[cell]['macro_f1']:.4f} "
              f"· 실패 {failures[cell]['n_parse_fail']} "
              f"· 경계이탈 박스 {rep.n_boxes_out_of_bounds}/{rep.n_boxes}")

    # --- 좌표계 건강 판정 (다섯 칸 전부 같은 방법) ---------------------------------
    health = {tag: coord_health(m) for tag, m in metrics.items()}

    # --- 회복률 ---------------------------------------------------------------------
    def f1(tag: str) -> float:
        return float(metrics[tag]["macro_f1"])

    local_mean = float(np.mean([f1("sep_local_C1"), f1("sep_local_C2"), f1("sep_local_C3")]))
    sep_denom = f1("sep_central") - local_mean
    recovery = {
        "basis": "macro_f1",
        "separated": {
            "central": f1("sep_central"), "fed": f1("sep_fed"), "local_mean": local_mean,
            "locals": {k: f1(k) for k in ("sep_local_C1", "sep_local_C2", "sep_local_C3")},
            "denominator": sep_denom,
            "recovery_pct": (f1("sep_fed") - local_mean) / sep_denom * 100
            if sep_denom > 0 else None,
        },
        "unified": {
            "central": f1("uni_central"), "fed": f1("uni_fed"),
            "local_mean": None,
            "recovery_pct": None,
            "retention_pct": f1("uni_fed") / f1("uni_central") * 100
            if f1("uni_central") > 0 else None,
            "note": (
                "통합·로컬은 실험 구조상 '제외' 칸이라 회복률 분모(중앙−로컬)가 존재하지 "
                "않는다. 회복률 대신 연합/중앙 유지율만 낸다 — 두 값은 다른 양이다"
            ),
        },
        "caveat": "시드 1세트 · 표본 3,279 · R×E=N=6 (본실험의 1/17). 결론으로 쓰지 않는다",
    }

    # --- 사전등록 대조 (표본 상대) --------------------------------------------------
    samples = [
        MetaSample(
            image_id=r["image_id"], width_px=int(r["width_px"]),
            height_px=int(r["height_px"]), file_bytes=0, n_channels=1, quant_table_id=0,
            iso_codes=tuple(sorted(gold_codes.get(r["image_id"], ()))),
        )
        for r in ev
    ]
    bound = trivial_bound(samples, classes)

    # --- P9 (일곱 모델 전부, 같은 러너) ---------------------------------------------
    with (snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}
    contexts, ctx_missing = contexts_from_snapshot(ev, prov)
    p9 = p9_all_cells(all_records, contexts)

    # --- 통합형 인용 진단: 지어낸 조항인가 ------------------------------------------
    index_ids = _index_clause_ids()
    citation_diag = {}
    for cell, cited in citations.items():
        flat = [c for v in cited.values() for c in v]
        citation_diag[cell] = {
            "n_citations": len(flat),
            "n_images_citing": sum(1 for v in cited.values() if v),
            "n_in_index": sum(1 for c in flat if c in index_ids),
            "n_not_in_index": sum(1 for c in flat if c not in index_ids),
            "distinct": sorted(set(flat))[:20],
            "note": (
                "정답 조항 쌍은 두께 부재로 0건이라 무근거 인용률의 2차(무관) 분해는 "
                "산출 불가다. 1차(미실존)만 낸다 — 색인에 없는 조항을 지어냈는가"
            ),
        }

    payload = {
        "snapshot": str(snapshot), "seed": SEED, "n_eval": len(ev),
        "scorer": "evaluation.score.score_records (단일)",
        "metrics": metrics,
        "failures": failures,
        "adapters": adapters,
        "coord_health": health,
        "recovery": recovery,
        "prereg": {
            "sample_trivial_bound": bound,
            "gate_basis": "표본 상대(relative)",
            "checks": {
                tag: {"macro_f1": m["macro_f1"], "above_trivial": m["macro_f1"] > bound}
                for tag, m in metrics.items()
            },
        },
        "p9": p9.as_dict(),
        "p9_context_missing": list(ctx_missing),
        "citation_diagnostic": citation_diag,
        "regression_vs_65": regression,
    }
    dest = out / "score_all_cells_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")
    return 0


def _index_clause_ids() -> set[str]:
    """색인에 실제로 있는 조항 ID. `_meta` 헤더 레코드는 건너뛴다."""
    from rag.index import load_chunks, load_rag_config

    cfg = load_rag_config()
    return {c.chunk_id for c in load_chunks(cfg.chunk_meta)}


if __name__ == "__main__":
    raise SystemExit(main())
