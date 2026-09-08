"""**다섯 칸 채점 단일 진입점.** 77번 과제 6.

    uv run python scripts/probe/score_cells.py score          # 다섯 칸 채점 + 65·66 대조
    uv run python scripts/probe/score_cells.py gate           # 게이트 재대조 (A 상수 도착 시)
    uv run python scripts/probe/score_cells.py gate --gate 0.5940
    uv run python scripts/probe/score_cells.py predict        # 스윕용 하한 추론 (CPU)
    uv run python scripts/probe/score_cells.py sweep          # 임계 스윕 + RQ2 판정

65·66번은 스크립트 두 벌로 나뉘어 있었고 임계·시드·칸 목록이 각자 살아 있었다.
지금은 전부 `evaluation.params`·`evaluation.cells` 에서 온다. 옛 스크립트는 이 파일로
넘기는 얇은 껍데기로 남겨 두었다 — 65·66 보고서의 재현 명령을 깨지 않기 위해서다.

**리팩토링의 통과 조건은 값 불변이다.** `score` 가 65번(`score_detection_v1.json`)과
66번(`score_all_cells_v1.json`)의 저장 지표를 완전 일치로 대조하고, 어긋나면 0 이
아닌 코드로 죽는다. 저장 파일을 덮어쓰지 않는다 — 증거를 지우면 대조가 무의미해진다.

**차단은 종료 코드다** (13번 D-7). `run_scoring_gates` 의 `blocking_failures` 가 비어
있지 않으면 산출물은 증거로 남기되 `score` 가 2 로 죽는다. 이전 판은 출력만 하고 0 을
돌려줘 "차단 ○" 열이 기록 이상이 아니었다. 종료 코드 표는 `evaluation/README.md`.

`score` 는 전역 지표 옆에 **같은 산출물 안에** id 구간 층화 블록을 싣고(총괄 판정 6 ·
13번 D-1), 채점 디렉터리에 `prereg_recomputed_v1.json` 이 없으면 동결본에서 재산출해
선배치한다(13번 D-8 파생). 둘 다 게이트가 매 채점마다 확인한다.

GPU 를 쓰지 않는다. `predict` 만 CPU 추론이고 나머지는 순수 채점이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import csv

from data.label_map import load_label_map
from evaluation.cells import (
    ALL_TAGS,
    DET_TAGS,
    UNI_TAGS,
    Population,
    gate_check,
    load_detection_records,
    load_population,
    load_unified_records,
    record_path,
    regression_diffs,
    score,
)
from evaluation.gates import GateContext, run_scoring_gates
from evaluation.params import (
    FROZEN_SNAPSHOT,
    ScoringParams,
    add_common_args,
    params_from_args,
    resolve_gate_status,
)
from evaluation.probes.metadata_probe import MetaSample, trivial_bound
from evaluation.probes.p9_runner import contexts_from_snapshot, p9_all_cells
from evaluation.score import coord_health, failure_breakdown
from evaluation.strata import (
    DEFAULT_K,
    ID_GRANULARITY,
    SHORTCUT_TAG,
    STRATUM_AXIS,
    stratified_table,
)

EXIT_OK = 0
EXIT_REGRESSION = 1
"""저장 지표(65·66번) 대조 불일치 — 리팩토링이 값을 바꿨다."""
EXIT_GATE_BLOCKED = 2
"""차단 게이트 실패. 산출물은 쓰이지만 **이 채점을 결과로 쓰지 마라.**"""


def exit_code(gates: dict, regressions: dict) -> tuple[int, str]:
    """`score` 의 종료 코드와 사유. **차단 게이트가 대조 불일치보다 앞선다** — 게이트
    실패는 채점 신뢰의 전제가 빈 것이고, 대조 불일치는 그 위의 정의 변경 문제다."""
    blocked = list(gates.get("blocking_failures") or [])
    if blocked:
        return EXIT_GATE_BLOCKED, (
            f"차단 게이트 실패 {blocked} — 산출물은 증거로 남겼으나 이 채점을 결과로 쓰지 마라")
    bad = [k for k, v in regressions.items() if v.get("checked") and not v["identical"]]
    if bad:
        return EXIT_REGRESSION, f"저장 지표 대조 불일치 {bad}"
    return EXIT_OK, "차단 실패 없음 · 저장 지표 대조 일치"


def stratified_block(pop: Population, by_cell: dict,
                     frozen: Path = Path(FROZEN_SNAPSHOT)) -> dict:
    """전역 지표 옆에 **같은 산출물 안에** 층화 지표를 싣는다 (총괄 판정 6 · 13번 D-1).

    절단점은 동결본 train+val 에서만 온다(A 의 `data.id_strata`). 채점 모집단이 파일럿
    부분집합이든 동결 평가셋 전량이든 같은 절단점을 받는다. K 사다리 전체를 내되 기본
    K 를 표시한다 — `stratified_scoring` 게이트가 기본 K 의 표와 지름길 행을 본다.
    구간별 상세는 `stratified_compare.py --ladder` 가 낸다(여기서는 표만).
    """
    gold = {i: sorted(pop.gold_codes.get(i, ())) for i in pop.eval_ids}
    preds = {tag: {r.image_id: sorted(r.iso_codes) for r in recs}
             for tag, recs in by_cell.items()}
    by_k = stratified_table(preds, gold, pop.classes, ID_GRANULARITY, snapshot=frozen)
    return {
        "axis": STRATUM_AXIS,
        "default_k": DEFAULT_K,
        "ladder": list(ID_GRANULARITY),
        "cut_points_from": f"{frozen} train+val (data.id_strata)",
        "by_k": by_k,
        "note": (
            "층화 Macro-F1 은 지시 문면, lift 는 완료 기준이 겨냥한 수다(83번 §1-2). "
            "지름길 규칙(__shortcut__) 행의 lift 가 0 이 아니면 층 정의나 기준선이 틀린 것이다"
        ),
    }


def raw_record_path(params: ScoringParams, tag: str) -> Path:
    """하한(export floor) 레코드. `adapt_main_detections.py` 와 `sweep_detection_conf.py` 가
    같은 자리를 쓴다."""
    return params.out / "sweep" / f"{tag}_raw_s{params.seed}.jsonl"


def _image_level(pred_codes: dict, gold: dict, classes) -> dict:
    """곡선 한 점 — **이미지 수준 지표만.** IoU 매칭·COCO 평가를 돌리지 않는다.

    곡선의 목적은 임계 의존성을 보이는 것이고, 그 의존성이 사는 축은 Macro-F1·놓침처럼
    **임계에서 잘라 세는** 지표다. 위치 축의 임계 독립 값은 하한 한 점에서 따로 낸다
    (`threshold_independent_block`) — 임계마다 mAP 를 다시 내면 잘린 PR 곡선의 그림자를
    14번 그리는 셈이고 채점 시간만 늘어난다.
    """
    from evaluation.metrics.detection import class_jaccard, score_detection

    det = score_detection(pred_codes, gold, classes)
    return {
        "macro_f1": det.macro_f1,
        "miss_rate": 1.0 - det.defect_recall,
        "defect_recall": det.defect_recall,
        "class_jaccard": class_jaccard(pred_codes, gold, classes),
    }


def _recovery_of(values: dict[str, float]) -> dict:
    """`(연합 − 로컬평균) / (중앙집중 − 로컬평균)`. 칸이 없으면 None 을 돌려준다."""
    need = ("sep_central", "sep_fed", "sep_local_C1", "sep_local_C2", "sep_local_C3")
    if not all(t in values and values[t] is not None for t in need):
        return {"recovery_pct": None, "note": "검출 5모델이 모두 있어야 산출된다"}
    locals_ = [values[t] for t in need[2:]]
    local_mean = sum(locals_) / len(locals_)
    denom = values["sep_central"] - local_mean
    return {
        "central": values["sep_central"], "fed": values["sep_fed"],
        "local_mean": local_mean, "denominator": denom,
        "recovery_pct": ((values["sep_fed"] - local_mean) / denom * 100)
        if abs(denom) > 1e-12 else None,
    }


def curve_block(params: ScoringParams, pop: Population, det_tags) -> dict:
    """**등록된 격자 전 구간의 곡선.** 채점의 정규 단계다 (총괄 판정 1 · 22번 §1-2).

    단일 임계 한 점은 확증 기준이 아니다 — 17번 §3-5 에서 연합↔로컬평균의 대소가 두 축
    모두 뒤집혔고, 총괄은 헤드라인을 곡선 + 임계 독립 지표로 옮겼다. 그래서 이 블록은
    옵션이 아니라 매 채점마다 나온다: 곡선이 없으면 `sweep_curve_recorded` 게이트가 막는다.

    입력은 **하한 레코드**(`sweep/{tag}_raw_s{seed}.jsonl`)다. 하한 1회 + 사후 필터가
    임계별 재추론과 같다는 동치 위에 서 있고, 그 동치는 파일럿 비트 대조 + 본실험 표본
    재확인으로 받친다(`scripts/probe/verify_filter_parity.py`).
    """
    from evaluation.adapters import read_records
    from evaluation.detect_infer import filter_by_conf

    gold = {k: sorted(v) for k, v in pop.gold_codes.items()}
    by_tag: dict[str, dict] = {}
    missing: list[str] = []
    for tag in det_tags:
        src = raw_record_path(params, tag)
        if not src.exists():
            missing.append(tag)
            continue
        raw = read_records(src.read_text(encoding="utf-8").splitlines())
        points: dict[str, dict] = {}
        for thr in params.conf_sweep:
            cut = filter_by_conf(raw, thr)
            codes = {r.image_id: sorted(r.iso_codes) for r in cut}
            points[f"{thr:.2f}"] = {
                "n_boxes": sum(len(r.defects) for r in cut),
                **_image_level(codes, gold, pop.classes),
            }
        by_tag[tag] = {"n_boxes_floor": sum(len(r.defects) for r in raw),
                       "by_threshold": points}

    # 임계별 회복률과 **부호가 뒤집히는지**. 뒤집히면 단일 임계 결론이 성립하지 않는다.
    recovery: dict[str, dict] = {}
    flips: dict[str, dict] = {}
    if not missing and by_tag:
        for key in ("macro_f1", "miss_rate"):
            per_thr = {}
            signs_fl, signs_fc = set(), set()
            for thr in params.conf_sweep:
                k = f"{thr:.2f}"
                vals = {t: by_tag[t]["by_threshold"][k][key] for t in by_tag}
                per_thr[k] = _recovery_of(vals)
                lm = per_thr[k]["local_mean"]
                f, c = vals["sep_fed"], vals["sep_central"]
                signs_fl.add((f > lm) - (f < lm))
                signs_fc.add((f > c) - (f < c))
            recovery[key] = per_thr
            flips[key] = {
                "fed_vs_local_mean_flips": len(signs_fl) > 1,
                "fed_vs_central_flips": len(signs_fc) > 1,
                "signs_fed_vs_local_mean": sorted(signs_fl),
                "signs_fed_vs_central": sorted(signs_fc),
            }

    return {
        "grid": list(params.conf_sweep),
        "grid_source": params.conf_sweep_source,
        "input": "하한 레코드(sweep/{tag}_raw_s{seed}.jsonl) + filter_by_conf",
        "metrics_computed": ["n_boxes", "macro_f1", "miss_rate", "defect_recall",
                             "class_jaccard"],
        "by_tag": by_tag,
        "missing_tags": missing,
        "recovery_by_threshold": recovery,
        "threshold_dependence": flips,
        "note": (
            "단일 임계 한 점은 확증 기준이 아니다(총괄 판정 1, 22번). 격자는 사전등록 "
            "대상이고 단일 임계는 아니다 — 격자는 결과와 무관하게 정할 수 있어 사후 선택이 "
            "아니기 때문이다. 대소가 뒤집히는 축은 threshold_dependence 가 말한다"
        ),
    }


def threshold_independent_block(params: ScoringParams, pop: Population, det_tags) -> dict:
    """**임계 독립 헤드라인 지표.** 하한 레코드 한 점에서만 낸다 (총괄 판정 1).

    - `map_50`·`map_50_95`: PR 곡선 전 구간 적분(위치 축)
    - `macro_ap`: 이미지 수준 macro Average Precision(분류 축 — Macro-F1 의 임계 독립 대응물)

    **"임계 독립"의 정확한 범위**(22번 과제 2가 요구한 주석): 운용 conf 임계에는 독립이다.
    그러나 아래에는 **여전히 의존한다** — 이 값들을 "모든 설정에 불변"으로 읽으면 안 된다.

    1. **export 하한 자체**(`conf_floor`, C 가 0.01 로 내보냈다). 그 아래 박스는 존재하지 않는다
    2. **NMS IoU**(Ultralytics 기본값) — 추론 시점에 이미 적용돼 레코드에 반영돼 있다
    3. **매칭 IoU**(mAP@50 은 0.50, mAP@50:95 는 0.50~0.95 평균)
    4. **`max_det`**(300) — 이미지당 상한에 걸리면 재현율 꼬리가 잘린다
    5. 모집단·정답(동결 스냅샷)과 채점 클래스 4종

    즉 임계 독립은 **"운용점을 고르지 않아도 된다"**는 뜻이지 "자유 매개변수가 없다"가 아니다.
    """
    from evaluation.adapters import read_records

    per_tag: dict[str, dict] = {}
    for tag in det_tags:
        src = raw_record_path(params, tag)
        if not src.exists():
            continue
        recs = read_records(src.read_text(encoding="utf-8").splitlines())
        m = score(pop, recs)
        per_tag[tag] = {
            "n_boxes_floor": sum(len(r.defects) for r in recs),
            "map_50": m["map_50"], "map_50_95": m["map_50_95"],
            "macro_ap": m["macro_ap"], "per_class_ap": m["per_class_ap"],
            "scores_present": m["scores_present"],
        }
    recovery = {
        key: _recovery_of({t: v[key] for t, v in per_tag.items()})
        for key in ("map_50", "map_50_95", "macro_ap")
    } if per_tag else {}
    return {
        "computed_at": f"conf >= {params.conf_floor} (export 하한, 임계 미적용)",
        "primary": "map_50",
        "per_tag": per_tag,
        "recovery": recovery,
        "independent_of": "운용 conf 임계 (PR 곡선 전 구간 적분)",
        "still_depends_on": {
            "conf_floor": params.conf_floor,
            "nms_iou": "Ultralytics 기본값 — 추론 시점에 적용됨",
            "match_iou": "map_50=0.50 · map_50_95=0.50~0.95 · macro_ap 은 이미지 수준이라 IoU 무관",
            "max_det": params.max_det,
            "population": str(params.snapshot),
            "classes": list(params.class_names),
        },
        "note": ("임계 독립은 '운용점을 고르지 않아도 된다'는 뜻이지 자유 매개변수가 "
                 "없다는 뜻이 아니다 (22번 과제 2)"),
    }


def discrimination_block(pop: Population, by_cell: dict, prov: dict) -> dict:
    """**출처 고정 판별력 Δ** — 지름길이 정의상 통과할 수 없는 유일한 축 (22번 §5 과제 4).

    `evaluation/discrimination.py` 는 09-02 에 구현·시험까지 끝났는데 본채점 진입점이
    부르지 않아 값이 산출물에 실린 적이 없었다. 의사결정로그가 이 지표를 "지름길이 정의상
    통과할 수 없는 유일한 축" 으로 적어 둔 이상, 무내용 대조가 문제가 된 지금이 그 값을
    표에 올릴 자리다.

        판별력 Δ = (결함 이미지 발화율) − (정상 이미지 발화율)   [출처 고정, N-crop]

    출처를 상수로 묶은 구간 안에서는 출처만 읽는 예측기가 상수 예측기로 퇴화하므로 Δ 가
    정확히 0 이 된다. **0 을 넘으려면 이미지를 봐야 한다.** 0 은 실패선이 아니라 기준선이고,
    음수는 정상에서 더 발화했다는 뜻이다(역전).

    맥락은 P9 와 달리 **결함·정상 둘 다** 필요하다 — P9 는 정상만 본다.
    """
    from evaluation.discrimination import CROP, score_discrimination_all_cells

    contexts = {
        r["image_id"]: (r["group_id"], r["has_defect"] == "True",
                        prov.get(r["image_id"], ""))
        for r in pop.rows
    }
    records = [rec for recs in by_cell.values() for rec in recs]
    reports = score_discrimination_all_cells(records, contexts, provenance=CROP)
    rows = [r.as_dict() for r in reports]
    return {
        "provenance": CROP,
        "definition": "결함 발화율 − 정상 발화율 (출처 고정, 묶음 클러스터 부트스트랩 CI)",
        "baseline": "0 = 이미지를 안 봤다. 지름길은 정의상 0 을 넘을 수 없다",
        "n_context": len(contexts),
        "results": rows,
        "promotion_note": (
            "헤드라인 보조지표 승격은 총괄 판정 사항이다(22번 §5 과제 4). 여기서는 값과 "
            "판정 재료만 낸다"
        ),
    }


def decomposition_block(params: ScoringParams, pop: Population, det_tags) -> dict:
    """**귀속 축 분해** — 회복률이 80% 미만일 때 결과표 템플릿이 요구하는 분해 (과제 5).

    **평가셋에는 클라이언트 열이 없다.** 회사별 분할보다 먼저 뗐기 때문이다(불변조건 1-3,
    실측: eval 12,461행의 `client` 열이 전부 빈 문자열). 그래서 3분할 클라이언트 분해는
    원리적으로 불가능하고, 대신 **재질 축**으로 분해한다 — 분할 설계상 `AL ≡ C3`(알루미늄
    단독 클라이언트)이고 `ST ≡ C1 ∪ C2` 다. C1 과 C2 는 둘 다 강재이고 Dirichlet 로 갈렸으므로
    평가셋에서 분리되지 않는다. 이 한계를 값과 함께 싣는다.

    분해는 **하한 레코드**(임계 독립 축)에서 낸다 — 헤드라인이 거기 있기 때문이다.
    """
    from evaluation.adapters import read_records

    groups: dict[str, list[str]] = {}
    for r in pop.rows:
        groups.setdefault(r["material"], []).append(r["image_id"])
    out: dict[str, dict] = {}
    for material, ids in sorted(groups.items()):
        idset = set(ids)
        gold_codes = {i: pop.gold_codes.get(i, set()) for i in ids}
        gold_boxes = {i: pop.gold_boxes.get(i, []) for i in ids}
        n_defect = sum(1 for i in ids if gold_codes[i])
        per_tag: dict[str, dict] = {}
        for tag in det_tags:
            src = raw_record_path(params, tag)
            if not src.exists():
                continue
            recs = [r for r in read_records(src.read_text(encoding="utf-8").splitlines())
                    if r.image_id in idset]
            from evaluation.score import score_records

            m = score_records(recs, gold_codes, gold_boxes, pop.classes)
            per_tag[tag] = {"macro_f1": m["macro_f1"], "miss_rate": m["miss_rate"],
                            "macro_ap": m["macro_ap"], "map_50": m["map_50"],
                            "n_images": len(recs)}
        out[material] = {
            "n_images": len(ids), "n_defect": n_defect, "n_normal": len(ids) - n_defect,
            "per_tag": per_tag,
            "recovery": {key: _recovery_of({t: v[key] for t, v in per_tag.items()})
                         for key in ("map_50", "macro_ap", "macro_f1", "miss_rate")}
            if per_tag else {},
        }
    return {
        "axis": "material",
        "computed_at": f"conf >= {params.conf_floor} (하한 — 임계 독립 축)",
        "client_mapping": {"AL": "C3 (알루미늄 단독 클라이언트)",
                           "ST": "C1 ∪ C2 (강재 두 클라이언트, 평가셋에서 분리 불가)"},
        "limitation": (
            "평가셋은 회사별 분할보다 먼저 뗐으므로 `client` 열이 비어 있다(실측 12,461행 전부). "
            "3분할 클라이언트 분해는 원리적으로 불가능하고 재질이 유일한 귀속 축이다"
        ),
        "by_group": out,
    }


def shortcut_footnote(p9: dict, strata: dict) -> dict:
    """**전역 지표 표에 자동으로 달리는 각주** (총괄 판정 2 · 22번 §2-2-2).

    "이 점수의 일부는 규격 지름길이다"를 사람이 기억해서 다는 문장으로 두지 않는다 —
    산출물이 스스로 달게 한다. 두 지름길을 각각의 실측에서 읽는다.

    - **함정 #11 규격 지름길**: 정상 이미지의 오탐이 출처(크롭/타일)로 쏠리는가. P9 의
      출처별 분해에서 읽는다. TOST 동등이 아니면 전역 점수의 일부가 "결함이라서"가 아니라
      "크롭이라서" 나온 것이다.
    - **함정 #12 id 축 지름길**: 구간 최빈 라벨 규칙만으로 전역 Macro-F1 이 얼마까지 가는가.
      층화 블록의 지름길 행에서 읽는다.

    문장을 만들되 **수치는 실측에서 온다** — 하드코딩한 경고문은 값이 바뀌어도 그대로 남는다.
    """
    lines: list[str] = []
    p9_rows = list(p9.get("results") or [])
    not_equiv = [r for r in p9_rows if not (r.get("tost") or {}).get("equivalent")]
    if p9_rows:
        worst = min(p9_rows, key=lambda r: (r.get("fp_rate_diff") or {}).get("point", 0.0))
        crop = (worst.get("fp_rate_crop") or {}).get("point")
        tile = (worst.get("fp_rate_tile") or {}).get("point")
        lines.append(
            f"함정 #11(규격 지름길): 정상 이미지 오탐이 출처로 쏠린다 — TOST 동등 아님 "
            f"{len(not_equiv)}/{len(p9_rows)}칸. 최대 격차 {worst.get('tag', worst.get('cell'))}: "
            f"N-crop {crop:.4f} 대 N-tile {tile:.4f}. **전역 지표의 일부는 결함이 아니라 "
            f"크롭 규격에서 온다.** 출처별 분리 표를 반드시 함께 읽는다"
            if crop is not None and tile is not None else
            "함정 #11: P9 출처별 분해가 비어 있다"
        )
    k0 = str(strata.get("default_k", ""))
    row = (strata.get("by_k", {}).get(k0, {}) or {}).get(SHORTCUT_TAG, {})
    if row:
        lines.append(
            f"함정 #12(id 축 지름길): 구간 최빈 라벨 규칙만으로 전역 Macro-F1 "
            f"{row.get('global_macro_f1'):.4f} (K={k0}, 평가셋 적합 상한). "
            f"**전역 Macro-F1 하나로는 픽셀을 본 모델과 id 구간을 본 규칙을 가를 수 없다** — "
            f"층화 lift 를 함께 읽는다"
        )
    return {
        "applies_to": "metrics (전역 지표 표)",
        "n_p9_cells_not_equivalent": len(not_equiv),
        "n_p9_cells": len(p9_rows),
        "lines": lines,
        "source": "P9 출처별 분해 + 층화 지름길 행 (실측에서 유도, 하드코딩 아님)",
    }


def selected_tags(args) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`--cells` → (검출 태그, 통합형 태그). **빠진 칸을 산출물이 말하게 한다.**

    본실험 시드 1 은 검출 3칸(5모델)만 돌았다 — 통합형 두 칸은 아직 학습 전이다. 그때
    `ALL_TAGS` 를 그대로 돌리면 통합형 원시 출력 부재로 채점이 죽고, 예외를 삼키면
    "다섯 칸 채점"이라는 이름 아래 세 칸만 채점된 산출물이 남는다. 선택을 인자로 올려
    산출물의 `cells_scored` 에 적는다.
    """
    if getattr(args, "cells", "all") == "det":
        return DET_TAGS, ()
    return DET_TAGS, UNI_TAGS


def score_all(
    params: ScoringParams, pop: Population, uni_tags: Sequence[str] = UNI_TAGS
) -> tuple[dict, dict, dict, list, dict]:
    """다섯 칸 전부 채점. 검출은 저장 레코드 되읽기, 통합형은 어댑터 경유."""
    known = set(load_label_map().iso_codes())
    metrics: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    adapters: dict[str, dict] = {}
    all_records: list = []

    by_cell: dict[str, list] = {}
    for tag in DET_TAGS:
        recs = load_detection_records(params, tag)
        all_records.extend(recs)
        by_cell[tag] = recs
        metrics[tag] = score(pop, recs)
        failures[tag] = failure_breakdown(recs)

    for cell in uni_tags:
        # 원시 생성문에서 매번 새로 어댑트한다. 저장본을 되읽으면 어댑터가 채점 경로에서
        # 빠져 "칸이 갈리는 유일한 지점"이 검증 대상 밖으로 나간다.
        rep = load_unified_records(params, pop, cell, known)
        recs = rep.records
        adapters[cell] = rep.as_dict()
        dest = record_path(params, cell)
        if dest.exists():
            stored = load_detection_records(params, cell)   # 같은 되읽기 경로
            adapters[cell]["stored_matches"] = (
                [r.model_dump_json() for r in stored] == [r.model_dump_json() for r in recs]
            )
        else:
            with dest.open("w", encoding="utf-8", newline="\n") as fh:
                for r in recs:
                    fh.write(r.model_dump_json() + "\n")
        missing = pop.eval_ids - {r.image_id for r in recs}
        extra = {r.image_id for r in recs} - pop.eval_ids
        if missing or extra:
            raise SystemExit(f"{cell}: 평가셋 불일치 결측 {len(missing)} 초과 {len(extra)}")
        all_records.extend(recs)
        by_cell[cell] = recs
        metrics[cell] = score(pop, recs)
        failures[cell] = failure_breakdown(recs)
        adapters[cell]["citations"] = rep.citations

    return metrics, failures, adapters, all_records, by_cell


def recovery(metrics: dict) -> dict:
    """회복률 = (연합 − 로컬) / (중앙집중 − 로컬). 헤드라인 숫자(개발규약).

    통합·로컬은 실험 구조상 '제외' 칸이라 분모가 없다. 그쪽은 유지율만 낸다 —
    **두 값은 다른 양이다.** 같은 표에 나란히 두면 읽는 사람이 섞는다.
    """
    def f1(tag: str) -> float:
        return float(metrics[tag]["macro_f1"])

    locals_ = ("sep_local_C1", "sep_local_C2", "sep_local_C3")
    local_mean = sum(f1(k) for k in locals_) / len(locals_)
    denom = f1("sep_central") - local_mean
    # 통합형 두 칸이 아직 없으면(검출 선행 구간) **0 을 지어내지 않는다.** 없는 것과
    # 0 인 것을 섞으면 유지율이 거짓 수를 낸다.
    uni = (
        {
            "central": f1("uni_central"), "fed": f1("uni_fed"),
            "local_mean": None, "recovery_pct": None,
            "retention_pct": f1("uni_fed") / f1("uni_central") * 100
            if f1("uni_central") > 0 else None,
            "note": "통합·로컬은 '제외' 칸이라 회복률 분모가 없다. 유지율만 낸다",
        }
        if {"uni_central", "uni_fed"} <= set(metrics)
        else {"central": None, "fed": None, "local_mean": None, "recovery_pct": None,
              "retention_pct": None,
              "note": "통합형 두 칸 미채점 — 이 채점에 없다(0 이 아니라 부재)"}
    )
    return {
        "basis": "macro_f1",
        "separated": {
            "central": f1("sep_central"), "fed": f1("sep_fed"), "local_mean": local_mean,
            "locals": {k: f1(k) for k in locals_},
            "denominator": denom,
            "recovery_pct": (f1("sep_fed") - local_mean) / denom * 100
            if denom > 0 else None,
        },
        "unified": uni,
        "caveat": "시드 1세트 — 시드 3세트 집계 전까지 경향만. 결론으로 쓰지 않는다",
    }


def diagnostics(params: ScoringParams, pop: Population, all_records: list,
                adapters: dict, uni_tags: Sequence[str] = UNI_TAGS) -> dict:
    """P9(규격 지름길) · 자명하한 · 통합형 인용 진단. 66번이 내던 것을 그대로 옮겼다."""
    with (params.snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}
    contexts, ctx_missing = contexts_from_snapshot(pop.rows, prov)
    p9 = p9_all_cells(all_records, contexts)

    samples = [
        MetaSample(
            image_id=r["image_id"], width_px=int(r["width_px"]),
            height_px=int(r["height_px"]), file_bytes=0, n_channels=1, quant_table_id=0,
            iso_codes=tuple(sorted(pop.gold_codes.get(r["image_id"], ()))),
        )
        for r in pop.rows
    ]

    from rag.index import load_chunks, load_rag_config

    index_ids = {c.chunk_id for c in load_chunks(load_rag_config().chunk_meta)}
    citation_diag = {}
    for cell in uni_tags:
        cited = adapters.get(cell, {}).get("citations", {}) or {}
        flat = [c for v in cited.values() for c in v]
        citation_diag[cell] = {
            "n_citations": len(flat),
            "n_images_citing": sum(1 for v in cited.values() if v),
            "n_in_index": sum(1 for c in flat if c in index_ids),
            "n_not_in_index": sum(1 for c in flat if c not in index_ids),
            "distinct": sorted(set(flat))[:20],
        }
    return {
        "p9": p9.as_dict(),
        "p9_context_missing": list(ctx_missing),
        "sample_trivial_bound": trivial_bound(samples, pop.classes),
        "citation_diagnostic": citation_diag,
        "_provenance": prov,      # 판별력 Δ 가 같은 출처 표를 쓴다(두 번 읽지 않는다)
    }


def check_regressions(params: ScoringParams, metrics: dict) -> dict:
    """65·66번 저장 지표와의 완전 일치 대조. 저장 파일은 읽기만 한다."""
    out: dict[str, dict] = {}
    for label, fname, tags in (
        ("65번", "score_detection_v1.json", DET_TAGS),
        ("66번", "score_all_cells_v1.json", ALL_TAGS),
    ):
        p = params.out / fname
        if not p.exists():
            out[label] = {"checked": False, "reason": f"{fname} 없음"}
            continue
        stored = json.loads(p.read_text(encoding="utf-8"))["metrics"]
        diffs = regression_diffs(stored, metrics, tags)
        out[label] = {"checked": True, "n_cells": len(tags), "diffs": diffs,
                      "identical": not diffs}
    return out


def cmd_score(args) -> int:
    params = params_from_args(args)
    params.out.mkdir(parents=True, exist_ok=True)
    pop = load_population(params)
    det_tags, uni_tags = selected_tags(args)
    tags = (*det_tags, *uni_tags)
    print(f"평가셋 {pop.n_eval}장 (정상 {pop.n_normal}) · 칸 {len(tags)}개 {list(tags)}")

    metrics, failures, adapters, all_records, by_cell = score_all(params, pop, uni_tags)
    for tag in tags:
        m = metrics[tag]
        print(f"[{tag}] macroF1 {m['macro_f1']:.4f} · miss {m['miss_rate']:.4f} "
              f"· IoU {m['bbox_iou']:.4f}")

    reg = check_regressions(params, metrics)
    diag = diagnostics(params, pop, all_records, adapters, uni_tags)
    strata = stratified_block(pop, by_cell)

    # 총괄 판정 1 (22번) — 곡선과 임계 독립 지표는 **정규 단계**다. 옵션이 아니다.
    indep = threshold_independent_block(params, pop, det_tags)
    curve = curve_block(params, pop, det_tags)
    footnote = shortcut_footnote(diag["p9"], strata)
    # 22번 §5 — 판별력 Δ 배선(과제 4)과 귀속 축 분해(과제 5).
    discrim = discrimination_block(pop, by_cell, diag.pop("_provenance"))
    decomp = decomposition_block(params, pop, det_tags)
    for r in discrim["results"]:
        tag = r["cell"] if not r.get("client") else f"{r['cell']}_{r['client']}"
        print(f"[판별력] {tag}: Δ {r['delta']['point']:+.4f} "
              f"[{r['delta']['ci_lo']:+.4f}, {r['delta']['ci_hi']:+.4f}] "
              f"(결함 {r['fire_rate_defect']:.4f} · 정상 {r['fire_rate_normal']:.4f})")
    ti = indep.get("recovery", {}).get(indep["primary"], {})
    if ti.get("recovery_pct") is not None:
        print(f"[임계 독립] {indep['primary']} 회복률 {ti['recovery_pct']:.1f}% "
              f"(하한 {params.conf_floor} 에서 산출 · 분모 {ti['denominator']:.4f})")
    else:
        print("[임계 독립] 회복률 산출 불가 — 검출 5모델이 모두 있어야 한다")
    for key, f in curve.get("threshold_dependence", {}).items():
        print(f"[곡선] {key}: 연합↔로컬평균 "
              f"{'뒤집힘' if f['fed_vs_local_mean_flips'] else '유지'} · 연합↔중앙집중 "
              f"{'뒤집힘' if f['fed_vs_central_flips'] else '유지'}")
    k0 = str(strata["default_k"])
    for tag in [*tags, SHORTCUT_TAG]:
        s = strata["by_k"][k0][tag]
        print(f"[{tag}] 층화(K={k0}) macroF1 {s['stratified_macro_f1']:.4f} · "
              f"lift {s['stratified_lift']:+.5f} · 비순수 lift {s['stratified_lift_impure']:+.5f}")
    gates = run_scoring_gates(GateContext(
        metrics=metrics,
        records_by_cell=by_cell,
        expected_coord_space=params.coord_space,
        population_bound=diag["sample_trivial_bound"],
        n_eval=pop.n_eval,
        n_scored={t: len(v) for t, v in by_cell.items()},
        recovery=recovery(metrics),
        seed_sd=None,                 # 시드 1세트 — 게이트가 그 사실을 판정으로 남긴다
        env=None,                     # 실제 프로세스 환경을 본다
        tags=None,                    # 채점 단계에는 run 태그가 없다. 차단하지 않는다
        gate_status=resolve_gate_status(),
        extra={
            "gate_pass_line": params.gate_pass_line,
            "measured_prereg": _measured_prereg(params),
            "stratified": strata,
            "curve": curve,
            "threshold_independent": indep,
            "p9": diag["p9"],
            "shortcut_footnote": footnote,
        },
    ))
    code, why = exit_code(gates, reg)
    payload = {
        "params": params.as_dict(),
        "n_eval": pop.n_eval,
        "cells_scored": list(tags),
        "cells_selection": getattr(args, "cells", "all"),
        "scorer": "evaluation.score.score_records (단일)",
        "metrics": metrics,
        "metrics_role": (
            f"운용점 예시 conf={params.conf.value} — **확증적 기준 아님**(총괄 판정 1, "
            "22번 §1-2-3). 헤드라인은 threshold_independent, 칸 비교는 curve 로 한다"
        ),
        "threshold_independent": indep,
        "curve": curve,
        "discrimination": discrim,
        "decomposition": decomp,
        "shortcut_footnote": footnote,
        "stratified": strata,
        "failures": failures,
        "adapters": adapters,
        "coord_health": {t: coord_health(m) for t, m in metrics.items()},
        "recovery": recovery(metrics),
        "gate": gate_check(metrics, params),
        "regression": reg,
        "gates_evaluated": gates,
        "exit_code": code,
        "exit_reason": why,
        **diag,
    }
    dest = params.out / "score_cells_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")

    print(f"게이트 {gates['n_evaluated']}/{gates['n_registered']} 평가 · "
          f"차단 실패 {gates['blocking_failures'] or '없음'} · "
          f"건너뜀 {gates['n_skipped']}")
    for r in gates["results"]:
        if not r["passed"] and not r["skipped"]:
            mark = "차단" if r["blocking"] else "기록"
            print(f"  [{mark}] {r['name']}: {r['detail'][:120]}")

    bad = [k for k, v in reg.items() if v.get("checked") and not v["identical"]]
    for k in bad:
        print(f"{k} 대조 불일치: {reg[k]['diffs'][:3]}")
    if not bad:
        print("65·66번 대조: 완전 일치" if any(v.get("checked") for v in reg.values())
              else "65·66번 대조: 저장본 없음 — 미대조")
    print(f"게이트: {payload['gate']['verdict']} (선 {params.gate_pass_line}, "
          f"{params.gate.source})")
    print(f"종료 코드 {code} — {why}")
    return code


def cmd_gate(args) -> int:
    """게이트 상수만 갈아 끼워 다섯 칸을 다시 판정한다. 채점 자체는 건드리지 않는다.

    A 의 content-free 천장 재산출이 도착하면 `--gate <값>` 한 번, 혹은
    `configs/base.yaml` 에 키가 생기면 플래그 없이도 그 값으로 돈다.
    """
    params = params_from_args(args)
    src = params.out / "score_cells_v1.json"
    if not src.exists():
        print(f"{src} 없음 — 먼저 `score` 를 돌린다")
        return 1
    metrics = json.loads(src.read_text(encoding="utf-8"))["metrics"]
    result = gate_check(metrics, params)
    best = best_case_gate(params)
    dest = params.out / f"gate_recheck_g{params.gate.value:.4f}_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({"params": params.as_dict(), "gate_result": result,
                   "best_case_over_sweep": best}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    for tag, v in result["cells"].items():
        mark = "통과" if v["above_pass_line"] else "미달"
        print(f"[{tag}] {v['macro_f1']:.4f} {mark} (여유 {v['margin_vs_pass_line']:+.4f})")
    print(f"판정(운용 임계 {params.conf.value}): {result['verdict']} · "
          f"통과선 {result['pass_line']} ({params.gate.source})")
    if best.get("checked"):
        print(f"판정(임계 최선): {best['verdict']} — "
              f"{best['n_above_pass_line']}/{best['n_cells']} 통과")
    print(f"저장: {dest}")
    return 0


def best_case_gate(params: ScoringParams) -> dict:
    """**임계를 가장 유리하게 잡아도** 게이트를 넘는 칸이 있는가.

    운용 임계 하나에서의 미달은 "임계를 잘못 골라서"라는 반론을 받는다. 스윕 전 구간의
    칸별 최댓값으로 대면 그 반론이 닫힌다 — 분리형에 최대한 유리한 조건이다.
    통합형은 임계가 없으므로 한 점 그대로다.
    """
    p = params.out / "sweep_detection_conf_v1.json"
    if not p.exists():
        return {"checked": False, "reason": "sweep_detection_conf_v1.json 없음"}
    d = json.loads(p.read_text(encoding="utf-8"))
    cells: dict[str, dict] = {}
    for tag, s in d["sweep"].items():
        arg, val = max(((k, v["macro_f1"]) for k, v in s["by_threshold"].items()),
                       key=lambda kv: kv[1])
        cells[tag] = {"macro_f1": val, "at_conf": arg,
                      "above_pass_line": val > params.gate_pass_line}
    for cell, u in d["unified"].items():
        val = u["metrics"]["macro_f1"]
        cells[cell] = {"macro_f1": val, "at_conf": "임계 없음",
                       "above_pass_line": val > params.gate_pass_line}
    n = sum(1 for v in cells.values() if v["above_pass_line"])
    return {
        "checked": True, "pass_line": params.gate_pass_line,
        "sweep_grid": d["params"]["conf_sweep"],
        "n_cells": len(cells), "n_above_pass_line": n, "cells": cells,
        "verdict": ("임계를 가장 유리하게 잡아도 전 칸 미달" if n == 0
                    else f"임계 최선에서 {n}/{len(cells)} 통과"),
    }


def cmd_predict(args) -> int:
    """검출 3칸 CPU 추론.

    기본은 **스윕용 하한**(`conf_floor`)이다. `--at-conf` 를 주면 운용 임계로 65번과
    같은 레코드를 만든다 — 임계가 인자인 것이 이 리팩토링의 핵심이다. 65번은 이 값을
    모듈 상수로 박아 분리형만 잘린 뒤 채점됐다(감사 D-1).
    """
    from evaluation.detect_infer import (
        cell_tag,
        checkpoint_paths,
        load_yolo_from_npz,
        predict_cell,
    )
    from evaluation.params import CONF_FLOOR
    from tracking.mlflow_local import reject_best_checkpoint

    params = params_from_args(args)
    pop = load_population(params)
    conf = params.conf.value if args.at_conf else params.conf_floor
    sub = params.out if args.at_conf else params.out / "sweep"
    sub.mkdir(parents=True, exist_ok=True)
    print(f"평가셋 {pop.n_eval}장 · conf {conf} "
          f"({'운용' if args.at_conf else f'하한 {CONF_FLOOR}'}) · "
          f"프로파일 {params.profile} ({params.model_cfg} · imgsz {params.imgsz} · "
          f"청크 {params.predict_chunk})")

    for (cell, client), ckpt in checkpoint_paths(params.pilot).items():
        reject_best_checkpoint(ckpt)
        if not ckpt.exists():
            print(f"체크포인트 없음: {ckpt}")
            return 1
        tag = cell_tag(cell, client)
        yolo = load_yolo_from_npz(ckpt, params.class_names, params.imgsz,
                                  model_cfg=params.model_cfg)
        recs = predict_cell(yolo, pop.rows, Path(args.root), cell, client, params, conf=conf)
        name = f"{tag}_s{params.seed}.jsonl" if args.at_conf \
            else f"{tag}_raw_s{params.seed}.jsonl"
        with (sub / name).open("w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(r.model_dump_json() + "\n")
        print(f"[{tag}] {sum(len(r.defects) for r in recs)}박스 -> {name}")
    return 0


def cmd_sweep(args) -> int:
    from scripts.probe.sweep_detection_conf import main as sweep_main

    argv = ["sweep_detection_conf.py", "--stage", "sweep",
            "--snapshot", args.snapshot, "--pilot", args.pilot, "--out", args.out]
    if args.conf is not None:
        argv += ["--conf", str(args.conf)]
    sys.argv = argv
    return sweep_main()


def main() -> int:
    ap = argparse.ArgumentParser(description="다섯 칸 채점 단일 진입점")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("score", cmd_score), ("gate", cmd_gate),
                     ("predict", cmd_predict), ("sweep", cmd_sweep)):
        p = sub.add_parser(name)
        add_common_args(p)
        p.add_argument("--root", default=".")
        p.add_argument("--at-conf", action="store_true",
                       help="predict: 하한이 아니라 운용 임계로 추론한다(65번 레코드 생성)")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


def _measured_prereg(params: ScoringParams) -> dict:
    """채점 디렉터리의 `prereg_recomputed_v1.json`. **없으면 동결본에서 재산출해 선배치한다.**

    13번 D-8 파생: 파일이 없으면 `prereg_constants_reproduced` 게이트가 `skipped` 로
    갈렸다 — 검증에서 실측된 구멍이다. 상수는 동결 스냅샷의 결정론적 함수이므로 채점기가
    스스로 만들어 두는 것이 사람 손 선배치보다 안전하다. 파일은 남겨 다음 채점이 되읽고,
    digest 와 출처를 게이트 value 에 실어 어느 동결본에서 왔는지가 표에 남게 한다.
    """
    from scripts.probe.recompute_prereg import FILE_NAME, recompute, write_payload

    p = params.out / FILE_NAME
    if not p.exists():
        write_payload(recompute(), p)
        print(f"사전등록 상수 재산출 → 선배치 {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = d["populations"]["frozen_total"]
    return {
        "all_positive_macro_f1": tot["all_positive_macro_f1"],
        "spec_only_macro_f1": tot["spec_only_macro_f1"],
        "snapshot_digest": d.get("snapshot_digest"),
        "source": str(p),
    }


if __name__ == "__main__":
    raise SystemExit(main())

