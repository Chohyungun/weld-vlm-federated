"""다섯 칸 채점의 공통 배관 — 진입점이 몇 개든 같은 함수를 타게 한다. 77번 과제 6.

감사 전까지 채점 코드는 세 곳에 흩어져 있었다.

    scripts/probe/score_detection_cells.py   추론 + 채점 (65번)
    scripts/probe/score_all_cells.py         되읽기 + 어댑터 + 채점 (66번)
    evaluation/score.py                      실제 지표 함수

`evaluation/score.py` 가 단일 채점기인 것은 맞았지만, **정답을 만드는 경로·칸 목록·
임계가 스크립트마다 따로 살아 있었다.** 단일 채점기 주장은 지표 함수 하나로 서지
않는다 — 같은 정답, 같은 모집단, 같은 파라미터를 통과해야 성립한다. 그 세 가지를
여기 모은다.

- 정답·모집단: `load_population()` 한 지점
- 칸 목록: `DET_TAGS` · `UNI_TAGS`
- 파라미터: `evaluation.params`
- 지표: `evaluation.score.score_records`

칸 이름으로 분기하는 코드는 여기에도 없다. 칸이 갈리는 곳은 **레코드를 만드는 방식**
하나뿐이고(검출은 추론·되읽기, 통합형은 어댑터), 계약 #4 레코드가 만들어진 뒤로는
다섯 칸이 비트 단위로 같은 경로를 탄다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data.label_map import load_label_map
from evaluation.adapters import adapt_unified_generations, read_records
from evaluation.detect_infer import CHECKPOINTS, cell_tag
from evaluation.eval_set import eval_rows as select_eval
from evaluation.eval_set import image_sizes, read_gold, read_manifest
from evaluation.params import ScoringParams
from evaluation.schema import PredictionRecord
from evaluation.score import score_records

DET_TAGS: tuple[str, ...] = tuple(cell_tag(c, cl) for c, cl, _ in CHECKPOINTS)
UNI_TAGS: tuple[str, ...] = ("uni_central", "uni_fed")
ALL_TAGS: tuple[str, ...] = DET_TAGS + UNI_TAGS

REGRESSION_KEYS: tuple[str, ...] = (
    "macro_f1", "defect_recall", "miss_rate", "class_jaccard",
    "bbox_iou", "bbox_iou_matched_only", "n_gold", "n_matched",
    "coord_suspect", "map_50", "map_50_95",
)
"""채점기를 옮겨도 변하면 안 되는 키. 부동소수 근사 비교를 하지 않고 완전 일치를 본다."""


@dataclass(frozen=True)
class Population:
    """채점 모집단과 정답. **다섯 칸이 같은 인스턴스를 쓴다는 것이 공정성의 한 축이다.**"""

    rows: list[dict]
    eval_ids: set[str]
    gold_codes: dict[str, set[str]]
    gold_boxes: dict[str, list]
    sizes: dict[str, tuple[int, int]]
    classes: list[str]

    @property
    def n_eval(self) -> int:
        return len(self.rows)

    @property
    def n_normal(self) -> int:
        return sum(1 for r in self.rows if r["has_defect"] == "False")


def load_population(params: ScoringParams) -> Population:
    """동결 스냅샷 → 평가 모집단. 정답을 만드는 경로가 여기 하나뿐이다."""
    lm = load_label_map()
    rows = select_eval(read_manifest(params.snapshot))
    eval_ids = {r["image_id"] for r in rows}
    gold_codes, gold_boxes = read_gold(params.snapshot, eval_ids)
    for iid in eval_ids:
        gold_codes.setdefault(iid, set())
    return Population(
        rows=rows, eval_ids=eval_ids,
        gold_codes=gold_codes, gold_boxes=gold_boxes,
        sizes=image_sizes(rows),
        classes=[lm.iso_code(n) for n in params.class_names],
    )


def score(pop: Population, records: Sequence[PredictionRecord]) -> dict:
    """단일 채점기 호출 지점. 칸을 인자로 받지 않는다 — 받을 수 없게 두는 게 요점이다."""
    return score_records(records, pop.gold_codes, pop.gold_boxes, pop.classes)


def record_path(params: ScoringParams, tag: str) -> Path:
    return params.out / f"{tag}_s{params.seed}.jsonl"


def load_detection_records(params: ScoringParams, tag: str) -> list[PredictionRecord]:
    """65번이 저장한 계약 #4 레코드를 되읽는다. **재추론하지 않는다.**"""
    p = record_path(params, tag)
    if not p.exists():
        raise FileNotFoundError(f"검출 산출물 없음: {p}")
    return read_records(p.read_text(encoding="utf-8").splitlines())


def load_unified_records(
    params: ScoringParams, pop: Population, cell: str, known_codes: set[str]
):
    """통합형 원시 생성문 → 계약 #4. 칸이 갈리는 유일한 지점(어댑터)이다."""
    src = params.pilot / "predictions" / f"{cell}.generations.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"통합형 원시 출력 없음: {src}")
    return adapt_unified_generations(
        src.read_text(encoding="utf-8").splitlines(),
        cell=cell, seed=params.seed, known_iso_codes=known_codes, image_size=pop.sizes,
    )


def regression_diffs(stored: dict, fresh: dict, tags: Sequence[str]) -> list[dict]:
    """저장된 지표와 새 지표의 완전 일치 대조.

    **값이 바뀌면 옮긴 코드가 같은 코드가 아니다.** 리팩토링의 통과 조건이 이것이다.
    """
    diffs = []
    for tag in tags:
        if tag not in stored:
            continue
        for k in REGRESSION_KEYS:
            a, b = stored[tag].get(k), fresh[tag].get(k)
            if a != b:
                diffs.append({"cell": tag, "key": k, "stored": a, "fresh": b})
    return diffs


def gate_check(metrics: dict, params: ScoringParams) -> dict:
    """게이트 대조 — 칸이 사전등록 통과선을 넘는가.

    게이트 값의 출처를 함께 싣는다. A 의 content-free 천장 재산출이 도착하기 전이면
    `source` 가 그 사실을 밝힌다(74번 A-1). **넘지 못한 것도 결과다** — 사진을 안 보고
    도달 가능한 상한을 제대로 재 보니 어떤 칸도 넘지 못했다는 것은 방법 절에 실린다.
    """
    cells = {
        tag: {
            "macro_f1": m["macro_f1"],
            "above_gate": m["macro_f1"] > params.gate.value,
            "above_pass_line": m["macro_f1"] > params.gate_pass_line,
            "margin_vs_pass_line": round(m["macro_f1"] - params.gate_pass_line, 6),
        }
        for tag, m in metrics.items()
    }
    n_pass = sum(1 for v in cells.values() if v["above_pass_line"])
    return {
        "gate": params.gate.as_dict(),
        "tolerance": params.gate_tolerance.as_dict(),
        "pass_line": params.gate_pass_line,
        "n_cells": len(cells),
        "n_above_pass_line": n_pass,
        "cells": cells,
        "verdict": (
            "전 칸 미달" if n_pass == 0
            else ("전 칸 통과" if n_pass == len(cells) else f"{n_pass}/{len(cells)} 통과")
        ),
    }
