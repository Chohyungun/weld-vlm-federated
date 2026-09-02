"""어댑터 대칭 — 체크리스트 13 (80번 D7·D8).

어댑터는 "칸을 구분해도 되는 유일한 지점"이지만 그건 **형식**의 차이를 흡수하라는
뜻이지 **정책**을 갈라도 된다는 뜻이 아니었다. 실제로 갈려 있었고, 같은 이미지가
칸에 따라 "전량 미검출"과 "일부 검출"로 갈렸다.

여기 시험은 **정책이 한 곳에서 온다**는 것을 두 방향으로 확인한다 — 통합형 어댑터가
항목 단위로 버리는가, 그리고 그 판단이 분리형과 같은 함수에서 나오는가.
"""

from __future__ import annotations

import json

import pytest

from evaluation.adapters import adapt_unified_generations
from evaluation.policy import (
    DEFECT_ITEM_POLICY,
    ITEM_LEVEL,
    RECORD_LEVEL,
    filter_defect_items,
)

KNOWN = ["2011", "100", "2012", "402"]
SCORING = ["2011", "100"]
GOOD = {"iso_code": "2011", "bbox_px": [10, 10, 20, 20]}
GOOD2 = {"iso_code": "100", "bbox_px": [30, 30, 40, 40]}
BAD_BOX = {"iso_code": "2011", "bbox_px": [20, 20, 10, 10]}      # 역전
OUT_OF_SCOPE = {"iso_code": "2012", "bbox_px": [50, 50, 60, 60]}
UNKNOWN = {"iso_code": "9999", "bbox_px": [50, 50, 60, 60]}


def _line(image_id: str, defects: list[dict]) -> str:
    return json.dumps({
        "image_id": image_id,
        "coord_space": "ABS_ORIG",
        "bbox_px_parsed": {"verdict": "불합격", "cited_clauses": [], "defects": defects},
    }, ensure_ascii=False)


def _adapt(defects: list[dict]):
    return adapt_unified_generations(
        [_line("i", defects)], cell="uni_central", seed=1,
        known_iso_codes=KNOWN, scoring_iso_codes=SCORING,
    )


# --------------------------------------------------------------------------------------
# 정책 상수 자체
# --------------------------------------------------------------------------------------

def test_policy_is_item_level() -> None:
    """레코드 단위 폐기는 쓰지 않는다. 상수를 바꾸면 아래 시험들이 깨진다."""
    assert DEFECT_ITEM_POLICY == ITEM_LEVEL
    assert DEFECT_ITEM_POLICY != RECORD_LEVEL


# --------------------------------------------------------------------------------------
# D7 — 항목 하나가 깨져도 나머지는 산다
# --------------------------------------------------------------------------------------

def test_one_bad_item_does_not_discard_the_record() -> None:
    rep = _adapt([GOOD, BAD_BOX, GOOD2])
    assert len(rep.records) == 1
    rec = rep.records[0]
    assert rec.parse_ok is True
    assert len(rec.defects) == 2, "깨진 항목 하나가 레코드 전체를 폐기했다(80번 D7)"
    assert rep.n_bad_items == 1


def test_all_items_bad_still_yields_a_record() -> None:
    """전부 깨져도 레코드는 남고 빈 예측이 된다 — 미검출로 계상된다."""
    rep = _adapt([BAD_BOX])
    assert len(rep.records) == 1
    assert rep.records[0].defects == []
    assert rep.n_bad_items == 1


def test_detection_and_unified_drop_the_same_item() -> None:
    """분리형·통합형이 **같은 함수**로 판단한다. 결과가 항목 단위로 일치해야 한다."""
    items = [GOOD, BAD_BOX, GOOD2, OUT_OF_SCOPE, UNKNOWN]
    shared = filter_defect_items(items, known_codes=KNOWN, scoring_codes=SCORING)
    uni = _adapt(items)
    assert [d["iso_code"] for d in shared.kept] == [d.iso_code for d in uni.records[0].defects]
    assert shared.n_bad_item == uni.n_bad_items
    assert shared.n_out_of_scope == uni.n_out_of_scope
    assert shared.n_unknown_code == uni.n_unknown_code


# --------------------------------------------------------------------------------------
# D8 — 채점 클래스 밖 코드
# --------------------------------------------------------------------------------------

def test_out_of_scope_code_is_dropped_and_counted() -> None:
    rep = _adapt([GOOD, OUT_OF_SCOPE])
    assert [d.iso_code for d in rep.records[0].defects] == ["2011"]
    assert rep.n_out_of_scope == 1
    assert rep.out_of_scope_codes == {"2012": 1}


def test_out_of_scope_code_no_longer_inflates_class_jaccard() -> None:
    """분리형은 nc=4 라 못 내는 코드가 통합형에만 벌점이 되던 경로를 막았다 (80번 D8)."""
    from evaluation.metrics.detection import class_jaccard

    gold = {"i": ["2011"]}
    with_extra = {"i": ["2011", "2012"]}
    assert class_jaccard(with_extra, gold) == pytest.approx(0.5)
    assert class_jaccard(with_extra, gold, SCORING) == pytest.approx(1.0)


def test_unknown_code_is_still_reported() -> None:
    """조용히 버리지 않는다 — 환각 신호는 남아야 한다."""
    rep = _adapt([UNKNOWN])
    assert rep.n_unknown_code == 1
    assert rep.records[0].defects == []


def test_report_dict_exposes_the_policy() -> None:
    d = _adapt([GOOD, BAD_BOX, OUT_OF_SCOPE]).as_dict()
    assert d["discard_policy"] == ITEM_LEVEL
    assert d["n_bad_items_dropped"] == 1
    assert d["n_out_of_scope_dropped"] == 1
