"""출처 고정 판별력 지표 — 71번 과제 2. 헤드라인 보조지표 승격 전제로 정의를 고정한다.

이 지표의 존재 이유는 하나다: **출처만 읽는 예측기가 정확히 0 을 받는다.** 그 성질이
깨지면 지표가 다른 네 지표와 같은 운명이 되므로, 여기서 시험으로 못박는다.
"""

from __future__ import annotations

import pytest

from evaluation.discrimination import (
    CROP,
    fires,
    score_discrimination,
    score_discrimination_all_cells,
)
from evaluation.probes.p9_runner import is_false_positive
from evaluation.schema import PredictionRecord, failed_record

TILE = "N-tile"


def rec(image_id: str, codes, *, cell="sep_central", client=None, seed=1, ok=True):
    if not ok:
        return failed_record(image_id, cell, seed, "truncated", client=client)
    return PredictionRecord(
        schema_version="1.3", image_id=image_id, cell=cell, client=client, seed=seed,
        defects=[{"iso_code": c} for c in codes],
        verdict="판정불가", cited_clauses=[], parse_ok=True,
    )


def ctx(items):
    """image_id → (group_id, has_defect, provenance)"""
    return {i: (g, d, p) for i, g, d, p in items}


def test_발화_정의가_P9_와_같은_함수다() -> None:
    """정의가 갈리면 두 지표의 차가 모델 차이가 아니라 정의 차이를 잰다."""
    fired = rec("a", ["2011"])
    silent = rec("b", [])
    broken = rec("c", [], ok=False)
    for r in (fired, silent, broken):
        assert fires(r) is is_false_positive(r)
    assert fires(fired) and not fires(silent) and not fires(broken)


def test_출처만_읽는_규칙은_정확히_0_을_받는다() -> None:
    """지름길이 통과할 수 없다는 것이 이 지표의 존재 이유다."""
    contexts = ctx([
        ("d1", "g1", True, CROP), ("d2", "g1", True, CROP),
        ("n1", "g2", False, CROP), ("n2", "g2", False, CROP),
        ("t1", "g3", False, TILE),
    ])
    # N-crop 이면 기공, 아니면 없음 — 이미지를 열지 않는다
    records = [
        rec(i, ["2011"] if p == CROP else [])
        for i, (_, _, p) in contexts.items()
    ]
    r = score_discrimination(records, contexts)
    assert r.fire_rate_defect == 1.0
    assert r.fire_rate_normal == 1.0
    assert r.delta.point == 0.0
    assert (r.delta.lo, r.delta.hi) == (0.0, 0.0)
    assert r.verdict.startswith("지름길선")


def test_섞인_묶음에서도_규칙은_0_이다() -> None:
    """실제 평가셋은 결함·정상이 같은 묶음에 섞여 있다(68번 §2-3-f: 98.9%)."""
    contexts = ctx([
        ("d1", "g1", True, CROP), ("n1", "g1", False, CROP),
        ("d2", "g2", True, CROP), ("n2", "g2", False, CROP),
        ("d3", "g3", True, CROP), ("n3", "g3", False, CROP),
    ])
    records = [rec(i, ["2011"]) for i in contexts]     # 전부 N-crop → 전부 발화
    r = score_discrimination(records, contexts)
    assert r.delta.point == 0.0
    assert (r.delta.lo, r.delta.hi) == (0.0, 0.0)
    assert r.delta.n_undefined == 0                    # 층이 비는 재표집이 없다


def test_침묵하는_모델도_0_을_받는다() -> None:
    contexts = ctx([("d1", "g1", True, CROP), ("n1", "g2", False, CROP)])
    r = score_discrimination([rec("d1", []), rec("n1", [])], contexts)
    assert r.delta.point == 0.0


def test_완벽한_판별은_1_이다() -> None:
    contexts = ctx([
        ("d1", "g1", True, CROP), ("n1", "g1", False, CROP),
        ("d2", "g2", True, CROP), ("n2", "g2", False, CROP),
    ])
    records = [rec("d1", ["2011"]), rec("d2", ["2011"]), rec("n1", []), rec("n2", [])]
    r = score_discrimination(records, contexts)
    assert r.delta.point == pytest.approx(1.0)
    assert r.verdict.startswith("판별 있음")


def test_한쪽_층이_빈_재표집은_0_으로_대치하지_않고_뺀다() -> None:
    """0 대치는 CI 를 0 쪽으로 끌어 '판별 불성립'을 공짜로 만든다."""
    contexts = ctx([
        ("d1", "g1", True, CROP), ("d2", "g2", True, CROP),
        ("n1", "g3", False, CROP), ("n2", "g4", False, CROP),
    ])
    records = [rec("d1", ["2011"]), rec("d2", ["2011"]), rec("n1", []), rec("n2", [])]
    r = score_discrimination(records, contexts)
    assert r.delta.point == pytest.approx(1.0)
    assert r.delta.lo == pytest.approx(1.0)      # 0 대치였다면 하한이 0 이 된다
    assert r.delta.n_undefined > 0
    assert "주의" in r.verdict                   # 버린 사실을 숨기지 않는다


def test_정상에서_더_발화하면_역전으로_판정한다() -> None:
    contexts = ctx([
        ("d1", "g1", True, CROP), ("n1", "g1", False, CROP),
        ("d2", "g2", True, CROP), ("n2", "g2", False, CROP),
    ])
    records = [rec("d1", []), rec("d2", []), rec("n1", ["2011"]), rec("n2", ["2011"])]
    r = score_discrimination(records, contexts)
    assert r.delta.point == pytest.approx(-1.0)
    assert "역전" in r.verdict


def test_고정_출처_밖_이미지는_계산에_들어가지_않는다() -> None:
    """N-tile 정상을 섞으면 출처 축이 다시 살아나 지름길이 통과한다."""
    contexts = ctx([
        ("d1", "g1", True, CROP), ("n1", "g2", False, CROP),
        ("t1", "g3", False, TILE), ("t2", "g3", False, TILE),
    ])
    records = [rec("d1", ["2011"]), rec("n1", ["2011"]), rec("t1", []), rec("t2", [])]
    r = score_discrimination(records, contexts)
    assert r.n_defect == 1 and r.n_normal == 1
    assert r.delta.point == 0.0        # N-tile 을 셌다면 +0.5 가 나왔을 것이다


def test_파싱_실패는_발화_아님으로_들어간다() -> None:
    contexts = ctx([("d1", "g1", True, CROP), ("n1", "g2", False, CROP)])
    r = score_discrimination([rec("d1", [], ok=False), rec("n1", [])], contexts)
    assert r.fire_rate_defect == 0.0
    assert r.n_defect == 1             # 분모에서 사라지지 않는다


def test_레코드_없는_이미지는_결측으로_세고_분모에서_뺀다() -> None:
    contexts = ctx([
        ("d1", "g1", True, CROP), ("d2", "g1", True, CROP), ("n1", "g2", False, CROP),
    ])
    r = score_discrimination([rec("d1", ["2011"]), rec("n1", [])], contexts)
    assert r.n_missing_prediction == 1
    assert r.n_defect == 1             # 발화 0 으로 채웠다면 2 가 됐을 것이다
    assert r.fire_rate_defect == 1.0


def test_결함_또는_정상이_없으면_산출_불가다() -> None:
    contexts = ctx([("d1", "g1", True, CROP)])
    r = score_discrimination([rec("d1", ["2011"])], contexts)
    assert "산출 불가" in r.verdict


def test_모델이_섞이면_거부한다() -> None:
    contexts = ctx([("d1", "g1", True, CROP), ("n1", "g2", False, CROP)])
    mixed = [rec("d1", ["2011"], cell="sep_local", client="C1"),
             rec("n1", [], cell="sep_local", client="C2")]
    with pytest.raises(ValueError, match="섞여 있다"):
        score_discrimination(mixed, contexts)


def test_중복_레코드를_조용히_덮지_않는다() -> None:
    contexts = ctx([("d1", "g1", True, CROP), ("n1", "g2", False, CROP)])
    with pytest.raises(ValueError, match="중복"):
        score_discrimination([rec("d1", ["2011"]), rec("d1", [])], contexts)


def test_전_칸을_모델별로_갈라_계산한다() -> None:
    contexts = ctx([("d1", "g1", True, CROP), ("n1", "g2", False, CROP)])
    records = [
        rec("d1", ["2011"], cell="sep_local", client="C1"),
        rec("n1", [], cell="sep_local", client="C1"),
        rec("d1", [], cell="sep_local", client="C2"),
        rec("n1", ["2011"], cell="sep_local", client="C2"),
    ]
    out = score_discrimination_all_cells(records, contexts)
    assert len(out) == 2
    assert {r.client for r in out} == {"C1", "C2"}
    assert {round(r.delta.point, 4) for r in out} == {1.0, -1.0}


def test_CI_는_묶음_단위로_재표집한다() -> None:
    """같은 묶음 이미지를 독립 표본으로 세면 CI 가 좁아진다(stats 모듈 머리말)."""
    # 묶음 2개, 각 묶음 안에서 결과가 완전히 일치 → 묶음 재표집이면 폭이 넓다
    contexts = ctx([
        ("d1", "g1", True, CROP), ("d2", "g1", True, CROP),
        ("d3", "g2", True, CROP), ("d4", "g2", True, CROP),
        ("n1", "g1", False, CROP), ("n2", "g2", False, CROP),
    ])
    records = [
        rec("d1", ["2011"]), rec("d2", ["2011"]),    # g1 결함 발화
        rec("d3", []), rec("d4", []),                # g2 결함 침묵
        rec("n1", []), rec("n2", []),
    ]
    r = score_discrimination(records, contexts)
    assert r.n_groups == 2
    assert r.delta.hi - r.delta.lo > 0.5             # 묶음 2개 → 매우 넓다
    assert "판별 불성립" in r.verdict or "역전" in r.verdict
