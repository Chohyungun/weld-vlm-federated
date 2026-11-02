"""2 차 조항 축 — 정답 조항을 허용치 표에서 두께 없이 낸다 · AL 빈 집합 · 일치율 · 정밀도 · 무근거(지어냄 · 무관) · 코드별 · 대조선.

기대는 허용치 표(`corpus/rules/limits_v0_pilot.csv`, sha256 아래)의 행에서 손으로 냈다. 표의 바이트가 바뀌면 이 시험이 먼저 멈춘다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from corpus.rules import limits_loader
from evaluation.clause_axis import (
    citations_from_codes,
    clause_axis_draw_parts,
    clause_axis_draws,
    clause_axis_report,
    clause_axis_weighted,
    clauses_by_code,
    entries_from_limits,
    gold_pairs_without_thickness,
)
from evaluation.metrics.clause import GoldPair

REPO = Path(__file__).resolve().parents[1]
LIMITS = REPO / "corpus/rules/limits_v0_pilot.csv"
LIMITS_SHA256 = "334c78811677accb86a2ec14dad24716e16add3b6a14b47abe33ca6fe9e0cccb"
PAIRS = REPO / "data/processed/pairs_main_v1/pairs.jsonl"


@pytest.fixture(scope="module")
def table():
    assert hashlib.sha256(LIMITS.read_bytes()).hexdigest() == LIMITS_SHA256
    return limits_loader.load_limits(LIMITS, pilot=True)


@pytest.fixture(scope="module")
def entries(table):
    return entries_from_limits(table)


def test_재질_코드에서_조항을_두께_없이_낸다(entries) -> None:
    assert clauses_by_code(entries, material="ST") == {
        "100": ("KRA27-3D",), "2011": ("KRA27-T15",), "301": ("KRA27-T16",), "401": ("KRA27-T16",)}
    assert clauses_by_code(entries, material="AL") == {}
    assert clauses_by_code(entries, material="ST", inspection_method="VT") == {"2011": ("KRA27-S",)}


def test_정답_쌍은_정답_코드에서만_나고_조항_없는_코드는_센다(entries) -> None:
    material = {"a": "ST", "b": "ST", "c": "ST", "d": "ST", "e": "AL", "f": "ST"}
    gt = {"a": ["2011", "301"], "b": ["401"], "c": ["100"], "d": ["402"], "e": ["2011"], "f": []}
    pairs, skipped = gold_pairs_without_thickness(entries, material, gt)
    assert pairs == (GoldPair("a", "2011", "KRA27-T15"), GoldPair("a", "301", "KRA27-T16"),
                     GoldPair("b", "401", "KRA27-T16"), GoldPair("c", "100", "KRA27-3D"))
    assert skipped == {"정답 조항 없음": 2}                     # d(강재 402) · e(AL)


def test_재질을_모르면_쌍을_지어내지_않는다(entries) -> None:
    with pytest.raises(KeyError):
        gold_pairs_without_thickness(entries, {}, {"x": ["2011"]})


def test_조항_축_셋과_코드별(entries) -> None:
    material = {"a": "ST", "b": "ST", "c": "ST", "d": "ST", "e": "AL", "f": "ST"}
    gt = {"a": ["2011", "301"], "b": ["401"], "c": ["100"], "d": ["402"], "e": ["2011"], "f": []}
    pairs, _ = gold_pairs_without_thickness(entries, material, gt)
    cited = {"a": ["KRA27-T15"], "b": ["KRA27-T16", "KRA27-T16"], "c": ["KRA27-T99"], "d": [],
             "e": ["KRA27-T15"], "f": ["KRA27-S"]}
    index = {e.clause_id for e in entries} | {"KRA27-S"}
    rep = clause_axis_report(cited, pairs, index)
    assert rep["citation_match_rate"] == {"value": 0.5, "state": "finite", "numerator": 2, "denominator": 4}
    assert rep["citation_precision"] == {"value": 0.5, "state": "finite", "numerator": 3, "denominator": 6}
    ung = rep["ungrounded_rate"]
    assert (ung["value"], ung["nonexistent"], ung["irrelevant"], ung["denominator"]) == (0.5, 1, 2, 6)
    assert {k: (v["numerator"], v["denominator"]) for k, v in rep["per_code"].items()} == {
        "100": (0, 1), "2011": (1, 1), "301": (0, 1), "401": (1, 1)}


def fixture_six(entries):
    material = {"a": "ST", "b": "ST", "c": "ST", "d": "ST", "e": "AL", "f": "ST"}
    gt = {"a": ["2011", "301"], "b": ["401"], "c": ["100"], "d": ["402"], "e": ["2011"], "f": []}
    pairs, _ = gold_pairs_without_thickness(entries, material, gt)
    cited = {"a": ["KRA27-T15"], "b": ["KRA27-T16", "KRA27-T16"], "c": ["KRA27-T99"], "d": [],
             "e": ["KRA27-T15"], "f": ["KRA27-S"]}
    return cited, pairs, {e.clause_id for e in entries} | {"KRA27-S"}


def test_설계_가중은_단위마다_그_이미지의_가중을_곱한다(entries) -> None:
    """손셈(04 §7-10 ①): 가중 a 1 · b 2 · c 3 · d 4 · e 5 · f 6. 일치 쌍 a-2011(1) · b-401(2) / 쌍 7, 맞는 인용 a(1) · b(2 × 2) / 인용
    19, 지어냄 c(3) · 무관 e(5) · f(6)."""
    cited, pairs, index = fixture_six(entries)
    rep = clause_axis_weighted(cited, pairs, index, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6})
    assert (rep["citation_match_rate"]["numerator"], rep["citation_match_rate"]["denominator"]) == (3.0, 7.0)
    assert rep["citation_match_rate"]["value"] == pytest.approx(3 / 7)
    assert (rep["citation_precision"]["numerator"], rep["citation_precision"]["denominator"]) == (5.0, 19.0)
    ung = rep["ungrounded_rate"]
    assert (ung["nonexistent"], ung["irrelevant"], ung["denominator"]) == (3.0, 11.0, 19.0)
    assert ung["value"] == pytest.approx(14 / 19)
    assert {k: (v["numerator"], v["denominator"]) for k, v in rep["per_code"].items()} == {
        "100": (0.0, 3.0), "2011": (1.0, 1.0), "301": (0.0, 1.0), "401": (2.0, 2.0)}
    assert (rep["n_gold_pairs"], rep["n_citations"]) == (4, 6)                  # 단위의 수는 무가중과 같다


def test_가중이_모두_1_이면_무가중_셈과_값이_같다(entries) -> None:
    cited, pairs, index = fixture_six(entries)
    ones = clause_axis_weighted(cited, pairs, index, dict.fromkeys(cited, 1.0))
    plain = clause_axis_report(cited, pairs, index)
    for key in ("citation_match_rate", "citation_precision", "ungrounded_rate"):
        assert ones[key]["value"] == plain[key]["value"], key
    assert {k: v["value"] for k, v in ones["per_code"].items()} == {k: v["value"] for k, v in plain["per_code"].items()}


def test_가중이_없는_이미지는_메우지_않는다(entries) -> None:
    cited, pairs, index = fixture_six(entries)
    with pytest.raises(KeyError, match="메우지"):
        clause_axis_weighted(cited, pairs, index, {k: 1.0 for k in cited if k != "e"})
    assert clause_axis_weighted({}, (), {"KRA27-T15"}, {})["citation_match_rate"]["state"] == "해당 없음"


SIX = ("a", "b", "c", "d", "e", "f")
W6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
"""fixture_six 의 이미지 차례와 설계 가중(손셈 시험과 같은 값)."""


def test_항등_추첨은_점추정이고_같은_W_는_같은_추첨을_낸다(entries) -> None:
    """지시 1023d 과제 2 — 중복도 1 의 추첨(설계 가중 = W 의 한 행)은 설계 가중 점추정과, 그 행을 가중으로 나눈 것(1)은 무가중 점추정과 같다."""
    cited, pairs, index = fixture_six(entries)
    design = clause_axis_weighted(cited, pairs, index, dict(zip(SIX, W6, strict=True)))
    plain = clause_axis_report(cited, pairs, index)
    one_d = clause_axis_draws(cited, pairs, index, SIX, W6[None, :])
    one_u = clause_axis_draws(cited, pairs, index, SIX, (W6 / W6)[None, :])
    for name in ("citation_match_rate", "citation_precision", "ungrounded_rate"):
        assert one_d[name][0] == pytest.approx(design[name]["value"], rel=1e-12)
        assert one_u[name][0] == plain[name]["value"]                          # 정수 셈 — 같은 수다
    rng = np.random.default_rng(7)
    M = rng.multinomial(6, np.full(6, 1 / 6), size=50).astype(float)
    a = clause_axis_draws(cited, pairs, index, SIX, M * W6[None, :])
    b = clause_axis_draws(cited, pairs, index, SIX, M * W6[None, :])
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def test_균형_추첨의_분자_분모_평균은_점추정의_분자_분모다(entries) -> None:
    """추첨마다 한 이미지에 중복도 6 을 몰면(여섯 추첨) 이미지마다 평균 중복도가 1 — 분자 · 분모는 W 에 선형이라 평균이 점추정의 것과 같다."""
    cited, pairs, index = fixture_six(entries)
    M = np.eye(6) * 6.0
    design = clause_axis_weighted(cited, pairs, index, dict(zip(SIX, W6, strict=True)))
    plain = clause_axis_report(cited, pairs, index)
    for W, point in ((M * W6[None, :], design), (M, plain)):
        parts = clause_axis_draw_parts(cited, pairs, index, SIX, W)
        for name in ("citation_match_rate", "citation_precision"):
            n, d = parts[name]
            assert (n.mean(), d.mean()) == pytest.approx((point[name]["numerator"], point[name]["denominator"]), rel=1e-12)
        n, d = parts["ungrounded_rate"]
        ung = point["ungrounded_rate"]
        assert (n.mean(), d.mean()) == pytest.approx((ung["nonexistent"] + ung["irrelevant"], ung["denominator"]), rel=1e-12)


def test_분모가_빈_추첨은_NaN_이다(entries) -> None:
    """d 만 뽑은 추첨은 정답 쌍도 인용도 없어 셋 다 미정의, f 만 뽑은 추첨은 정답 쌍이 없어 일치율만 미정의(인용 S 는 무관)."""
    cited, pairs, index = fixture_six(entries)
    only = np.eye(6)
    got = clause_axis_draws(cited, pairs, index, SIX, only[[3, 5]] * 6.0)
    assert np.isnan(got["citation_match_rate"]).all()
    assert np.isnan(got["citation_precision"][0]) and np.isnan(got["ungrounded_rate"][0])
    assert (got["citation_precision"][1], got["ungrounded_rate"][1]) == (0.0, 1.0)


def test_추첨_입력의_꼴이_틀리면_거부한다(entries) -> None:
    cited, pairs, index = fixture_six(entries)
    with pytest.raises(KeyError, match="모집단 밖"):
        clause_axis_draws(cited, pairs, index, SIX[:-1], np.ones((1, 5)))
    for bad in (np.ones((1, 5)), -np.ones((1, 6)), np.full((1, 6), np.nan)):
        with pytest.raises(ValueError):
            clause_axis_draws(cited, pairs, index, SIX, bad)


def test_분모가_비면_해당_없음이다() -> None:
    rep = clause_axis_report({}, (), {"KRA27-T15"})
    for key in ("citation_match_rate", "citation_precision", "ungrounded_rate"):
        assert rep[key]["value"] is None and rep[key]["state"] == "해당 없음"


def test_대조선의_인용은_코드에서_낸다(entries) -> None:
    got = citations_from_codes(entries, {"a": "ST", "e": "AL", "d": "ST"},
                               {"a": ["301", "2011", "401"], "e": ["2011"], "d": ["402"]})
    assert got == {"a": ["KRA27-T15", "KRA27-T16"], "d": [], "e": []}


def test_학습_풀의_골격_조항과_같다(entries) -> None:
    """두 구현 대조 — 생성 쪽 골격의 `clauses` 와 이 길이 낸 정답 조항 집합이 학습 풀 전 행에서 같다(03 2판 §2-4 와 같은 대조)."""
    if not PAIRS.exists():
        pytest.skip(f"학습 풀이 없는 체크아웃: {PAIRS.relative_to(REPO)}")
    n = same = 0
    with PAIRS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            codes = [str(d["type"]) for d in r["skeleton"]["defects"]]
            table = clauses_by_code(entries, material=r["material"])
            mine = sorted({c for code in codes for c in table.get(code, ())})
            n += 1
            same += mine == sorted(r["skeleton"].get("clauses") or [])
    assert n > 0 and same == n
