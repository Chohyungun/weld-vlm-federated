"""층화 채점 — 총괄 판정 6 (76번 을안).

두 가지를 시험이 강제한다.

1. **층은 A 의 `data.id_strata` 에서 온다.** 절단점을 만드는 곳은 한 군데여야 한다(84번
   §1-3). 채점 경로가 실제로 그 모듈을 거치는지, D 쪽에 우회 산식이 다시 생기지 않았는지를
   **기계가** 잡는다. 그 모듈과 `recompute_baselines.py` 의 정합은 A 의
   `tests/test_id_strata.py` 가 맡는다.
2. **판별력.** 지름길 규칙(구간→최빈 라벨)이 층화 지표에서 0 으로 떨어지는가.
   이것이 완료 기준이고, 동시에 이 지표가 무이빨이 아니라는 증거다.
"""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.strata import (
    DEFAULT_K,
    ID_GRANULARITY,
    bins_for,
    majority_codeset,
    shortcut_pred,
    stratified_score,
)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "data" / "interim" / "manifest_v1"
CLASSES = ["2011", "100", "401", "301"]


@pytest.fixture(scope="module")
def frozen_eval_ids() -> list[str]:
    if not (FROZEN / "SNAPSHOT.sha256").exists():
        pytest.skip("동결 스냅샷 없음")
    from evaluation.eval_set import read_manifest

    return sorted(r["image_id"] for r in read_manifest(FROZEN) if r["split"] == "eval")


# --------------------------------------------------------------------------------------
# 1. 층은 A 의 모듈에서 온다 — 배선 시험
# --------------------------------------------------------------------------------------

def test_granularity_ladder_matches_track_a() -> None:
    """`ID_GRANULARITY` 가 A 의 `GRANULARITY["idq"]` 에서 None 뺀 것과 같은가."""
    sys.path.insert(0, str(REPO))
    from scripts.recompute_baselines import GRANULARITY

    a_ladder = tuple(g for g in GRANULARITY["idq"] if g is not None)
    assert ID_GRANULARITY == a_ladder, (
        f"A 의 사다리 {a_ladder} 와 다르다 — 층 정의가 갈리면 게이트와 층화 지표가 "
        "다른 축 위에 선다"
    )
    assert DEFAULT_K in ID_GRANULARITY


def test_bins_come_from_track_a_module(frozen_eval_ids) -> None:
    """채점 경로의 층 번호가 A 의 `stratum_of` 와 **전 이미지에서** 같다."""
    from data.id_strata import stratum_of

    for k in (16, DEFAULT_K):
        mine = bins_for(frozen_eval_ids, k, snapshot=FROZEN)
        theirs = stratum_of(frozen_eval_ids, k, FROZEN)
        assert np.array_equal(np.array([mine[i] for i in frozen_eval_ids]), theirs), (
            f"K={k} 에서 채점기의 층이 A 의 모듈과 갈린다"
        )


def test_bins_match_track_a_materialized_table(frozen_eval_ids) -> None:
    """A 가 실체화한 `id_strata_k{K}.csv` 와 같은 층 번호.

    함수 호출 결과뿐 아니라 **디스크에 남은 A 의 산출표**와도 맞아야 한다 — 어느 판을
    썼는지가 파일 하나로 고정되는 것이 그 표의 존재 이유다.
    """
    checked = 0
    for k in (16, DEFAULT_K):
        table_path = FROZEN / f"id_strata_k{k}.csv"
        if not table_path.exists():
            continue
        with table_path.open(encoding="utf-8", newline="") as fh:
            table = {
                row["image_id"]: int(row["stratum"])
                for row in csv.DictReader(fh) if row["split"] == "eval"
            }
        mine = bins_for(frozen_eval_ids, k, snapshot=FROZEN)
        assert set(mine) == set(table), f"K={k} 평가셋 id 집합이 A 의 표와 다르다"
        assert mine == table, f"K={k} 층 번호가 A 의 표와 다르다"
        checked += 1
    if checked == 0:
        pytest.skip("A 의 id_strata_k*.csv 가 없다")


def test_no_local_cut_formula_in_scoring_module() -> None:
    """**우회 금지.** `evaluation/strata.py` 가 분위 절단점을 스스로 만들지 않는다 (AST).

    산식이 두 벌이면 언젠가 갈린다. 처음 판이 정확히 그 상태였다 — 시험으로 정합을
    대조했지만 산식 자체는 두 벌이었다. 이제 D 쪽에는 산식이 없어야 하고, A 의 모듈을
    실제로 import 해야 한다.
    """
    src = (REPO / "evaluation" / "strata.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"quantile", "percentile", "searchsorted", "digitize"}
    hits = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in banned]
    hits += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in banned]
    assert not hits, f"D 쪽에 절단점 산식이 다시 생겼다: {hits}"

    imported = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "data.id_strata"
        for alias in n.names
    }
    assert "stratum_of" in imported, "채점기가 A 의 `stratum_of` 를 부르지 않는다"


# --------------------------------------------------------------------------------------
# 2. 판별력 — 완료 기준
# --------------------------------------------------------------------------------------

def _toy():
    """구간 2개 × 이미지 6장. 구간 안에 변별할 것이 남아 있는 인공 예다."""
    gold = {
        "s:1": ["2011"], "s:2": ["2011"], "s:3": ["100"],       # bin 0
        "s:11": ["100"], "s:12": ["100"], "s:13": ["2011"],     # bin 1
    }
    bins = {"s:1": 0, "s:2": 0, "s:3": 0, "s:11": 1, "s:12": 1, "s:13": 1}
    return gold, bins


def test_shortcut_rule_scores_exactly_zero_lift() -> None:
    """**완료 기준.** 구간→최빈 라벨 규칙의 층화 lift 는 정의상 0 이다."""
    gold, bins = _toy()
    sc = shortcut_pred(gold, CLASSES, bins)
    rep = stratified_score(sc, gold, CLASSES, bins, k=2)
    assert rep.lift_mean == pytest.approx(0.0, abs=1e-12)
    assert rep.lift_weighted == pytest.approx(0.0, abs=1e-12)


def test_stratified_macro_f1_alone_does_not_kill_the_shortcut() -> None:
    """**지시 문면만으로는 완료 기준을 못 만족한다는 사실을 시험으로 고정한다.**

    층 안 Macro-F1 을 그대로 평균하면 최빈 라벨 예측기가 0 근처가 아니라 높은 값을
    받는다 — 순수 구간에서 전건 정답이기 때문이다. lift 를 함께 내야 하는 이유다.
    """
    gold, bins = _toy()
    sc = shortcut_pred(gold, CLASSES, bins)
    rep = stratified_score(sc, gold, CLASSES, bins, k=2)
    assert rep.macro_f1_mean > 0.3, (
        "층화 Macro-F1 이 지름길을 0 으로 떨어뜨렸다면 이 보고의 §판정이 틀렸다"
    )


def test_a_better_model_gets_positive_lift() -> None:
    """지름길보다 잘하는 모델은 양의 lift 를 받는다. 지표가 무이빨이 아니라는 증거."""
    gold, bins = _toy()
    perfect = {i: list(v) for i, v in gold.items()}
    rep = stratified_score(perfect, gold, CLASSES, bins, k=2)
    assert rep.lift_mean > 0.0
    assert rep.macro_f1_mean == pytest.approx(1.0)


def test_a_worse_model_gets_negative_lift() -> None:
    gold, bins = _toy()
    wrong = {i: ["401"] for i in gold}
    rep = stratified_score(wrong, gold, CLASSES, bins, k=2)
    assert rep.lift_mean < 0.0


def test_pure_strata_are_counted() -> None:
    """순수 구간은 층 안 변별이 원리적으로 불가능하다. 건수가 보고에 남아야 한다."""
    gold = {"s:1": ["2011"], "s:2": ["2011"], "s:11": ["100"], "s:12": ["2011"]}
    bins = {"s:1": 0, "s:2": 0, "s:11": 1, "s:12": 1}
    rep = stratified_score(gold, gold, CLASSES, bins, k=2)
    assert rep.n_pure_strata == 1
    assert rep.frac_images_in_pure == pytest.approx(0.5)


def test_majority_codeset_is_deterministic() -> None:
    """동률에서 결정론적이어야 한다 — 채점기가 비결정적이면 비트 일치 요구가 무너진다."""
    gold = {"s:1": ["2011"], "s:2": ["100"]}
    a = majority_codeset(["s:1", "s:2"], gold, CLASSES)
    b = majority_codeset(["s:2", "s:1"], gold, CLASSES)
    assert a == b


def test_empty_gold_stratum_is_skipped_not_zeroed() -> None:
    """정상만 있는 구간을 0.0 으로 채우면 '못 맞혔다'로 오독된다."""
    gold = {"s:1": [], "s:2": [], "s:11": ["2011"], "s:12": ["2011"]}
    bins = {"s:1": 0, "s:2": 0, "s:11": 1, "s:12": 1}
    rep = stratified_score(gold, gold, CLASSES, bins, k=2)
    assert rep.n_strata == 2
    assert rep.n_strata_scored == 1
    assert rep.macro_f1_mean == pytest.approx(1.0)
    assert rep.as_dict()["axis"] == "idq"
