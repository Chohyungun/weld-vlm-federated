"""임계 독립 판별력 `Δ_AUC = 2·AUROC − 1` — 승격 후보의 성질 (17번 §12-5).

이 파일이 지키는 것은 셋이다.

1. **상수 점수는 정확히 0** — 출처만 읽는 예측기의 성질이다. 촬영 ID 등 다른
   메타데이터를 쓰는 규칙은 같은 출처 안에서도 양수가 가능하다. 반례도 함께 시험한다.
2. **임계와 무관** — 같은 순위를 주는 점수라면 값이 같아야 한다. 이 지표의 존재 이유다.
3. **`image_score` 가 발화 정의의 연속판** — `image_score(r) >= c` 가 임계 `c` 로 자른 뒤의
   `fires(r)` 와 **항상** 같아야 한다. 어긋나면 곡선과 임계 독립판이 다른 것을 재게 된다.
"""

from __future__ import annotations

import math
import random

import pytest

from evaluation.detect_infer import filter_by_conf
from evaluation.discrimination import fires, gini, image_score
from evaluation.schema import Defect, PredictionRecord


def _rec(image_id: str, scores: list[float], *, parse_ok: bool = True) -> PredictionRecord:
    return PredictionRecord(
        schema_version="1.3",
        image_id=image_id,
        cell="sep_central",
        seed=1,
        defects=[
            Defect(iso_code="100", bbox_px=(1.0, 1.0, 2.0, 2.0), score=s)
            for s in scores
        ],
        verdict="불합격" if scores else "합격",
        cited_clauses=[],
        parse_ok=parse_ok,
        parse_error=None if parse_ok else "schema_violation",
    )


# --------------------------------------------------------------------------------------
# 1 — 상수 점수의 기준선과 비상수 메타데이터 반례
# --------------------------------------------------------------------------------------

def test_상수_점수_예측기는_정확히_0():
    """출처만 읽는 예측기는 구간 안에서 상수가 된다. 근사 0 이 아니라 정확히 0 이어야 한다."""
    sc = dict.fromkeys([f"i{n}" for n in range(50)], 0.7)
    d, n = [f"i{n}" for n in range(30)], [f"i{n}" for n in range(30, 50)]
    assert gini(sc, d, n) == 0.0


def test_같은_출처의_메타데이터_규칙도_양수일_수_있다(monkeypatch):
    """train 빈도만 적합해도 ID 구간과 결함이 교락하면 영상 없이 완전 분리한다."""
    from evaluation import content_free

    bins = {"train_a": 0, "train_b": 1, "eval_a": 0, "eval_b": 1}
    monkeypatch.setattr(content_free, "bins_for", lambda ids, k, snapshot: {i: bins[i] for i in ids})
    rule = content_free.fit_idq(2, ["train_a", "train_b"],
                                {"train_a": ["100"], "train_b": []}, ["100"])
    values, _ = content_free.scored(rule, ["eval_a", "eval_b"], content_free.FREQ)
    scores = {i: max(s for _, s in items) for i, items in values.items()}
    # 두 평가 이미지는 같은 출처다. 정답은 점수 적합에 쓰지 않는다.
    assert gini(scores, ["eval_a"], ["eval_b"]) == 1.0


def test_보고서가_메타데이터_교락을_경고한다():
    from evaluation.discrimination import score_threshold_free

    report = score_threshold_free({"metadata": {"d": 1.0, "n": 0.0}},
                                  ["d"], ["n"], {"g": ["d", "n"]})
    assert "메타데이터" in report["limitation"]
    assert "반례" in report["status"]


def test_전부_미발화도_정확히_0():
    """아무것도 예측하지 않는 칸 — 이것도 상수라 0 이다. `image_score` 가 −1 로 묶는다."""
    recs = [_rec(f"i{n}", []) for n in range(20)]
    sc = {r.image_id: image_score(r) for r in recs}
    assert gini(sc, [f"i{n}" for n in range(10)],
                [f"i{n}" for n in range(10, 20)]) == 0.0


def test_파싱_실패는_어떤_임계에서도_발화하지_않는다():
    r = _rec("x", [0.99], parse_ok=False)
    assert image_score(r) < 0.0
    assert not fires(filter_by_conf([r], 0.01)[0])


# --------------------------------------------------------------------------------------
# 2 — 임계 독립성과 부호
# --------------------------------------------------------------------------------------

def test_완전_분리는_1():
    sc = {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}
    assert gini(sc, ["a", "b"], ["c", "d"]) == pytest.approx(1.0)


def test_완전_역전은_음수_1():
    """정상에서 더 높게 발화하면 음수다 — 0 이 바닥이 아니다."""
    sc = {"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9}
    assert gini(sc, ["a", "b"], ["c", "d"]) == pytest.approx(-1.0)


def test_단조_변환에_불변():
    """순위만 보는 지표다 — 점수를 단조 변환해도 값이 같아야 한다."""
    sc = {"a": 0.9, "b": 0.4, "c": 0.5, "d": 0.1}
    d, n = ["a", "b"], ["c", "d"]
    warped = {k: v**3 for k, v in sc.items()}
    assert gini(warped, d, n) == pytest.approx(gini(sc, d, n))


def test_동점은_절반으로_센다():
    """결함 1장과 정상 1장이 동점이면 AUROC 0.5 → Δ_AUC 0."""
    assert gini({"a": 0.5, "b": 0.5}, ["a"], ["b"]) == 0.0


def test_한쪽_층이_비면_nan():
    """재표집에서 한쪽이 통째로 안 뽑히는 경우다. 0 으로 대치하면 CI 가 0 쪽으로 끌린다."""
    assert math.isnan(gini({"a": 0.5}, ["a"], []))


# --------------------------------------------------------------------------------------
# 3 — image_score 가 발화 정의의 연속판이다
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("conf", [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.9, 1.0])
def test_점수_이상이_임계_필터_후_발화와_같다(conf):
    """`image_score(r) >= c` ⟺ `fires(filter_by_conf([r], c)[0])`.

    이 동치가 깨지면 Δ 곡선(임계로 자름)과 Δ_AUC(점수 순위)가 서로 다른 발화 개념을 쓴다.
    """
    recs = [
        _rec("a", [0.9, 0.3]),
        _rec("b", [0.05]),
        _rec("c", []),
        _rec("d", [0.25]),
        _rec("e", [0.5, 0.5]),
        _rec("f", [1.0], parse_ok=False),
    ]
    for r in recs:
        cut = filter_by_conf([r], conf)[0]
        assert (image_score(r) >= conf) == fires(cut), f"{r.image_id} @ {conf}"


# --------------------------------------------------------------------------------------
# 4 — numpy 구현이 기준 구현과 같은 값을 낸다
# --------------------------------------------------------------------------------------

def _reference_gini(scores, defect, normal):
    """순위를 손으로 세는 기준 구현. **느리지만 읽으면 정의가 보인다.**

    본체는 부트스트랩 안에서 수만 번 돌아야 해서 numpy 로 썼다(22번 §6-2-5 로 승격
    후보가 되어 본채점이 매 시드 부른다). 최적화가 정의를 바꾸지 않았다는 것은 시험이 본다.
    """
    d = [scores[i] for i in defect if i in scores]
    n = [scores[i] for i in normal if i in scores]
    if not d or not n:
        return float("nan")
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in d for b in n)
    return 2 * (wins / (len(d) * len(n))) - 1


@pytest.mark.parametrize("seed", range(8))
def test_numpy_구현이_기준_구현과_같다(seed):
    """무작위 점수 · 동점 다수 · 층 크기 불균형에서 두 구현이 일치해야 한다."""
    rng = random.Random(seed)
    n_img = rng.randint(4, 60)
    # 동점을 일부러 많이 만든다 — 동점 처리가 두 구현이 갈리기 쉬운 자리다
    sc = {f"i{k}": rng.choice([0.0, 0.25, 0.5, 0.5, 0.75, 1.0, -1.0]) for k in range(n_img)}
    ids = list(sc)
    rng.shuffle(ids)
    cut = rng.randint(1, n_img - 1)
    d, n = ids[:cut], ids[cut:]
    assert gini(sc, d, n) == pytest.approx(_reference_gini(sc, d, n), abs=1e-12)


def test_전부_동점이면_기준_구현도_0():
    """지름길 성질의 교차 확인 — 두 구현 모두 정확히 0 이어야 한다."""
    sc = dict.fromkeys([f"i{k}" for k in range(20)], 0.3)
    d, n = [f"i{k}" for k in range(12)], [f"i{k}" for k in range(12, 20)]
    assert gini(sc, d, n) == 0.0 == _reference_gini(sc, d, n)
