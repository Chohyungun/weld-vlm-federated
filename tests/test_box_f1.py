"""M1 — 클래스 일치 박스 매칭 F1@IoU 0.5. 07번 미니스펙 §2-4 의 시험 목록.

여기서 지키는 것 넷.

1. **TP 는 최대 매칭 크기다.** 무차별 대입으로 셈을 맞댄다 — 겹침 합 최대 배정과 갈리는 배치를 포함해서.
2. **계수가 2차 기준·입력 순서에 흔들리지 않는다.** 흔들리면 채점기가 비결정적이라는 뜻이다.
3. **정상 이미지가 모집단에 있다.** 빠지면 그 이미지의 오탐이 지표에서 사라진다.
4. **미정의를 0 으로 채우지 않는다.** 분모가 0 인 클래스는 값이 없는 것이지 0 점이 아니다.
"""

from __future__ import annotations

import itertools
import random
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from evaluation.metrics.box_f1 import (
    COORD_NUMERIC_CONVENTION,
    CountTable,
    build_counts,
    cluster_weights,
    is_candidate,
    max_matching,
    score_m1,
    sensitivity_counts,
)
from evaluation.metrics.localization import match_image

C = ("100", "2011", "301", "401")


def box(x, y, w=10, h=10):
    return (float(x), float(y), float(x + w), float(y + h))


def brute_max_matching(gold, pred) -> int:
    """모든 부분 배정을 열거한 최대 짝 수. 작은 경우에만 쓴다."""
    for r in range(min(len(gold), len(pred)), 0, -1):
        for gs in itertools.combinations(range(len(gold)), r):
            for ps in itertools.permutations(range(len(pred)), r):
                if all(is_candidate(gold[g], pred[p])[0] for g, p in zip(gs, ps, strict=True)):
                    return r
    return 0


# ---------------------------------------------------------------- 최대 매칭

def test_무작위_배치에서_무차별_대입과_같다() -> None:
    rng = random.Random(20260921)
    for _ in range(300):
        g = [box(rng.randint(0, 80), rng.randint(0, 80), rng.randint(5, 40), rng.randint(5, 40))
             for _ in range(rng.randint(0, 4))]
        p = [box(rng.randint(0, 80), rng.randint(0, 80), rng.randint(5, 40), rng.randint(5, 40))
             for _ in range(rng.randint(0, 4))]
        assert max_matching(g, p)[0] == brute_max_matching(g, p)


def test_겹침_합_최대_배정은_짝을_덜_짓는다() -> None:
    """**기존 매칭을 M1 에 그대로 쓸 수 없는 이유**를 실물로 고정한다.

    겹침 표가 `[[1.0, 0.5385], [0.5385, 0.25]]` 인 배치다.
    겹침 합을 최대화하면 `1.0 + 0.25 = 1.25` 를 골라 문턱을 넘는 쌍이 **하나**뿐이고,
    엇갈려 묶으면 `0.5385` 두 쌍이라 **둘**이다. TP 가 1 과 2 로 갈린다.
    """
    g = [(0.0, 0.0, 20.0, 20.0), (-6.0, 0.0, 14.0, 20.0)]
    p = [(0.0, 0.0, 20.0, 20.0), (6.0, 0.0, 26.0, 20.0)]
    cand = [[is_candidate(a, b)[0] for b in p] for a in g]
    assert cand == [[True, True], [True, False]], "전제가 깨졌다"

    assert max_matching(g, p)[0] == 2, "짝 수 최대"

    ms, _ = match_image([("2011", b) for b in p], [("2011", b) for b in g], "i")
    assert sum(1 for m in ms if m.iou >= 0.5) == 1, "겹침 합 최대는 하나만 넘긴다"


def test_겹침_0_은_어떤_경우에도_후보가_아니다() -> None:
    assert max_matching([box(0, 0)], [box(50, 50)]) == (0, 0)


def test_빈_쪽은_0() -> None:
    assert max_matching([], [box(0, 0)]) == (0, 0)
    assert max_matching([box(0, 0)], []) == (0, 0)


# ---------------------------------------------------------------- 경계

def test_정확히_절반이면_후보다() -> None:
    ok, boundary = is_candidate((0.0, 0.0, 2.0, 1.0), (0.0, 0.0, 1.0, 1.0))
    assert ok and boundary, "IoU 가 정확히 1/2 — 유리수로 되봐서 후보다"


def test_절반_바로_아래는_후보가_아니다() -> None:
    g = (0.0, 0.0, 2.0, 1.0)
    p = (0.0, 0.0, 1.0, 1.0)
    below = (0.0, 0.0, float(np.nextafter(1.0, 0.0)), 1.0)
    assert is_candidate(g, p)[0]
    assert not is_candidate(g, below)[0]


def test_경계쌍을_세어_보고한다() -> None:
    t = build_counts({"i": [("2011", (0.0, 0.0, 1.0, 1.0))]},
                     {"i": [("2011", (0.0, 0.0, 2.0, 1.0))]}, C)
    assert t.n_boundary_pairs == 1
    assert t.boundary_cells == (("i", "2011"),)


# ---------------------------------------------------------------- 계수

def test_중복_예측은_각각_센다() -> None:
    g = {"i": [("2011", box(0, 0))]}
    p = {"i": [("2011", box(0, 0)), ("2011", box(0, 0)), ("2011", box(0, 0))]}
    t = build_counts(p, g, C)
    c = t.classes.index("2011")
    assert (t.tp[0, c], t.fp[0, c], t.fn[0, c]) == (1, 2, 0)


def test_클래스가_다르면_자리가_같아도_못_짓는다() -> None:
    t = build_counts({"i": [("100", box(0, 0))]}, {"i": [("2011", box(0, 0))]}, C)
    assert t.tp.sum() == 0 and t.fp.sum() == 1 and t.fn.sum() == 1


def test_정상_이미지의_오탐이_센다() -> None:
    t = build_counts({"n": [("2011", box(0, 0))]}, {"n": []}, C)
    assert t.fp.sum() == 1 and t.tp.sum() == 0 and t.fn.sum() == 0


def test_정상_이미지의_빈_예측은_아무것도_늘리지_않는다() -> None:
    t = build_counts({"n": []}, {"n": []}, C)
    assert t.tp.sum() == t.fp.sum() == t.fn.sum() == 0


def test_모집단은_정답의_키_전량이다() -> None:
    t = build_counts({}, {"a": [], "b": [("2011", box(0, 0))]}, C)
    assert t.image_ids == ("a", "b")
    assert t.fn.sum() == 1


def test_정답에_없는_예측은_오류다() -> None:
    with pytest.raises(ValueError, match="정답에 없는"):
        build_counts({"x": []}, {"a": []}, C)


def test_입력_순서를_섞어도_계수가_같다() -> None:
    g = [("2011", box(0, 0)), ("2011", box(30, 30)), ("100", box(60, 60))]
    p = [("2011", box(1, 1)), ("100", box(61, 61)), ("2011", box(31, 31))]
    a = build_counts({"i": p}, {"i": g}, C)
    b = build_counts({"i": list(reversed(p))}, {"i": list(reversed(g))}, C)
    assert np.array_equal(a.tp, b.tp) and np.array_equal(a.fp, b.fp) and np.array_equal(a.fn, b.fn)


# ---------------------------------------------------------------- 합산과 부분집합

def test_표를_나눠_더하면_전체와_같다() -> None:
    gold = {f"i{n}": [("2011", box(n, n))] for n in range(10)}
    pred = {f"i{n}": [("2011", box(n, n))] if n % 2 else [] for n in range(10)}
    t = build_counts(pred, gold, C)
    left = t.subset([f"i{n}" for n in range(5)])
    right = t.subset([f"i{n}" for n in range(5, 10)])
    for a, b, whole in zip(left.totals(), right.totals(), t.totals(), strict=True):
        assert np.array_equal(a + b, whole)


def test_가중이_전부_1_이면_가중_없는_합과_같다() -> None:
    gold = {f"i{n}": [("2011", box(n, n))] for n in range(6)}
    t = build_counts({"i0": [("2011", box(0, 0))]}, gold, C)
    plain = t.totals()
    weighted = t.totals(np.ones(len(t.image_ids)))
    for a, b in zip(plain, weighted, strict=True):
        assert np.allclose(a, b)


def test_가중의_길이가_다르면_오류다() -> None:
    t = build_counts({}, {"a": [], "b": []}, C)
    with pytest.raises(ValueError, match="가중의 길이"):
        t.totals(np.ones(3))


def test_덩어리_가중을_이미지로_편다() -> None:
    w = cluster_weights(("a", "b", "c"), {"a": "g1", "b": "g1", "c": "g2"}, {"g1": 2.0, "g2": 5.0})
    assert list(w) == [2.0, 2.0, 5.0]


def test_모양이_맞지_않는_표는_만들_수_없다() -> None:
    with pytest.raises(ValueError, match="모양"):
        CountTable(("a",), ("i", "j"), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))


# ---------------------------------------------------------------- M1

def test_완벽한_예측은_1_이다() -> None:
    gold = {"i": [("2011", box(0, 0)), ("100", box(30, 30))]}
    r = score_m1(build_counts(gold, gold, C))
    assert r.macro_f1 == 1.0 and r.micro_f1 == 1.0
    assert r.classes_in_macro == ("100", "2011")
    assert set(r.classes_without_support) == {"301", "401"}


def test_평균_대상은_정답_지지량으로_정한다() -> None:
    """예측으로 정하면 모델이 어떤 클래스를 안 내는 것만으로 점수가 오른다."""
    gold = {"i": [("2011", box(0, 0)), ("100", box(30, 30))]}
    pred = {"i": [("2011", box(0, 0))]}          # 균열을 아예 내지 않았다
    r = score_m1(build_counts(pred, gold, C))
    assert "100" in r.classes_in_macro and r.per_class_f1["100"] == 0.0
    assert r.macro_f1 == pytest.approx(0.5)


def test_지지량이_없는_클래스는_미정의다() -> None:
    r = score_m1(build_counts({"i": []}, {"i": []}, C))
    assert r.macro_f1 is None and r.micro_f1 is None
    assert all(v is None for v in r.per_class_f1.values())
    assert r.classes_in_macro == ()


def test_넷보다_적으면_4결함_매크로라_부르지_않는다() -> None:
    gold = {"i": [("2011", box(0, 0))]}
    assert score_m1(build_counts(gold, gold, C)).label == "1결함 매크로"


def test_micro_는_계수를_통째로_더한_값이다() -> None:
    gold = {"a": [("2011", box(0, 0))], "b": [("100", box(0, 0))]}
    pred = {"a": [("2011", box(0, 0))], "b": []}
    r = score_m1(build_counts(pred, gold, C))
    assert r.micro_f1 == pytest.approx(2 * 1 / (2 * 1 + 0 + 1))


# ---------------------------------------------------------------- 민감도

def test_정상_이미지의_실패에만_오탐을_더한다() -> None:
    gold = {"n": [], "d": [("2011", box(0, 0))]}
    base = build_counts({"n": [], "d": []}, gold, C)
    s = sensitivity_counts(base, gold_codes={"n": [], "d": ["2011"]},
                           failed_image_ids=["n", "d"])
    idx = s.index_of()
    assert list(s.fp[idx["n"]]) == [1, 1, 1, 1], "정상 실패는 채점 4클래스에 각 1건"
    assert list(s.fp[idx["d"]]) == [0, 0, 0, 0], "결함 실패는 주 채점과 같다"
    assert np.array_equal(s.tp, base.tp) and np.array_equal(s.fn, base.fn)
    assert s.notes["n_normal_failures_penalised"] == 1


def test_민감도는_주_채점을_바꾸지_않는다() -> None:
    gold = {"n": []}
    base = build_counts({"n": []}, gold, C)
    before = base.fp.copy()
    sensitivity_counts(base, gold_codes=gold, failed_image_ids=["n"])
    assert np.array_equal(base.fp, before), "원본 표를 건드리면 두 값이 같은 배열을 본다"


# ================================================================ 후속 검수 대응
#
# 아래는 §27-24·26 의 지적마다 하나씩 대응하는 작은 회귀다. 건수로 종결하지 않는다 —
# 어느 지적에 어느 시험이 서는지 이름과 설명으로 잇는다.


# ---------------------------------------------------------------- 십진/이진 규약

def test_십진으로_정확히_절반인_쌍이_이진에서는_빠진다() -> None:
    """**규약이 갈리는 실물**을 고정한다. 숨기지 않고 시험으로 남긴다.

    정답 `[0, 0, 0.3, 0.1]` · 예측 `[0.1, 0, 0.4, 0.1]` 은 십진 기하로 IoU 가 정확히 1/2 이다.
    배정밀도에서는 `2·교집합 − 합집합 = −5.55e-18` 로 1/2 **미만**이라 후보가 아니다.
    정본은 이진이다(모듈 머리말). 바꾸려면 어댑터의 `parse_float` 까지 함께 바꾸고 새 지문으로 등록한다.
    """
    g = (0.0, 0.0, 0.3, 0.1)
    p = (0.1, 0.0, 0.4, 0.1)

    dec = [Fraction(Decimal(v)) for v in ("0", "0", "0.3", "0.1", "0.1", "0", "0.4", "0.1")]
    ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = dec
    inter = (min(ax2, bx2) - max(ax1, bx1)) * (min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    assert 2 * inter - union == 0, "십진에서는 정확히 경계다"

    ok, boundary = is_candidate(g, p)
    assert boundary is True, "경계에서 되본 쌍이라는 사실은 남는다"
    assert ok is False, "이진 규약에서는 후보가 아니다"


def test_규약을_산출물에_박는다() -> None:
    t = build_counts({}, {"i": []}, C)
    assert t.notes["coord_numeric_convention"] == COORD_NUMERIC_CONVENTION == "binary64"


# ---------------------------------------------------------------- 부분집합 진단 메타

def test_전체_id_로_다시_고르면_진단이_그대로다() -> None:
    """총계만 들고 있으면 여기서 경계쌍이 0 으로 보고된다 — 거짓 진단이 산출물로 나간다."""
    pred = {"a": [("2011", (0.0, 0.0, 1.0, 1.0))], "b": [("2011", box(50, 50))]}
    gold = {"a": [("2011", (0.0, 0.0, 2.0, 1.0))], "b": []}
    t = build_counts(pred, gold, C)
    assert t.n_boundary_pairs == 1 and t.n_pred_boxes == 2

    same = t.subset(t.image_ids)
    assert same.n_boundary_pairs == t.n_boundary_pairs
    assert same.boundary_cells == t.boundary_cells
    assert same.n_pred_boxes == t.n_pred_boxes
    assert same.notes["n_images"] == t.notes["n_images"], "원래 notes 를 덮지 않는다"


def test_경계쌍이_없는_쪽을_고르면_0_이_맞다() -> None:
    pred = {"a": [("2011", (0.0, 0.0, 1.0, 1.0))], "b": [("2011", box(50, 50))]}
    gold = {"a": [("2011", (0.0, 0.0, 2.0, 1.0))], "b": []}
    t = build_counts(pred, gold, C)
    assert t.subset(["a"]).n_boundary_pairs == 1
    assert t.subset(["b"]).n_boundary_pairs == 0, "경계쌍이 실제로 없는 부분집합"
    assert t.subset(["b"]).n_pred_boxes == 1


def test_부분집합의_진단을_나눠_더하면_전체와_같다() -> None:
    gold = {f"i{n}": [("2011", (0.0, 0.0, 2.0, 1.0))] for n in range(6)}
    pred = {f"i{n}": [("2011", (0.0, 0.0, 1.0, 1.0))] for n in range(6)}
    t = build_counts(pred, gold, C)
    left, right = t.subset([f"i{n}" for n in range(3)]), t.subset([f"i{n}" for n in range(3, 6)])
    assert left.n_boundary_pairs + right.n_boundary_pairs == t.n_boundary_pairs == 6
    assert left.n_pred_boxes + right.n_pred_boxes == t.n_pred_boxes == 6


def test_민감도_변형도_진단을_보존한다() -> None:
    pred = {"n": [("2011", (0.0, 0.0, 1.0, 1.0))]}
    gold = {"n": [("2011", (0.0, 0.0, 2.0, 1.0))]}
    base = build_counts(pred, gold, C)
    s = sensitivity_counts(base, gold_codes={"n": ["2011"]}, failed_image_ids=[])
    assert s.n_boundary_pairs == base.n_boundary_pairs == 1
    assert s.n_pred_boxes == base.n_pred_boxes == 1
    assert s.notes["variant"] == "sensitivity"
    assert s.notes["n_images"] == base.notes["n_images"], "원래 notes 가 남는다"


# ---------------------------------------------------------------- 고정 평균 대상

def test_가중이_지지량을_없애면_그_추첨은_미정의다() -> None:
    """**점수가 좋아진 것이 아니라 잴 수 없는 것이다.**

    정답이 각각 A·B 인 이미지 둘에서 A 만 맞춘 표는 비가중 macro 가 0.5 다. 가중 `[1, 0]` 을
    주면 B 의 지지량이 0 이 되어 B 가 빠지고 macro 가 1 로 올라간다. 재표집이 이 경로를 타면
    지지량이 얇은 추첨마다 점수가 부풀려진다.
    """
    gold = {"a": [("2011", box(0, 0))], "b": [("100", box(0, 0))]}
    pred = {"a": [("2011", box(0, 0))], "b": []}
    t = build_counts(pred, gold, C)
    fixed = ("100", "2011")

    assert score_m1(t, fixed_classes=fixed).macro_f1 == pytest.approx(0.5)

    w = np.array([1.0, 0.0])
    loose = score_m1(t, w)
    assert loose.macro_f1 == pytest.approx(1.0), "대상을 다시 정하면 1 로 올라간다"
    assert loose.macro_support_rule == "gold_support"

    strict = score_m1(t, w, fixed_classes=fixed)
    assert strict.macro_f1 is None
    assert strict.undefined_reason == "support"
    assert strict.macro_support_rule == "fixed"
    assert strict.classes_without_support == ("100",)
    assert strict.classes_in_macro == fixed, "의도한 대상을 그대로 들고 있다"
    assert strict.micro_f1 is not None, "micro 는 여전히 정의된다"


def test_모집단을_바꿔_대상이_달라지는_것은_정당하다() -> None:
    """점추정에서는 고정 대상을 주지 않는다 — 그때의 대상 변경은 모집단 변경이다."""
    gold = {"a": [("2011", box(0, 0))], "b": [("100", box(0, 0))]}
    t = build_counts(gold, gold, C).subset(["a"])
    r = score_m1(t)
    assert r.classes_in_macro == ("2011",) and r.macro_f1 == pytest.approx(1.0)
    assert r.undefined_reason is None


def test_고정_대상이_전부_살아_있으면_그_수로_나눈다() -> None:
    gold = {"a": [("2011", box(0, 0))], "b": [("100", box(0, 0))]}
    pred = {"a": [("2011", box(0, 0))], "b": []}
    r = score_m1(build_counts(pred, gold, C), fixed_classes=("100", "2011"))
    assert r.macro_f1 == pytest.approx(0.5)
    assert r.label == "2결함 매크로"


def test_계수표에_없는_클래스를_고정_대상으로_줄_수_없다() -> None:
    t = build_counts({}, {"i": []}, C)
    with pytest.raises(ValueError, match="고정 평균 대상"):
        score_m1(t, fixed_classes=("9999",))


def test_지지량이_아예_없으면_사유를_남긴다() -> None:
    r = score_m1(build_counts({"i": []}, {"i": []}, C))
    assert r.macro_f1 is None and r.undefined_reason == "no_support"


# ---------------------------------------------------------------- 가중 검증

@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_유한하지_않거나_음수인_가중은_거부한다(bad: float) -> None:
    """음수 가중은 FP 를 음수로 만들어 F1 을 1 위로 올린다."""
    gold = {"a": [("2011", box(0, 0))], "b": []}
    t = build_counts({"a": [("2011", box(0, 0))], "b": [("2011", box(0, 0))]}, gold, C)
    with pytest.raises(ValueError, match="유한하지 않은|음수"):
        t.totals(np.array([1.0, bad]))


def test_0_가중은_정상이다() -> None:
    """재표집에서 안 뽑힌 이미지와 부표본 밖이 0 이다."""
    gold = {"a": [("2011", box(0, 0))], "b": [("2011", box(0, 0))]}
    t = build_counts(gold, gold, C)
    tp, fp, fn = t.totals(np.zeros(2))
    assert tp.sum() == fp.sum() == fn.sum() == 0


def test_전부_0_가중은_지지량_없음으로_접힌다() -> None:
    gold = {"a": [("2011", box(0, 0))]}
    r = score_m1(build_counts(gold, gold, C), np.zeros(1))
    assert r.macro_f1 is None and r.undefined_reason == "no_support"


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_덩어리_가중도_유한_비음수를_본다(bad: float) -> None:
    with pytest.raises(ValueError, match="유한하지 않은|음수"):
        cluster_weights(("a", "b"), {"a": "g1", "b": "g2"}, {"g1": 1.0, "g2": bad})


def test_음수_계수인_표는_만들_수_없다() -> None:
    """직접 만든 계수표도 믿지 않는다."""
    with pytest.raises(ValueError, match="음수"):
        CountTable(("a",), ("i",), np.array([[-1]]), np.zeros((1, 1)), np.zeros((1, 1)))


def test_유한하지_않은_계수인_표는_만들_수_없다() -> None:
    with pytest.raises(ValueError, match="유한하지 않은"):
        CountTable(("a",), ("i",), np.array([[np.nan]]), np.zeros((1, 1)), np.zeros((1, 1)))
