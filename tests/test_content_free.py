"""무내용 대조선 — 점수 구성 H·F 와 적합 격리 (22번 §5, 사전등록 17번 §11).

이 파일이 지키는 것은 셋이다.

1. **`AP = P·R`** — 하드 예측기의 AP 는 정의상 정밀도×재현율이다. 이 항등식이 깨지면
   대조선 값이 다른 규칙의 값이 된다(총괄 검증의 추정이 선 자리이기도 하다).
2. **적합이 평가셋을 열지 않는다** — 적합 모집단의 정답만으로 규칙이 만들어져야 한다.
   평가셋 정답이 새면 대조선이 오라클이 되어 게이트가 무의미해진다.
3. **H 와 F 의 구분** — H 는 예측하지 않은 클래스에 점수를 주지 않는다(부재 ≠ 0점).
   0점을 주면 H 가 F 의 열화판이 되어 두 구성이 같아진다.
"""

from __future__ import annotations

import pytest

from evaluation.content_free import (
    FREQ,
    HARD,
    fit_idq,
    fit_trivial,
    predict_codes,
    scored,
)
from evaluation.metrics.detection import image_level_ap, score_detection

CLASSES = ("100", "2011")


def _ids(n: int, start: int = 0) -> list[str]:
    return [f"aihub71761:{10_000_000 + start + i}" for i in range(n)]


# --------------------------------------------------------------------------------------
# 1 — AP = P·R (하드 예측기)
# --------------------------------------------------------------------------------------

def test_하드_예측기의_AP_는_정밀도x재현율이다():
    """총괄 검증의 추정이 서는 항등식. 깨지면 대조선 값이 다른 규칙의 값이다."""
    ids = _ids(10)
    # 앞 6장이 양성, 규칙은 앞 5장에만 100 을 주장 → P = 5/5, R = 5/6
    gold = {i: (["100"] if n < 6 else []) for n, i in enumerate(ids)}
    pred = {i: ([("100", 1.0)] if n < 5 else []) for n, i in enumerate(ids)}
    ap = image_level_ap(pred, gold, ["100"])["per_class_ap"]["100"]
    assert ap == pytest.approx(1.0 * (5 / 6))


def test_P_R_은_F1_제곱_이상이다():
    """`P·R ≥ F1²` — (P−R)² ≥ 0 과 같은 말이다. 대조선 하한 추정의 근거."""
    for p, r in ((0.9, 0.5), (0.3, 0.99), (0.7, 0.7), (0.2, 0.05)):
        f1 = 2 * p * r / (p + r)
        assert p * r >= f1**2 - 1e-12


# --------------------------------------------------------------------------------------
# 2 — 적합 격리: 평가셋 정답이 새지 않는다
# --------------------------------------------------------------------------------------

def test_적합은_적합_모집단_정답만_쓴다():
    """평가셋 정답을 바꿔도 규칙이 변하지 않아야 한다. 변하면 대조선이 오라클이다."""
    fit, ev = _ids(200), _ids(60, start=500_000)
    gold_fit = {i: (["100"] if n % 2 == 0 else ["2011"]) for n, i in enumerate(fit)}
    rule_a = fit_idq(4, fit, gold_fit, CLASSES)

    # 평가셋 정답을 완전히 뒤집어도 같은 규칙이 나온다 — 적합에 안 들어가기 때문이다
    rule_b = fit_idq(4, fit, {**gold_fit, **{i: ["2011"] for i in ev}}, CLASSES)
    assert {b: (r.majority, tuple(sorted(r.freq.items())))
            for b, r in rule_a.bins.items()} == \
           {b: (r.majority, tuple(sorted(r.freq.items())))
            for b, r in rule_b.bins.items()}
    assert rule_a.fit_population == "train+val"


def test_구간_빈도는_적합_모집단의_비율이다():
    fit = _ids(100)
    gold = {i: (["100"] if n < 40 else []) for n, i in enumerate(fit)}
    rule = fit_idq(2, fit, gold, CLASSES)
    total = sum(br.n_fit for br in rule.bins.values())
    assert total == 100
    assert rule.global_freq["100"] == pytest.approx(0.40)
    assert rule.global_freq["2011"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# 3 — H 와 F 의 구분
# --------------------------------------------------------------------------------------

def test_H_는_예측하지_않은_클래스에_점수를_주지_않는다():
    """부재는 0점이 아니다. 0점을 주면 H 가 F 의 열화판이 된다."""
    fit = _ids(100)
    gold = {i: ["100"] for i in fit}          # 전부 100 → 최빈 집합 = {100}
    rule = fit_idq(2, fit, gold, CLASSES)
    s, _ = scored(rule, fit[:5], HARD)
    for v in s.values():
        assert [c for c, _ in v] == ["100"]   # 2011 은 아예 없다
        assert all(score == 1.0 for _, score in v)


def test_F_는_네_클래스_전부에_점수를_준다_0_포함():
    fit = _ids(100)
    gold = {i: ["100"] for i in fit}
    rule = fit_idq(2, fit, gold, CLASSES)
    s, diag = scored(rule, fit[:5], FREQ)
    for v in s.values():
        assert sorted(c for c, _ in v) == sorted(CLASSES)
        assert dict(v)["2011"] == pytest.approx(0.0)   # 0 점도 순위에 들어간다
    assert diag["construction"] == FREQ


def test_상수_예측기는_H_와_F_가_같은_AP_를_준다():
    """순위가 없는 예측기라 두 구성이 같아야 한다 — 표의 정합 검사다."""
    fit, ev = _ids(200), _ids(80, start=900_000)
    gold_fit = {i: (["100"] if n % 3 == 0 else []) for n, i in enumerate(fit)}
    gold_ev = {i: (["100"] if n % 3 == 0 else []) for n, i in enumerate(ev)}
    rule = fit_trivial("all_positive", CLASSES, fit, gold_fit, CLASSES)
    aps = {}
    for name in (HARD, FREQ):
        s, _ = scored(rule, ev, name)
        aps[name] = image_level_ap(s, gold_ev, CLASSES)["per_class_ap"]["100"]
    assert aps[HARD] == pytest.approx(aps[FREQ])


def test_전량양성의_AP_는_유병률이다():
    """P = 유병률, R = 1 → AP = 유병률. 자명하한이 그 값이어야 한다."""
    fit, ev = _ids(100), _ids(100, start=700_000)
    gold_fit = {i: (["100"] if n < 25 else []) for n, i in enumerate(fit)}
    gold_ev = {i: (["100"] if n < 30 else []) for n, i in enumerate(ev)}
    rule = fit_trivial("all_positive", CLASSES, fit, gold_fit, CLASSES)
    s, _ = scored(rule, ev, HARD)
    assert image_level_ap(s, gold_ev, ["100"])["per_class_ap"]["100"] == pytest.approx(0.30)


# --------------------------------------------------------------------------------------
# 규칙 자체
# --------------------------------------------------------------------------------------
@pytest.fixture
def half_bins(monkeypatch):
    """id 를 앞뒤 절반으로 가르는 구간 배정을 주입한다.

    **구간 배정 자체는 A 소관**(`data.id_strata`)이고 그 정합은
    `tests/test_strata_alignment.py` 가 본다. 여기서 보는 것은 그 위의 적합·예측 논리라,
    합성 id 가 동결본 절단점에서 한 구간으로 몰리는 것에 시험이 좌우되지 않게 주입한다.
    """
    import evaluation.content_free as cf

    def fake(ids, k, snapshot=None):
        ordered = sorted(ids)
        half = len(ordered) // 2
        return {i: (0 if n < half else 1) for n, i in enumerate(ordered)}

    monkeypatch.setattr(cf, "bins_for", fake)
    return fake


def test_예측은_구간_최빈_집합이다(half_bins):
    fit = _ids(100)
    # 앞 절반은 100, 뒤 절반은 아무것도 없음 → 구간마다 최빈이 다르다
    gold = {i: (["100"] if n < 50 else []) for n, i in enumerate(fit)}
    rule = fit_idq(2, fit, gold, CLASSES)
    codes = predict_codes(rule, fit)
    assert codes[fit[0]] == ["100"]
    assert codes[fit[-1]] == []


def test_구간_최빈_규칙이_상수_예측기보다_F1_이_높다(half_bins):
    """구간을 쓰는 것이 값을 하는지 — 안 그러면 대조선이 자명하한과 같아진다."""
    fit = _ids(200)
    gold = {i: (["100"] if n < 100 else ["2011"]) for n, i in enumerate(fit)}
    idq = predict_codes(fit_idq(2, fit, gold, CLASSES), fit)
    const = predict_codes(fit_trivial("c", ["100"], fit, gold, CLASSES), fit)
    g = {k: sorted(v) for k, v in gold.items()}
    assert (score_detection(idq, g, CLASSES).macro_f1
            > score_detection(const, g, CLASSES).macro_f1)


def test_구간_배정은_A_의_절단점을_쓴다():
    """주입 없이는 `evaluation.strata.bins_for` 를 그대로 부른다 — 절단점을 D 가 다시
    만들지 않는다(동결본 train+val 분위 단일 출처)."""
    import inspect

    import evaluation.content_free as cf

    src = inspect.getsource(cf)
    assert "from evaluation.strata import bins_for" in src
    assert "quantile" not in src and "searchsorted" not in src


def test_알_수_없는_점수_구성은_거부된다():
    fit = _ids(20)
    rule = fit_trivial("x", CLASSES, fit, {i: [] for i in fit}, CLASSES)
    with pytest.raises(ValueError, match="점수 구성"):
        scored(rule, fit, "Z")
