"""묶음 클러스터 부트스트랩 · TOST · 시드 분산 진단 · P9 · D-3 테스트. §5-3·§5-4.

**이미지 단위로 재표집하면 CI 가 좁아진다**는 것이 이 모듈의 존재 이유이고, 그 사실을
테스트가 직접 보인다.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.probes.cross_source import (
    CROP,
    TILE,
    NormalImage,
    fp_breakdown_by_source,
    p9_cross_source,
)
from evaluation.stats import (
    MAX_RECOVERY_HALF_WIDTH,
    Interval,
    cluster_bootstrap,
    rate_by_group,
    recovery_denominator_verdict,
    seed_sd_diagnostic,
    tost_equivalence,
)

# --- 클러스터 부트스트랩 ---------------------------------------------------------

def test_bootstrap_point_matches_direct_computation():
    per_image = {f"i{i}": float(i % 2) for i in range(20)}
    groups = {f"i{i}": f"g{i // 2}" for i in range(20)}
    ci = cluster_bootstrap(sorted(set(groups.values())), rate_by_group(per_image, groups))
    assert ci.point == pytest.approx(0.5)


def test_bootstrap_is_deterministic():
    """시드 고정 — CI 가 흔들리면 게이트가 흔들린다."""
    per_image = {f"i{i}": float(i % 3 == 0) for i in range(30)}
    groups = {f"i{i}": f"g{i // 3}" for i in range(30)}
    stat = rate_by_group(per_image, groups)
    units = sorted(set(groups.values()))
    a = cluster_bootstrap(units, stat)
    b = cluster_bootstrap(units, stat)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_cluster_ci_is_wider_than_image_level_ci():
    """같은 데이터를 이미지 단위로 재표집하면 CI 가 좁아진다.
    묶음 안이 완전상관인데 독립 표본처럼 세기 때문이다."""
    n_groups, per = 40, 5
    per_image, groups = {}, {}
    rng = np.random.default_rng(0)
    for g in range(n_groups):
        v = float(rng.integers(0, 2))          # 묶음 안은 값이 같다(완전상관)
        for k in range(per):
            img = f"i{g}_{k}"
            per_image[img] = v
            groups[img] = f"g{g}"

    cluster = cluster_bootstrap(
        sorted(set(groups.values())), rate_by_group(per_image, groups)
    )
    image_units = sorted(per_image)
    image_level = cluster_bootstrap(
        image_units, rate_by_group(per_image, {i: i for i in image_units})
    )
    assert cluster.half_width > image_level.half_width * 1.5


def test_empty_input_gives_degenerate_interval():
    ci = cluster_bootstrap([], lambda _: 0.0)
    assert (ci.n_clusters, ci.half_width) == (0, 0.0)


# --- TOST ------------------------------------------------------------------------

def test_tost_equivalent_when_ci_inside_margin():
    ci = Interval(0.01, -0.03, 0.05, 2000, 100)
    assert tost_equivalence(ci, 0.10).equivalent


def test_tost_not_equivalent_when_ci_exceeds_margin():
    ci = Interval(0.02, -0.02, 0.15, 2000, 100)
    r = tost_equivalence(ci, 0.10)
    assert not r.equivalent and "벗어난다" in r.verdict


def test_wide_ci_cannot_claim_equivalence():
    """검정력이 낮을수록 '차이 없음'이 쉬워지는 역인센티브를 TOST 가 막는다."""
    wide = Interval(0.0, -0.50, 0.50, 2000, 100)
    assert not tost_equivalence(wide, 0.10).equivalent


def test_underpowered_reports_undecidable_not_pass():
    """'판정 불가'는 통과가 아니다."""
    ci = Interval(0.0, -0.01, 0.01, 2000, 5)
    r = tost_equivalence(ci, 0.10, min_clusters=20)
    assert not r.equivalent
    assert "판정 불가" in r.verdict and "지름길 없음" in r.verdict


# --- 시드 분산 진단 (3시드 검정력 문제) ---------------------------------------------

def test_three_seed_sd_is_flagged_unreliable():
    d = seed_sd_diagnostic([0.80, 0.82, 0.81])
    assert not d.reliable
    assert d.cv == pytest.approx(0.5227, abs=0.001)


def test_three_seed_sd_ci_is_very_wide():
    """자유도 2 — σ 의 95% CI 상한이 점추정의 3배를 넘는다."""
    d = seed_sd_diagnostic([0.80, 0.82, 0.81])
    assert d.ci_hi > d.s * 3


def test_sd_bias_correction_inflates_point_estimate():
    """E[s] = c4·σ 라 n=3 에서 s 는 σ 를 평균 11.4% 과소추정한다."""
    d = seed_sd_diagnostic([0.80, 0.82, 0.81])
    assert d.bias_corrected > d.s


def test_single_seed_cannot_estimate_variance():
    assert not seed_sd_diagnostic([0.80]).reliable


# --- 회복률 분모 판정 -------------------------------------------------------------

def test_wide_denominator_is_reportable():
    v = recovery_denominator_verdict(0.80, [0.60], seed_values=[0.60, 0.61, 0.60])
    assert v.tripwire_pass and v.reportable


def test_narrow_denominator_blocks_reporting():
    v = recovery_denominator_verdict(0.62, [0.60], seed_values=[0.58, 0.62, 0.60])
    assert not v.tripwire_pass and not v.reportable
    assert "세 절대값만" in v.detail


def test_wide_recovery_ci_blocks_even_if_tripwire_passes():
    """3σ 트립와이어만으로는 검정력이 낮다 — CI 반폭이 보조 판정을 맡는다."""
    wide = Interval(0.85, 0.40, 1.30, 2000, 300)
    v = recovery_denominator_verdict(
        0.80, [0.60], seed_values=[0.60, 0.601, 0.60], recovery_ci=wide
    )
    assert v.tripwire_pass
    assert v.ci_pass is False and not v.reportable


def test_narrow_recovery_ci_passes_secondary_check():
    tight = Interval(0.90, 0.82, 0.98, 2000, 300)
    v = recovery_denominator_verdict(
        0.80, [0.60], seed_values=[0.60, 0.601, 0.60], recovery_ci=tight
    )
    assert v.ci_pass is True and v.reportable
    assert tight.half_width <= MAX_RECOVERY_HALF_WIDTH


def test_verdict_carries_seed_diagnostic():
    v = recovery_denominator_verdict(0.80, [0.60], seed_values=[0.60, 0.62, 0.61])
    assert v.diagnostic is not None
    assert "불안정" in v.diagnostic.note


# --- P9 교차 출처 ------------------------------------------------------------------

def normals(n_crop: int, n_tile: int, fp_crop: float, fp_tile: float,
            per_group: int = 1) -> list[NormalImage]:
    out = []
    for i in range(n_crop):
        out.append(NormalImage(f"c{i}", f"gc{i // per_group}", CROP, i < n_crop * fp_crop))
    for i in range(n_tile):
        out.append(NormalImage(f"t{i}", f"gt{i // per_group}", TILE, i < n_tile * fp_tile))
    return out


def test_p9_equivalent_when_rates_match():
    r = p9_cross_source(normals(200, 400, 0.10, 0.10))
    assert r.tost.equivalent and r.reportable


def test_p9_not_equivalent_on_large_gap():
    """출처가 새 지름길이 된 경우."""
    r = p9_cross_source(normals(200, 400, 0.05, 0.60))
    assert not r.tost.equivalent


def test_p9_underpowered_is_not_reportable():
    """N-crop 이 ≈65 묶음뿐이라 실재하는 위험이다."""
    r = p9_cross_source(normals(10, 400, 0.10, 0.10), min_clusters=20)
    assert not r.reportable
    assert "판정 불가" in r.tost.verdict


def test_p9_headline_states_margin_and_verdict():
    r = p9_cross_source(normals(200, 400, 0.10, 0.10))
    h = r.headline()
    assert "TOST" in h and "0.1" in h and "95% CI" in h


# --- D-3 출처축 분해 ---------------------------------------------------------------

def test_breakdown_separates_sources():
    b = fp_breakdown_by_source(normals(100, 100, 0.10, 0.50))
    assert b[CROP]["fp_rate"] == pytest.approx(0.10)
    assert b[TILE]["fp_rate"] == pytest.approx(0.50)


def test_total_is_dominated_by_majority_source():
    """N-tile 이 8배라 총계는 사실상 N-tile 값이 된다 — 분해 보고가 필요한 이유."""
    b = fp_breakdown_by_source(normals(65, 520, 0.02, 0.40))
    assert abs(b["전체"]["fp_rate"] - b[TILE]["fp_rate"]) < 0.05
    assert b[CROP]["fp_rate"] < 0.05


def test_breakdown_counts_groups_not_only_images():
    b = fp_breakdown_by_source(normals(100, 100, 0.0, 0.0, per_group=10))
    assert b[CROP]["n_groups"] == 10
