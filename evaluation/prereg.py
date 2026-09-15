"""사전등록 상수 — 모든 결과표의 머리행에 박는다. `40_규격지름길_대응판정.md` §5-1.

학습 착수 전에 커밋하고 **이후 수정하지 않는다.** 결과를 보고 기준을 고르는 것이
사후 합리화이기 때문이다.

**모든 게이트는 헤드라인 과제 위에 놓는다** — 이미지 수준 다중라벨 **4결함** Macro-F1.
정상은 클래스가 아니라 평가 모집단에 남아 오탐(FP) 원천으로만 들어간다
(`evaluation.metrics.detection` 계약, 게이트 #11 확정).

이진 정확도나 5클래스 지표를 게이트에 쓰면 **미처리 원본이 게이트를 통과한다** —
계약 #4가 채점하지 않는 과제의 값이기 때문이다.

    from evaluation.prereg import PREREG, all_positive_macro_f1, spec_only_macro_f1
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

RT_TOTAL = 62_308
"""RT 전량 — **동결 스냅샷 `aihub71761_rt_v1` 실측.**

이전 값 62,998 은 동결 전 집계였고 690장 많았다. 자명하한은 유병률의 함수라 N 이 틀리면
게이트선이 통째로 틀어진다 — 실제로 등록 통과선 0.2131 이 **동결 평가셋 자명하한
0.21595 보다 낮아** 과엄격 헛경보가 될 상태였다(80번 D2).
"""

R1_COUNT = 62_308
"""1280×720 이미지 수. 규격전용 예측기의 예측 양성 수.

**재인코딩 후 전량이 1280×720 이다.** 그래서 규격전용 규칙은 전량양성과 같은 예측기가
되고 지름길 순 기여가 0 이 된다 — 함정 #11 대응이 성공했다는 증거다. 이전 값 37,814 는
처리 전 값이며, 같은 이름의 다른 양이다.
"""

TOLERANCE = 0.005
"""P1′ 통과선의 여유. 전량양성 자명하한 + 이 값 이하여야 지름길이 죽은 것으로 본다."""

EVAL_ALL_POSITIVE = 0.21595238
"""**동결 평가셋(12,461장)** 의 전량양성 자명하한.

전량(62,308) 값과 다르다. 채점은 평가셋에서 하므로 게이트를 세울 때는 이쪽을 쓴다 —
전량 값으로 선을 그으면 채점 모집단에서 자명하한 아래에 선이 놓인다.
`pass_line_for()` 를 통해 쓴다.
"""

TRAINVAL_ALL_POSITIVE = 0.20989820
"""동결 train+val(49,847장) 자명하한. 참고값."""

BEFORE_PREPROCESS = {
    "RT_TOTAL": 62_998, "R1_COUNT": 37_814,
    "all_positive_macro_f1": 0.2081, "spec_only_macro_f1": 0.3025,
    "shortcut_contribution": 0.0944,
}
"""**처리 전 등록값.** 지우지 않는다 — "지름길이 죽었다"는 주장은 전후 대조이고,
전 값을 지우면 그 대조가 사라진다. 게이트 판정에는 쓰지 않는다.
"""


@dataclass(frozen=True)
class PreregisteredConstants:
    """§5-1 표. **동결 평가셋에서 재산출해 이 값을 재현해야 계측이 옳다.**

    값의 출처: `scripts/probe/recompute_prereg.py` →
    `outputs/pilot_d/prereg_recomputed_v1.json` (동결 digest 1f80e98b…).
    """

    all_positive_macro_f1: float = 0.21111837
    """전량양성 자명하한 — 규격을 무시하고 모든 이미지에 4코드를 주장. 동결 전량 기준."""
    spec_only_macro_f1: float = 0.21111837
    """규격전용 최적 — 1280×720 에만 4코드를 주장. **처리 후 전량양성과 같다.**"""
    shortcut_contribution: float = 0.0
    """규격 지름길의 순 기여. 위 둘의 차. **0 이 목표값이었다**(처리 전 0.0944)."""
    sds_rt: float = 0.9484
    sds_st: float = 0.9613
    sds_al: float = 0.871
    """규격 결정력 — "1280×720 → 결함" 이진 규칙의 정확도. **헤드라인이 아니라 참고값**
    이며 게이트로 쓰지 않는다. AL 은 최빈 반올림에서 유도해 M0 로 확정한다."""

    @property
    def p1_prime_gate(self) -> float:
        """P1′ 통과선(동결 전량 기준). 채점 모집단이 평가셋이면 `pass_line_for` 를 쓴다."""
        return round(self.all_positive_macro_f1 + TOLERANCE, 4)

    def as_header_rows(self) -> tuple[tuple[str, str], ...]:
        """결과표 머리행. 지표 함수는 건드리지 않고 표에만 얹는다(D-2)."""
        return (
            ("전량양성 자명하한 (4결함 Macro-F1)", f"{self.all_positive_macro_f1:.4f}"),
            ("규격전용 최적 (처리 전)", f"{self.spec_only_macro_f1:.4f}"),
            ("규격 지름길 순 기여 (처리 전)", f"+{self.shortcut_contribution:.4f}"),
            ("규격 결정력 SDS (RT/ST/AL, 참고값)",
             f"{self.sds_rt:.4f} / {self.sds_st:.4f} / {self.sds_al:.3f}"),
        )


PREREG = PreregisteredConstants()


def _macro(values: Mapping[str, float]) -> float:
    return sum(values.values()) / len(values) if values else 0.0


def all_positive_macro_f1(
    class_counts: Mapping[str, int], n_images: int = RT_TOTAL
) -> tuple[float, dict[str, float]]:
    """전량양성 예측기의 4결함 Macro-F1.

    모든 이미지에 전 클래스를 주장하므로 클래스별 예측 양성 = `n_images`,
    TP = `n_c` 이고 `F1_c = 2·n_c / (n_images + n_c)` 다. **자명하한**이며, 어떤 모델도
    이 값을 못 넘으면 학습이 아무 일도 하지 않은 것이다.
    """
    per = {c: 2 * n / (n_images + n) for c, n in class_counts.items()}
    return _macro(per), per


def spec_only_macro_f1(
    class_counts: Mapping[str, int],
    n_predicted_positive: int = R1_COUNT,
    tp_counts: Mapping[str, int] | None = None,
) -> tuple[float, dict[str, float]]:
    """규격전용 최적 예측기의 4결함 Macro-F1.

    1280×720 이미지에만 전 클래스를 주장한다. 결함이 사실상 전량 1280×720 이므로
    기본값은 `TP_c = n_c` 이며, 비-R1 결함이 있으면 `tp_counts` 로 넘긴다.

    처리 전 이 값이 자명하한보다 높다는 것이 **규격이 지름길이라는 정량 증거**이고,
    처리 후 자명하한으로 내려오는 것이 지름길이 죽었다는 증거다.
    """
    tp = dict(tp_counts) if tp_counts else dict(class_counts)
    per = {
        c: 2 * tp.get(c, 0) / (n_predicted_positive + n)
        for c, n in class_counts.items()
    }
    return _macro(per), per


def shortcut_contribution(
    class_counts: Mapping[str, int],
    n_images: int = RT_TOTAL,
    n_predicted_positive: int = R1_COUNT,
) -> float:
    """규격 지름길의 순 기여 = 규격전용 최적 − 전량양성 자명하한."""
    spec, _ = spec_only_macro_f1(class_counts, n_predicted_positive)
    base, _ = all_positive_macro_f1(class_counts, n_images)
    return spec - base


def verify_against_prereg(
    measured_all_positive: float,
    measured_spec_only: float,
    *,
    atol: float = 0.001,
) -> tuple[bool, str]:
    """동결 평가셋 재산출값이 사전등록 상수를 재현하는지 확인한다(§5-1 마지막 줄).

    재현하지 못하면 지름길 판정 이전에 **계측이 틀린 것**이므로, 프로브 결과를 해석하지
    않는다.
    """
    d_base = abs(measured_all_positive - PREREG.all_positive_macro_f1)
    d_spec = abs(measured_spec_only - PREREG.spec_only_macro_f1)
    if d_base <= atol and d_spec <= atol:
        return True, "사전등록 상수 재현 확인"
    return False, (
        f"사전등록 상수 미재현 — 전량양성 {measured_all_positive:.4f}"
        f"(등록 {PREREG.all_positive_macro_f1:.4f}, 차 {d_base:.4f}) / "
        f"규격전용 {measured_spec_only:.4f}"
        f"(등록 {PREREG.spec_only_macro_f1:.4f}, 차 {d_spec:.4f}). "
        "프로브 결과를 해석하기 전에 계측을 먼저 고친다."
    )


def recovery_denominator_ok(
    central: float, local_mean: float, seed_sd: float
) -> tuple[bool, float, str]:
    """회복률 분모 규칙(§5-4, 사전등록).

    지름길이 죽으면 다섯 칸 절대 성능이 모두 내려가고 분모 D=(중앙−로컬)이 좁아진다.
    **D < 3 × 시드 sd 이면 회복률을 산출하지 않고 세 절대값만 CI 와 함께 보고한다.**
    분산이 폭증한 비율을 헤드라인으로 싣지 않기 위해서다.
    """
    d = central - local_mean
    threshold = 3 * seed_sd
    if d >= threshold and d > 0:
        return True, d, f"분모 D={d:.4f} ≥ 3·sd={threshold:.4f} — 회복률 산출 가능"
    return False, d, (
        f"분모 D={d:.4f} < 3·sd={threshold:.4f} — 회복률을 산출하지 않고 "
        "세 절대값만 95% CI 와 함께 보고한다"
    )


def pass_line_for(population_bound: float) -> float:
    """**채점 모집단의 자명하한 위에** 통과선을 세운다.

    고정 상수를 다른 유병률의 집단에 그대로 대면 선이 틀린다. 실제로 등록 통과선
    0.2131 이 동결 평가셋 자명하한 0.21595 보다 낮았다 — 자명하한도 못 넘는 예측기가
    "통과"로 찍히는 상태다(80번 D2 재검 정정).
    """
    return round(float(population_bound) + TOLERANCE, 6)


# --------------------------------------------------------------------------------------
# 회복률 CI 판정 규칙 — 36번 미니스펙 · 총괄 게이트 2026-09-16 00:35 (조건부 통과)
#
# **공식 사전등록이 아니다.** 값을 보기 전 제안(41번 2026-08-24 · 36번 2026-09-16)을
# 3세트 채점 뒤 등록한 것이다 — 격자 등록 때와 같은 표기다. 등록 시점에 CI 값은 아직
# 산출되지 않았고, 그 사실이 이 상수들이 사후 규칙이 아닌 근거다. 의사결정로그 09-16 항목이
# 같은 내용을 담는다(총괄). 값을 바꾸는 것은 판정 규칙을 바꾸는 것이라 총괄 판정 사항이다.
# --------------------------------------------------------------------------------------

RECOVERY_CI_MAX_HALF_WIDTH = 0.25
"""규칙 ② — 판정 통계량(3세트 평균 회복률 R̄)의 95% CI 반폭 상한. 비율 단위(0.25 = 25 pp).

41번 §4 가 사전등록 §5-4 의 델타법 예(D=0.08 → 반폭 31 pp)에서 "못 쓰는 구간" 경계로 잡은
값을 **그대로** 등록했다. `evaluation.stats.MAX_RECOVERY_HALF_WIDTH` 와 같은 값이어야 하며
그 동일성은 시험이 본다.
"""

RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO = True
"""규칙 ③ — 분모 D_s 의 95% CI 가 **세 시드 모두** 0 을 배제해야 한다.

사전등록 §5-4 "D 를 CI 와 병기" 의 판정형. 분모가 0 을 품으면 비율은 정의 불가 구간을
지나므로 반폭만으로는 판정이 서지 않는다. 36번 신설 제안, 총괄 채택.
"""

RECOVERY_CI_MAX_UNDEFINED_FRACTION = 0.10
"""규칙 ④ — 미정의 재표집(D_s ≤ 0) 비율이 이 값을 넘으면 CI 를 헤드라인 근거로 쓰지 않는다.

버린 재표집이 많을수록 백분위 구간은 살아남은 표본만의 구간이 된다. 36번 신설 제안, 총괄 채택.
"""

RECOVERY_CI_REGISTRATION = {
    "rules": {"2_max_half_width": RECOVERY_CI_MAX_HALF_WIDTH,
              "3_denominator_ci_excludes_zero": RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO,
              "4_max_undefined_fraction": RECOVERY_CI_MAX_UNDEFINED_FRACTION},
    "statistic": "R̄ = 3세트 평균 회복률 (mAP@50, export 하한). R_s·D_s·칸별 mAP 는 병기",
    "proposed": "41번 (2026-08-24) · 36번 미니스펙 (2026-09-16)",
    "registered": "총괄 게이트 2026-09-16 00:35 — 3세트 채점 뒤, CI 산출 전",
    "status": "값을 보기 전 제안, 3세트 채점 뒤 등록. 공식 사전등록이 아니다",
    "caveat": "시드 3개는 분산 추정이 아니다 — CI 는 평가셋 묶음의 재표집 분산이지 학습 시드의 분산이 아니다",
}
"""산출물 머리에 통째로 싣는다 — 읽는 사람이 규칙의 출처와 지위를 값 옆에서 본다."""
