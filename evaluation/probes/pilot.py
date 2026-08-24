"""파일럿 산출물로 프로브를 돌리는 준비. `52_한사이클_파일럿_계획.md` §2·§5.

파일럿은 묶음 단위 층화 표본 약 3,000장이다. 본실험 RT 62,998 과 **결함 유병률이 다를 수
있으므로**, 사전등록 상수(전량양성 0.2081 / 통과선 0.2131)를 그대로 대면 통과선이 틀린다.
`trivial_bound` 의 `relative` 옵션을 쓰고, 어느 근거를 썼는지 보고에 남긴다.

P2·P3 은 학습이 필요해 GPU 를 쓴다. 이 모듈은 **GPU 없이 되는 부분**을 맡는다.

- 표본 구성 점검과 통과선 계산 (`prepare_pilot_gates`)
- P2 출처 판별 결과를 받아 게이트 판정 (`judge_p2`)
- P3 패치셔플·저해상 결과를 받아 게이트 판정 (`judge_p3`)
- 파일럿 채점 결과의 사전등록 상수 대조 (`verify_pilot_constants`)

**파일럿이라고 규칙을 풀지 않는다.** 조기 종료·best 채점·재시도는 여전히 금지다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evaluation.prereg import PREREG, TOLERANCE, all_positive_macro_f1
from evaluation.probes.metadata_probe import MetaSample, trivial_bound

P2_AUC_GATE = 0.60
"""P2 출처 판별 AUC 통과선. 초과하면 후퇴 사다리로 간다."""
P2_CI_UPPER_GATE = 0.65
"""묶음 클러스터 부트스트랩 CI 상한 통과선."""
P3_SHUFFLE_AUC_GATE = 0.65
"""P3 패치셔플 조건의 출처 판별 AUC 통과선."""


@dataclass(frozen=True)
class PilotGates:
    """파일럿 표본에서 실제로 적용할 통과선. 본실험 상수와 다를 수 있다."""

    n_samples: int
    prevalence: dict[str, float]
    trivial_bound: float
    p1_prime_gate: float
    gate_basis: str
    matches_prereg: bool
    note: str
    class_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "class_counts": self.class_counts,
            "prevalence": self.prevalence,
            "trivial_bound": self.trivial_bound,
            "p1_prime_gate": self.p1_prime_gate,
            "gate_basis": self.gate_basis,
            "matches_prereg": self.matches_prereg,
            "note": self.note,
        }


def prepare_pilot_gates(
    samples: Sequence[MetaSample],
    classes: Sequence[str],
    *,
    atol: float = 0.005,
) -> PilotGates:
    """축소 표본의 유병률에서 통과선을 계산한다.

    `F1 = 2p/(1+p)` 라 자명하한은 **클래스 유병률에만 의존**한다. 층화 추출이 비율을
    보존하면 본실험 상수와 같아지고, 그렇지 않으면 달라진다. 어느 쪽인지를 판정해
    `matches_prereg` 로 남긴다. 값이 같더라도 **어떻게 얻었는지**를 기록해야 나중에
    "고정 상수를 그냥 갖다 썼다"와 구분된다.
    """
    n = len(samples)
    counts = {c: sum(1 for s in samples if c in s.iso_codes) for c in classes}
    present = {c: v for c, v in counts.items() if v > 0}
    prevalence = {c: (v / n if n else 0.0) for c, v in counts.items()}
    bound = trivial_bound(samples, classes)
    gate = round(bound + TOLERANCE, 4)
    matches = abs(bound - PREREG.all_positive_macro_f1) <= atol

    if not present:
        note = "결함 표본이 0건이다. 통과선을 계산할 수 없다"
    elif matches:
        note = (
            f"표본 자명하한 {bound:.4f} 이 사전등록 상수 "
            f"{PREREG.all_positive_macro_f1:.4f} 와 일치한다. 층화가 비율을 보존했다"
        )
    else:
        note = (
            f"표본 자명하한 {bound:.4f} 이 사전등록 상수 "
            f"{PREREG.all_positive_macro_f1:.4f} 와 다르다. "
            f"축소 표본의 유병률이 본실험과 달라 통과선을 {gate:.4f} 로 다시 잡는다"
        )
    return PilotGates(
        n_samples=n,
        prevalence=prevalence,
        trivial_bound=bound,
        p1_prime_gate=gate,
        gate_basis="표본 상대(relative)",
        matches_prereg=matches,
        note=note,
        class_counts=counts,
    )


@dataclass(frozen=True)
class ProbeVerdict:
    probe: str
    passed: bool | None
    detail: str
    values: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "probe": self.probe, "passed": self.passed,
            "detail": self.detail, **self.values,
        }


def judge_p2(
    auc: float,
    ci_upper: float | None = None,
    *,
    n_clusters: int = 0,
    min_clusters: int = 20,
) -> ProbeVerdict:
    """P2 출처 판별 게이트. "이 1280×720 은 원본 결함 크롭인가, 파노라마 타일인가".

    통과선은 AUC ≤ 0.60 이고, 묶음 클러스터 부트스트랩 CI 상한이 0.65 미만이어야 한다.
    묶음이 부족하면 **통과가 아니라 판정 불가**다. 미도달을 통과처럼 쓰지 않는다.
    """
    if n_clusters and n_clusters < min_clusters:
        return ProbeVerdict(
            "P2", None,
            f"판정 불가. 묶음 {n_clusters}개는 검정력 부족(최소 {min_clusters})",
            {"auc": auc, "ci_upper": ci_upper, "n_clusters": n_clusters},
        )
    auc_ok = auc <= P2_AUC_GATE
    ci_ok = ci_upper is None or ci_upper < P2_CI_UPPER_GATE
    passed = auc_ok and ci_ok
    if passed:
        detail = f"AUC {auc:.4f} ≤ {P2_AUC_GATE}. 출처가 판별되지 않는다"
    elif not auc_ok:
        detail = (
            f"AUC {auc:.4f} > {P2_AUC_GATE}. 출처가 판별된다. "
            "후퇴 사다리로 간다"
        )
    else:
        detail = (
            f"AUC 점추정 {auc:.4f} 는 통과하나 CI 상한 {ci_upper:.4f} "
            f"≥ {P2_CI_UPPER_GATE}. 통과로 보지 않는다"
        )
    return ProbeVerdict(
        "P2", passed, detail,
        {"auc": auc, "ci_upper": ci_upper, "n_clusters": n_clusters},
    )


def judge_p3(
    shuffle_auc: float,
    lowres_macro_f1: float,
    gates: PilotGates,
) -> ProbeVerdict:
    """P3 패치셔플·저해상 게이트.

    16×16 패치를 섞으면 텍스처만 남고, 32×32 로 줄이면 전역 통계만 남는다. 두 조건에서도
    출처나 결함이 판별되면 규격 외의 채널이 살아 있다는 뜻이다.

    저해상 조건의 통과선은 **파일럿 표본의 자명하한**을 쓴다. 본실험 상수(0.2131)를 그대로
    대면 유병률이 다른 표본에서 통과선이 틀린다.
    """
    shuffle_ok = shuffle_auc <= P3_SHUFFLE_AUC_GATE
    lowres_ok = lowres_macro_f1 <= gates.p1_prime_gate
    passed = shuffle_ok and lowres_ok
    shuffle_sign = "≤" if shuffle_ok else ">"
    lowres_sign = "≤" if lowres_ok else ">"
    parts = [
        (f"셔플 출처 AUC {shuffle_auc:.4f} {shuffle_sign} {P3_SHUFFLE_AUC_GATE}"),
        (f"저해상 4결함 Macro-F1 {lowres_macro_f1:.4f} {lowres_sign} "
         f"{gates.p1_prime_gate:.4f} ({gates.gate_basis})"),
    ]
    if not passed:
        parts.append("규격 외 채널이 살아 있다")
    return ProbeVerdict(
        "P3", passed, ". ".join(parts),
        {
            "shuffle_auc": shuffle_auc,
            "lowres_macro_f1": lowres_macro_f1,
            "gate": gates.p1_prime_gate,
            "gate_basis": gates.gate_basis,
        },
    )


def verify_pilot_constants(
    measured_all_positive: float, gates: PilotGates, *, atol: float = 0.001
) -> ProbeVerdict:
    """파일럿 채점 결과가 **표본 자명하한**을 재현하는지 확인한다.

    본실험이라면 사전등록 상수를 재현해야 하지만, 축소 표본에서는 그 표본의 자명하한이
    기준이다. 재현하지 못하면 지름길 판정 이전에 계측이 틀린 것이다.
    """
    d = abs(measured_all_positive - gates.trivial_bound)
    ok = d <= atol
    return ProbeVerdict(
        "prereg-check", ok,
        (
            f"전량양성 실측 {measured_all_positive:.4f} 이 표본 자명하한 "
            f"{gates.trivial_bound:.4f} 를 재현한다"
            if ok else
            f"전량양성 실측 {measured_all_positive:.4f} 이 표본 자명하한 "
            f"{gates.trivial_bound:.4f} 와 {d:.4f} 어긋난다. "
            "프로브 결과를 해석하기 전에 계측을 먼저 고친다"
        ),
        {"measured": measured_all_positive, "expected": gates.trivial_bound},
    )


def pilot_probe_report(verdicts: Sequence[ProbeVerdict], gates: PilotGates) -> dict:
    """파일럿 프로브 결과를 한 덩어리로 묶는다.

    `판정 불가`(None)를 통과로 세지 않는다. 통과·불통과·판정 불가를 각각 센다.
    """
    passed = sum(1 for v in verdicts if v.passed is True)
    failed = sum(1 for v in verdicts if v.passed is False)
    undecided = sum(1 for v in verdicts if v.passed is None)
    return {
        "gates": gates.as_dict(),
        "verdicts": [v.as_dict() for v in verdicts],
        "n_passed": passed,
        "n_failed": failed,
        "n_undecided": undecided,
        "all_clear": failed == 0 and undecided == 0 and passed > 0,
        "note": (
            "파일럿 수치는 보고하되 결론으로 쓰지 않는다. 표본 3,000 에 시드 1세트다"
        ),
    }


def expected_constants_note(samples: Sequence[MetaSample], classes: Sequence[str]) -> str:
    """보고서에 넣을 한 줄. 본실험 상수와 파일럿 통과선을 나란히 적는다."""
    counts = {c: sum(1 for s in samples if c in s.iso_codes) for c in classes}
    present = {c: v for c, v in counts.items() if v > 0}
    pilot_bound, _ = (
        all_positive_macro_f1(present, len(samples)) if present else (0.0, {})
    )
    return (
        f"본실험 자명하한 {PREREG.all_positive_macro_f1:.4f} / 통과선 "
        f"{PREREG.p1_prime_gate:.4f}. 파일럿 표본 자명하한 {pilot_bound:.4f} / 통과선 "
        f"{round(pilot_bound + TOLERANCE, 4):.4f}"
    )
