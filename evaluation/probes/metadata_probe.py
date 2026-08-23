"""P0 메타데이터 전용 베이스라인 · P1′ 규격전용 분류기. §5-2.

**픽셀을 보지 않고** `{w, h, MP, 종횡비, 파일 바이트수, DQT 해시, 채널수}` 만으로 결함을
맞힐 수 있는지 잰다. 맞힐 수 있으면 그만큼이 규격 지름길이다.

두 프로브의 지표는 **헤드라인과 동일**하다 — 이미지 수준 다중라벨 4결함 Macro-F1.
이진 정확도로 재면 계약 #4가 채점하지 않는 과제를 재게 되고, 그 값으로 게이트를 세우면
미처리 원본이 통과한다.

- **P0**(처리 전): 고쳐야 할 것의 크기. 기대 ≈ 0.30.
- **P1′**(처리 후): 통과선 ≤ 0.2131 (= 전량양성 자명하한 0.2081 + 0.005).
  미처리 원본은 0.3025 라 통과하지 못한다.

GPU 를 쓰지 않는다. 시드를 고정해 결정론적으로 돌린다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from evaluation.metrics.detection import score_detection
from evaluation.prereg import PREREG, TOLERANCE, all_positive_macro_f1

SEED = 20260825
FEATURE_NAMES = (
    "width_px", "height_px", "megapixels", "aspect_ratio",
    "file_bytes", "quant_table_id", "n_channels",
)


@dataclass(frozen=True)
class MetaSample:
    """프로브 입력 한 건. **픽셀은 들어오지 않는다.**"""

    image_id: str
    width_px: int
    height_px: int
    file_bytes: int
    n_channels: int
    quant_table_id: int
    iso_codes: tuple[str, ...] = ()
    """정답 결함 코드들. 정상 이미지는 빈 튜플 — 클래스가 아니라 모집단에 남는다."""

    def features(self) -> list[float]:
        mp = self.width_px * self.height_px / 1e6
        aspect = self.width_px / self.height_px if self.height_px else 0.0
        return [
            float(self.width_px), float(self.height_px), mp, aspect,
            float(self.file_bytes), float(self.quant_table_id), float(self.n_channels),
        ]


@dataclass(frozen=True)
class MetaProbeResult:
    probe: str
    macro_f1: float
    per_class: dict[str, float]
    gate: float | None
    passed: bool | None
    n_train: int
    n_test: int
    detail: str
    feature_importance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "probe": self.probe, "macro_f1": self.macro_f1, "gate": self.gate,
            "passed": self.passed, "n_train": self.n_train, "n_test": self.n_test,
            "per_class": self.per_class, "detail": self.detail,
            "feature_importance": self.feature_importance,
        }


def _fit_predict(
    train: Sequence[MetaSample],
    test: Sequence[MetaSample],
    classes: Sequence[str],
    model: str,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """클래스마다 이진 분류기를 세우고 이미지별 예측 집합을 만든다.

    다중라벨이므로 one-vs-rest 다. 정상 이미지는 전 클래스 음성으로 학습된다.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    x_train = np.asarray([s.features() for s in train], dtype=float)
    x_test = np.asarray([s.features() for s in test], dtype=float)
    pred: dict[str, list[str]] = {s.image_id: [] for s in test}
    importance: dict[str, float] = dict.fromkeys(FEATURE_NAMES, 0.0)

    for code in classes:
        y = np.asarray([1 if code in s.iso_codes else 0 for s in train], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            continue                      # 한쪽만 있으면 학습 불가 — 그 클래스는 건너뛴다
        if model == "tree":
            clf = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        elif model == "logistic":
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
        else:
            clf = GradientBoostingClassifier(random_state=SEED)
        clf.fit(x_train, y)
        yhat = clf.predict(x_test)
        for s, hit in zip(test, yhat, strict=True):
            if hit:
                pred[s.image_id].append(code)
        if hasattr(clf, "feature_importances_"):
            for name, v in zip(FEATURE_NAMES, clf.feature_importances_, strict=True):
                importance[name] += float(v) / len(classes)
    return pred, importance


def run_metadata_probe(
    train: Sequence[MetaSample],
    test: Sequence[MetaSample],
    classes: Sequence[str],
    *,
    probe: str = "P0",
    model: str = "tree",
    gate: float | None = None,
) -> MetaProbeResult:
    """P0 / P1′ 공통 실행부. 지표는 헤드라인과 같은 채점기를 그대로 쓴다.

    Args:
        gate: 통과선. `None` 이면 판정하지 않고 값만 낸다(P0 는 "크기를 재는" 프로브라
            통과·불통과가 없다).
    """
    gold = {s.image_id: list(s.iso_codes) for s in test}
    pred, importance = _fit_predict(train, test, classes, model)
    report = score_detection(pred, gold, classes)
    per = {c.iso_code: (c.f1 or 0.0) for c in report.per_class if c.support > 0}
    passed = None if gate is None else report.macro_f1 <= gate
    if gate is None:
        detail = f"메타데이터만으로 4결함 Macro-F1 {report.macro_f1:.4f} — 고쳐야 할 크기"
    elif passed:
        detail = (
            f"{report.macro_f1:.4f} ≤ {gate:.4f} — 메타데이터로는 자명하한을 넘지 못한다"
        )
    else:
        detail = (
            f"{report.macro_f1:.4f} > {gate:.4f} — 규격 지름길이 살아 있다. "
            "타일링 처리를 재검토한다"
        )
    return MetaProbeResult(
        probe=probe,
        macro_f1=report.macro_f1,
        per_class=per,
        gate=gate,
        passed=passed,
        n_train=len(train),
        n_test=len(test),
        detail=detail,
        feature_importance={k: round(v, 4) for k, v in importance.items() if v},
    )


def p0_baseline(
    train: Sequence[MetaSample], test: Sequence[MetaSample], classes: Sequence[str]
) -> MetaProbeResult:
    """P0 — 처리 **전** 메타데이터 전용 베이스라인. 깊이 4 결정트리."""
    return run_metadata_probe(train, test, classes, probe="P0", model="tree", gate=None)


def trivial_bound(samples: Sequence[MetaSample], classes: Sequence[str]) -> float:
    """이 표본 집합에서의 전량양성 자명하한.

    `F1_c = 2·(n_c/N) / (1 + n_c/N)` 이라 **클래스 유병률에만 의존**한다. 층화 추출이면
    비율이 보존돼 사전등록 상수가 그대로 재현되지만, 표본 구성이 달라지면 값이 달라진다.
    고정 상수를 다른 비율의 집단에 그대로 대면 통과선이 틀린다.
    """
    counts = {c: sum(1 for s in samples if c in s.iso_codes) for c in classes}
    counts = {c: n for c, n in counts.items() if n > 0}
    if not counts:
        return 0.0
    macro, _ = all_positive_macro_f1(counts, len(samples))
    return macro


def p1_prime(
    train: Sequence[MetaSample],
    test: Sequence[MetaSample],
    classes: Sequence[str],
    *,
    model: str = "gbm",
    gate: float | None = None,
    relative: bool = False,
) -> MetaProbeResult:
    """P1′ — 처리 **후** 규격전용 분류기. 기본 통과선 ≤ 0.2131 (사전등록 상수).

    Args:
        relative: True 면 통과선을 `test` 표본의 자명하한 + 여유로 계산한다. 동결 평가셋이
            층화 추출이라 비율이 보존되면 두 값이 같지만, 축소 파일럿처럼 비율이 다른
            표본에서는 고정 상수가 맞지 않는다. **어느 쪽을 썼는지 보고에 남긴다.**
    """
    if gate is None:
        gate = (
            round(trivial_bound(test, classes) + TOLERANCE, 4)
            if relative
            else PREREG.p1_prime_gate
        )
    result = run_metadata_probe(
        train, test, classes, probe="P1'", model=model, gate=gate
    )
    basis = "표본 상대" if relative else "사전등록 상수"
    return MetaProbeResult(
        **{**result.__dict__, "detail": f"{result.detail} (통과선 근거: {basis})"}
    )


def paired_report(before: MetaProbeResult, after: MetaProbeResult) -> dict:
    """전/후 쌍 보고(§5 원칙 1). **전 단계 수치가 없으면 "죽었다"가 아니라 "낮다"밖에
    못 쓴다.**"""
    return {
        "before": before.as_dict(),
        "after": after.as_dict(),
        "delta_macro_f1": round(after.macro_f1 - before.macro_f1, 4),
        "headline": (
            f"RT 규격전용 최적 예측기의 4결함 Macro-F1 "
            f"{before.macro_f1:.4f} → {after.macro_f1:.4f}"
        ),
    }
