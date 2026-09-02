"""출처 고정 판별력 지표 — **지름길이 정의상 통과할 수 없는 유일한 축.** 68번 §5-(나).

## 왜 이 지표가 필요한가

66번이 실측한 것: "N-crop 이면 기공, 아니면 없음"이라는 **이미지를 한 번도 열지 않는
규칙**이 Macro-F1·결함 놓침·Class-Jaccard·결함 recall 네 지표 중 셋에서 일곱 모델을
전부 이긴다. 네 지표가 전부 출처 축에 지배되기 때문이다.

이 모듈의 지표는 **출처가 상수인 구간 안에서만** 계산한다. 그 구간에서 출처만 읽는
예측기는 상수 예측기로 퇴화하므로 결함 이미지와 정상 이미지에서 같은 발화율을 갖고,
차는 **정확히 0** 이 된다. 통과할 방법이 없다 — 통과하려면 이미지를 봐야 한다.

    판별력 Δ = (결함 이미지 발화율) − (정상 이미지 발화율)   [출처 고정, 기본 N-crop]

## 정의 (헤드라인 보조지표 승격 전제로 고정한다)

| 항목 | 정의 | 이유 |
|---|---|---|
| 계산 모집단 | 지정 출처(기본 `N-crop`)의 평가셋 이미지 **전량**. 결함·정상 둘 다 | 출처를 상수로 고정하는 것이 이 지표의 존재 이유다 |
| 결함/정상 판정 | 매니페스트 `has_defect` (GT). 예측으로 정하지 않는다 | 예측으로 나누면 분모가 모델에 따라 달라져 칸 간 비교가 깨진다 |
| 발화 | `parse_ok` 이고 예측 결함 집합이 비어 있지 않다 | **P9 와 같은 함수**(`fires`)를 쓴다. 아래 `is_false_positive` 가 이 함수를 호출한다 |
| 파싱 실패 | 발화 아님(빈 예측) | 스펙 §4-1. 실패율은 별도 지표가 이미 센다 |
| 결측 | 레코드 없는 이미지는 **결측으로 세고 분모에서 뺀다**. 발화 0으로 채우지 않는다 | 채점 누락을 '깨끗함'으로 읽으면 지표가 조용히 좋아진다 |
| CI | 묶음(`group_id`) 클러스터 부트스트랩, 차 자체를 재표집 | 이미지 단위로 내면 CI 가 좁아진다(stats 모듈 머리말) |
| 부호 | 양수 = 결함에서 더 발화. 음수 = **정상에서 더 발화**(역전) | 0 이 지름길선이고 음수는 그 아래다 |

**0 은 실패선이 아니라 기준선이다.** 0 이하는 "이 구간에서 이미지 내용을 쓰지 않았다"는
뜻이고, 그것이 학습이 일어났는지에 대한 직접 진술이다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from evaluation.schema import PredictionRecord
from evaluation.stats import Interval, cluster_bootstrap

CROP = "N-crop"
"""기본 고정 출처. 결함 이미지가 전량 여기 있으므로 다른 출처에서는 결함 분모가 0이다."""


def fires(record: PredictionRecord) -> bool:
    """**다섯 칸 공통 단일 발화 정의.** P9 의 오탐 정의와 같은 함수다.

    정상 이미지에서 True 면 오탐이고, 결함 이미지에서 True 면 검출 시도다. 두 경우에
    다른 정의를 쓰면 판별력 차가 정의 차이를 재게 된다.
    """
    return record.parse_ok and bool(record.iso_codes)


@dataclass(frozen=True)
class DiscriminationSample:
    """판별력 계산 대상 이미지 한 장."""

    image_id: str
    group_id: str
    has_defect: bool
    fired: bool


@dataclass(frozen=True)
class DiscriminationReport:
    cell: str
    client: str | None
    seed: int
    provenance: str
    n_defect: int
    n_normal: int
    n_groups: int
    fire_rate_defect: float
    fire_rate_normal: float
    delta: Interval
    n_missing_prediction: int
    verdict: str

    def as_dict(self) -> dict:
        return {
            "cell": self.cell, "client": self.client, "seed": self.seed,
            "provenance": self.provenance,
            "n_defect": self.n_defect, "n_normal": self.n_normal,
            "n_groups": self.n_groups,
            "fire_rate_defect": self.fire_rate_defect,
            "fire_rate_normal": self.fire_rate_normal,
            "delta": self.delta.as_dict(),
            "n_missing_prediction": self.n_missing_prediction,
            "verdict": self.verdict,
        }

    def headline(self) -> str:
        return (
            f"출처 고정 판별력 Δ {self.delta.point:+.4f} "
            f"[95% CI {self.delta.lo:+.4f}, {self.delta.hi:+.4f}] "
            f"({self.provenance} 결함 {self.n_defect} · 정상 {self.n_normal})"
        )


def _delta_statistic(samples: Sequence[DiscriminationSample]):
    """차 자체를 재표집한다. 두 CI 를 따로 내서 겹침을 보면 상관이 무시된다."""
    by_group: dict[str, list[DiscriminationSample]] = {}
    for s in samples:
        by_group.setdefault(s.group_id, []).append(s)

    def statistic(groups: Sequence[str]) -> float:
        d_hit = d_tot = n_hit = n_tot = 0
        for g in groups:
            for s in by_group.get(g, ()):
                if s.has_defect:
                    d_hit += s.fired
                    d_tot += 1
                else:
                    n_hit += s.fired
                    n_tot += 1
        if d_tot == 0 or n_tot == 0:
            # 한쪽 층이 통째로 안 뽑힌 재표집이다. 0 으로 대치하면 CI 가 0 쪽으로
            # 끌려가 "판별 불성립"이 공짜로 나온다 — 정의되지 않음으로 남기고 뺀다.
            return float("nan")
        return d_hit / d_tot - n_hit / n_tot

    return statistic, sorted(by_group)


UNDEFINED_WARN_RATIO = 0.05
"""재표집의 이 비율 이상이 정의되지 않으면 CI 폭을 액면 그대로 읽지 않는다."""


def _verdict(delta: Interval, n_defect: int, n_normal: int) -> str:
    if n_defect == 0 or n_normal == 0:
        return "산출 불가 — 고정 출처 안에 결함 또는 정상 이미지가 없다"
    if delta.point == 0.0 and delta.lo == 0.0 and delta.hi == 0.0:
        base = (
            "지름길선 — Δ 가 정확히 0. 이 구간에서 출력이 이미지 내용에 반응하지 않는다"
        )
    elif delta.hi < 0:
        base = "역전 — 정상에서 더 발화한다. 이 구간에서 이미지 내용을 쓰지 않았다"
    elif delta.lo <= 0:
        base = "판별 불성립 — CI 가 0 을 포함한다. 지름길선과 구분되지 않는다"
    else:
        base = "판별 있음 — CI 하한이 0 위다. 출처로 설명되지 않는 신호가 있다"
    if delta.n_resamples and delta.n_undefined / delta.n_resamples >= UNDEFINED_WARN_RATIO:
        base += (
            f" (주의: 재표집 {delta.n_undefined}/{delta.n_resamples} 가 한쪽 층 결측으로 "
            "제외됐다 — 묶음이 결함·정상으로 순수하게 갈려 있다)"
        )
    return base


def samples_from_records(
    records: Iterable[PredictionRecord],
    contexts: Mapping[str, tuple[str, bool, str]],
    *,
    provenance: str = CROP,
) -> tuple[list[DiscriminationSample], int]:
    """레코드 + 이미지 맥락 → 계산 표본.

    Args:
        contexts: image_id → (group_id, has_defect, provenance). 매니페스트·`tiles.csv`
            에서 온다. **GT 기준**이며 예측을 쓰지 않는다.

    Returns:
        `(표본들, 결측 수)`. 맥락에 있는데 레코드가 없는 이미지가 결측이다.
    """
    by_id = {r.image_id: r for r in records}
    out: list[DiscriminationSample] = []
    missing = 0
    for image_id, (group_id, has_defect, prov) in sorted(contexts.items()):
        if prov != provenance:
            continue
        rec = by_id.get(image_id)
        if rec is None:
            missing += 1
            continue
        out.append(DiscriminationSample(image_id, group_id, has_defect, fires(rec)))
    return out, missing


def score_discrimination(
    records: Iterable[PredictionRecord],
    contexts: Mapping[str, tuple[str, bool, str]],
    *,
    provenance: str = CROP,
) -> DiscriminationReport:
    """모델 하나(칸·클라이언트·시드)의 출처 고정 판별력.

    P9 러너와 같은 이유로 **모델이 섞이면 거부한다** — 같은 image_id 가 두 모델에서
    오면 마지막 것만 남아 조용히 틀린 값이 나온다.
    """
    recs = list(records)
    keys = {(r.cell, r.client, r.seed) for r in recs}
    if len(keys) > 1:
        raise ValueError(
            f"칸/클라이언트/시드가 섞여 있다: {sorted(map(str, keys))} — "
            "판별력은 모델 하나씩 계산한다"
        )
    seen: set[str] = set()
    for r in recs:
        if r.image_id in seen:
            raise ValueError(f"이미지 {r.image_id} 레코드가 중복이다 — 모델이 섞였다")
        seen.add(r.image_id)

    samples, missing = samples_from_records(recs, contexts, provenance=provenance)
    defect = [s for s in samples if s.has_defect]
    normal = [s for s in samples if not s.has_defect]
    rate_d = sum(s.fired for s in defect) / len(defect) if defect else 0.0
    rate_n = sum(s.fired for s in normal) / len(normal) if normal else 0.0

    stat, groups = _delta_statistic(samples)
    delta = (
        cluster_bootstrap(groups, stat, drop_undefined=True)
        if defect and normal else Interval(0.0, 0.0, 0.0, 0, len(groups))
    )

    cell, client, seed = next(iter(keys)) if keys else ("?", None, -1)
    return DiscriminationReport(
        cell=cell, client=client, seed=seed, provenance=provenance,
        n_defect=len(defect), n_normal=len(normal), n_groups=len(groups),
        fire_rate_defect=rate_d, fire_rate_normal=rate_n,
        delta=delta, n_missing_prediction=missing,
        verdict=_verdict(delta, len(defect), len(normal)),
    )


def score_discrimination_all_cells(
    records: Iterable[PredictionRecord],
    contexts: Mapping[str, tuple[str, bool, str]],
    *,
    provenance: str = CROP,
) -> tuple[DiscriminationReport, ...]:
    """전 칸·전 시드를 (칸, 클라이언트, 시드)로 갈라 각각 계산한다.

    **칸 이름으로 분기하는 코드는 없다** — 다섯 칸이 같은 함수를 탄다(불변조건 3-7).
    """
    grouped: dict[tuple[str, str, int], list[PredictionRecord]] = {}
    for r in records:
        grouped.setdefault((r.cell, r.client or "", r.seed), []).append(r)
    return tuple(
        score_discrimination(grouped[k], contexts, provenance=provenance)
        for k in sorted(grouped)
    )
