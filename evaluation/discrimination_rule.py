"""출처 진단을 채택 구간 규약으로 — 07번 §33-6 의 적용 범위.

`evaluation.discrimination` 은 계약 파일이라 바이트가 고정돼 있다 — D1 은 새 기능을 새 파일에 둔다(07번 §14-2 의 1,
`tests/test_d1_baseline_frozen.py`). 그래서 방식 인자를 그 파일에 더하지 않고 여기 둔다. 표본 · 통계량 · 검증 · 판정 문구는
그 모듈의 것을 그대로 부르고, **구간만** `cluster_bootstrap(…, method=…)` 로 낸다.

- `method=None` 이면 그 모듈의 함수를 그대로 부른다 — 값도 꼴도 같다.
- 방식을 주면(기본 — 통합형 산출물) 구간에 규약 id 와 끝점 · 반폭의 상태 칸이 실리고, ±무한 추첨은 거부한다
  (미정의를 빼는 통계량 — §33-6). 정의되지 않는 추첨은 세고 뺀다.
- 정의된 추첨이 하나도 없으면 구간은 `absent` 이고 판정 문구는 **산출 불가**다. 그 모듈의 판정 문구는 끝점이 `NaN` 이면
  비교가 모두 거짓이라 "판별 있음" 으로 떨어진다 — 방식 경로에서는 그 자리에 닿기 전에 가른다.
- `ci.min_defined_draws` 의 하한은 여기서 걸지 않는다 — 수를 등록이 정하고(§33-5) 판정은 등록 판정 모듈의 몫이다.
  산출의 `n_undefined_resamples` 가 그 판정의 입력이다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from evaluation import discrimination as D
from evaluation.percentile_rule import PERCENTILE_RULE, convention_id
from evaluation.schema import PredictionRecord
from evaluation.stats import Interval, cluster_bootstrap

NO_INTERVAL = "산출 불가 — 정의된 재표집이 없어 구간이 없다"


def _verdict(delta: Interval, n_defect: int, n_normal: int) -> str:
    if n_defect and n_normal and delta.rule is not None and delta.rule.lo.state != "finite":
        return NO_INTERVAL
    return D._verdict(delta, n_defect, n_normal)


def score_discrimination(
    records: Iterable[PredictionRecord],
    contexts: Mapping[str, tuple[str, bool, str]],
    *,
    provenance: str = D.CROP,
    method: str | None = PERCENTILE_RULE,
) -> D.DiscriminationReport:
    """모델 하나의 출처 고정 판별력 — `evaluation.discrimination.score_discrimination` 에 구간 방식을 더한 것."""
    recs = list(records)
    base = D.score_discrimination(recs, contexts, provenance=provenance)  # 검증 · 표본 · 발화율 — 그 모듈 그대로
    if method is None:
        return base
    samples, _missing = D.samples_from_records(recs, contexts, provenance=provenance)
    stat, groups = D._delta_statistic(samples)
    units = groups if base.n_defect and base.n_normal else []
    delta = cluster_bootstrap(units, stat, drop_undefined=True, method=method)
    return replace(base, delta=delta, verdict=_verdict(delta, base.n_defect, base.n_normal))


def score_threshold_free(
    scores_by_cell: Mapping[str, Mapping[str, float]],
    defect: Iterable[str],
    normal: Iterable[str],
    by_group: Mapping[str, Sequence[str]],
    *,
    central: str = "sep_central",
    fed: str = "sep_fed",
    local_prefix: str = "sep_local",
    method: str | None = PERCENTILE_RULE,
) -> dict:
    """칸별 `Δ_AUC` 와 칸 대비 — `evaluation.discrimination.score_threshold_free` 에 구간 방식을 더한 것.

    설명 칸(정의 · 기준선 · 한계 · 상태)은 그 모듈이 빈 입력으로 낸 것을 그대로 쓴다 — 글을 옮겨 적지 않는다.
    계산은 그 모듈과 같은 통계량(`gini`)과 같은 짝지음(같은 씨앗의 재표집)이다. 회복률은 두 점추정이 유한하고 분모가 양수일 때만 낸다.
    """
    kw = {"central": central, "fed": fed, "local_prefix": local_prefix}
    if method is None:
        return D.score_threshold_free(scores_by_cell, defect, normal, by_group, **kw)
    out = D.score_threshold_free({}, [], [], {}, **kw)  # 설명 칸만 — 계산 없음
    d_set, n_set = set(defect), set(normal)
    units = sorted(by_group)
    cells = sorted(scores_by_cell)

    def sub(cell: str, gs: Sequence[str]) -> float:
        ids = [i for g in gs for i in by_group[g]]
        return D.gini(scores_by_cell[cell], [i for i in ids if i in d_set], [i for i in ids if i in n_set])

    def ci(fn) -> dict:
        return cluster_bootstrap(units, fn, drop_undefined=True, method=method).as_dict()

    by_cell = {c: ci(lambda gs, k=c: sub(k, gs)) for c in cells}
    locals_ = [c for c in cells if c.startswith(local_prefix)]
    contrasts: dict[str, dict] = {}
    recovery = None
    if central in scores_by_cell and fed in scores_by_cell and locals_:
        def _mean_local(gs: Sequence[str]) -> float:
            return sum(sub(c, gs) for c in locals_) / len(locals_)

        pairs = {
            "fed_minus_central": lambda gs: sub(fed, gs) - sub(central, gs),
            "fed_minus_local_mean": lambda gs: sub(fed, gs) - _mean_local(gs),
            "central_minus_local_mean": lambda gs: sub(central, gs) - _mean_local(gs),
        }
        for k, fn in pairs.items():
            contrasts[k] = ci(fn)
        den = contrasts["central_minus_local_mean"]["point"]
        num = contrasts["fed_minus_local_mean"]["point"]
        recovery = None if den is None or num is None or den <= 0 else 100.0 * num / den
    out.update({"n_groups": len(units), "by_cell": by_cell, "contrasts": contrasts, "recovery_pct": recovery,
                "convention_id": convention_id(method)})
    return out
