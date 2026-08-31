"""P9 교차 출처 오탐 시험 — 공통 스키마 출력에서 계산하는 브리지. 파일럿 필수 산출물.

R3 잔존 수용의 조건이 P9 다. 출처 신호(밝기의 공간 배치)가 데이터에 남아 있으므로,
**본 모델이 그 신호를 실제로 오탐에 쓰는지**를 다섯 칸 각각에 대해 잰다.

- 대상: 평가셋 **정상** 이미지. FP 율을 출처별로 분해한다
  (N-crop 654 / N-tile 4,697 / N-band 24 — 동결 스냅샷 실측).
- N-band 는 표본 부족으로 집계에서 제외하되 건수는 보고한다.
- 판정: TOST 동등성 검정, δ = 0.10. 동등하면 잔존은 한계 문단으로,
  동등하지 않으면(오탐이 출처로 쏠리면) 크롭 한정본 승격을 재검토한다.
- **다섯 칸 공통 채점기 출력(공통 예측 스키마)에서 계산한다.** 칸마다 따로 구현하면
  칸마다 오탐의 정의가 갈릴 수 있고, 그 순간 비교가 무너진다(불변조건 3-7 과 같은 이유).

오탐의 정의: `parse_ok` 이고 예측 결함 집합이 비어 있지 않은 정상 이미지.
파싱 실패 레코드는 예측 집합을 공집합으로 간주하므로(스펙 §4-1) 오탐이 아니다 —
실패율은 별도 지표가 이미 센다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from evaluation.probes.cross_source import (
    BAND,
    CrossSourceReport,
    NormalImage,
    fp_breakdown_by_source,
    p9_cross_source,
)
from evaluation.schema import PredictionRecord

P9_MARGIN = 0.10
"""사전등록 마진 δ. 수정 금지."""


@dataclass(frozen=True)
class EvalNormalContext:
    """평가셋 정상 이미지 한 장의 조회 맥락. 동결 스냅샷에서 온다."""

    image_id: str
    group_id: str
    provenance: str


def contexts_from_snapshot(
    manifest_rows: Iterable[Mapping[str, str]],
    provenance: Mapping[str, str],
) -> tuple[tuple[EvalNormalContext, ...], tuple[str, ...]]:
    """동결 스냅샷의 매니페스트 + `tiles.csv` 출처에서 평가셋 정상 목록을 만든다.

    정상 여부는 **GT(매니페스트 `has_defect`)** 로 정한다. 예측으로 정하면 오탐 많은
    칸일수록 분모가 줄어드는 순환이 생긴다. 출처 미상은 조용히 버리지 않고 돌려준다.
    """
    out: list[EvalNormalContext] = []
    missing: list[str] = []
    for r in manifest_rows:
        if r.get("split") != "eval" or str(r.get("has_defect")) != "False":
            continue
        iid = str(r["image_id"])
        src = provenance.get(iid)
        if src is None:
            missing.append(iid)
            continue
        out.append(EvalNormalContext(iid, str(r["group_id"]), src))
    return tuple(out), tuple(missing)


def is_false_positive(record: PredictionRecord) -> bool:
    """정상 이미지에 대한 오탐 여부. **다섯 칸 공통 단일 정의.**"""
    return record.parse_ok and bool(record.iso_codes)


@dataclass(frozen=True)
class P9CellResult:
    """칸 하나(×시드 하나)의 P9 결과."""

    cell: str
    seed: int
    report: CrossSourceReport
    breakdown: dict[str, dict[str, float | int]]
    n_band_excluded: int
    n_missing_prediction: int
    verdict: str

    def as_dict(self) -> dict:
        return {
            "cell": self.cell,
            "seed": self.seed,
            **self.report.as_dict(),
            "fp_breakdown": self.breakdown,
            "n_band_excluded": self.n_band_excluded,
            "n_missing_prediction": self.n_missing_prediction,
            "verdict": self.verdict,
        }


def p9_for_cell(
    records: Iterable[PredictionRecord],
    contexts: Iterable[EvalNormalContext],
    *,
    margin: float = P9_MARGIN,
    min_clusters: int = 20,
) -> P9CellResult:
    """칸 하나의 예측 레코드에서 P9 를 계산한다.

    레코드가 없는 정상 이미지는 **오탐 아님으로 채우지 않고 결측으로 센다** —
    채점이 누락된 이미지를 '깨끗하다'로 읽으면 오탐률이 조용히 내려간다.
    """
    ctx_list = list(contexts)
    by_id: dict[str, PredictionRecord] = {}
    cells: set[str] = set()
    seeds: set[int] = set()
    for rec in records:
        by_id[rec.image_id] = rec
        cells.add(rec.cell)
        seeds.add(rec.seed)
    if len(cells) > 1 or len(seeds) > 1:
        raise ValueError(
            f"칸/시드가 섞여 있다: cells={sorted(cells)} seeds={sorted(seeds)} — "
            "P9 는 (칸, 시드) 하나씩 계산한다"
        )

    images: list[NormalImage] = []
    n_band = 0
    n_missing = 0
    for ctx in ctx_list:
        rec = by_id.get(ctx.image_id)
        if rec is None:
            n_missing += 1
            continue
        if ctx.provenance == BAND:
            n_band += 1          # 집계 제외, 건수만 보고
            continue
        images.append(
            NormalImage(
                image_id=ctx.image_id,
                group_id=ctx.group_id,
                source=ctx.provenance,
                false_positive=is_false_positive(rec),
            )
        )

    report = p9_cross_source(images, margin=margin, min_clusters=min_clusters)
    breakdown = fp_breakdown_by_source(images)
    if report.tost.equivalent:
        verdict = (
            f"동등 (TOST δ={margin}) — 출처 잔존 신호를 본 모델이 오탐에 쓰지 않는다. "
            "잔존은 한계 문단으로 간다"
        )
    elif "판정 불가" in report.tost.verdict:
        verdict = report.tost.verdict
    else:
        verdict = (
            f"동등 아님 — 오탐이 출처로 쏠린다 "
            f"(차 {report.diff.point:+.4f} [{report.diff.lo:+.4f}, {report.diff.hi:+.4f}]). "
            "크롭 한정본 승격 재검토 대상"
        )
    cell = next(iter(cells)) if cells else "?"
    seed = next(iter(seeds)) if seeds else -1
    return P9CellResult(
        cell=cell, seed=seed, report=report, breakdown=breakdown,
        n_band_excluded=n_band, n_missing_prediction=n_missing, verdict=verdict,
    )


@dataclass(frozen=True)
class P9Summary:
    results: tuple[P9CellResult, ...]
    caveats: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict:
        return {
            "results": [r.as_dict() for r in self.results],
            "all_equivalent": all(r.report.tost.equivalent for r in self.results),
            "caveats": list(self.caveats),
        }


def p9_all_cells(
    records: Iterable[PredictionRecord],
    contexts: Iterable[EvalNormalContext],
    *,
    margin: float = P9_MARGIN,
    min_clusters: int = 20,
) -> P9Summary:
    """전 칸·전 시드의 레코드를 (칸, 시드)로 갈라 각각 P9 를 낸다.

    **같은 함수가 다섯 칸을 전부 처리한다.** 칸 이름으로 분기하는 코드는 없다 —
    channel 이 다르면 결과가 다를 뿐, 오탐의 정의와 판정 규칙은 하나다.
    """
    ctx_list = list(contexts)
    grouped: dict[tuple[str, int], list[PredictionRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.cell, rec.seed), []).append(rec)
    results = tuple(
        p9_for_cell(grouped[key], ctx_list, margin=margin, min_clusters=min_clusters)
        for key in sorted(grouped)
    )
    caveats = (
        ("P9 통과는 잔여 신호가 없다는 증명이 아니라, 본 모델의 오탐이 출처로 쏠리지 "
         "않았다는 측정 기록이다."),
        f"N-band 는 표본 부족으로 집계에서 제외했다 (칸당 건수는 결과에 있다). δ={margin}.",
    )
    return P9Summary(results=results, caveats=caveats)
