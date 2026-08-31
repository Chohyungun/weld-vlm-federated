"""P9 교차 출처 오탐 시험 · D-3 출처축 분해 보고. §5-2·§5-5.

타일링 후 정상 이미지는 두 출신으로 갈린다.

- **N-crop** — 원래부터 1280×720 이던 정상 (≈646장 / ≈65 묶음)
- **N-tile** — 파노라마에서 잘라낸 타일 (≈5,033장)

둘의 오탐률이 다르면 규격이 아니라 **출처**가 새 지름길이 된 것이다. 규격을 통일해도
압축 세대·납글자·필름 여백 같은 채널은 감쇠할 뿐 사라지지 않는다(§5-6).

**등가성은 TOST 로 판정한다.** CI 겹침으로 판정하면 검정력이 낮을수록 "차이 없음"이
쉬워지는 역인센티브가 생긴다. N-crop 이 65 묶음뿐이라 이 위험이 실재한다 —
검정력이 부족하면 "지름길 없음"이 아니라 **"판정 불가"** 로 보고한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evaluation.stats import (
    Interval,
    TostResult,
    cluster_bootstrap,
    tost_equivalence,
)

CROP = "N-crop"
TILE = "N-tile"
BAND = "N-band"
"""밴드 중심 크롭. 표본이 적어 P9 집계에서 제외하고 건수만 보고한다."""
DEFAULT_MARGIN = 0.10
"""§5-2 P9 사전등록 마진 δ."""


@dataclass(frozen=True)
class NormalImage:
    """평가셋 정상 이미지 한 장."""

    image_id: str
    group_id: str
    source: str
    """`N-crop` / `N-tile`. 매니페스트의 타일 출신 표시에서 온다."""
    false_positive: bool
    """이 정상 이미지에 결함을 하나라도 주장했는가."""


@dataclass(frozen=True)
class CrossSourceReport:
    crop: Interval
    tile: Interval
    diff: Interval
    tost: TostResult
    n_crop: int
    n_tile: int

    @property
    def reportable(self) -> bool:
        """`판정 불가` 는 통과가 아니다 — 미도달을 통과처럼 쓰지 않는다."""
        return self.tost.equivalent

    def as_dict(self) -> dict:
        return {
            "fp_rate_crop": self.crop.as_dict(),
            "fp_rate_tile": self.tile.as_dict(),
            "fp_rate_diff": self.diff.as_dict(),
            "tost": self.tost.as_dict(),
            "n_crop": self.n_crop, "n_tile": self.n_tile,
        }

    def headline(self) -> str:
        """§5-5 고정 문장의 P9 부분."""
        return (
            f"교차 출처 오탐률 차 {self.diff.point:+.4f} "
            f"[95% CI {self.diff.lo:+.4f}, {self.diff.hi:+.4f}] "
            f"[TOST δ={self.tost.margin} {'등가' if self.tost.equivalent else '불성립'}]"
        )


def _fp_statistic(images: Sequence[NormalImage]):
    members: dict[str, list[bool]] = {}
    for im in images:
        members.setdefault(im.group_id, []).append(im.false_positive)

    def statistic(groups: Sequence[str]) -> float:
        hits = total = 0
        for g in groups:
            for fp in members.get(g, ()):
                hits += fp
                total += 1
        return hits / total if total else 0.0

    return statistic, sorted(members)


def _diff_statistic(images: Sequence[NormalImage]):
    """출처별 오탐률의 차를 하나의 통계량으로 낸다.

    두 CI 를 따로 내서 겹침을 보는 대신 **차 자체를 재표집**한다 — 그래야 TOST 가 성립하고
    두 집단의 상관도 자연히 반영된다.
    """
    by_group: dict[str, list[NormalImage]] = {}
    for im in images:
        by_group.setdefault(im.group_id, []).append(im)

    def statistic(groups: Sequence[str]) -> float:
        c_hit = c_tot = t_hit = t_tot = 0
        for g in groups:
            for im in by_group.get(g, ()):
                if im.source == CROP:
                    c_hit += im.false_positive
                    c_tot += 1
                elif im.source == TILE:
                    t_hit += im.false_positive
                    t_tot += 1
        c = c_hit / c_tot if c_tot else 0.0
        t = t_hit / t_tot if t_tot else 0.0
        return t - c

    return statistic, sorted(by_group)


def p9_cross_source(
    images: Sequence[NormalImage],
    *,
    margin: float = DEFAULT_MARGIN,
    min_clusters: int = 20,
) -> CrossSourceReport:
    """P9 — 출처별 오탐률을 묶음 클러스터 부트스트랩으로 비교하고 TOST 로 판정한다.

    N-crop 묶음이 적으면(설계상 ≈65) 검정력이 부족할 수 있다. 그 경우 `판정 불가` 이며
    **불합격과 구분해 보고**한다 — "찾지 못했다"와 "없다"는 다르다.
    """
    crop_imgs = [i for i in images if i.source == CROP]
    tile_imgs = [i for i in images if i.source == TILE]

    crop_stat, crop_groups = _fp_statistic(crop_imgs)
    tile_stat, tile_groups = _fp_statistic(tile_imgs)
    diff_stat, all_groups = _diff_statistic(images)

    crop_ci = cluster_bootstrap(crop_groups, crop_stat)
    tile_ci = cluster_bootstrap(tile_groups, tile_stat)
    diff_ci = cluster_bootstrap(all_groups, diff_stat)
    # 검정력 판단은 희소한 쪽(N-crop) 묶음 수로 한다.
    diff_for_tost = Interval(
        diff_ci.point, diff_ci.lo, diff_ci.hi, diff_ci.n_resamples, len(crop_groups)
    )
    tost = tost_equivalence(diff_for_tost, margin, min_clusters=min_clusters)
    return CrossSourceReport(
        crop=crop_ci, tile=tile_ci, diff=diff_ci, tost=tost,
        n_crop=len(crop_imgs), n_tile=len(tile_imgs),
    )


def fp_breakdown_by_source(
    images: Sequence[NormalImage],
) -> dict[str, dict[str, float | int]]:
    """D-3 — 정상 오탐률을 출처축으로 분해한다. **스키마 변경 없이 보고만 확장**한다.

    총계 하나만 실으면 두 출신의 차이가 평균에 묻힌다. N-tile 이 N-crop 의 8배라
    총계는 사실상 N-tile 값이 된다.
    """
    out: dict[str, dict[str, float | int]] = {}
    for src in (CROP, TILE):
        subset = [i for i in images if i.source == src]
        n = len(subset)
        fp = sum(1 for i in subset if i.false_positive)
        out[src] = {
            "n_images": n,
            "n_groups": len({i.group_id for i in subset}),
            "n_false_positive": fp,
            "fp_rate": fp / n if n else 0.0,
        }
    total = len(images)
    fp_total = sum(1 for i in images if i.false_positive)
    out["전체"] = {
        "n_images": total,
        "n_groups": len({i.group_id for i in images}),
        "n_false_positive": fp_total,
        "fp_rate": fp_total / total if total else 0.0,
    }
    return out
