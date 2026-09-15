"""회복률의 묶음 클러스터 부트스트랩 CI — 매칭 1회 + 집계 재표집 (36번 미니스펙, 총괄 게이트 09-16).

## 왜 이렇게 하나

대표 축 `mAP@50` 은 COCO 평가라 재표집마다 `COCOeval` 을 다시 돌리면 2,000회 × 칸 5 × 시드 3 =
30,000번, 며칠이다. `COCOeval` 은 두 단계다 — `evaluate()` 가 **이미지 안에서 닫히는** 매칭을
`evalImgs` 에 남기고, `accumulate()` 가 그것을 이어 붙여 AP 를 낸다. 다른 이미지가 있든 없든
한 이미지의 매칭은 같다. 그래서 매칭은 (칸, 시드)마다 한 번만 하고 재표집마다 집계만 다시 한다.

## 중복 = 제자리 반복

묶음 재표집은 이미지별 **중복도** `w_i ∈ {0, 1, 2, …}` 로 바뀐다. `accumulate` 는 이미지 순서로
검출을 이어 붙인 뒤 `-score` 로 **안정 정렬(mergesort)** 하므로, 이미지 `i` 의 검출·GT 항목을
`w_i` 번 **블록으로 제자리 반복**한 것은 같은 이미지가 인접한 id 로 `w_i` 번 존재하는 것과
정확히 같은 계산이다(동점 순서까지). 원소 단위 반복(d1 d1 d2 d2)은 다르다 — 같은 이미지 안의
동점 tp/fp 순서가 바뀌어 PR 곡선이 달라진다. 이 모듈은 블록 반복만 한다.

## 항등 검사가 이 모듈의 전부다

중복도가 전부 1 인 "재표집" 은 `coco_map()` 이 낸 `map_50`·`map_50_95` 와 **부동소수 마지막
자리까지** 같아야 한다. 어긋나면 이 집계 재구현이 pycocotools 와 다른 것이고, 그때는 CI 를 내지
않고 멈춘다(`IdentityCheckFailed`). 근사값으로 판정하지 않는다.

pycocotools 의 `evalImgs` 구조와 `accumulate` 의 정렬·보간 규칙은 비공개 API 다. 버전이 바뀌면
깨질 수 있고, 항등 검사와 합성 중복 시험(`tests/test_recovery_ci.py`)이 그때 잡는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from evaluation.metrics.localization import build_cocoeval

MAX_DETS = 100
"""`map_50`·`map_50_95` 가 쓰는 pycocotools 기본 `maxDets` 의 마지막 값. `stats[0]`·`stats[1]`."""

AREA_ALL = 0
"""`areaRng` 의 'all' 인덱스. 두 대표 지표는 이 조각만 쓴다."""


class IdentityCheckFailed(RuntimeError):
    """중복도 전부 1 의 집계가 채점기의 mAP 와 다르다 — 집계 재구현이 pycocotools 와 갈렸다."""


@dataclass
class CategoryCache:
    """카테고리 하나의 이미지별 매칭을 **이미지 순서로 평탄화**한 것.

    이미지 `i` 의 검출은 `[start[i], start[i] + length[i])` 구간이다. `evaluateImg` 가 이미
    점수 내림차순으로 정렬해 뒀고 `maxDets` 로 잘랐다.
    """

    scores: np.ndarray          # (D,) float
    dt_match: np.ndarray        # (T, D) bool — 어느 GT 에 매칭됐는가 (0 = 미매칭)
    dt_ignore: np.ndarray       # (T, D) bool
    start: np.ndarray           # (I,) int — 이미지별 시작 오프셋
    length: np.ndarray          # (I,) int — 이미지별 검출 수
    npig: np.ndarray            # (I,) int — 이미지별 무시 아닌 GT 수


@dataclass
class EvalCache:
    """(칸, 시드) 하나의 매칭 캐시. 이미지 순서는 `coco_map` 과 같은 `sorted(gold)` 다."""

    image_ids: list[str]
    classes: list[str]
    n_iou: int
    rec_thrs: np.ndarray
    per_class: list[CategoryCache] = field(default_factory=list)
    n_gt_boxes: int = 0
    n_pred_boxes: int = 0

    @property
    def n_images(self) -> int:
        return len(self.image_ids)


def cache_from_cocoeval(ev, image_ids: Sequence[str], classes: Sequence[str],
                        n_gt: int, n_dt: int) -> EvalCache:
    """`COCOeval.evaluate()` 가 남긴 `evalImgs` 에서 'all'·`maxDets=100` 조각을 평탄화한다.

    `evalImgs[k*A*I + a*I + i]` 는 (카테고리 k, 영역 a, 이미지 i) 의 딕셔너리이거나, 그 이미지에
    GT 도 검출도 없으면 `None` 이다. `None` 은 길이 0 · GT 0 으로 둔다 — `accumulate` 도
    `None` 을 건너뛴다.
    """
    p = ev._paramsEval
    img_ids = list(p.imgIds)
    cat_ids = list(p.catIds)
    n_img, n_area = len(img_ids), len(p.areaRng)
    t_count = len(p.iouThrs)
    if n_img != len(image_ids) or len(cat_ids) != len(classes):
        raise ValueError("COCOeval 파라미터가 입력과 맞지 않는다")

    per_class: list[CategoryCache] = []
    for k in range(len(cat_ids)):
        scores, dtm, dtig, length, npig = [], [], [], [], []
        base = k * n_area * n_img + AREA_ALL * n_img
        for i in range(n_img):
            e = ev.evalImgs[base + i]
            if e is None:
                length.append(0)
                npig.append(0)
                continue
            s = np.asarray(e["dtScores"][:MAX_DETS], dtype=float)
            m = np.asarray(e["dtMatches"][:, :MAX_DETS]) != 0
            ig = np.asarray(e["dtIgnore"][:, :MAX_DETS]).astype(bool)
            scores.append(s)
            dtm.append(m.reshape(t_count, -1))
            dtig.append(ig.reshape(t_count, -1))
            length.append(int(s.size))
            npig.append(int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)))
        length_arr = np.asarray(length, dtype=np.int64)
        start = np.concatenate(([0], np.cumsum(length_arr)[:-1])) if n_img else np.zeros(0, int)
        per_class.append(CategoryCache(
            scores=np.concatenate(scores) if scores else np.zeros(0, float),
            dt_match=(np.concatenate(dtm, axis=1) if dtm
                      else np.zeros((t_count, 0), bool)),
            dt_ignore=(np.concatenate(dtig, axis=1) if dtig
                       else np.zeros((t_count, 0), bool)),
            start=start, length=length_arr, npig=np.asarray(npig, dtype=np.int64),
        ))
    return EvalCache(
        image_ids=list(image_ids), classes=list(classes), n_iou=t_count,
        rec_thrs=np.asarray(p.recThrs, dtype=float), per_class=per_class,
        n_gt_boxes=n_gt, n_pred_boxes=n_dt,
    )


def build_cache(
    pred: Mapping[str, Sequence[tuple[str, tuple[float, float, float, float], float]]],
    gold: Mapping[str, Sequence[tuple[str, tuple[float, float, float, float]]]],
    classes: Sequence[str],
) -> EvalCache | None:
    """`coco_map` 과 **같은 구축 함수**(`build_cocoeval`)로 매칭을 얻어 캐시한다. GT 0 이면 None."""
    built = build_cocoeval(pred, gold, classes)
    if built is None:
        return None
    ev, n_gt, n_dt = built
    return cache_from_cocoeval(ev, sorted(gold), classes, n_gt, n_dt)


# --------------------------------------------------------------------------------------
# 가중 집계 — pycocotools.accumulate 의 'all'·maxDets=100 조각을 중복도로
# --------------------------------------------------------------------------------------

def _block_repeat_index(start: np.ndarray, length: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """이미지별 검출 구간을 중복도만큼 **블록으로** 반복한 전역 인덱스. 파이썬 루프 없음.

    `img_rep` 가 이미지 i 를 w_i 번 연속으로 나열하고, 각 원소의 구간 `[start, start+length)`
    를 이어 붙인다 — 벡터화된 다중 range 연결이다.
    """
    img_rep = np.repeat(np.arange(len(length)), weights)
    lens = length[img_rep]
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = start[img_rep]
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return np.arange(total, dtype=np.int64) + offsets


def _precision_grid(scores: np.ndarray, dtm: np.ndarray, dtig: np.ndarray, npig: int,
                    rec_thrs: np.ndarray) -> np.ndarray:
    """카테고리 하나의 보간 정밀도 격자 `(T, R)` — `accumulate` 의 내부 루프를 그대로.

    `-score` 안정 정렬 → tp/fp 누적 → 정밀도 우측 포락선 → 재현율 101점 `searchsorted(left)`
    보간. 검출이 0 이면 pycocotools 처럼 0 행(예외 경로가 `q` 를 0 으로 남긴다). GT 가 0 이면
    `-1` 행 — 카테고리 부재 표시이고 평균에서 빠진다.

    **AP 를 여기서 평균하지 않는다.** pycocotools 는 `(T, R, K)` 격자를 통째로 평탄화해
    한 번에 평균한다. 카테고리별로 먼저 평균하면 수학적으로는 같아도 부동소수 마지막
    자리가 달라져 항등 검사가 깨진다.
    """
    t_count, r_count = dtm.shape[0], rec_thrs.size
    if npig == 0:
        return -np.ones((t_count, r_count))
    inds = np.argsort(-scores, kind="mergesort")
    dtm, dtig = dtm[:, inds], dtig[:, inds]
    tps = np.logical_and(dtm, ~dtig)
    fps = np.logical_and(~dtm, ~dtig)
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    grid = np.zeros((t_count, r_count))
    nd = tp_sum.shape[1]
    if nd == 0:
        return grid
    for t in range(t_count):
        rc = tp_sum[t] / npig
        pr = tp_sum[t] / (fp_sum[t] + tp_sum[t] + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]            # 우측 포락선
        idx = np.searchsorted(rc, rec_thrs, side="left")
        grid[t] = np.where(idx < nd, pr[np.minimum(idx, nd - 1)], 0.0)
    return grid


def weighted_map(cache: EvalCache, weights: np.ndarray) -> tuple[float, float]:
    """중복도 `weights`(이미지별, 정수 ≥ 0) 로 집계한 `(map_50, map_50_95)`.

    평균은 pycocotools `_summarize` 와 **같은 순서**로 낸다 — `(T, R, K)` 격자에서 `-1` 이
    아닌 항목을 C 순서로 평탄화해 한 번에 `np.mean`. 그래서 `weights` 가 전부 1 이면
    `coco_map` 과 비트 단위로 같다(`identity_check` 가 본다).
    """
    weights = np.asarray(weights, dtype=np.int64)
    if weights.shape != (cache.n_images,):
        raise ValueError("중복도 길이가 이미지 수와 다르다")
    t_count, r_count, k_count = cache.n_iou, cache.rec_thrs.size, len(cache.per_class)
    precision = -np.ones((t_count, r_count, k_count))
    for k, c in enumerate(cache.per_class):
        idx = _block_repeat_index(c.start, c.length, weights)
        npig = int((c.npig * weights).sum())
        precision[:, :, k] = _precision_grid(
            c.scores[idx], c.dt_match[:, idx], c.dt_ignore[:, idx], npig, cache.rec_thrs)
    s50 = precision[0]
    s50 = s50[s50 > -1]
    sall = precision[precision > -1]
    map_50 = float(np.mean(s50)) if s50.size else float("nan")
    map_50_95 = float(np.mean(sall)) if sall.size else float("nan")
    return map_50, map_50_95


def identity_check(cache: EvalCache, expected_map_50: float, expected_map_50_95: float | None = None,
                   *, atol: float = 0.0) -> dict:
    """중복도 전부 1 → 채점기 값과 **비트 단위로** 일치해야 한다. 아니면 예외 — CI 를 내지 않는다.

    기본 허용 오차 0. 평균 순서까지 pycocotools 와 맞췄으므로 같은 입력이면 같은 비트가 나온다.
    한 자리라도 다르면 집계가 갈린 것이고, 그 상태로 CI 를 내면 판정이 근사 위에 선다.
    """
    ones = np.ones(cache.n_images, dtype=np.int64)
    m50, m5095 = weighted_map(cache, ones)
    d50 = abs(m50 - expected_map_50)
    ok = d50 <= atol
    d5095 = None
    if expected_map_50_95 is not None:
        d5095 = abs(m5095 - expected_map_50_95)
        ok = ok and d5095 <= atol
    result = {"map_50": m50, "expected_map_50": expected_map_50, "abs_diff_map_50": d50,
              "map_50_95": m5095, "expected_map_50_95": expected_map_50_95,
              "abs_diff_map_50_95": d5095, "atol": atol, "passed": ok}
    if not ok:
        raise IdentityCheckFailed(
            f"항등 검사 실패 — 집계 재구현 map_50 {m50!r} vs 채점기 {expected_map_50!r} "
            f"(차 {d50:.3e}); map_50_95 차 {d5095}. CI 를 내지 않는다")
    return result


# --------------------------------------------------------------------------------------
# 짝지은 묶음 부트스트랩
# --------------------------------------------------------------------------------------

def group_weights(group_of_image: np.ndarray, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """묶음을 복원추출해 이미지별 중복도로 바꾼다. `cluster_bootstrap` 과 같은 추첨 규칙
    (묶음 n 개를 n 번 뽑는다)."""
    drawn = rng.integers(0, n_groups, size=n_groups)
    g_w = np.bincount(drawn, minlength=n_groups)
    return g_w[group_of_image]


def recovery_from(central: float, fed: float, locals_: Sequence[float]) -> tuple[float, float]:
    """`(R, D)`. D ≤ 0 이면 R 은 nan — 버리고 세는 규약(`drop_undefined`)의 신호다."""
    lm = float(np.mean(locals_))
    d = central - lm
    return (float("nan") if d <= 0 else (fed - lm) / d), d
