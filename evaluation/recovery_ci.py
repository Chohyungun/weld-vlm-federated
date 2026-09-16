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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

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


# ======================================================================================
# CI 산출물의 입력 결속 · 규칙 재판정 (외부 검토 codex_reply §18, 09-16)
#
# CI 생성기(`recovery_bootstrap.py`)와 집계기(`aggregate_seeds.py`)가 **같은 함수**로 결속
# 정보를 만든다. 둘이 각자 만들면 필드 하나가 어긋나도 "일치" 가 나올 수 있다.
# ======================================================================================

CI_BINDING_VERSION = 1
"""결속 블록의 판. 이 판이 없거나 다르면 집계기는 CI 를 쓰지 않는다(구판 CI 포함)."""

POINT_TOLERANCE = 1e-12
"""CI 에 기록된 점추정과 집계기가 **지금 읽은 채점 파일**로 다시 계산한 값의 허용 절대 차.

**근거 (09-16 v2 실측과 규모 계산).**

- 계산 경로 잡음: v2 세 시드에서 CI 기록과 집계기 재계산의 최대 차는 R_s 5.6e-17 · R̄ 2.8e-17 ·
  D_s 0 · map_50 0 이었다. `map_50` 은 항등 검사로 비트가 같다. 로컬 평균은 CI 가 `np.mean`,
  집계기가 `statistics.fmean` 이라 마지막 자리(1 ulp ≈ 5.6e-17) 하나가 다를 수 있다.
  1e-12 는 이 잡음의 약 1.8만 배다 — 경로 차이로는 걸리지 않는다.
- 내용 변화의 크기: 검출 한 건이 바뀌면 재현율 한 칸이 1/22,549 ≈ 4.4e-5 움직이고(네 클래스
  평균 ≈ 1.1e-5), 회복률은 1/D ≈ 3.4 배로 더 움직인다. 이런 변화는 1e-12 보다 일곱 자릿수 크다.
- 한계: 정밀도 한 점의 최소 변화는 1/(n(n+1)) 이고(n = 그 순위까지의 검출 수) 101점·4클래스
  평균으로 희석되면 n = 3만에서 약 2.7e-12 까지 작아진다. 포락선이 변화를 아예 지울 수도 있다.
  그래서 **점추정 대조만으로 모든 내용 변화를 잡는다고 주장하지 않는다.** 한 바이트 변화는
  sha256 결속이 잡고, 이 대조는 "이 파일에서 이 CI 의 수가 나오는가" 를 따로 확인한다.
  허용 오차를 잡음이 허락하는 한 작게(4자리 여유) 둔 이유다.

비교는 전부 **비율 단위**(회복률 0.2477, 백분율 아님)로 한다 — 단위가 섞이면 같은 허용 오차가
두 자릿수 다른 뜻이 된다. 한쪽만 미정의(None·nan)면 불일치, 양쪽 다 미정의면 일치로 본다.
"""

_BOUND_FIELDS = ("sha256", "seed_value", "scorer_code_combined", "scorer_code_rule",
                 "newline_normalized")
"""시드별로 CI 기록과 집계 입력이 **전부** 같아야 하는 필드."""


def read_artifact(path: Path) -> tuple[dict, dict]:
    """채점 산출물을 **한 번만** 읽어 `(payload, 결속 항목)` 을 낸다.

    바이트를 읽어 그 바이트로 해싱하고 그 바이트를 파싱한다 — 해싱과 파싱 사이에 파일이 바뀌어도
    둘이 갈리지 않는다. 해시는 **원시 바이트**다. 데이터 파일이라 줄끝 정규화를 하지 않는다
    (소스 지문과 다르다). 한 바이트만 바뀌어도 다른 값이다.
    """
    import hashlib
    import json

    from evaluation.provenance import relpath

    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    sc = payload.get("scorer_code") or {}
    entry = {
        "path": relpath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "seed_value": (payload.get("params") or {}).get("seed"),
        "scorer_code_combined": sc.get("combined"),
        "scorer_code_rule": sc.get("rule"),
        "newline_normalized": bool(sc.get("newline_normalized", False)),
        "scorer_code_stable": sc.get("stable"),
    }
    return payload, entry


def make_binding(artifact_name: str, entries: Mapping[int, dict]) -> dict:
    """CI 가 어떤 입력으로 만들어졌는지 — 산출물 이름 · 시드 목록 · 시드별 산출물 해시 · 채점기 지문."""
    seeds = sorted(int(s) for s in entries)
    return {
        "version": CI_BINDING_VERSION,
        "artifact_name": artifact_name,
        "seeds": seeds,
        "artifacts": {str(s): dict(entries[s]) for s in seeds},
    }


def check_binding(ci_payload: dict, observed: dict) -> list[str]:
    """CI 에 기록된 결속과 집계기가 읽은 입력이 같은가. **다른 점을 전부** 사유로 낸다.

    빈 목록이면 결속 성립. 하나라도 있으면 그 CI 는 이 집계의 입력으로 만든 것이 아니다.
    """
    reasons: list[str] = []
    recorded = ci_payload.get("binding")
    if not isinstance(recorded, dict):
        return [("CI 에 입력 결속 블록(binding)이 없다 — 결속 이전 판 CI 라 어느 입력으로 "
                 "만들었는지 확인할 수 없다")]
    if recorded.get("version") != CI_BINDING_VERSION:
        reasons.append(f"결속 판이 다르다: CI {recorded.get('version')} · 집계기 {CI_BINDING_VERSION}")
    if recorded.get("artifact_name") != observed["artifact_name"]:
        reasons.append(f"산출물 이름이 다르다: CI {recorded.get('artifact_name')} · "
                       f"집계 {observed['artifact_name']}")
    if list(recorded.get("seeds") or []) != list(observed["seeds"]):
        reasons.append(f"시드 집합이 다르다: CI {recorded.get('seeds')} · 집계 {observed['seeds']}")

    rec_arts = recorded.get("artifacts") or {}
    obs_arts = [observed["artifacts"][str(s)] for s in observed["seeds"]]
    for s, obs in zip(observed["seeds"], obs_arts, strict=True):
        if obs.get("scorer_code_combined") is None:
            reasons.append(f"시드 {s} 채점 산출물에 채점기 지문이 없다 — 결속할 수 없다")
        got = rec_arts.get(str(s))
        if got is None:
            reasons.append(f"시드 {s} 가 CI 결속에 없다")
            continue
        for f in _BOUND_FIELDS:
            if got.get(f) != obs.get(f):
                reasons.append(f"시드 {s} 의 {f} 가 다르다: CI {str(got.get(f))[:16]} · "
                               f"집계 {str(obs.get(f))[:16]}")

    # CI 를 계산한 코드가 채점한 코드와 같은가 — 지문이 CI 생성기까지 덮는다
    ci_code = ci_payload.get("scorer_code") or {}
    combos = {a.get("scorer_code_combined") for a in obs_arts}
    if len(combos) != 1:
        reasons.append(f"시드 사이에 채점기 지문이 다르다: {sorted(str(c)[:16] for c in combos)}")
    elif ci_code.get("combined") not in combos:
        reasons.append(f"CI 를 계산한 코드의 지문 {str(ci_code.get('combined'))[:16]} 이 "
                       "채점 지문과 다르다")
    if ci_code.get("stable") is not True:
        reasons.append("CI 계산 중 코드 트리가 바뀌었거나 안정성 기록이 없다(scorer_code.stable ≠ true)")
    if ci_code.get("rule") not in {a.get("scorer_code_rule") for a in obs_arts}:
        reasons.append("CI 를 계산한 코드의 지문 규칙 표기가 채점 산출물과 다르다")
    if bool(ci_code.get("newline_normalized", False)) not in {
            a.get("newline_normalized") for a in obs_arts}:
        reasons.append("CI 를 계산한 코드와 채점 산출물의 줄끝 정규화 여부가 다르다")
    return reasons


def rejudge_ci_rules(ci_payload: dict, seeds: Sequence[int]) -> tuple[dict | None, list[str]]:
    """**기록된 `ci_pass` 를 믿지 않고** 구간 · 반폭 · 미정의 수에서 규칙 ②③④를 다시 판정한다.

    Returns:
        `(판정, 사유)`. CI 기록이 자기모순이거나 등록 조건(재표집 횟수 · 난수 시드)과 다르면
        판정 없이 사유만 낸다 — 그 CI 로는 판정할 수 없다는 뜻이다.
    """
    from evaluation.prereg import (
        RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO,
        RECOVERY_CI_MAX_HALF_WIDTH,
        RECOVERY_CI_MAX_UNDEFINED_FRACTION,
    )
    from evaluation.stats import BOOTSTRAP_N, BOOTSTRAP_SEED

    reasons: list[str] = []
    stat = ci_payload.get("statistic") or {}
    n_res = stat.get("n_resamples")
    if n_res != BOOTSTRAP_N:
        reasons.append(f"재표집 횟수가 등록값과 다르다: {n_res} ≠ {BOOTSTRAP_N}")
    if stat.get("rng_seed") != BOOTSTRAP_SEED:
        reasons.append(f"재표집 난수 시드가 등록값과 다르다: {stat.get('rng_seed')} ≠ {BOOTSTRAP_SEED}")

    rb = ((ci_payload.get("ci") or {}).get("R_bar")) or {}
    lo, hi, hw, n_undef = rb.get("ci_lo"), rb.get("ci_hi"), rb.get("half_width"), rb.get("n_undefined")
    finite = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                 for x in (lo, hi, hw))
    if not finite or not isinstance(n_undef, int) or isinstance(n_undef, bool) or n_undef < 0:
        reasons.append("R̄ 구간 · 반폭 · 미정의 수가 수치가 아니다")
        return None, reasons
    half = (hi - lo) / 2
    if isinstance(n_res, int) and n_undef > n_res:
        reasons.append(f"미정의 수 {n_undef} 가 재표집 수 {n_res} 보다 크다 — 기록이 자기모순이다")
    if lo > hi:
        reasons.append(f"R̄ 구간이 뒤집혀 있다: [{lo}, {hi}]")
    if abs(half - hw) > 1e-12:
        reasons.append(f"기록된 반폭 {hw} 이 구간에서 계산한 반폭 {half} 과 다르다 — 기록이 자기모순이다")

    d_by = ((ci_payload.get("ci") or {}).get("D_by_seed")) or {}
    d_lo: dict[int, float] = {}
    for s in seeds:
        v = (d_by.get(str(s)) or {}).get("ci_lo")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            reasons.append(f"시드 {s} 분모 CI 하한이 없다")
        else:
            d_lo[s] = float(v)
    if reasons:
        return None, reasons

    undef_frac = n_undef / n_res
    rule2 = half <= RECOVERY_CI_MAX_HALF_WIDTH
    rule3_by = {str(s): d_lo[s] > 0 for s in seeds}
    rule3 = all(rule3_by.values())
    rule4 = undef_frac <= RECOVERY_CI_MAX_UNDEFINED_FRACTION
    ci_pass = bool(rule2 and (rule3 or not RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO) and rule4)
    recorded = (ci_payload.get("verdict") or {}).get("ci_pass")
    return {
        "2_half_width": {"value": half, "max": RECOVERY_CI_MAX_HALF_WIDTH, "pass": rule2},
        "3_denominator_ci_excludes_zero": {"by_seed": rule3_by,
                                           "required": RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO,
                                           "pass": rule3},
        "4_undefined_fraction": {"value": undef_frac, "max": RECOVERY_CI_MAX_UNDEFINED_FRACTION,
                                 "pass": rule4},
        "ci_pass": ci_pass,
        "recorded_ci_pass": recorded,
        "recorded_agrees": recorded is ci_pass,
        "source": "집계기가 CI 의 구간 · 반폭 · 미정의 수에서 다시 판정(기록된 ci_pass 미사용)",
    }, []


def check_points(ci_payload: dict, seeds: Sequence[int], observed_points: dict) -> list[str]:
    """CI 의 점추정이 집계기가 **지금 읽은 채점 파일**로 다시 계산한 값과 같은가(요구 6).

    `observed_points` = `{"R_bar": 비율, "R": {seed: 비율}, "D": {seed}, "map_50": {seed: {tag}}}`.
    해시·시드·지문이 같으면 따라오는 성질이지만 **그것으로 대신하지 않는다** — 두 구현이 같은
    파일에서 같은 수를 내는지를 따로 맞댄다. 허용 오차는 `POINT_TOLERANCE`(근거는 그 docstring).
    """
    reasons: list[str] = []
    pt = ci_payload.get("point") or {}

    def _num(x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    def _undefined(x) -> bool:        # D ≤ 0 인 시드 — CI 는 nan, 집계기는 None 으로 적는다
        return x is None or (isinstance(x, float) and math.isnan(x))

    def _check(label: str, a, b) -> None:
        if _undefined(a) and _undefined(b):
            return
        if not (_num(a) and _num(b)):
            reasons.append(f"{label} 점추정을 비교할 수 없다: CI {a!r} · 집계 {b!r}")
            return
        d = abs(a - b)
        if d > POINT_TOLERANCE:
            reasons.append(f"{label} 점추정이 다르다: CI {a!r} · 집계 {b!r} (차 {d:.3e} > {POINT_TOLERANCE:g})")

    _check("R̄", pt.get("R_bar"), observed_points["R_bar"])
    for s in seeds:
        k = str(s)
        _check(f"시드 {s} 회복률", (pt.get("R_by_seed") or {}).get(k), observed_points["R"][s])
        _check(f"시드 {s} 분모", (pt.get("D_by_seed") or {}).get(k), observed_points["D"][s])
        cm = (pt.get("map_50_by_seed") or {}).get(k) or {}
        for tag, val in observed_points["map_50"][s].items():
            _check(f"시드 {s} {tag} map_50", cm.get(tag), val)
    return reasons


def check_identity(ci_payload: dict, seeds: Sequence[int], tags: Sequence[str],
                   observed_map: Mapping[int, Mapping[str, Mapping[str, float]]]) -> list[str]:
    """T1 항등 검사 결과가 CI 산출물에 **있고, 전부 통과했고, 차가 0** 인가(요구 5).

    항등 검사는 이 CI 의 집계 재구현이 pycocotools 와 같다는 유일한 근거다. 기록이 없거나
    한 칸이라도 실패했거나 차가 0 이 아니면, 그 CI 의 재표집 값은 다른 계산의 값이다.

    또한 각 항목의 **기준값(`expected_map_50`·`expected_map_50_95`)이 지금 읽은 채점 파일의 값과
    같은지** 본다 — 항등 검사가 다른 파일을 기준으로 돌았으면 통과 기록이 이 집계를 대변하지 않는다.

    Args:
        observed_map: `{seed: {tag: {"map_50": .., "map_50_95": ..}}}` — 지금 읽은 파일의 값.
    """
    reasons: list[str] = []
    ic = ci_payload.get("identity_check")
    if not isinstance(ic, Mapping) or not ic:
        return ["CI 에 T1 항등 검사 결과가 없다 — 집계 재구현이 pycocotools 와 같다는 근거가 없다"]
    expected_n = len(seeds) * len(tags)
    got_n = 0
    for s in seeds:
        per = ic.get(str(s))
        if not isinstance(per, Mapping):
            reasons.append(f"시드 {s} 의 항등 검사 기록이 없다")
            continue
        for t in tags:
            c = per.get(t)
            if not isinstance(c, Mapping):
                reasons.append(f"시드 {s} {t} 의 항등 검사 기록이 없다")
                continue
            got_n += 1
            if c.get("passed") is not True:
                reasons.append(f"시드 {s} {t} 항등 검사가 통과하지 않았다")
            for key in ("abs_diff_map_50", "abs_diff_map_50_95"):
                v = c.get(key)
                if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0.0):
                    reasons.append(f"시드 {s} {t} 항등 {key} 가 0 이 아니다: {v!r}")
            obs = (observed_map.get(s) or {}).get(t)
            if obs is None:
                reasons.append(f"시드 {s} {t} 의 채점 파일 값이 없어 항등 기준값을 맞댈 수 없다")
                continue
            for key, okey in (("expected_map_50", "map_50"), ("expected_map_50_95", "map_50_95")):
                if c.get(key) != obs[okey]:
                    reasons.append(f"시드 {s} {t} 항등 검사의 기준 {okey} 가 지금 채점 파일과 다르다: "
                                   f"{c.get(key)!r} · {obs[okey]!r}")
    if got_n != expected_n:
        reasons.append(f"항등 검사 {got_n}/{expected_n} — 칸 × 시드 전부가 아니다")
    return reasons
