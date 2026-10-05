"""프레임 진단 검사기 — 생성한 좌표가 어느 프레임에 있는가. 07번 계약 §30-2 · §31-2 · §32-5(결합 순서와 후보 서명의 정본).

**겹침 없는 검사가 먼저다.** 이미지 수준의 중심 비 · 기울기는 예측과 정답이 겹치지 않아도 선다. 쌍(서로의 최대 IoU 상대)의
축 배율은 그 뒤다 — 박스가 작은 리사이즈 프레임은 겹침을 잃어도 이미지 수준에서 걸린다.

## 정의

| 항목 | 정의 |
|---|---|
| 결함 장 | 정답 박스가 하나 이상인 장 |
| 대상 장 | 결함 장 가운데 **예측 박스 수 = 정답 박스 수** 인 장. 예측이 없는 장 · 수가 다른 장은 빼고 그 수를 싣는다. 클래스는 보지 않는다 |
| 이미지 중심 | 그 장의 박스 중심의 산술평균 — 예측과 정답을 따로 |
| 중심 비 `r` | 축마다 `예측 중심 ÷ 정답 중심` 의 대상 장 사이 **중앙값** |
| 섞임 통계 `r̄` | 배율이 1 이 아닌 축마다 같은 비의 **절사 평균**(양쪽에서 `trim` 씩, 0 이면 평균) |
| 기울기 `s` | 축마다 정답 중심에 예측 중심을 댄 최소제곱 기울기(절편 있음). **교차 기울기** `s×` — 예측 x 를 정답 y 에, 예측 y 를 정답 x 에 |
| 쌍 | 예측과 정답이 하나 이상인 결함 장 안에서, 예측 `p` 와 정답 `g` 가 **서로의 최대 IoU 상대**이고 IoU > 0. 최대가 동률이면 쌍이 아니다 |
| 축 배율 `m` | 쌍마다 중심 좌표 비(예측 ÷ 정답)의 축별 중앙값 |
| 표준오차 | 모든 통계에 **`group_id` 단위 부트스트랩** — 한 묶음의 장과 쌍이 함께 흔들린다. 횟수와 씨앗은 규칙에서 |
| 독립 묶음 | 대상 장이 든 묶음의 수 · 쌍이 든 묶음의 수. 묶음이 적으면 부트스트랩이 늘 비슷한 집합을 뽑아 표준오차가 작게 나온다 — 하나뿐이면 0 이다 |
| 유효 재표집 | 그 통계가 유한한 재표집의 수. 규칙의 비율(`min_valid_boot × n_boot`)에 못 미치면 그 표준오차를 쓰지 않는다(게이트 불통과로 본다) |

## 결합 — 순서대로(§32-5)

| 순서 | 검사 | 어긋나면 |
|---|---|---|
| 1 | 대상 장 수 ≥ `N_img` 이고 대상 장의 묶음 수 ≥ `min_groups` | 판정 불가 — 표본 부족 |
| 2 | 두 축 `s ≥ s_min` | 두 축 `s×` 가 `1 ± w_s` 안이고 그 표준오차가 `w_s/3` 이하(유효 재표집 요건 포함)면 **불통과 — 뒤바뀜**. 아니면 판정 불가 — 위치를 따라가지 않는다 |
| 3 | `r` · `r̄` · `s` 의 표준오차 게이트(폭의 3분의 1 · 유효 재표집 요건) | 판정 불가 — 표본 부족 |
| 4 | 두 축 `\\|r − 1\\| ≤ w_r` 이고 `\\|r̄ − 1\\| ≤ w_mix` | **불통과** — 어긋난 축과 서명 |
| 5 | 두 축 `\\|s − 1\\| ≤ w_s` | 판정 불가 — 위치를 덜 따라간다 |
| 6 | 쌍 수 ≥ `min_pairs` · 쌍의 묶음 수 ≥ `min_pair_groups` 이고 두 축 `m` 의 표준오차 게이트 | 판정 불가 — 표본 부족 |
| 7 | 두 축 `\\|m − 1\\| ≤ axis_ratio_width` | **불통과** — 어긋난 축과 서명 |
| — | 일곱을 다 지나면 | **통과** |

**불통과는 표준오차 게이트를 지난 값으로만 낸다.** 판정 불가는 통과가 아니다.

## 후보 서명(§32-5 · M-6)

`k = (w_in/W, h_in/H)`(등록 프로세서의 `model_input_wh`), 정규화 `n = (1000/W, 1000/H)`.
**리사이즈** — 두 축의 `r` · `s` 가 `k ± δ` 안(배율이 1 인 축은 `1 ± δ`). 두 축 배율이 모두 1 이면 서지 않는다.
**정규화** — 두 축의 `r` · `s` 가 `n ± δ` 안. **리사이즈 섞임** — 배율이 1 이 아닌 축에서 `r` 은 `1 ± w_r` 안인데 `r̄` 이 `1 ± w_mix` 밖이고
`k` 쪽. **그 밖은 "후보 없음"**. 원본과 리사이즈가 겹치지 않으려면 적어도 한 축에서 `δ < |1 − k| / 2` 여야 한다(규칙이 검사한다).

## 내는 것

`FrameVerdict` — **상태 · 후보 이름 · 사유 · 대상 장 수 · 쌍 수 · 묶음 수**와 빼낸 장의 수, 그리고 규칙의 표본 요건마다
**충족 · 미달**(`samples`). 기울기 · 비 · 표준오차의 값은 싣지 않는다(§30-2).
통계 값은 `frame_stats` 가 따로 낸다 — 문턱을 세우는 절차(합성 사례 위)만 쓴다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

PASS, FAIL, INDETERMINATE = "통과", "불통과", "판정 불가"
STATUSES = (PASS, FAIL, INDETERMINATE)
REASON_SAMPLE = "표본 부족"
REASON_NOT_TRACKING = "위치를 따라가지 않는다"
REASON_WEAK_TRACKING = "위치를 덜 따라간다"
REASON_SIZE_MIXED = "크기 혼재"
REASON_NO_DEFECT = "결함 장이 없다"
CAND_RESIZE, CAND_NORM, CAND_MIX, CAND_SWAP, CAND_NONE = "리사이즈", "정규화", "리사이즈 섞임", "뒤바뀜", "후보 없음"
AXES = ("x", "y")

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameImage:
    image_id: str
    group_id: str
    width: int
    height: int
    pred: tuple[Box, ...]
    gold: tuple[Box, ...]


@dataclass(frozen=True)
class FrameRules:
    """문턱과 계산 설정. 값은 **규칙 파일에서만** 온다(§31-1) — 이 객체는 그 파일을 읽은 꼴이다."""

    n_img: int
    s_min: float
    w_s: float
    w_r: float
    w_mix: float
    delta: float
    trim: float
    n_boot: int
    boot_seed: int
    axis_ratio_width: float = 0.01
    min_pairs: int = 100
    se_divisor: float = 3.0
    min_groups: int = 20
    """대상 장이 든 묶음의 하한 — 이미지 수준 검사(1~5)의 독립 표본."""
    min_pair_groups: int = 20
    """쌍이 든 묶음의 하한 — 쌍 검사(6 · 7)의 독립 표본."""
    min_valid_boot: float = 0.95
    """표준오차를 쓰려면 그 통계가 유한한 재표집이 `n_boot` 의 이 비율 이상이어야 한다."""

    def problems(self, scale: tuple[float, float]) -> list[str]:
        out = []
        if type(self.n_img) is not int or self.n_img < 1:
            out.append("n_img 는 1 이상의 정수")
        for name in ("s_min", "w_s", "w_r", "w_mix", "delta", "axis_ratio_width"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                out.append(f"{name} 는 양수")
        if not (0 <= self.trim < 0.5):
            out.append("trim 은 [0, 0.5)")
        if type(self.n_boot) is not int or self.n_boot < 2:
            out.append("n_boot 는 2 이상의 정수")
        for name in ("min_pairs", "min_groups", "min_pair_groups"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                out.append(f"{name} 는 2 이상의 정수")
        if not (isinstance(self.min_valid_boot, (int, float)) and 0 < self.min_valid_boot <= 1):
            out.append("min_valid_boot 는 (0, 1]")
        moved = [abs(1 - k) for k in scale if k != 1]
        if moved and not any(self.delta < m / 2 for m in moved):
            out.append("delta 는 적어도 한 축에서 |1 − 배율|/2 보다 작아야 한다 — 원본과 리사이즈의 서명이 겹친다")
        return out


@dataclass(frozen=True)
class FrameVerdict:
    status: str
    candidate: str | None
    reason: str
    n_images: int
    n_pairs: int
    n_no_pred: int
    n_count_mismatch: int
    n_groups: int
    n_image_groups: int = 0
    n_pair_groups: int = 0
    samples: dict = field(default_factory=dict)
    """규칙의 표본 요건마다 `충족` · `미달` — 대상 장 · 대상 장의 묶음 · 쌍 · 쌍의 묶음. 요구 수와 표준오차는 싣지 않는다."""
    step: int | None = field(default=None, compare=False)
    """걸린 순서(1~7, 통과면 `None`). 시험과 문턱 절차가 본다 — 산출물에는 싣지 않는다."""

    def as_dict(self) -> dict:
        return {"status": self.status, "candidate": self.candidate, "reason": self.reason,
                "n_images": self.n_images, "n_pairs": self.n_pairs, "n_no_pred": self.n_no_pred,
                "n_count_mismatch": self.n_count_mismatch, "n_groups": self.n_groups,
                "n_image_groups": self.n_image_groups, "n_pair_groups": self.n_pair_groups,
                "samples": dict(self.samples)}


MET, UNMET = "충족", "미달"


# ---------------------------------------------------------------- 통계

def _centers(boxes: Sequence[Box]) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(p, 4) × (g, 4) → (p, g)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def mutual_pairs(pred: Sequence[Box], gold: Sequence[Box]) -> list[tuple[int, int]]:
    """서로의 최대 IoU 상대이고 IoU > 0 인 `(예측, 정답)` 쌍. 최대가 동률이면 쌍이 아니다."""
    if not pred or not gold:
        return []
    m = _iou(np.asarray(pred, dtype=float), np.asarray(gold, dtype=float))
    out = []
    for i in range(m.shape[0]):
        row = m[i]
        j = int(np.argmax(row))
        if row[j] <= 0 or np.count_nonzero(row == row[j]) > 1:
            continue
        col = m[:, j]
        if int(np.argmax(col)) != i or np.count_nonzero(col == col[j]) > 1:
            continue
        out.append((i, j))
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    vx = float(np.var(x))
    if vx == 0:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx)


def _trimmed_mean(v: np.ndarray, trim: float) -> float:
    if v.size == 0:
        return float("nan")
    s = np.sort(v)
    k = int(math.floor(trim * s.size))
    s = s[k:s.size - k] if s.size - 2 * k > 0 else s
    return float(s.mean())


@dataclass(frozen=True)
class _Prepared:
    """대상 장 · 쌍을 한 번 풀어 둔 것. 부트스트랩이 장 번호로 다시 고른다."""

    gc: np.ndarray            # (n, 2) 정답 이미지 중심
    pc: np.ndarray            # (n, 2) 예측 이미지 중심
    img_group: np.ndarray     # (n,) 묶음 번호
    pair_ratio: np.ndarray    # (q, 2) 쌍 중심 비
    pair_group: np.ndarray    # (q,) 묶음 번호
    n_groups: int


def _stats_of(gc, pc, pair_ratio, scale, trim) -> dict:
    ratio = pc / gc
    out = {"r": [float(np.median(ratio[:, a])) if ratio.shape[0] else float("nan") for a in (0, 1)],
           "rbar": [(_trimmed_mean(ratio[:, a], trim) if scale[a] != 1 else float("nan")) for a in (0, 1)],
           "s": [_slope(gc[:, a], pc[:, a]) for a in (0, 1)],
           "sx": [_slope(gc[:, 1], pc[:, 0]), _slope(gc[:, 0], pc[:, 1])],
           "m": [float(np.median(pair_ratio[:, a])) if pair_ratio.shape[0] else float("nan") for a in (0, 1)]}
    return out


@dataclass(frozen=True)
class FrameStats:
    """문턱 절차용 통계 — **산출물에 싣지 않는다.** 축마다 `[x, y]`."""

    n_images: int
    n_pairs: int
    n_no_pred: int
    n_count_mismatch: int
    n_groups: int
    size: tuple[int, int] | None
    size_mixed: bool
    value: dict
    se: dict
    n_image_groups: int = 0
    """대상 장이 든 묶음 수."""
    n_pair_groups: int = 0
    """쌍이 든 묶음 수."""
    n_valid: dict = field(default_factory=dict)
    """통계 · 축마다 유한한 재표집의 수."""


def frame_stats(images: Sequence[FrameImage], model_input_wh: tuple[int, int], *, n_boot: int, boot_seed: int,
                trim: float) -> FrameStats:
    defect = [im for im in images if im.gold]
    sizes = {(im.width, im.height) for im in defect}
    size = next(iter(sizes)) if len(sizes) == 1 else None
    target = [im for im in defect if len(im.pred) == len(im.gold)]
    n_no_pred = sum(1 for im in defect if not im.pred)
    n_mismatch = sum(1 for im in defect if im.pred and len(im.pred) != len(im.gold))
    groups = sorted({im.group_id for im in defect})
    gidx = {g: i for i, g in enumerate(groups)}
    if size is None:
        empty = {k: [float("nan")] * 2 for k in ("r", "rbar", "s", "sx", "m")}
        zero = {k: [0, 0] for k in empty}
        return FrameStats(len(target), 0, n_no_pred, n_mismatch, len(groups), None, len(sizes) > 1, empty, empty,
                          len({im.group_id for im in target}), 0, zero)
    scale = (model_input_wh[0] / size[0], model_input_wh[1] / size[1])
    gc = np.array([_centers(im.gold).mean(axis=0) for im in target]).reshape(-1, 2)
    pc = np.array([_centers(im.pred).mean(axis=0) for im in target]).reshape(-1, 2)
    img_group = np.array([gidx[im.group_id] for im in target], dtype=int)
    pr, pg = [], []
    for im in defect:
        if not im.pred:
            continue
        pcs, gcs = _centers(im.pred), _centers(im.gold)
        for i, j in mutual_pairs(im.pred, im.gold):
            pr.append(pcs[i] / gcs[j])
            pg.append(gidx[im.group_id])
    pair_ratio = np.array(pr, dtype=float).reshape(-1, 2)
    pair_group = np.array(pg, dtype=int)
    value = _stats_of(gc, pc, pair_ratio, scale, trim)
    # 묶음 단위 부트스트랩 — 묶음을 복원추출하고, 뽑힌 횟수만큼 그 묶음의 장과 쌍을 넣는다
    rng = np.random.default_rng(boot_seed)
    G = len(groups)
    img_by_g = [np.flatnonzero(img_group == g) for g in range(G)]
    pair_by_g = [np.flatnonzero(pair_group == g) for g in range(G)]
    boots = {k: [[], []] for k in value}
    for _ in range(n_boot):
        drawn = rng.integers(0, G, size=G) if G else np.array([], dtype=int)
        ii = np.concatenate([img_by_g[g] for g in drawn]) if G else np.array([], dtype=int)
        pp = np.concatenate([pair_by_g[g] for g in drawn]) if G else np.array([], dtype=int)
        st = _stats_of(gc[ii], pc[ii], pair_ratio[pp], scale, trim)
        for k in st:
            for a in (0, 1):
                boots[k][a].append(st[k][a])
    se = {k: [_sd(boots[k][a]) for a in (0, 1)] for k in boots}
    n_valid = {k: [int(np.isfinite(np.asarray(boots[k][a], dtype=float)).sum()) for a in (0, 1)] for k in boots}
    return FrameStats(len(target), int(pair_ratio.shape[0]), n_no_pred, n_mismatch, G, size, False, value, se,
                      len(set(img_group.tolist())), len(set(pair_group.tolist())), n_valid)


def _sd(xs: list[float]) -> float:
    v = np.asarray(xs, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if v.size >= 2 else float("inf")


def sample_status(st: FrameStats, rules: FrameRules) -> dict:
    """규칙의 표본 요건 넷의 충족 · 미달. 관측 수와 규칙의 하한만 맞댄다."""
    pairs = [("images", st.n_images, rules.n_img), ("image_groups", st.n_image_groups, rules.min_groups),
             ("pairs", st.n_pairs, rules.min_pairs), ("pair_groups", st.n_pair_groups, rules.min_pair_groups)]
    return {name: MET if seen >= need else UNMET for name, seen, need in pairs}


# ---------------------------------------------------------------- 서명과 결합

def _within(v: float, center: float, width: float) -> bool:
    return math.isfinite(v) and abs(v - center) <= width


def signature(st: FrameStats, rules: FrameRules, model_input_wh: tuple[int, int]) -> str:
    """불통과일 때 붙이는 이름 — 맞을 때만. 그 밖은 "후보 없음"(학습 부족 · 치우침 의심)."""
    W, H = st.size
    k = (model_input_wh[0] / W, model_input_wh[1] / H)
    n = (1000 / W, 1000 / H)
    v, d = st.value, rules.delta
    moved = [a for a in (0, 1) if k[a] != 1]
    if moved and all(_within(v["r"][a], k[a], d) and _within(v["s"][a], k[a], d) for a in (0, 1)):
        return CAND_RESIZE
    if all(_within(v["r"][a], n[a], d) and _within(v["s"][a], n[a], d) for a in (0, 1)):
        return CAND_NORM
    for a in moved:
        rb = v["rbar"][a]
        if _within(v["r"][a], 1, rules.w_r) and math.isfinite(rb) and not _within(rb, 1, rules.w_mix) \
                and (rb - 1) * (k[a] - 1) > 0:
            return CAND_MIX
    return CAND_NONE


def _axes(bad: list[int]) -> str:
    return "·".join(AXES[a] for a in bad)


def judge(images: Sequence[FrameImage], model_input_wh: tuple[int, int], rules: FrameRules) -> FrameVerdict:
    """프레임 진단 한 번. 문턱은 `rules` 에서만 온다."""
    st = frame_stats(images, model_input_wh, n_boot=rules.n_boot, boot_seed=rules.boot_seed, trim=rules.trim)
    return verdict_of(st, rules, model_input_wh)


def verdict_of(st: FrameStats, rules: FrameRules, model_input_wh: tuple[int, int]) -> FrameVerdict:
    def out(status, reason, candidate=None, step=None):
        return FrameVerdict(status, candidate, reason, st.n_images, st.n_pairs, st.n_no_pred,
                            st.n_count_mismatch, st.n_groups, st.n_image_groups, st.n_pair_groups,
                            sample_status(st, rules), step)

    if st.size_mixed:
        return out(INDETERMINATE, REASON_SIZE_MIXED, step=0)
    if st.size is None:
        return out(INDETERMINATE, REASON_NO_DEFECT, step=0)
    bad_rules = rules.problems((model_input_wh[0] / st.size[0], model_input_wh[1] / st.size[1]))
    if bad_rules:
        raise ValueError("규칙이 서지 않는다: " + " · ".join(bad_rules))
    v, se, q = st.value, st.se, rules.se_divisor
    moved = [a for a in (0, 1) if model_input_wh[a] != st.size[a]]
    need_valid = math.ceil(rules.min_valid_boot * rules.n_boot)

    def se_ok(key: str, a: int, width: float) -> bool:
        # 유한한 재표집이 모자라면 표준오차를 쓰지 않는다 — 남은 몇 개로 낸 값이 게이트를 지나지 않게
        return st.n_valid[key][a] >= need_valid and se[key][a] <= width / q

    # 1 표본 수 — 장과 독립 묶음
    if st.n_images < rules.n_img or st.n_image_groups < rules.min_groups:
        return out(INDETERMINATE, REASON_SAMPLE, step=1)
    # 2 기울기 — 하한 아래면 뒤바뀜인지 본다
    if not all(math.isfinite(v["s"][a]) and v["s"][a] >= rules.s_min for a in (0, 1)):
        if all(_within(v["sx"][a], 1, rules.w_s) and se_ok("sx", a, rules.w_s) for a in (0, 1)):
            return out(FAIL, CAND_SWAP, CAND_SWAP, step=2)
        return out(INDETERMINATE, REASON_NOT_TRACKING, step=2)
    # 3 이미지 수준 표준오차 게이트
    gate = [se_ok("r", a, rules.w_r) and se_ok("s", a, rules.w_s) for a in (0, 1)]
    gate += [se_ok("rbar", a, rules.w_mix) for a in moved]
    if not all(gate):
        return out(INDETERMINATE, REASON_SAMPLE, step=3)
    # 4 중심 비와 섞임 통계
    bad = [a for a in (0, 1) if not _within(v["r"][a], 1, rules.w_r)
           or (a in moved and not _within(v["rbar"][a], 1, rules.w_mix))]
    if bad:
        cand = signature(st, rules, model_input_wh)
        return out(FAIL, f"{_axes(bad)} · {cand}", cand, step=4)
    # 5 기울기의 폭
    if not all(_within(v["s"][a], 1, rules.w_s) for a in (0, 1)):
        return out(INDETERMINATE, REASON_WEAK_TRACKING, step=5)
    # 6 쌍의 표본 — 쌍과 독립 묶음
    if st.n_pairs < rules.min_pairs or st.n_pair_groups < rules.min_pair_groups \
            or not all(se_ok("m", a, rules.axis_ratio_width) for a in (0, 1)):
        return out(INDETERMINATE, REASON_SAMPLE, step=6)
    # 7 축 배율
    bad = [a for a in (0, 1) if not _within(v["m"][a], 1, rules.axis_ratio_width)]
    if bad:
        cand = signature(st, rules, model_input_wh)
        return out(FAIL, f"{_axes(bad)} · {cand}", cand, step=7)
    return out(PASS, "", step=None)
