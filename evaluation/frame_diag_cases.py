"""프레임 진단의 합성 사례 — 07번 계약 §30-2 · §31-2 · §32-5 의 열셋(가~파)과 사다리.

**모델 출력은 어느 것도 쓰지 않는다**(§32-5 M-7). 정답은 요약값으로 만든다 — 실제 좌표를 커밋하지 않는다.
예측은 정답에서 사례의 식으로 낸다. 기대 판정은 `frame_diag_expected-20261001-1.json` 에 **문턱보다 먼저** 적었다(§30-2 문턱 절차 2).

| 요약값 | 출처 |
|---|---|
| 중심 x 640±102 · y 359±28 | 07번 §14-3 의 1(train+val 정답 1개 이미지 596장의 읽기 전용 모의) |
| 박스 높이 | 파일럿 페어 라벨(train/val 만)의 높이 분포 — 4,560 개 가운데 0~20 px 387 · 20~40 px 1,589 · 40~80 px 1,736 · 80 px 이상 848, 중앙값 44 · 최소 4(`docs/dev_log/2026-09-20-설계개정/11_계측셋_C.md` §2-3). 칸 안은 균등. **가정** 둘 — 맨 위 칸의 상한(160 px), 폭은 높이와 같은 분포에서 따로 뽑는다(폭 분포는 기록에 없다) |
| 장마다 박스 수 | **가정** — 1 개 60 % · 2 개 30 % · 3 개 10 % |
| 묶음 | **가정** — 연속한 1~3 장 |
| 잡음 사다리 | 중심 잡음의 표준편차 = 박스 크기 × 계단(규칙 파일에 적는다) |

이미지는 1280×720, 프로세서 입력은 등록의 `model_input_wh`(지금 1280×704) — 배율 `k = (1, 704/720)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from evaluation.frame_diag import FrameImage

W, H = 1280, 720
MODEL_INPUT_WH = (1280, 704)


@dataclass(frozen=True)
class Summary:
    cx_mean: float = 640.0
    cx_sd: float = 102.0
    cy_mean: float = 359.0
    cy_sd: float = 28.0
    box_bins: tuple[tuple[float, float, int], ...] = ((4.0, 20.0, 387), (20.0, 40.0, 1589), (40.0, 80.0, 1736),
                                                      (80.0, 160.0, 848))
    """(하한, 상한, 개수) — 높이 분포의 칸. 출처는 모듈 머리말."""
    boxes_per_image: tuple[float, float, float] = (0.6, 0.3, 0.1)


SUMMARY = Summary()


def gold_set(n_images: int, seed: int, summary: Summary = SUMMARY, *, box_range: tuple[float, float] | None = None,
             single_box: bool = False) -> list[FrameImage]:
    """합성 정답 — 예측은 비워 둔다(사례가 채운다)."""
    rng = np.random.default_rng(seed)
    bins = np.array([(a, b) for a, b, _ in summary.box_bins])
    weights = np.array([c for _, _, c in summary.box_bins], dtype=float)
    weights /= weights.sum()

    def side() -> float:
        if box_range is not None:
            return float(rng.uniform(*box_range))
        a, b = bins[rng.choice(len(bins), p=weights)]
        return float(rng.uniform(a, b))
    out, g, left = [], 0, 0
    for i in range(n_images):
        if left == 0:
            g, left = g + 1, int(rng.integers(1, 4))
        left -= 1
        k = 1 if single_box else int(rng.choice([1, 2, 3], p=summary.boxes_per_image))
        boxes = []
        for _ in range(k):
            bw, bh = side(), side()
            cx = float(np.clip(rng.normal(summary.cx_mean, summary.cx_sd), bw / 2 + 1, W - bw / 2 - 1))
            cy = float(np.clip(rng.normal(summary.cy_mean, summary.cy_sd), bh / 2 + 1, H - bh / 2 - 1))
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        out.append(FrameImage(f"syn{i:05d}", f"g{g:05d}", W, H, (), tuple(boxes)))
    return out


# ---------------------------------------------------------------- 예측의 식

def _scale(b, kx, ky):
    return (b[0] * kx, b[1] * ky, b[2] * kx, b[3] * ky)


def _shift(b, dx, dy):
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _toward(b, slope, mean):
    """중심을 평균 쪽으로 — 예측 중심 = slope·정답 중심 + (1 − slope)·평균. 크기는 그대로."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    nx, ny = slope * cx + (1 - slope) * mean[0], slope * cy + (1 - slope) * mean[1]
    return _shift(b, nx - cx, ny - cy)


def _mean_center(golds) -> tuple[float, float]:
    c = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for im in golds for b in im.gold])
    return float(c[:, 0].mean()), float(c[:, 1].mean())


def predict(golds: list[FrameImage], fn) -> list[FrameImage]:
    return [replace(im, pred=tuple(fn(b) for b in im.gold)) for im in golds]


def identity(golds):
    return predict(golds, lambda b: b)


def resize_y(golds, wh=MODEL_INPUT_WH):
    return predict(golds, lambda b: _scale(b, wh[0] / W, wh[1] / H))


def normalized(golds):
    return predict(golds, lambda b: _scale(b, 1000 / W, 1000 / H))


def constant(golds):
    m = _mean_center(golds)
    side = 48.0
    return predict(golds, lambda b: (m[0] - side / 2, m[1] - side / 2, m[0] + side / 2, m[1] + side / 2))


def toward_mean(golds, slope):
    m = _mean_center(golds)
    return predict(golds, lambda b: _toward(b, slope, m))


def noisy(golds, step, seed):
    """중심에 잡음 — 표준편차 = 그 박스의 (폭+높이)/2 × step. 크기는 그대로."""
    rng = np.random.default_rng(seed)

    def f(b):
        s = step * ((b[2] - b[0]) + (b[3] - b[1])) / 2
        return _shift(b, float(rng.normal(0, s)), float(rng.normal(0, s)))
    return predict(golds, f)


def y_bias(golds, px):
    return predict(golds, lambda b: _shift(b, 0.0, px))


def mixed_resize(golds, fraction, seed, wh=MODEL_INPUT_WH):
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(len(golds), size=int(round(fraction * len(golds))), replace=False).tolist())
    return [replace(im, pred=tuple(_scale(b, wh[0] / W, wh[1] / H) if i in pick else b for b in im.gold))
            for i, im in enumerate(golds)]


def count_mismatch(golds, seed, *, no_pred_fraction=0.5):
    """장마다 예측 박스 하나만, 장의 일부는 예측 없음."""
    rng = np.random.default_rng(seed)
    out = []
    for im in golds:
        if rng.random() < no_pred_fraction:
            out.append(replace(im, pred=()))
        else:
            out.append(replace(im, pred=(im.gold[0],)))
    return out


def swapped(golds):
    return predict(golds, lambda b: (b[1], b[0], b[3], b[2]))


# ---------------------------------------------------------------- 사례 — 이름 · 식 · 크기

@dataclass(frozen=True)
class Case:
    id: str
    label: str
    n_images: int
    build: object
    """`(정답 목록) → 예측을 채운 목록`."""
    box_range: tuple[float, float] | None = None
    single_box: bool = False
    seed: int = 20261001


def cases(n: int = 800) -> list[Case]:
    """기본 크기 `n` 의 열셋과 사다리. 마 와 쌍 수 경계는 크기를 따로 둔다."""
    out = [
        Case("가", "원본 프레임", n, identity),
        Case("나", "리사이즈 프레임", n, resize_y),
        Case("다", "정규화 좌표", n, normalized),
        Case("라", "평균 위치만 내는 출력", n, constant),
        Case("마", "짝 부족 — 대상 장은 문턱 이상, 쌍은 100 미만", 99, identity, single_box=True),
        # 99 장 — 쌍이 100 미만인 가장 큰 크기. 이 사례의 크기가 `N_img` 의 상한을 정하지 않게 한다(문턱 절차)
        Case("바", "덜 배운 모델 — 기울기 0.5", n, lambda g: toward_mean(g, 0.5)),
        Case("아", "박스가 작은 리사이즈", n, resize_y, box_range=(8.0, 16.0)),
        Case("자", "y 가산 치우침 −8 px", n, lambda g: y_bias(g, -8.0)),
        Case("차", "중심 쪽 균일 수축 — 기울기 0.8", n, lambda g: toward_mean(g, 0.8)),
        Case("카20", "섞인 프레임 — 20 % 리사이즈", n, lambda g: mixed_resize(g, 0.2, 7)),
        Case("카40", "섞인 프레임 — 40 % 리사이즈", n, lambda g: mixed_resize(g, 0.4, 7)),
        Case("타", "박스 수 불일치 — 박스 하나만 · 절반은 예측 없음", n, lambda g: count_mismatch(g, 11)),
        Case("파", "x·y 뒤바뀜", n, swapped),
    ]
    out += [Case(f"사{step}", f"잡음이 큰 옳은 프레임 — 중심 잡음 = 박스 크기 × {step}", n,
                 lambda g, step=step: noisy(g, step, 13)) for step in NOISE_LADDER]
    out += [Case(f"기울기{s}", f"기울기 사다리 {s}", n, lambda g, s=s: toward_mean(g, s)) for s in (0.3, 0.7, 0.9)]
    out += [Case(f"자{px}", f"y 가산 치우침 {px} px", n, lambda g, px=px: y_bias(g, float(px))) for px in (-4, -12)]
    out += [Case("카10", "섞인 프레임 — 10 % 리사이즈", n, lambda g: mixed_resize(g, 0.1, 7))]
    out += [Case(f"가박스{lo}", f"원본 프레임 — 박스 {lo}~{hi} px", n, identity, box_range=(float(lo), float(hi)))
            for lo, hi in ((8, 16), (16, 32))]
    out += [Case(f"나박스{lo}", f"리사이즈 — 박스 {lo}~{hi} px", n, resize_y, box_range=(float(lo), float(hi)))
            for lo, hi in ((16, 32),)]
    out += [Case(f"쌍{k}", f"원본 프레임 — 한 장 한 박스 {k} 장(쌍 {k})", k, identity, single_box=True) for k in (99, 100)]
    out += [Case("타30", "박스 수 불일치 — 30 % 예측 없음", n, lambda g: count_mismatch(g, 11, no_pred_fraction=0.3))]
    return out


NOISE_LADDER = (0.05, 0.1, 0.2, 0.4)
"""사 의 잡음 계단 — 중심 잡음의 표준편차를 박스 크기(폭과 높이의 평균)에 곱하는 수. **가정**이다(§32-5 M-7)."""


def build(case: Case, summary: Summary = SUMMARY) -> list[FrameImage]:
    golds = gold_set(case.n_images, case.seed, summary, box_range=case.box_range, single_box=case.single_box)
    return case.build(golds)
