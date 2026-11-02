"""평가 층화 표본의 묶음 재표집 — 본채점(엄격 · 표본 모드)의 구간이 쓰는 추첨. 25번 §3-1 · 07번 미니스펙 §3-1 · §15-10.

## 무엇을 하나

1. **층 안 묶음 복원추출.** 층 `h` 에서 뽑힌 묶음 `n_h` 개를 같은 수만큼 복원추출한다(다항 추첨). 추첨마다 묶음의 **중복도**를
   정수로 낸다(`draw_multiplicities` — 추첨 × 묶음). 이미지 가중은 그 중복도에 설계 가중 `G_h / n_h` 를 곱한 것이다
   (`weights_from`). 층 사이로 묶음을 옮기지 않는다 — 층마다 뽑는 수 `n_h` 와 가중이 추첨마다 그대로다. 25번 §3-1 의 크기 어림이
   쓴 방법과 같고, 표본이 층화 추출이라 따라 나오는 재표집이다(07번 §39-4-1 이 열어 둔 칸을 **표본 모드에서** 이렇게 닫는다 —
   35번 §1-3). **보정이 없는(naive) 꼴이다** — 층마다 분산이 `(n_h − 1) / n_h` 배로 작게 잡힌다(`n_h = 2` 이면 절반). Rao–Wu 꼴의
   보정은 등록 칸이 없어 쓰지 않았다. 산출이 층마다 `n_h` 를 싣는다.
2. **한 추첨을 모든 모델에.** 가중 행렬 `W`(추첨 × 이미지) 하나를 다섯 모델 · 모든 시드 · 대조선이 같이 쓴다 — 대응 표집(07번 §3-1).
3. **M1 을 행렬로.** 계수표의 클래스별 합을 `W @ tp` 로 낸다 — 추첨마다 매칭을 다시 풀지 않는다(계수표 설계, `box_f1`).
   평균 대상은 **점추정의 대상으로 고정**한다. 그 대상의 어느 클래스든 정답 지지량이 0 인 추첨은 잴 수 없다(`support`) —
   대상을 추첨마다 다시 정하면 클래스가 빠져 점수가 오른다(`box_f1.score_m1` 의 고정 대상 모드와 같은 계약).

## 결정성 · 검사

난수는 `numpy.random.default_rng(rng_seed)` 하나다 — 값은 등록(`ci.rng_seed`)에서만 온다. 층은 이름의 사전순, 층 안 묶음은 id 의
사전순으로 돈다. 같은 목록 · 곁 파일 · 매니페스트 묶음 · 등록 · NumPy 판이면 같은 중복도다 — 산출이 중복도 행렬의 sha256 과 NumPy 판을
싣는다(NumPy 는 난수열을 판 사이에 보증하지 않는다). 중복도마다 층의 합이 `n_h` 인지 실행 때 본다(`multiplicity_problems`).
**항등 추첨**(중복도 전부 1)을 재표집과 **같은 길**(`weights_from`)로 가중에 옮기면 설계 가중과 같아야 하고, 그 M1 은 점추정과
같아야 한다(`identity_problems`) — 중복도 → 열 대응과 `G_h / n_h` 곱이 틀리면 여기서 드러난다.

**층은 곁 파일에서만 온다** — 매니페스트의 열을 읽지 않는다(진입 관문의 매니페스트 판독은 네 열뿐이다). 묶음은 매니페스트의
`group_id` 다(표본의 추첨 단위, 25번 §2-1) — 재표집 단위 `CLUSTER_UNIT` 이고, 등록의 `cluster_unit` 이 이 값이어야 한다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from evaluation.metrics.box_f1 import CountTable, score_m1
from evaluation.sample_weights import SideRow

IDENTITY_TOL = 1e-12
"""항등 추첨의 M1 과 점추정의 차의 허용 — 합의 순서만 다른 두 부동소수 계산이다."""
CLUSTER_UNIT = "group"
"""재표집 단위 — 표본의 추첨 단위(매니페스트 `group_id`). 등록 · 평가 등록 설정의 `cluster_unit` 이 이 값이어야 한다."""


class SampleDesignError(ValueError):
    """표본의 묶음 · 층이 곁 파일과 맞지 않는다. 사유에는 수만 싣는다(id 를 싣지 않는다)."""


def strata_groups(image_ids: Sequence[str], group_of: Mapping[str, str],
                  side: Mapping[str, SideRow]) -> dict[str, list[str]]:
    """층 → 뽑힌 묶음(id 사전순). 층 이름의 사전순으로 돌려준다.

    거부: 곁 파일에 없는 이미지 · 매니페스트에 묶음이 없는 이미지 · **두 층에 걸친 묶음** · 층의 뽑힌 묶음 수가 곁 파일의 `n_h` 와
    다른 층. 마지막이 어긋나면 가중 `G_h / n_h` 와 재표집의 뽑는 수가 서로 다른 것을 센다.
    """
    stratum_of_group: dict[str, str] = {}
    n_h_of: dict[str, set[int]] = {}
    for i in image_ids:
        row = side.get(i)
        if row is None:
            raise SampleDesignError("곁 파일에 없는 이미지가 있다")
        g = group_of.get(i)
        if g is None:
            raise SampleDesignError("매니페스트에 묶음이 없는 이미지가 있다")
        if stratum_of_group.setdefault(g, row.stratum) != row.stratum:
            raise SampleDesignError("한 묶음이 두 층에 걸친다 — 층은 묶음 단위다(25번 §2-2)")
        n_h_of.setdefault(row.stratum, set()).add(row.n_h)
    by: dict[str, list[str]] = {}
    for g, h in stratum_of_group.items():
        by.setdefault(h, []).append(g)
    out: dict[str, list[str]] = {}
    for h in sorted(by):
        gs = sorted(by[h])
        if n_h_of[h] != {len(gs)}:
            raise SampleDesignError(f"층 하나의 뽑힌 묶음 수 {len(gs)} 가 곁 파일의 n_h {sorted(n_h_of[h])} 와 다르다")
        out[h] = gs
    return out


def design_weights(image_ids: Sequence[str], side: Mapping[str, SideRow]) -> np.ndarray:
    """이미지 순서대로 설계 가중 `G_h / n_h` — 항등 추첨의 가중이다."""
    return np.array([float(side[i].weight) for i in image_ids], dtype=np.float64)


def draw_multiplicities(groups: Mapping[str, Sequence[str]], *, n_resamples: int, rng_seed: int) -> np.ndarray:
    """(추첨 × 묶음) 정수 중복도. 열은 `groups` 의 층 순서 · 층 안 묶음 순서다. 층마다 다항 추첨 한 번."""
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError(f"재표집 수는 1 이상의 정수다(등록 ci.n_resamples): {n_resamples!r}")
    if type(rng_seed) is not int or rng_seed < 0:
        raise ValueError(f"난수 시드는 0 이상의 정수다(등록 ci.rng_seed): {rng_seed!r}")
    rng = np.random.default_rng(rng_seed)
    blocks = [rng.multinomial(len(gs), np.full(len(gs), 1.0 / len(gs)), size=n_resamples) for gs in groups.values()]
    return np.concatenate(blocks, axis=1).astype(np.int64)


def multiplicity_problems(M: np.ndarray, groups: Mapping[str, Sequence[str]]) -> list[str]:
    """중복도의 불변식 — 0 이상의 정수이고, 추첨마다 층의 합이 그 층의 뽑힌 묶음 수다."""
    out = []
    if M.ndim != 2 or M.shape[1] != sum(len(gs) for gs in groups.values()):
        return [f"중복도 행렬의 모양 {M.shape} 가 묶음 수와 맞지 않는다"]
    if not np.issubdtype(M.dtype, np.integer) or (M < 0).any():
        out.append("중복도가 0 이상의 정수가 아니다")
    start = 0
    for h, gs in groups.items():
        sums = M[:, start:start + len(gs)].sum(axis=1)
        if (sums != len(gs)).any():
            out.append(f"층 {h} 의 중복도 합이 n_h {len(gs)} 가 아닌 추첨이 {int((sums != len(gs)).sum())}개다")
        start += len(gs)
    return out


def multiplicity_sha256(M: np.ndarray) -> str:
    """중복도 행렬의 지문 — 리틀 엔디언 int64 의 C 순서 바이트. 같은 등록으로 같은 추첨을 냈는지 판이 바뀐 뒤에도 맞대게."""
    return hashlib.sha256(np.ascontiguousarray(M, dtype="<i8").tobytes()).hexdigest()


def weights_from(M: np.ndarray, image_ids: Sequence[str], group_of: Mapping[str, str], side: Mapping[str, SideRow],
                 groups: Mapping[str, Sequence[str]]) -> np.ndarray:
    """중복도 → (추첨 × 이미지) 가중. 열은 `image_ids` 의 순서다 — 이미지 가중 = 그 묶음의 중복도 × `G_h / n_h`."""
    order = [g for gs in groups.values() for g in gs]
    if M.ndim != 2 or M.shape[1] != len(order):
        raise ValueError(f"중복도 행렬의 열 {M.shape} 이 묶음 수 {len(order)} 와 다르다")
    col = {g: k for k, g in enumerate(order)}
    idx = np.array([col[group_of[i]] for i in image_ids], dtype=np.int64)
    return M[:, idx].astype(np.float64) * design_weights(image_ids, side)[None, :]


def draw_weights(image_ids: Sequence[str], group_of: Mapping[str, str], side: Mapping[str, SideRow], *,
                 n_resamples: int, rng_seed: int) -> np.ndarray:
    """(추첨 × 이미지) 가중 — `strata_groups` → `draw_multiplicities` → `weights_from`."""
    groups = strata_groups(image_ids, group_of, side)
    return weights_from(draw_multiplicities(groups, n_resamples=n_resamples, rng_seed=rng_seed),
                        image_ids, group_of, side, groups)


def m1_draws(table: CountTable, W: np.ndarray, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """추첨마다 고정 대상 macro F1 과 그 추첨이 잴 수 있는가(`support`).

    `classes` 는 점추정의 평균 대상(`classes_in_macro`)이다. 잴 수 없는 추첨의 값은 `NaN` 이다 — 0 이나 다른 수로 채우지 않는다.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != len(table.image_ids):
        raise ValueError(f"가중 행렬의 모양 {W.shape} 가 계수표의 이미지 수 {len(table.image_ids)} 와 맞지 않는다")
    if not np.isfinite(W).all() or (W < 0).any():
        raise ValueError("가중 행렬은 유한한 0 이상의 수다")
    if not classes:
        raise ValueError("평균 대상이 비었다 — 점추정이 정의되지 않은 표다")
    idx = [table.classes.index(c) for c in classes]
    tp = W @ table.tp[:, idx].astype(np.float64)
    fp = W @ table.fp[:, idx].astype(np.float64)
    fn = W @ table.fn[:, idx].astype(np.float64)
    ok = ((tp + fn) > 0).all(axis=1)
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1.0), np.nan)
    macro = np.full(W.shape[0], np.nan)
    macro[ok] = f1[ok].sum(axis=1) / len(idx)
    return macro, ok


def identity_problems(table: CountTable, side: Mapping[str, SideRow], group_of: Mapping[str, str],
                      classes: Sequence[str]) -> list[str]:
    """항등 추첨 — 중복도 1 을 **재표집과 같은 길**(`weights_from`)로 가중에 옮긴다. 그 가중이 설계 가중과 같고, 그 위의 M1 이
    점추정(`score_m1` 에 설계 가중)과 같은가. 어긋나면 사유 — 재표집의 배관(묶음 → 열 대응 · 가중 곱 · 행렬 M1)이 틀린 것이다."""
    groups = strata_groups(table.image_ids, group_of, side)
    ones = np.ones((1, sum(len(gs) for gs in groups.values())), dtype=np.int64)
    w1 = weights_from(ones, table.image_ids, group_of, side, groups)
    w = design_weights(table.image_ids, side)
    if not np.array_equal(w1[0], w):
        return ["항등 중복도의 가중이 설계 가중과 다르다 — 묶음 → 열 대응이나 G_h/n_h 곱이 틀렸다"]
    point = score_m1(table, w).macro_f1
    got, ok = m1_draws(table, w1, classes)
    if point is None:
        return [] if not ok[0] else ["점추정은 미정의인데 항등 추첨은 값이 있다"]
    if not ok[0]:
        return ["점추정은 값이 있는데 항등 추첨이 지지량 미정의다"]
    if abs(float(got[0]) - point) > IDENTITY_TOL:
        return [f"항등 추첨의 M1 이 점추정과 {abs(float(got[0]) - point):.3e} 만큼 다르다"]
    return []
