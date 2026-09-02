"""content-free 기준선 재산출 — 동결본 정본값. 68번 §5-(가) → 74번 감사 A-1·A-2 반영.

content-free 예측기란 **이미지 화소를 보지 않고 취득 메타데이터만으로** 예측하는 규칙이다.
이 계열의 최댓값이 게이트 선이 된다 — 자명하한 하나로는 지름길 구간이 걸러지지 않는다는
것이 66번의 소득이었다.

    uv run python scripts/recompute_baselines.py

## 74번 감사가 고치라고 한 두 가지

**A-1 (규칙 공간이 좁았다).** 이전 판은 출처·재질 두 축만 봤고 그 최댓값 0.3465 를
"content-free 계열의 최댓값"이라 진술했다. 사실이 아니다. `image_id`(촬영 순서) 축 하나만
더 세워도 값이 올라간다. 이 판은 **다섯 축의 모든 조합**을 훑는다.

**A-2 (게이트 상수가 평가셋 정답으로 적합됐다).** 이전 판의 `best_content_free()` 는
평가셋 `gold` 로 셀별 최적 코드 집합을 골랐다. 시험 문제의 답을 보고 커트라인을 정한 것이다.
이 판은 **규칙을 train+val 에서만 적합하고 평가셋에서는 채점만 한다.** 평가셋에서 적합한
값(오라클 상한)은 따로 내되 키에 `fit_eval` 을 박고 **게이트 판정에 쓰지 않는다.**

## 키 이름 규약

    <계열>__sel_<족선택 모집단>__fit_<규칙적합 모집단>__score_<채점 모집단>

`sel_val__fit_trainval` 만 게이트 재료다. `sel_eval`·`fit_eval` 이 붙은 값은 평가셋 유도이며
참고값이다. 상수 박스처럼 적합이 없는 값은 `derived_from` 문자열로 유도원을 밝힌다.
**이름만 보고 유도원이 판별되지 않으면 A-2 가 재발한다.**
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.manifest_io import load_snapshot
from evaluation.schema import Defect, PredictionRecord
from evaluation.score import score_records

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "data/interim/manifest_v1"
GATE_TOLERANCE = 0.005


# ======================================================================================
# 정답·규칙
# ======================================================================================


def build_gold(snap, image_ids: Sequence[str]):
    """모집단의 정답. 정상 이미지는 빈 집합으로 들어간다(오탐 원천)."""
    keep = set(image_ids)
    ann = snap.annotations[snap.annotations["image_id"].isin(keep)]
    codes: dict[str, list[str]] = {i: [] for i in image_ids}
    boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {
        i: [] for i in image_ids
    }
    n_nobox = 0
    for r in ann.itertuples():
        codes[r.image_id].append(str(r.iso_code))
        vals = (r.bbox_x1_px, r.bbox_y1_px, r.bbox_x2_px, r.bbox_y2_px)
        if any(pd.isna(v) for v in vals):
            n_nobox += 1        # N1(위치 없음). 코드는 살리고 박스 축에서만 빠진다
            continue
        boxes[r.image_id].append((str(r.iso_code), tuple(float(v) for v in vals)))
    if n_nobox:
        print(f"  bbox 결측 어노테이션 {n_nobox:,}건 — 코드 축에는 남기고 위치 축에서만 제외")
    return {k: sorted(set(v)) for k, v in codes.items()}, boxes


def records_from_rule(
    image_ids: Sequence[str], cell_of: Mapping[str, str], subsets: Mapping[str, tuple[str, ...]],
    box: tuple[float, float, float, float] | None = None,
    fallback: tuple[str, ...] = (),
) -> list[PredictionRecord]:
    """셀 → 코드 집합 규칙을 계약 #4 레코드로 만든다. 화소를 보지 않는다.

    적합 모집단에 없던 셀은 `fallback`(축 없는 전역 규칙)으로 떨어진다. 규칙을 train+val
    에서 적합하고 평가셋에서 채점하므로 미관측 셀이 실제로 생긴다.
    """
    out = []
    for img in image_ids:
        codes = subsets.get(cell_of[img], fallback)
        defects = [
            Defect(iso_code=c, bbox_px=list(box) if box else None,
                   score=1.0 if box else None)
            for c in codes
        ]
        out.append(PredictionRecord(
            schema_version="1.3", image_id=img, cell="sep_central", seed=20260825,
            defects=defects, verdict="판정불가", cited_clauses=[], parse_ok=True,
        ))
    return out


def analytic_macro_f1(
    cells: Sequence[str], classes: Sequence[str],
    n_by_cell: Mapping[str, int], pos_by_cell: Mapping[tuple[str, str], int],
    chosen: Mapping[str, tuple[str, ...]],
    fallback: tuple[str, ...] = (),
) -> float:
    """셀별 코드 집합이 주어졌을 때의 Macro-F1 해석값 (채점기 검산용).

    클래스 c: TP = Σ_{셀에 c 포함} n_{셀,c}, 예측양성 = Σ_{셀에 c 포함} N_셀,
    실양성 = Σ_셀 n_{셀,c}. F1 = 2TP / (예측양성 + 실양성).
    """
    per = []
    for c in classes:
        actual = sum(pos_by_cell.get((g, c), 0) for g in cells)
        if actual == 0:
            continue                       # GT 0 인 클래스는 macro 에서 빠진다(채점기 계약)
        sel = [g for g in cells if c in chosen.get(g, fallback)]
        tp = sum(pos_by_cell.get((g, c), 0) for g in sel)
        pp = sum(n_by_cell[g] for g in sel)
        per.append(0.0 if tp == 0 else 2 * tp / (pp + actual))
    return float(np.mean(per)) if per else 0.0


def best_content_free(
    cells: Sequence[str], classes: Sequence[str],
    n_by_cell: Mapping[str, int], pos_by_cell: Mapping[tuple[str, str], int],
) -> dict[str, tuple[str, ...]]:
    """셀 축만 쓰는 예측기의 **정확한** 최적해. O(|셀| log |셀|).

    Macro-F1 은 클래스별 F1 의 평균이고 클래스 c 의 F1 은 "어느 셀에서 c 를 주장하는가"
    에만 의존한다. 따라서 클래스끼리 독립이다. 클래스 c 에서 실양성 A 가 고정이므로

        F1(S) = 2·Σ_{g∈S} tp_g / (Σ_{g∈S} n_g + A)

    를 부분집합 S 에 대해 최대화하면 된다. 이건 비율 최대화라 Dinkelbach 로 풀린다 —
    최적 λ* 에서 S* = {g : tp_g − λ*·n_g > 0} = {g : tp_g/n_g > λ*} 이므로 **최적해는
    항상 tp_g/n_g 내림차순 정렬의 접두사**다. 접두사 |셀|+1 개만 보면 전역 최적이다.

    이전 판은 부분집합 2^|셀| 를 전수 열거했다(축 2개·6셀에서만 가능). 축을 다섯으로 넓히면
    셀이 수백 개라 열거가 불가능하다. 접두사 정리로 바꾸면서 소규모 사례에서 열거 결과와
    일치함을 `--verify-enumeration` 으로 확인한다.
    """
    include: dict[str, set[str]] = {g: set() for g in cells}
    for c in classes:
        actual = sum(pos_by_cell.get((g, c), 0) for g in cells)
        if actual == 0:
            continue
        order = sorted(
            cells,
            key=lambda g: (-(pos_by_cell.get((g, c), 0) / max(n_by_cell[g], 1)), g),
        )
        tp = pp = 0
        best_f1, best_k = 0.0, 0
        for k, g in enumerate(order, start=1):
            tp += pos_by_cell.get((g, c), 0)
            pp += n_by_cell[g]
            f1 = 0.0 if tp == 0 else 2 * tp / (pp + actual)
            if f1 > best_f1 + 1e-15:
                best_f1, best_k = f1, k
        for g in order[:best_k]:
            include[g].add(c)
    return {g: tuple(sorted(include[g])) for g in cells}


def brute_force_content_free(
    cells: Sequence[str], classes: Sequence[str],
    n_by_cell: Mapping[str, int], pos_by_cell: Mapping[tuple[str, str], int],
) -> dict[str, tuple[str, ...]]:
    """부분집합 전수 열거. **접두사 정리의 검산 전용**이라 소규모 셀에서만 부른다."""
    import itertools

    include: dict[str, set[str]] = {g: set() for g in cells}
    for c in classes:
        actual = sum(pos_by_cell.get((g, c), 0) for g in cells)
        if actual == 0:
            continue
        best_f1, best_sel = 0.0, ()
        for k in range(len(cells) + 1):
            for sel in itertools.combinations(cells, k):
                tp = sum(pos_by_cell.get((g, c), 0) for g in sel)
                pp = sum(n_by_cell[g] for g in sel)
                f1 = 0.0 if tp == 0 else 2 * tp / (pp + actual)
                if f1 > best_f1:
                    best_f1, best_sel = f1, sel
        for g in best_sel:
            include[g].add(c)
    return {g: tuple(sorted(include[g])) for g in cells}


def cell_stats(rows: pd.DataFrame, gold: Mapping[str, Sequence[str]], cell_col: str,
               classes: Sequence[str]):
    n_by_cell: dict[str, int] = {}
    pos_by_cell: dict[tuple[str, str], int] = {}
    for cell, part in rows.groupby(cell_col, observed=True):
        n_by_cell[str(cell)] = len(part)
        for c in classes:
            pos_by_cell[(str(cell), c)] = sum(
                1 for i in part["image_id"] if c in gold[i])
    return n_by_cell, pos_by_cell


# ======================================================================================
# 벡터 경로 — 규칙 족을 수백 개 훑으려면 셀별 파이썬 루프로는 끝나지 않는다
#
# 위의 dict 판(`cell_stats`·`analytic_macro_f1`·`best_content_free`)은 **읽기 쉬운 정의**로
# 남겨 두고 검산에만 쓴다. 탐색은 아래 정수 경로로 돈다. 두 경로가 같은 값을 내는지는
# `--verify-enumeration` 에서 함께 확인한다.
#
# 셀 이름을 문자열로 붙이면 factorize 비용이 족마다 붙는다. 축마다 정수 코드를 한 번
# 만들어 두고 자릿수 조합으로 셀 id 를 만들면 **두 모집단에서 같은 정수가 같은 셀**이 되어
# 대조도 공짜가 된다.
# ======================================================================================


class Population:
    """한 모집단의 축 코드·클래스 지시행렬을 미리 만들어 둔다."""

    def __init__(self, rows: pd.DataFrame, gold: Mapping[str, Sequence[str]],
                 classes: Sequence[str], cuts: Mapping[str, np.ndarray]):
        self.rows = rows
        self.ids = list(rows["image_id"])
        self.classes = list(classes)
        self.n = len(rows)
        self.cls = np.zeros((self.n, len(classes)), dtype=np.float64)
        for j, c in enumerate(classes):
            self.cls[:, j] = [1.0 if c in gold[i] else 0.0 for i in self.ids]
        self.code: dict[tuple[str, int], np.ndarray] = {}
        self.card: dict[tuple[str, int], int] = {}
        prov_map = {"N-crop": 0, "N-tile": 1, "N-band": 2}
        self.code[("prov", 3)] = rows["prov"].map(prov_map).fillna(2).to_numpy(dtype=np.int64)
        self.card[("prov", 3)] = 3
        self.code[("material", 2)] = (rows["material"].astype(str) == "AL").to_numpy(np.int64)
        self.card[("material", 2)] = 2
        for src, ax in (("id_num", "idq"), ("group_size", "gsz"), ("file_bytes", "fsz")):
            vals = rows[src].to_numpy(dtype=float)
            for k in (g for g in GRANULARITY[ax] if g is not None):
                self.code[(ax, k)] = np.searchsorted(cuts[f"{ax}__{k}"], vals, side="right")
                self.card[(ax, k)] = k

    def cell_ids(self, family: tuple[tuple[str, int], ...]) -> tuple[np.ndarray, int]:
        out = np.zeros(self.n, dtype=np.int64)
        size = 1
        for ax, k in family:
            key = (ax, int(k))
            out = out * self.card[key] + self.code[key]
            size *= self.card[key]
        return out, max(size, 1)

    def counts(self, family) -> tuple[np.ndarray, np.ndarray]:
        """(셀별 장수 [S], 셀×클래스 양성 수 [S, C]). 빈 셀은 0 으로 남는다."""
        ids, size = self.cell_ids(family)
        n = np.bincount(ids, minlength=size).astype(np.int64)
        pos = np.empty((size, len(self.classes)), dtype=np.int64)
        for j in range(len(self.classes)):
            pos[:, j] = np.bincount(ids, weights=self.cls[:, j], minlength=size).astype(np.int64)
        return n, pos


def fit_rule(n: np.ndarray, pos: np.ndarray, min_support: int) -> np.ndarray:
    """접두사 최적해를 셀×클래스 불리언 마스크로 낸다. `best_content_free` 의 벡터판."""
    n_cells, n_cls = pos.shape
    sel = np.zeros((n_cells, n_cls), dtype=bool)
    usable = n >= min_support
    for j in range(n_cls):
        actual = int(pos[:, j].sum())
        if actual == 0:
            continue
        idx = np.flatnonzero(usable)
        if idx.size == 0:
            continue
        ratio = pos[idx, j] / np.maximum(n[idx], 1)
        order = idx[np.lexsort((idx, -ratio))]           # 동률은 셀 id 로 결정적으로 깬다
        tp = np.cumsum(pos[order, j])
        pp = np.cumsum(n[order])
        f1 = np.where(tp == 0, 0.0, 2 * tp / (pp + actual))
        k = int(np.argmax(f1))
        if f1[k] > 0:
            sel[order[: k + 1], j] = True
    return sel


def apply_rule(n: np.ndarray, pos: np.ndarray, sel: np.ndarray) -> float:
    """주어진 마스크를 채점 모집단의 (n, pos) 에 적용한 Macro-F1."""
    per = []
    for j in range(pos.shape[1]):
        actual = int(pos[:, j].sum())
        if actual == 0:
            continue                       # GT 0 인 클래스는 macro 에서 빠진다(채점기 계약)
        mask = sel[:, j]
        tp = int(pos[mask, j].sum())
        pp = int(n[mask].sum())
        per.append(0.0 if tp == 0 else 2 * tp / (pp + actual))
    return float(np.mean(per)) if per else 0.0


def rule_to_dict(sel: np.ndarray, n: np.ndarray, classes: Sequence[str],
                 family, pop: Population) -> dict[str, tuple[str, ...]]:
    """셀 정수 id 를 사람이 읽는 이름으로 되돌린다. 보고서·검산용."""
    out: dict[str, tuple[str, ...]] = {}
    for cid in np.flatnonzero(n > 0):
        out[cell_label(int(cid), family, pop)] = tuple(
            c for j, c in enumerate(classes) if sel[cid, j])
    return out


#: 정수 코드를 사람이 읽는 축값으로 되돌린다. 분위 축은 구간 번호가 곧 의미라 그대로 쓴다.
CODE_LABEL: dict[str, dict[int, str]] = {
    "prov": {0: "N-crop", 1: "N-tile", 2: "N-band"},
    "material": {0: "ST", 1: "AL"},
}


def cell_label(cid: int, family, pop: Population) -> str:
    """셀 정수 id → `prov=N-crop|idq16=3` 문자열.

    라벨은 정수 코드와 족만으로 정해지므로 **모집단이 달라도 같은 셀이면 같은 이름**이다.
    train+val 에서 만든 규칙을 eval 셀에 붙이는 일이 이름으로 성립한다.
    """
    parts: list[str] = []
    rem = cid
    for ax, k in reversed(family):
        card = pop.card[(ax, int(k))]
        v = rem % card
        rem //= card
        parts.append(f"{ax}{k}={CODE_LABEL.get(ax, {}).get(v, v)}")
    return "|".join(reversed(parts)) if parts else "all"


# ======================================================================================
# 규칙 공간 — 화소를 보지 않는 축 다섯
# ======================================================================================


@dataclass(frozen=True)
class Axis:
    """content-free 축 하나. `pixel_free` 가 False 면 화소에서 파생된 양이다."""

    key: str
    label: str
    pixel_free: bool


#: 축 카탈로그. 각 축의 값은 **train+val 에서 뽑은 절단점**으로 이산화한다 —
#: 평가셋 분위수를 쓰면 이산화 자체가 평가셋 유도가 된다(A-2 와 같은 종류의 사고).
AXES: dict[str, Axis] = {
    "prov": Axis("prov", "출처(N-crop/N-tile/N-band)", True),
    "material": Axis("material", "재질(ST/AL)", True),
    "idq": Axis("idq", "촬영 순서(image_id 분위)", True),
    "gsz": Axis("gsz", "중복 묶음 크기(group_size 분위)", True),
    # 파일 바이트 크기는 JPEG 압축률이라 **화소에서 파생된 양**이다. 화소를 디코딩하지
    # 않고 얻을 수 있어 "메타데이터"로 보이지만 내용의 함수다. 지시가 포함을 요구했으므로
    # 넣되, 순수 메타데이터 최댓값과 **분리해서** 보고한다.
    "fsz": Axis("fsz", "파일 바이트 크기 분위", False),
}

#: 축별 granularity 선택지. None 은 "축을 쓰지 않음".
GRANULARITY: dict[str, tuple[int | None, ...]] = {
    "prov": (None, 3),
    "material": (None, 2),
    # 분위 수를 사다리로 둔다. 잘게 쪼갤수록 적합 모집단에서는 값이 오르지만 어느 지점부터
    # 이월이 무너진다. 그 꺾이는 지점을 실측으로 찾는 것이 (가) 족 선택의 부산물이다.
    #
    # **512 에서 멈추는 이유.** 사다리는 512 까지도 완전히 꺾이지 않는다(실측: prov3xidqK 는
    # 256→512 에서 처음 내려가지만 idqK 단독은 계속 오른다). 무한정 올릴 수 있는 축이 아니라
    # 멈출 근거가 필요하다 — train+val 49,847장을 512 로 쪼개면 구간당 약 97장이고,
    # 묶음 크기 중앙값이 12 이므로 구간 하나에 취득 묶음이 8개 남는다. 이보다 잘게 쪼개면
    # 규칙이 "메타데이터 예측기"가 아니라 **묶음 조회표**로 퇴화한다 — 잰 대상이 달라진다.
    # 그래서 묶음이 한 자릿수로 떨어지는 512 를 상한으로 둔다. 이 선택이 결론을 바꾸지는
    # 않는다. 게이트가 도달 불가라는 판단은 이미 K=64(0.897)에서 확정된다.
    "idq": (None, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    "gsz": (None, 2, 4, 8),
    "fsz": (None, 2, 4, 8),
}


def make_cuts(fit_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """분위 절단점을 **train+val 한 곳에서만** 뽑는다.

    평가셋 분위수로 이산화하면 이산화 자체가 평가셋 유도가 된다 — A-2 와 같은 종류의
    사고이고, 그쪽이 더 눈에 안 띈다. 절단점은 여기서 한 번 만들고 모든 모집단이 받는다.
    """
    cuts: dict[str, np.ndarray] = {}
    for src, ax in (("id_num", "idq"), ("group_size", "gsz"), ("file_bytes", "fsz")):
        vals = fit_frame[src].to_numpy(dtype=float)
        for k in (g for g in GRANULARITY[ax] if g is not None):
            qs = np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1])
            uniq = np.unique(qs)
            # 절단점이 중복되면 실제 칸 수가 k 보다 적어진다. 그대로 두면 카디널리티가
            # 어긋나므로 뒤를 +inf 로 채워 자릿수를 유지한다(빈 칸은 n=0 으로 남는다).
            if uniq.size < k - 1:
                uniq = np.concatenate([uniq, np.full(k - 1 - uniq.size, np.inf)])
            cuts[f"{ax}__{k}"] = uniq
    return cuts


def enumerate_families(include_filesize: bool) -> list[tuple[tuple[str, int], ...]]:
    """축 granularity 의 데카르트 곱. 빈 조합(전역 규칙)도 포함한다."""
    import itertools

    keys = [k for k in GRANULARITY if include_filesize or k != "fsz"]
    choices = [[(k, g) for g in GRANULARITY[k] if g is not None] + [None] for k in keys]
    fams = []
    for combo in itertools.product(*choices):
        fams.append(tuple(c for c in combo if c is not None))
    return fams


def family_name(family: tuple[tuple[str, int], ...]) -> str:
    return "global" if not family else "x".join(f"{ax}{k}" for ax, k in family)


# ======================================================================================
# 탐색
# ======================================================================================


def search(fit: Population, score: Population, families, min_support: int, tag: str):
    """각 규칙 족을 `fit` 에서 적합하고 `score` 에서 채점한다(해석식).

    적합 모집단에서 `min_support` 미만인 셀은 규칙을 만들지 않고 전역 규칙으로 떨어뜨린다.
    표본 몇 개짜리 셀에 맞춘 규칙은 옮겨 붙지 않는다. 적합 모집단에 아예 없던 셀도 같다 —
    train+val 적합 / eval 채점이라 미관측 셀이 실제로 생긴다.
    """
    classes = fit.classes
    gn, gpos = fit.counts(())
    fb = fit_rule(gn, gpos, 1)[0]                      # 전역(축 없음) 규칙 = fallback

    results = []
    for fam in families:
        fn, fpos = fit.counts(fam)
        sel = fit_rule(fn, fpos, min_support)
        sn, spos = score.counts(fam)
        # 적합에서 규칙이 서지 않은 셀(미관측·과소표본)은 전역 규칙으로 떨어진다
        unseen = (fn < min_support) & (sn > 0)
        sel_eff = sel.copy()
        sel_eff[unseen] = fb
        mf1 = apply_rule(sn, spos, sel_eff)
        results.append({
            "_macro_f1_exact": float(mf1),      # 채점기 대조용. 반올림하면 1e-9 비교가 깨진다
            "family": family_name(fam), "axes": [list(a) for a in fam],
            "n_cells_fit": int((fn > 0).sum()),
            "n_cells_used": int((fn >= min_support).sum()),
            "n_cells_dropped_small": int(((fn > 0) & (fn < min_support)).sum()),
            "n_cells_score": int((sn > 0).sum()),
            "n_cells_unseen_in_fit": int(unseen.sum()),
            "n_images_on_fallback": int(sn[unseen].sum()),
            "pixel_free": all(AXES[a].pixel_free for a, _ in fam),
            "macro_f1": round(float(mf1), 6),
            "fallback": [c for j, c in enumerate(classes) if fb[j]],
            "tag": tag,
        })
    results.sort(key=lambda r: -r["macro_f1"])
    return results


def materialize(fit: Population, score: Population, fam, min_support: int):
    """족 하나를 사람이 읽는 규칙 dict + 셀 배정으로 펼친다. 등록·검산용."""
    classes = fit.classes
    gn, gpos = fit.counts(())
    fb = fit_rule(gn, gpos, 1)[0]
    fallback = tuple(c for j, c in enumerate(classes) if fb[j])
    fn, fpos = fit.counts(fam)
    sel = fit_rule(fn, fpos, min_support)
    chosen = {
        cell_label(int(cid), fam, fit): tuple(c for j, c in enumerate(classes) if sel[cid, j])
        for cid in np.flatnonzero(fn >= min_support)
    }
    ids, _ = score.cell_ids(fam)
    cell_of = {img: cell_label(int(cid), fam, score) for img, cid in zip(score.ids, ids)}
    return chosen, fallback, cell_of


def verify_with_scorer(score: Population, gold_codes, gold_boxes, classes,
                       chosen, fallback, cell_of) -> dict:
    """등록되는 값은 반드시 **다섯 칸과 같은 채점기**(evaluation/score.py)로 다시 낸다."""
    ids = score.ids
    recs = records_from_rule(ids, cell_of, chosen, fallback=fallback)
    return score_records(recs, {i: gold_codes[i] for i in ids},
                         {i: gold_boxes[i] for i in ids}, classes)


# ======================================================================================
# main
# ======================================================================================


def _iou(a, b) -> float:
    from evaluation.metrics.localization import iou
    return iou(tuple(a), tuple(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("-o", "--out", type=Path,
                    default=REPO_ROOT / "data/interim/manifest_v1/baselines_recomputed.json")
    ap.add_argument("--min-support", type=int, default=30,
                    help="적합 모집단에서 이 미만인 셀은 전역 규칙으로 떨어뜨린다")
    ap.add_argument("--verify-enumeration", action="store_true", default=True,
                    help="소규모 셀에서 접두사 최적해를 부분집합 전수 열거와 대조한다")
    args = ap.parse_args()

    snap = load_snapshot(args.snapshot)
    m, t = snap.manifest, snap.tiles
    prov = dict(zip(t["image_id"], t["provenance"], strict=True))
    m = m.assign(prov=m["image_id"].map(prov))
    m["id_num"] = m["image_id"].str.rsplit(":", n=1).str[-1].astype("int64")
    m["group_size"] = pd.to_numeric(m["group_size"], errors="coerce").fillna(1).astype("int64")

    print(f"동결 스냅샷 {snap.snapshot_id} · 전체 {len(m):,}장 "
          f"({m['split'].value_counts().to_dict()})")

    # --- 파일 바이트 크기. 화소에서 파생된 축이라 따로 표시한다 ---
    sizes = []
    missing = 0
    for rp in m["rel_path"]:
        p = REPO_ROOT / str(rp)
        try:
            sizes.append(p.stat().st_size)
        except OSError:
            sizes.append(-1)
            missing += 1
    m["file_bytes"] = sizes
    if missing:
        print(f"  !! 파일 {missing:,}건 stat 실패 — file_bytes 축을 신뢰하지 마라")

    gold_codes, gold_boxes = build_gold(snap, list(m["image_id"]))

    tv = m[m["split"] != "eval"].reset_index(drop=True)
    ev = m[m["split"] == "eval"].reset_index(drop=True)
    tr = tv[tv["split"] == "train"].reset_index(drop=True)
    va = tv[tv["split"] == "val"].reset_index(drop=True)
    cuts = make_cuts(tv)                   # 절단점은 train+val 한 곳에서만
    print(f"적합 모집단 train+val {len(tv):,}장 · 채점 모집단 eval {len(ev):,}장")

    classes_ev = sorted({c for i in ev["image_id"] for c in gold_codes[i]})
    classes_tv = sorted({c for i in tv["image_id"] for c in gold_codes[i]})
    print(f"클래스 — train+val {classes_tv} / eval {classes_ev}")
    if classes_tv != classes_ev:
        print("  !! 두 모집단의 클래스 집합이 다르다. macro 분모가 갈린다")

    P_tv = Population(tv, gold_codes, classes_ev, cuts)
    P_ev = Population(ev, gold_codes, classes_ev, cuts)
    P_tr = Population(tr, gold_codes, classes_ev, cuts)
    P_va = Population(va, gold_codes, classes_ev, cuts)

    report: dict[str, object] = {
        "snapshot_id": snap.snapshot_id,
        "snapshot_digest": snap.capabilities.get("snapshot_digest", ""),
        "classes": classes_ev,
        "min_support": args.min_support,
        "populations": {
            "trainval49847": {"n": len(tv), "classes": classes_tv},
            "eval12461": {"n": len(ev), "classes": classes_ev},
        },
        "constants": {},
        "rule_space": {},
    }

    # --- 검산 1: 접두사 최적해 대 부분집합 전수 열거 (출처×재질 6셀) ---
    # --- 검산 2: 정수 벡터 경로 대 dict 정의 경로 ---
    # 둘 다 통과해야 아래 값들을 신뢰할 수 있다. 벡터 경로는 빠르라고 있는 것이지
    # 정의를 바꾸라고 있는 게 아니다.
    if args.verify_enumeration:
        fam6 = (("prov", 3), ("material", 2))
        sm = ev.assign(_c=ev["prov"].astype(str) + "|" + ev["material"].astype(str))
        n_by, pos_by = cell_stats(sm, gold_codes, "_c", classes_ev)
        cells = sorted(n_by)
        a = best_content_free(cells, classes_ev, n_by, pos_by)
        b = brute_force_content_free(cells, classes_ev, n_by, pos_by)
        fa = analytic_macro_f1(cells, classes_ev, n_by, pos_by, a)
        fb = analytic_macro_f1(cells, classes_ev, n_by, pos_by, b)
        n6, pos6 = P_ev.counts(fam6)
        fc = apply_rule(n6, pos6, fit_rule(n6, pos6, 1))
        ok = abs(fa - fb) < 1e-12 and abs(fa - fc) < 1e-12
        print(f"\n검산 (출처×재질 {len(cells)}셀): 접두사 {fa:.6f} / 전수열거 {fb:.6f} / "
              f"벡터 {fc:.6f} → {'일치' if ok else '불일치'}")
        report["optimality_and_vector_check"] = {
            "cells": len(cells), "prefix_dict": round(fa, 10),
            "brute_force": round(fb, 10), "vector": round(fc, 10),
            "agree": bool(ok), "rules_identical": a == b,
        }
        if not ok:
            print("  !! 세 경로가 갈렸다. 이 판의 모든 값을 신뢰하지 마라")
            return 1

    # --- 검산 3: 74번 감사가 낸 숫자를 재현한다 ---
    # 감사는 `image_id` 4분위로 0.3877, 출처×재질×id4분위(11셀)로 0.5940 을 보고했다.
    # train+val 절단점으로는 그 값이 안 나온다(0.3541 / 0.4912). **감사는 분위 절단점을
    # 평가셋에서 뽑았다** — 그러면 3자리까지 재현된다. 감사가 A-2 로 지적한 것과 같은
    # 종류의 유도가 감사 자신의 진단 수치에도 한 겹 더 들어가 있었다는 뜻이다.
    # 결론(게이트가 너무 낮다)은 그대로 선다. 오히려 재산출본이 더 높은 천장을 낸다.
    audit_cuts = make_cuts(ev)
    P_ev_evcuts = Population(ev, gold_codes, classes_ev, audit_cuts)
    audit_repro = {}
    for fam, label, reported in (
        ((("idq", 4),), "idq4", 0.3877),
        ((("prov", 3), ("material", 2), ("idq", 4)), "prov3xmaterial2xidq4", 0.5940),
    ):
        ne, pe = P_ev_evcuts.counts(fam)
        v_ev = apply_rule(ne, pe, fit_rule(ne, pe, 1))
        nt, pt = P_ev.counts(fam)
        v_tv = apply_rule(nt, pt, fit_rule(nt, pt, 1))
        audit_repro[label] = {
            "audit_reported": reported,
            "eval_derived_cuts": round(v_ev, 6),
            "trainval_derived_cuts": round(v_tv, 6),
            "cells": int((ne > 0).sum()),
        }
        print(f"  감사 재현 {label:22s} 보고 {reported:.4f} · eval 절단점 {v_ev:.4f} · "
              f"train+val 절단점 {v_tv:.4f} ({int((ne > 0).sum())}셀)")
    report["audit_number_reproduction"] = {
        "finding": ("감사 수치는 평가셋 분위 절단점에서 나온다. train+val 절단점으로는 "
                    "재현되지 않는다 — 이산화 자체가 평가셋 유도였다."),
        "values": audit_repro,
    }

    fams_all = enumerate_families(include_filesize=True)

    # ------------------------------------------------------------------------------
    # (가) 족 선택 — train 적합 / val 채점. **평가셋을 열지 않는다.**
    #
    # 규칙의 파라미터만 train+val 에서 뽑고 어느 족을 게이트로 삼을지는 eval 최댓값으로
    # 고르면, 고르는 행위 자체가 평가셋 유도가 된다(A-2 와 같은 종류의 사고, 강도만 약하다).
    # 그래서 족은 train→val 로 고르고, 고른 족만 train+val 재적합해서 eval 에서 잰다.
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("(가) 족 선택 — fit_train / score_val (평가셋 미접근)")
    print("=" * 86)
    res_transfer = search(P_tr, P_va, fams_all, args.min_support, "fit_train__score_val")
    print(f"  규칙 족 {len(res_transfer):,}개 · train {len(tr):,} → val {len(va):,}. 상위 8개:")
    for r in res_transfer[:8]:
        print(f"  {r['family']:38s} {r['pixel_free']!s:>8s} {r['macro_f1']:9.4f}")
    sel_any = res_transfer[0]
    sel_pixfree = max((r for r in res_transfer if r["pixel_free"]), key=lambda r: r["macro_f1"])
    print(f"\n  선택(전체 축) {sel_any['family']}  ·  선택(화소무관 축만) {sel_pixfree['family']}")

    # --- 분위 사다리: 어느 granularity 에서 이월이 꺾이는지 ---
    ladder = []
    for k in (g for g in GRANULARITY["idq"] if g is not None):
        fam = (("prov", 3), ("idq", k))
        row_v = next(r for r in res_transfer if r["family"] == family_name(fam))
        ladder.append({"idq": k, "fit_train__score_val": row_v["macro_f1"]})
    print("  분위 사다리(prov3xidqK, train→val): "
          + " · ".join(f"K={r['idq']} {r['fit_train__score_val']:.4f}" for r in ladder))

    # ------------------------------------------------------------------------------
    # (나) 게이트 재료 — 선택된 족을 train+val 재적합, eval 에서 채점
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("(나) fit_trainval / score_eval12461  ← 게이트 재료")
    print("=" * 86)
    res_gate = search(P_tv, P_ev, fams_all, args.min_support,
                      "fit_trainval__score_eval12461")
    gate_by_fam = {r["family"]: r for r in res_gate}
    best_pixfree = gate_by_fam[sel_pixfree["family"]]     # 족은 val 로 골랐다
    best_any = gate_by_fam[sel_any["family"]]
    print(f"  {'족':38s} {'셀':>5s} {'미관측':>6s} {'화소무관':>8s} {'macro_f1':>9s}")
    for r in res_gate[:10]:
        mark = " ←선택" if r["family"] in (sel_any["family"], sel_pixfree["family"]) else ""
        print(f"  {r['family']:38s} {r['n_cells_used']:5d} {r['n_cells_unseen_in_fit']:6d} "
              f"{r['pixel_free']!s:>8s} {r['macro_f1']:9.4f}{mark}")
    posthoc_max = res_gate[0]
    print(f"\n  val 선택 족의 eval 값: 전체축 {best_any['macro_f1']:.4f} / "
          f"화소무관 {best_pixfree['macro_f1']:.4f}")
    print(f"  사후 eval 최댓값(참고, 게이트 아님): {posthoc_max['macro_f1']:.4f} "
          f"({posthoc_max['family']})")

    # ------------------------------------------------------------------------------
    # (다) 참고 — eval 에서 적합, eval 에서 채점 (오라클 상한, 게이트 미사용)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("(다) 오라클 상한 — fit_eval / score_eval12461  ← 평가셋 유도. 게이트에 쓰지 않는다")
    print("=" * 86)
    # **min_support 를 걸지 않는다.** 이 값의 역할은 "평가셋 정답을 봤다면 어디까지 갔을까"
    # 라는 도달 불가 상한이다. 적합 안정성 하한(30)은 적합 모집단 49,847장을 전제로 고른
    # 값이고, eval 12,461장에 그대로 걸면 잘게 쪼갠 족이 통째로 fallback 으로 떨어져
    # **상한이 게이트보다 낮게 나오는 역전**이 생긴다(실측: idq512 가 0.9285 → 0.7852).
    # 상한을 핸디캡 씌워 재면 상한이 아니다.
    res_oracle = search(P_ev, P_ev, fams_all, 1, "fit_eval__score_eval12461")
    print(f"  {'족':38s} {'셀':>5s} {'화소무관':>8s} {'macro_f1':>9s}")
    for r in res_oracle[:6]:
        print(f"  {r['family']:38s} {r['n_cells_used']:5d} "
              f"{r['pixel_free']!s:>8s} {r['macro_f1']:9.4f}")
    oracle_pixfree = max((r for r in res_oracle if r["pixel_free"]), key=lambda r: r["macro_f1"])

    # ------------------------------------------------------------------------------
    # (라) 등록값 — 단일 채점기로 재산출
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("(라) 등록값 재산출 — evaluation/score.py 단일 채점기")
    print("=" * 86)

    named: dict[str, dict] = {}
    ev_gold = {i: gold_codes[i] for i in ev["image_id"]}

    def register(name: str, fam, chosen, fallback, cell_of, analytic: float,
                 note: str) -> float:
        res = verify_with_scorer(P_ev, gold_codes, gold_boxes, classes_ev,
                                 chosen, fallback, cell_of)
        mf1 = float(res["macro_f1"])
        agree = abs(mf1 - analytic) < 1e-9
        flag = "" if agree else f"  !! 해석식 {analytic:.6f} 과 불일치"
        print(f"  {name:60s} macro_f1 {mf1:.4f}  miss {res['miss_rate']:.4f}{flag}")
        named[name] = {
            "macro_f1": round(mf1, 6), "analytic": round(analytic, 6),
            "miss_rate": round(float(res["miss_rate"]), 6),
            "defect_recall": round(float(res["defect_recall"]), 6),
            "class_jaccard": round(float(res["class_jaccard"]), 6),
            "family": family_name(fam), "note": note,
            "agrees_with_analytic": bool(agree),
            "n_cells": len(chosen),
            "fallback": list(fallback),
            # 규칙 전문. 셀이 많으면 요약만 남기고 전문은 rule_space 로 뺀다
            "rule": ({k: list(v) for k, v in sorted(chosen.items())}
                     if len(chosen) <= 64 else f"{len(chosen)}셀 — 전문 생략"),
        }
        return mf1

    allc = tuple(classes_ev)
    flat = dict.fromkeys(ev["image_id"], "all")
    n0, pos0 = P_ev.counts(())

    def const_analytic(codes: tuple[str, ...]) -> float:
        sel = np.array([[c in codes for c in classes_ev]], dtype=bool)
        return apply_rule(n0, pos0, sel)

    # 자명·상수 규칙 (적합 없음 — 유도원 문제가 애초에 없다)
    register("trivial_all_positive__score_eval12461", (), {"all": allc}, allc, flat,
             const_analytic(allc), "전량양성. 적합 없음")
    register("constant_porosity__score_eval12461", (), {"all": ("2011",)}, ("2011",), flat,
             const_analytic(("2011",)), "상수 기공. 적합 없음")

    def fam_tuple(r) -> tuple[tuple[str, int], ...]:
        return tuple((a, int(k)) for a, k in r["axes"])

    def reg_family(name: str, row, fit_pop: Population, note: str,
                   min_support: int | None = None) -> float:
        fam = fam_tuple(row)
        ms = args.min_support if min_support is None else min_support
        chosen, fallback, cell_of = materialize(fit_pop, P_ev, fam, ms)
        return register(name, fam, chosen, fallback, cell_of,
                        row["_macro_f1_exact"], note)

    gate_pixfree = reg_family(
        "content_free_pixelfree__sel_val__fit_trainval__score_eval12461",
        best_pixfree, P_tv, "화소와 무관한 축만. 족은 val, 규칙은 train+val")
    gate_any = reg_family(
        "content_free_any__sel_val__fit_trainval__score_eval12461",
        best_any, P_tv, "파일 크기 축 포함. 족은 val, 규칙은 train+val")
    posthoc = reg_family(
        "content_free_any__sel_eval__fit_trainval__score_eval12461",
        posthoc_max, P_tv, "**족 선택이 사후·평가셋 기반**. 참고 상한. 게이트 아님")
    orc_pixfree = reg_family(
        "content_free_pixelfree__sel_eval__fit_eval__score_eval12461",
        oracle_pixfree, P_ev, "**평가셋 유도 오라클**. 게이트에 쓰지 않는다", min_support=1)
    orc_any = reg_family(
        "content_free_any__sel_eval__fit_eval__score_eval12461",
        res_oracle[0], P_ev, "**평가셋 유도 오라클**. 게이트에 쓰지 않는다", min_support=1)

    # 옛 판이 게이트로 쓴 족(출처×재질) — 비교용으로 남긴다
    row_sm = next(r for r in res_gate if r["family"] == "prov3xmaterial2")
    reg_family("source_x_material__fit_trainval__score_eval12461", row_sm, P_tv,
               "69번이 게이트 0.3465 로 쓴 족. 이제 train+val 적합이라 값이 달라진다")
    row_sm_ev = next(r for r in res_oracle if r["family"] == "prov3xmaterial2")
    reg_family("source_x_material__fit_eval__score_eval12461", row_sm_ev, P_ev,
               "69번의 0.3465 재현 — 평가셋 적합이었다는 것이 A-2 의 지적", min_support=1)

    gate = max(gate_pixfree, gate_any)
    gate_family = best_any["family"] if gate == gate_any else best_pixfree["family"]
    pass_line = round(gate + GATE_TOLERANCE, 6)
    print(f"\n  게이트 = {gate:.4f} (족 {gate_family}) · 통과선 = {pass_line:.4f}")
    print(f"  참고 상한 — 사후 eval 족선택 {posthoc:.4f} · 평가셋 유도 오라클 "
          f"{max(orc_pixfree, orc_any):.4f} (둘 다 게이트 아님)")
    if posthoc > pass_line:
        print(f"  !! 통과선({pass_line:.4f})과 사후 상한({posthoc:.4f}) 사이 구간에서는 "
              "'모든 content-free 규칙을 이겼다'고 주장할 수 없다")

    # ------------------------------------------------------------------------------
    # (마) 놓침 축 · 위치 축
    # ------------------------------------------------------------------------------
    print("\n=== 놓침 축 ===")
    ids_ev = list(ev["image_id"])
    recs = records_from_rule(ids_ev, dict.fromkeys(ids_ev, "all"), {"all": allc})
    r_all = score_records(recs, ev_gold, {i: gold_boxes[i] for i in ids_ev}, classes_ev)
    print(f"  전량양성 miss_rate {r_all['miss_rate']:.4f} — 놓침 단독 게이트는 불가능")

    print("\n=== 위치 축 (상수 박스 — train+val GT 중앙값, eval 미사용) ===")
    train_ids = set(tv["image_id"])
    tr_ann = snap.annotations[snap.annotations["image_id"].isin(train_ids)]
    const_box = (
        float(tr_ann["bbox_x1_px"].median()), float(tr_ann["bbox_y1_px"].median()),
        float(tr_ann["bbox_x2_px"].median()), float(tr_ann["bbox_y2_px"].median()),
    )
    recs = records_from_rule(ids_ev, dict.fromkeys(ids_ev, "all"), {"all": allc}, box=const_box)
    res = score_records(recs, ev_gold, {i: gold_boxes[i] for i in ids_ev}, classes_ev)
    per_image_max = [max(_iou(const_box, b) for _, b in gold_boxes[i]) for i in ids_ev
                     if gold_boxes[i]]
    ge50 = int(res["n_matched_ge_50"]) / max(int(res["n_gold"]), 1)
    print(f"  박스 {const_box} (n={len(tr_ann):,})")
    print(f"  GT {int(res['n_gold']):,}개 중 IoU>=0.5 {int(res['n_matched_ge_50']):,} ({ge50*100:.2f}%)")
    print(f"  bbox_iou {res['bbox_iou']:.4f} · 이미지당 최대 IoU 평균 {np.mean(per_image_max):.4f}")

    # ---- 기록 ----
    report["constants"] = named
    report["constants"]["miss_rate__score_eval12461"] = {
        "trivial_all_positive": round(float(r_all["miss_rate"]), 6),
        "note": "전량양성이 0 을 달성한다 — 놓침만으로는 게이트를 세울 수 없다",
    }
    report["constants"]["constant_box__derived_trainval__score_eval12461"] = {
        "box_xyxy": list(const_box),
        "derived_from": "train+val GT 좌표별 중앙값 (eval 미사용)",
        "n_ann_fit": len(tr_ann),
        "n_gold": int(res["n_gold"]), "n_ge_50": int(res["n_matched_ge_50"]),
        "frac_ge_50": round(ge50, 6),
        "bbox_iou_mean_all": round(float(res["bbox_iou"]), 6),
        "per_image_max_iou_mean": round(float(np.mean(per_image_max)), 6),
        "map_50": round(float(res.get("map_50", 0.0)), 6),
    }
    report["gate"] = {
        "max_macro_f1": round(gate, 6),
        "gate_tolerance": GATE_TOLERANCE,
        "pass_line": pass_line,
        "family": gate_family,
        "family_selected_on": "val (5,001장). 평가셋 미접근",
        "rule_fitted_on": "train+val (49,847장). 평가셋 정답 라벨 미사용",
        "scored_on": "eval12461",
        "pixel_free_only": round(gate_pixfree, 6),
        "incl_filesize": round(gate_any, 6),
        "reference_posthoc_eval_selected": round(posthoc, 6),
        "reference_eval_fitted_oracle": round(max(orc_pixfree, orc_any), 6),
        "oracle_min_support": 1,
        "oracle_note": ("오라클은 min_support 를 걸지 않는다. 적합 안정성 하한 30 은 "
                        "train+val 49,847장 기준이라 eval 12,461장에 그대로 걸면 상한이 "
                        "게이트보다 낮게 나오는 역전이 생긴다."),
        "warning": ("통과선과 사후 상한 사이는 '모든 content-free 규칙을 이겼다'고 "
                    "주장할 수 없는 구간이다"),
    }
    report["rule_space"] = {
        "n_families": len(fams_all),
        "axes": {k: {"label": v.label, "pixel_free": v.pixel_free,
                     "granularity": [g for g in GRANULARITY[k] if g is not None]}
                 for k, v in AXES.items()},
        "idq_ladder_fit_train_score_val": ladder,
        "fit_train__score_val": res_transfer,
        "fit_trainval__score_eval12461": res_gate,
        "fit_eval__score_eval12461": res_oracle,
    }

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\n기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
