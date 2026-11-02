"""서술 지표 — 금지 주장률 · 유보 사유 표기율 · 상태율 · JSON 앞 글 비율과 보조 둘, 설계 가중 · 무가중, 대응 차 구간(보조).
03 2판 §2-3a · §6a 결정 4 · 04 §1-3 · §1-4 · §1-8.

## 분모

| 지표 | 분자 | 분모 |
|---|---|---|
| 금지 주장률 · 종류 k | 종류 k 가 걸린 이미지 | 서술 있는 이미지(잘림 + 완결) |
| 금지 주장률 · 전체 | 어느 종류든 걸린 이미지 | 같은 분모 |
| 유보 표기율 · 사유 r | r ∈ R 이고 r 을 적었다고 인정된 완결 서술 | r ∈ R 인 완결 서술 |
| 유보 표기율 · 전체 | R 의 사유를 모두 적은 완결 서술 | R ≠ ∅ 인 완결 서술 |
| 상태율 | 그 상태의 이미지 | 전 이미지 |
| JSON 앞 글 | 앞 글이 있는 이미지 | 구조 파싱 유효(미기재 + 잘림 + 완결 — P4) |

분모가 0 이면 값 대신 `해당 없음` 과 적격 건수다 — 0 % 나 100 % 로 메우지 않는다. R(필요 사유)은 검사기가 (재질, 구조 필드 코드 ∪ 서술
관찰 코드)에서 낸다. 보조 둘 — R 의 크기별 분포, 정답 코드로 낸 R 의 같은 율 — 을 늘 함께 싣는다: 분모가 모델이 주장한 코드에 기대므로,
결함을 놓친 모델의 높은 표기율을 진단 능력으로 읽지 않게(190 §3-2).

## 검사기

`Checker` 꼴이면 무엇이든 꽂는다 — 서술마다 `check(narrative, material=, gold_codes=, structured_codes=)` 가 걸린 종류 · 필요 사유 ·
인정된 사유를, `required_for(material, codes)` 가 정답 코드의 필요 사유를 낸다. 검사기 v3 가 서기 전에는 시안 v2 를 어댑터로 꽂아 틀만
돌린다 — 그 수는 결과로 싣지 않는다(지시 1022b 과제 4).

## 구간(보조, 결정 4)

M1 과 같은 추첨의 가중 행렬 `W`(추첨 × 이미지 — `evaluation.sample_bootstrap.draw_weights`)를 받는다. 추첨마다 분자 · 분모의 가중합을
다시 내고, 분모가 0 인 추첨은 `NaN`(미정의)이다. 두 모델의 대응 차는 두 분모가 모두 0 이 아닌 추첨에서만 정의된다. 미정의를 빼고
`percentile_rule.defined_interval` 로 구간을 내며, 유효 추첨이 등록 하한(`ci.min_defined_draws`)보다 적으면 방향을 말하지 않는다.
점추정의 분모가 0 이면 구간을 내지 않는다(`해당 없음`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from evaluation import percentile_rule as PR
from evaluation.narrative_extract import COMPLETE, MISSING, STATES, TRUNCATED, NarrativeExtract

KINDS = ("가", "나", "다", "라", "마")
REASONS = ("r_dim", "r_note", "r_scope", "r_uncov")
PRESENT = (TRUNCATED, COMPLETE)
"""서술 있음 — 금지 주장률의 분모."""
STRUCT_VALID = (MISSING, TRUNCATED, COMPLETE)
"""구조 파싱 유효 — JSON 앞 글 비율의 분모(P4)."""
NOT_APPLICABLE = "해당 없음"
DIRECTION_KO = {"above": "위", "below": "아래", "indeterminate": "구분 불가"}
"""`evaluation.strict_main.DIRECTION_KO` 와 같은 낱말 — 지표마다 말이 갈리지 않게."""


class CheckResult(Protocol):
    flagged: Iterable[str]
    required: Iterable[str]
    found: Iterable[str]


class Checker(Protocol):
    def check(self, narrative: str, *, material: str, gold_codes: Sequence[str],
              structured_codes: Sequence[str]) -> CheckResult: ...

    def required_for(self, material: str, codes: Sequence[str]) -> Iterable[str]: ...


@dataclass(frozen=True)
class ImageItem:
    """채점 이미지 한 장의 입력 — 추출 결과 · 재질 · 정답 코드(정답 뷰) · 구조 필드 코드(1.4 레코드) · 설계 가중."""

    image_id: str
    extract: NarrativeExtract
    material: str
    gold_codes: tuple[str, ...]
    structured_codes: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class ImageRow:
    image_id: str
    weight: float
    state: str
    front_text: bool | None
    flagged: frozenset[str]
    required: frozenset[str]
    found: frozenset[str]
    required_gold: frozenset[str]


def build_rows(items: Iterable[ImageItem], checker: Checker) -> list[ImageRow]:
    """이미지마다 검사기를 부른다 — 서술이 있는 이미지(잘림 · 완결)만. 미기재 · 실패 · 검증 오류는 검사하지 않는다."""
    rows: list[ImageRow] = []
    for it in items:
        ex = it.extract
        if ex.state not in STATES:
            raise ValueError(f"모르는 상태 {ex.state!r}")
        flagged = required = found = required_gold = frozenset()
        if ex.state in PRESENT:
            res = checker.check(ex.narrative, material=it.material, gold_codes=list(it.gold_codes),
                                structured_codes=list(it.structured_codes))
            flagged, required, found = frozenset(res.flagged), frozenset(res.required), frozenset(res.found)
            unknown = (flagged - set(KINDS)) | (required - set(REASONS)) | (found - set(REASONS))
            if unknown:
                raise ValueError(f"검사기가 모르는 이름을 냈다: {sorted(unknown)}")
            if ex.state == COMPLETE:
                required_gold = frozenset(checker.required_for(it.material, list(it.gold_codes)))
        rows.append(ImageRow(it.image_id, float(it.weight), ex.state, ex.front_text, flagged, required, found,
                             required_gold))
    return rows


def indicators(rows: Sequence[ImageRow]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """지표 이름 → (분자 표시, 분모 표시). 이미지 순서는 `rows` 의 순서다."""
    def col(pred) -> np.ndarray:
        return np.fromiter((bool(pred(r)) for r in rows), dtype=bool, count=len(rows))

    every = col(lambda r: True)
    present = col(lambda r: r.state in PRESENT)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in STATES:
        out[f"state.{s}"] = (col(lambda r, s=s: r.state == s), every)
    out["front_text"] = (col(lambda r: r.state in STRUCT_VALID and r.front_text), col(lambda r: r.state in STRUCT_VALID))
    for k in KINDS:
        out[f"forbidden.{k}"] = (col(lambda r, k=k: r.state in PRESENT and k in r.flagged), present)
    out["forbidden.any"] = (col(lambda r: r.state in PRESENT and bool(r.flagged)), present)
    for tag, attr in (("reservation", "required"), ("reservation_gold", "required_gold")):
        for rs in REASONS:
            out[f"{tag}.{rs}"] = (col(lambda r, rs=rs, a=attr: r.state == COMPLETE and rs in getattr(r, a) and rs in r.found),
                                  col(lambda r, rs=rs, a=attr: r.state == COMPLETE and rs in getattr(r, a)))
        out[f"{tag}.all"] = (col(lambda r, a=attr: r.state == COMPLETE and bool(getattr(r, a))
                                 and getattr(r, a) <= r.found),
                             col(lambda r, a=attr: r.state == COMPLETE and bool(getattr(r, a))))
    return out


def _point(num: np.ndarray, den: np.ndarray, w: np.ndarray) -> dict:
    n_eligible = int(den.sum())
    d = float(w[den].sum())
    n = float(w[num & den].sum())
    if n_eligible == 0 or d == 0:
        return {"value": None, "state": NOT_APPLICABLE, "numerator": n, "denominator": d, "n_eligible": n_eligible}
    return {"value": n / d, "state": "finite", "numerator": n, "denominator": d, "n_eligible": n_eligible}


def point_estimates(rows: Sequence[ImageRow]) -> dict:
    """점추정 — 설계 가중(`weight`)과 가중 없음을 함께. 분모가 0 이면 `해당 없음` + 적격 건수."""
    ind = indicators(rows)
    w = np.array([r.weight for r in rows], dtype=float)
    ones = np.ones(len(rows), dtype=float)
    out = {name: {"weighted": _point(num, den, w), "unweighted": _point(num, den, ones)} for name, (num, den) in ind.items()}
    sizes = Counter(len(r.required) for r in rows if r.state == COMPLETE)
    out["aux.required_size_distribution"] = {str(k): sizes[k] for k in sorted(sizes)}
    out["counts"] = {"n_images": len(rows), **{f"n_{s}": sum(r.state == s for r in rows) for s in STATES}}
    return out


def draws(rows: Sequence[ImageRow], W) -> dict[str, np.ndarray]:
    """추첨마다의 값 — 분모가 0 인 추첨은 `NaN`. `W` 는 (추첨 × 이미지), 열 순서는 `rows` 의 순서다."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[1] != len(rows):
        raise ValueError(f"가중 행렬의 모양 {W.shape} 가 이미지 수 {len(rows)} 와 맞지 않는다")
    if not np.isfinite(W).all() or (W < 0).any():
        raise ValueError("가중 행렬은 유한한 0 이상의 수다")
    out: dict[str, np.ndarray] = {}
    for name, (num, den) in indicators(rows).items():
        n = W @ (num & den).astype(float)
        d = W @ den.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)
    return out


def paired_difference(point_a: Mapping, point_b: Mapping, draws_a: np.ndarray, draws_b: np.ndarray, *,
                      alpha: float, min_defined_draws: int) -> dict:
    """두 모델의 같은 지표의 차(a − b)와 보조 구간. 점추정의 분모가 어느 쪽이든 0 이면 구간을 내지 않는다."""
    if point_a["state"] != "finite" or point_b["state"] != "finite":
        return {"diff": None, "state": NOT_APPLICABLE, "interval": None, "direction": None,
                "direction_reason": "점추정의 분모가 0 이다"}
    d = np.asarray(draws_a, dtype=float) - np.asarray(draws_b, dtype=float)    # 한쪽이라도 NaN 이면 NaN — 둘 다 정의된 추첨에서만
    iv, n_def, n_undef = PR.defined_interval(d, alpha)
    out = {"diff": point_a["value"] - point_b["value"], "state": "finite",
           "interval": {**iv.as_dict(), "n_defined": n_def, "n_undefined": n_undef}}
    if n_def < min_defined_draws or n_def == 0:
        out.update(direction=None, direction_reason=f"유효 추첨 {n_def} 이 등록 하한 {min_defined_draws} 보다 적다")
    elif iv.lo.state != "finite" or iv.hi.state != "finite":
        out.update(direction=None, direction_reason=f"구간 끝점이 유한하지 않다({iv.lo.state} · {iv.hi.state})")
    else:
        key = "above" if iv.lo.value > 0 else "below" if iv.hi.value < 0 else "indeterminate"
        out.update(direction=DIRECTION_KO[key], direction_reason=None)
    return out


__all__ = ["KINDS", "NOT_APPLICABLE", "PRESENT", "REASONS", "STRUCT_VALID", "Checker", "ImageItem", "ImageRow",
           "build_rows", "draws", "indicators", "paired_difference", "point_estimates"]
