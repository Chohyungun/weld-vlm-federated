"""평가 층화 표본의 설계 가중과 대표 지표 M1 의 산출 — 25번 §2-5 · 실험설계 8-5(2026-10-13 등록).

## 무엇을 하나

1. **곁 파일을 읽는다.** 머리줄 `image_id,stratum,G_h,n_h`, 목록과 같은 순서의 행, UTF-8 · LF · 끝 개행. 층마다 `G_h` · `n_h` 는
   하나의 값이어야 하고 `1 ≤ n_h ≤ G_h` 다. 목록 파일과 곁 파일의 id 열이 **같은 순서로 같아야** 한다.
2. **가중을 계수표에 맞춘다.** 이미지마다 `G_h / n_h`(층의 묶음 수 ÷ 뽑힌 묶음 수). 계수표의 이미지 집합과 표본 목록이
   같아야 한다 — 하나라도 다르면 거부한다(빠진 이미지는 그 오탐을 지표에서 지운다).
3. **대표 지표를 낸다.** 주 채점(레코드 실패는 빈 예측)과 민감도(실패 레코드마다 평가 클래스별 오탐 1)를 같은 가중으로.
   가중 없는 표본 값을 병기한다. 표본 값을 "전량 평가의 실측"이라 부르지 않는다 — 추정 대상은 초모집단이다(07번 §15-4).

가중은 계수표 **위층**에서 붙는다(07번 §15-10) — `box_f1` 의 매칭 · 계수는 그대로다. 전량이면 가중이 전부 1 이고
`weighting = "none"` 이다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from evaluation.eval_list import validate_eval_list
from evaluation.metrics.box_f1 import SENSITIVITY_REGISTERED, CountTable, score_m1, sensitivity_counts

SIDE_HEADER = "image_id,stratum,G_h,n_h"
REPRESENTATIVE_METRIC = "m1_box_f1_iou_1_2_macro4"
"""대표 지표의 이름 — 클래스 일치 박스 매칭 F1@IoU 1/2, 네 결함 macro(실험설계 8-5, 2026-10-13 사용자 승인)."""
WEIGHTING_SAMPLE = "stratified_g_over_n"
WEIGHTING_NONE = "none"


class SampleSideError(ValueError):
    """곁 파일 · 목록 · 계수표가 서로 맞지 않는다."""


@dataclass(frozen=True)
class SideRow:
    stratum: str
    G_h: int
    n_h: int

    @property
    def weight(self) -> Fraction:
        return Fraction(self.G_h, self.n_h)


def read_side(raw: bytes, list_raw: bytes) -> dict[str, SideRow]:
    """곁 파일을 읽어 `image_id → SideRow`. 목록 파일과 같은 순서 · 같은 id 여야 한다."""
    ev = validate_eval_list(list_raw, where="sample_list")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise SampleSideError("곁 파일은 BOM 없는 UTF-8 · LF · 끝 개행이어야 한다")
    lines = raw.decode("utf-8").split("\n")[:-1]
    if not lines or lines[0] != SIDE_HEADER:
        raise SampleSideError(f"곁 파일의 머리줄이 {SIDE_HEADER!r} 가 아니다")
    out: dict[str, SideRow] = {}
    per_stratum: dict[str, tuple[int, int]] = {}
    order: list[str] = []
    for k, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        if len(parts) != 4:
            raise SampleSideError(f"곁 파일 {k} 줄의 칸 수가 4 가 아니다")
        image_id, stratum, g, n = parts
        if not (g.isdigit() and n.isdigit()) or g != str(int(g)) or n != str(int(n)):
            raise SampleSideError(f"곁 파일 {k} 줄의 G_h · n_h 가 10진 정수가 아니다")
        G_h, n_h = int(g), int(n)
        if not 1 <= n_h <= G_h:
            raise SampleSideError(f"곁 파일 {k} 줄: 1 ≤ n_h ≤ G_h 가 아니다")
        if per_stratum.setdefault(stratum, (G_h, n_h)) != (G_h, n_h):
            raise SampleSideError(f"층 {stratum!r} 의 G_h · n_h 가 줄마다 다르다")
        if image_id in out:
            raise SampleSideError("곁 파일에 같은 id 가 두 번 있다")
        out[image_id] = SideRow(stratum, G_h, n_h)
        order.append(image_id)
    if tuple(order) != tuple(ev.ids):
        raise SampleSideError("곁 파일의 id 열이 목록 파일과 같은 순서로 같지 않다")
    return out


def weights_for(table: CountTable, side: Mapping[str, SideRow]) -> np.ndarray:
    """계수표의 이미지 순서대로 `G_h / n_h`. 계수표의 이미지 집합과 표본이 같아야 한다."""
    ids = set(table.image_ids)
    if ids != set(side):
        raise SampleSideError(f"계수표와 표본의 이미지 집합이 다르다(계수표에만 {len(ids - set(side))} · 표본에만 "
                              f"{len(set(side) - ids)})")
    return np.array([float(side[i].weight) for i in table.image_ids], dtype=np.float64)


def representative(table: CountTable, *, gold_codes: Mapping[str, Iterable[str]], failed_image_ids: Sequence[str],
                   side: Mapping[str, SideRow] | None) -> dict:
    """대표 지표 M1 — 주 채점 · 민감도, 표본이면 설계 가중 추정과 가중 없는 값을 함께.

    `side` 가 `None` 이면 전량 채점이다(가중 1, `weighting = "none"`).
    """
    sens = sensitivity_counts(table, gold_codes=gold_codes, failed_image_ids=failed_image_ids)
    if side is None:
        return {"representative_metric": REPRESENTATIVE_METRIC, "weighting": WEIGHTING_NONE,
                "main": score_m1(table).as_dict(), "sensitivity": score_m1(sens).as_dict(),
                "sensitivity_variant": SENSITIVITY_REGISTERED, "n_images": len(table.image_ids)}
    w = weights_for(table, side)
    return {"representative_metric": REPRESENTATIVE_METRIC, "weighting": WEIGHTING_SAMPLE,
            "main": score_m1(table, w).as_dict(), "sensitivity": score_m1(sens, w).as_dict(),
            "main_unweighted": score_m1(table).as_dict(), "sensitivity_unweighted": score_m1(sens).as_dict(),
            "sensitivity_variant": SENSITIVITY_REGISTERED, "n_images": len(table.image_ids),
            "n_strata": len({r.stratum for r in side.values()}),
            "estimand": "초모집단 — 표본 값은 전량 평가의 실측이 아니다(07번 §15-4)"}
