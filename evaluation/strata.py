"""id 구간 층화 채점 — 총괄 판정 6 (76번 을안). 전역 지표와 **병기**한다.

## 왜

동결 평가셋의 Macro-F1 축은 촬영 순서로 거의 결정된다. 화소를 한 번도 안 보는 규칙
(`idq512`)이 0.9149 를 낸다(76번 §1-3). 기제는 진단으로 확인됐다 — id 구간 64개의
최빈 라벨 점유율 중앙값이 **1.000**(무작위 기준 0.445), 순도 ≥0.99 구간이 이미지의
**59.2%**, 구간 안 train↔eval 클래스 분포 L1 중앙값이 **0.0000** 이다. 구간 안에서
학습 쪽과 평가 쪽 구성이 같으니 "구간 → 클래스"를 외운 것이 그대로 옮겨 붙는다.

그래서 전역 Macro-F1 은 **외운 것과 배운 것을 구분하지 못한다.** 층화 채점은 구간을
고정하고 그 안에서만 재므로 "촬영 순서를 알고도 못 맞히는 부분"만 남는다.

## 무엇을 내는가 — 두 수를 함께 낸다

지시의 완료 기준은 *"지름길 규칙(구간→최빈 라벨)이 층화 지표에서 정확히 0 근처로
떨어진다"* 였다. **층 안 Macro-F1 을 그대로 평균하면 그 기준을 만족하지 못한다** —
순수 구간에서 최빈 라벨 예측기는 전건 정답이라 1.0 을 받는다. 파일럿 실측에서 지름길
규칙의 층화 Macro-F1 은 K=64 에서 **0.8614** 였다 — 0 근처가 아니다.

기준을 만족하는 것은 **지름길 대비 이득(lift)** 이다.

    lift_s = MacroF1_s(모델) − MacroF1_s(구간 최빈 라벨 예측기)
    stratified_lift = mean_s lift_s

구간 최빈 라벨 예측기가 곧 지름길 규칙이므로 그 lift 는 **정의상 정확히 0** 이다.
따라서 두 수를 함께 싣는다 — 층화 Macro-F1(지시 문면)과 lift(완료 기준). 어느 쪽도
단독으로는 부족하다.

## 층은 D 가 만들지 않는다 — A 의 `data.id_strata` 를 부른다

절단점을 만드는 곳은 한 군데여야 한다(84번 §1-3, A). 처음 판은 여기서 같은 산식을
한 벌 더 갖고 시험으로 정합을 대조했는데, A 가 `data/id_strata.py` 를 **소비 지점**으로
올렸으므로 산식을 지우고 위임한다. 절단점은 **train+val 분위**에서만 나오고
(`load_cut_points` 가 평가셋을 열지 않는다), 배정은 `stratum_of` 가 한다.

**정합은 시험이 강제한다**(`tests/test_strata_alignment.py`): 채점 경로의 층 번호가 A 의
`stratum_of` 및 A 가 실체화한 `id_strata_k{K}.csv` 와 같고, 이 모듈 안에 분위·searchsorted
산식이 다시 생기지 않는다(AST). A 의 `tests/test_id_strata.py` 가 그 모듈과
`recompute_baselines.py` 의 정합을 맡는다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.id_strata import STRATUM_AXIS, stratum_of
from evaluation.metrics.detection import score_detection

ID_GRANULARITY: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)
"""A 의 `GRANULARITY["idq"]` 에서 `None` 을 뺀 것과 같아야 한다. 시험이 대조한다."""

DEFAULT_K = 64
"""파일럿 대조의 기본 구간 수.

A 의 게이트 족은 `idq512` 지만 그것은 **적합 모집단이 train+val 49,847장**일 때의
선택이다. 채점 쪽은 사정이 다르다 — 동결 평가셋 12,461장을 512 로 쪼개면 구간당 24장,
파일럿 653장이면 구간당 1.3장이라 층 안 Macro-F1 이 성립하지 않는다. 진단
(`diagnose_id_axis.py`)이 쓴 64 를 기본으로 두고 **사다리를 함께 낸다** — K 선택이
결론을 바꾸는지가 보여야 한다.
"""


def bins_for(
    image_ids: Sequence[str], k: int, snapshot: Path | None = None
) -> dict[str, int]:
    """이미지 → 층 번호(0..K-1). **절단점은 A 의 `data.id_strata` 가 만든다.**

    `snapshot` 은 절단점을 뽑을 동결본 경로(None 이면 A 의 기본 `manifest_v1`). 어느
    모집단을 채점하든 절단점은 그 동결본의 train+val 에서 온다 — 파일럿 653장은 동결
    평가셋의 부분집합이라 같은 절단점을 그대로 받는다. 축 이름은 `STRATUM_AXIS`("idq").
    """
    ids = list(image_ids)
    idx = stratum_of(ids, k, snapshot)
    return dict(zip(ids, (int(v) for v in idx), strict=True))


def majority_codeset(
    image_ids: Sequence[str], gold: Mapping[str, Iterable[str]], classes: Sequence[str]
) -> frozenset[str]:
    """구간의 **최빈 정답 코드 집합** — 지름길 규칙 그 자체.

    동률이면 정렬 순으로 앞선 것을 고른다(결정론). 채점 클래스 안으로 제한한다.
    """
    scope = frozenset(classes)
    counts: dict[frozenset[str], int] = {}
    for i in image_ids:
        key = frozenset(gold.get(i, ())) & scope
        counts[key] = counts.get(key, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], tuple(sorted(kv[0], reverse=True))))
    return best[0]


@dataclass(frozen=True)
class StratifiedReport:
    k: int
    n_strata: int
    n_strata_scored: int
    """정답 양성이 하나라도 있어 Macro-F1 이 정의되는 구간 수."""
    n_pure_strata: int
    """정답 코드 집합이 한 가지뿐인 구간. **층 안 변별이 원리적으로 불가능**하다."""
    frac_images_in_pure: float
    macro_f1_mean: float
    """층화 Macro-F1 — 구간별 Macro-F1 의 **비가중 평균**(지시 문면)."""
    macro_f1_weighted: float
    """구간 크기 가중 평균. 작은 구간이 값을 흔드는 정도를 보려고 함께 낸다."""
    lift_mean: float
    """지름길(구간 최빈 라벨) 대비 이득. **완료 기준이 겨냥한 수다.**"""
    lift_weighted: float
    baseline_mean: float
    """지름길 규칙 자신의 층화 Macro-F1. 1.0 에 가까울수록 축이 순서에 먹혔다는 뜻."""
    global_macro_f1: float
    n_strata_impure_scored: int = 0
    """**변별 가능한 구간 수.** 정답 코드 집합이 두 가지 이상인 구간만 센다."""
    macro_f1_impure: float = 0.0
    lift_impure: float = 0.0
    """순수 구간을 뺀 lift. **가장 정직한 수다.**

    순수 구간에서는 최빈 라벨 기준선이 전건 정답이라 어떤 모델도 이길 수 없다 —
    거기서의 음수 lift 는 "못 배웠다"가 아니라 "잴 것이 없다"는 뜻이다. 파일럿
    평가셋은 이미지의 63.7%(K=64)가 순수 구간에 있어 그 왜곡이 크다.
    """
    baseline_impure: float = 0.0
    per_stratum: tuple[dict, ...] = ()

    def as_dict(self, with_detail: bool = False) -> dict:
        d = {
            "axis": STRATUM_AXIS,
            "k": self.k,
            "n_strata": self.n_strata,
            "n_strata_scored": self.n_strata_scored,
            "n_pure_strata": self.n_pure_strata,
            "frac_images_in_pure": self.frac_images_in_pure,
            "stratified_macro_f1": self.macro_f1_mean,
            "stratified_macro_f1_weighted": self.macro_f1_weighted,
            "stratified_lift": self.lift_mean,
            "stratified_lift_weighted": self.lift_weighted,
            "shortcut_baseline_macro_f1": self.baseline_mean,
            "global_macro_f1": self.global_macro_f1,
            "n_strata_impure_scored": self.n_strata_impure_scored,
            "stratified_macro_f1_impure": self.macro_f1_impure,
            "stratified_lift_impure": self.lift_impure,
            "shortcut_baseline_impure": self.baseline_impure,
        }
        if with_detail:
            d["per_stratum"] = list(self.per_stratum)
        return d


def _macro_f1(
    pred: Mapping[str, Iterable[str]],
    gold: Mapping[str, Iterable[str]],
    classes: Sequence[str],
) -> float | None:
    """구간 하나의 Macro-F1. **전역과 같은 함수**(`score_detection`)를 쓴다.

    정답 양성이 0 인 구간은 `None` — 0.0 으로 채우면 정상만 있는 구간이 평균을 끌어내려
    "못 맞혔다"로 오독된다.
    """
    rep = score_detection(pred, gold, classes)
    if not [s for s in rep.per_class if s.support > 0]:
        return None
    return rep.macro_f1


def stratified_score(
    pred_codes: Mapping[str, Iterable[str]],
    gold_codes: Mapping[str, Iterable[str]],
    classes: Sequence[str],
    bins: Mapping[str, int],
    *,
    k: int,
    keep_detail: bool = False,
) -> StratifiedReport:
    """id 구간 안에서 Macro-F1 을 내고 평균한다. 지름길 대비 이득을 함께 낸다."""
    image_ids = sorted(gold_codes)
    by_bin: dict[int, list[str]] = {}
    for i in image_ids:
        by_bin.setdefault(bins[i], []).append(i)

    rows: list[dict] = []
    n_pure = 0
    n_img_pure = 0
    for b in sorted(by_bin):
        ids = by_bin[b]
        g = {i: gold_codes.get(i, ()) for i in ids}
        p = {i: pred_codes.get(i, ()) for i in ids}
        distinct = {frozenset(v) & frozenset(classes) for v in g.values()}
        pure = len(distinct) == 1
        if pure:
            n_pure += 1
            n_img_pure += len(ids)
        maj = majority_codeset(ids, gold_codes, classes)
        base_pred = {i: sorted(maj) for i in ids}
        model_f1 = _macro_f1(p, g, classes)
        base_f1 = _macro_f1(base_pred, g, classes)
        rows.append({
            "bin": b, "n": len(ids), "pure": pure,
            "majority_codes": sorted(maj),
            "macro_f1": model_f1, "baseline_macro_f1": base_f1,
            "lift": None if (model_f1 is None or base_f1 is None) else model_f1 - base_f1,
        })

    scored = [r for r in rows if r["macro_f1"] is not None and r["baseline_macro_f1"] is not None]
    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in scored])) if scored else 0.0

    def _wmean(key: str) -> float:
        if not scored:
            return 0.0
        w = np.array([r["n"] for r in scored], dtype=float)
        v = np.array([r[key] for r in scored], dtype=float)
        return float((w * v).sum() / w.sum())

    impure = [r for r in scored if not r["pure"]]

    def _imean(key: str) -> float:
        return float(np.mean([r[key] for r in impure])) if impure else 0.0

    global_f1 = score_detection(pred_codes, gold_codes, classes).macro_f1
    return StratifiedReport(
        k=k,
        n_strata=len(rows),
        n_strata_scored=len(scored),
        n_pure_strata=n_pure,
        frac_images_in_pure=n_img_pure / len(image_ids) if image_ids else 0.0,
        macro_f1_mean=_mean("macro_f1"),
        macro_f1_weighted=_wmean("macro_f1"),
        lift_mean=_mean("lift"),
        lift_weighted=_wmean("lift"),
        baseline_mean=_mean("baseline_macro_f1"),
        global_macro_f1=global_f1,
        n_strata_impure_scored=len(impure),
        macro_f1_impure=_imean("macro_f1"),
        lift_impure=_imean("lift"),
        baseline_impure=_imean("baseline_macro_f1"),
        per_stratum=tuple(rows) if keep_detail else (),
    )


SHORTCUT_TAG = "__shortcut__"
"""지름길 규칙(구간→최빈 라벨)을 하나의 예측기로 넣을 때 쓰는 태그. **매 채점마다** 같은
표에 실린다 — 판별력 시험이 일회성 검증이 아니라 상시 계측이 되게. 이 행의 lift 가 0 이
아니면 층 정의나 기준선 정의가 틀린 것이고, `stratified_scoring` 게이트가 그것을 본다."""


def stratified_table(
    preds_by_tag: Mapping[str, Mapping[str, Iterable[str]]],
    gold_codes: Mapping[str, Iterable[str]],
    classes: Sequence[str],
    ks: Sequence[int],
    *,
    snapshot: Path | None = None,
    detail_k: int | None = None,
) -> dict[str, dict[str, dict]]:
    """K 별 · 예측기별 층화 보고 — `{str(k): {tag: report_dict}}`.

    본채점 진입점(`score_cells.py score`)과 대조표 스크립트(`stratified_compare.py`)가
    **같은 함수**로 표를 만든다(13번 D-1 — 층화가 별도 스크립트에만 살아 있던 배선 공백).
    지름길 규칙은 `SHORTCUT_TAG` 예측기로 항상 함께 넣는다. `detail_k` 를 주면 그 K 의
    지름길 행에 구간별 상세를 남긴다.
    """
    ids = sorted(gold_codes)
    gold = {i: sorted(gold_codes[i]) for i in ids}
    out: dict[str, dict[str, dict]] = {}
    for k in ks:
        bins = bins_for(ids, k, snapshot=snapshot)
        shortcut = shortcut_pred(gold, classes, bins)
        rows: dict[str, dict] = {}
        for tag, pc in {**dict(preds_by_tag), SHORTCUT_TAG: shortcut}.items():
            detail = k == detail_k and tag == SHORTCUT_TAG
            rep = stratified_score(pc, gold, classes, bins, k=k, keep_detail=detail)
            rows[tag] = rep.as_dict(with_detail=detail)
        out[str(k)] = rows
    return out


def shortcut_pred(
    gold_codes: Mapping[str, Iterable[str]],
    classes: Sequence[str],
    bins: Mapping[str, int],
) -> dict[str, list[str]]:
    """지름길 예측기 — 각 이미지에 **그 구간의 최빈 코드 집합**을 주장한다.

    판별력 시험의 기준선이다. 이 예측기의 `stratified_lift` 는 정의상 0 이어야 한다.
    """
    by_bin: dict[int, list[str]] = {}
    for i in sorted(gold_codes):
        by_bin.setdefault(bins[i], []).append(i)
    out: dict[str, list[str]] = {}
    for ids in by_bin.values():
        maj = sorted(majority_codeset(ids, gold_codes, classes))
        for i in ids:
            out[i] = list(maj)
    return out
