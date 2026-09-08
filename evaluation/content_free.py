"""무내용(content-free) 대조선 — **화소를 한 번도 보지 않는 규칙**을 같은 채점기로 채점한다.

## 왜 이 모듈이 있나

이 연구는 규격·촬영 순서 같은 메타데이터만으로도 높은 점수가 나온다는 것을 실측해 두고
(함정 #11·#12), 모델이 그보다 나은지를 늘 대조해 왔다. 등록된 천장은 `idq512`(촬영 순서
512분위 단독)의 **Macro-F1 0.9149** 다(`configs/base.yaml`
`content_free__sel_val__fit_trainval__score_eval12461`).

22번이 분류 축 주 지표를 **macro-AP** 로 옮기면서 그 대조가 함께 옮겨지지 않았다. 이 모듈이
같은 규칙을 새 축에서 채점해 그 빈자리를 메운다. 점수 구성·적합 모집단·기준선 목록은
**산출 전에** 17번 §11 에 사전등록했고 이 모듈은 그 문서를 그대로 구현한다.

## 사전등록된 규약 (17번 §11-2·§11-3)

| 항목 | 확정 |
|---|---|
| 점수 구성 H | 규칙이 예측한 클래스에 1, 나머지는 **점수 없음**(0 이 아니라 부재) |
| 점수 구성 F | 구간의 **train+val 빈도**를 4클래스 전부에 준다(0 포함). **주 대조선** |
| 적합 모집단 | 동결 train+val 49,847. 평가셋을 열지 않는다 |
| 절단점 | A 의 `data.id_strata`(이미 train+val 분위) |
| 족 | 등록된 `idq512` 를 그대로. 여기서 다시 고르지 않는다 |

H 는 등록 상수 0.9149 와 **같은 예측기**라 `AP = P·R` 관계가 표에서 직접 읽힌다. F 는
순위 지표에서 무내용 규칙이 도달할 수 있는 최대라 게이트 질문("화소를 안 보고 도달 가능한
수준을 모델이 넘는가")에 답하는 쪽이다.

**이 모듈은 칸 점수를 바꾸지 않는다.** 기준선은 추가 산출물이고 채점 기준을 건드리지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from evaluation.strata import bins_for

FIT_SPLITS: tuple[str, ...] = ("train", "val")
"""적합 모집단. **`eval` 은 절대 들어가지 않는다** — 등록 절차 `fit_trainval` 그대로."""

HARD = "H"
FREQ = "F"


@dataclass(frozen=True)
class BinRule:
    """구간 하나에 적합된 무내용 규칙.

    `majority` 는 H 구성의 예측(구간 최빈 코드 집합), `freq` 는 F 구성의 점수
    (구간 안에서 그 클래스가 정답이었던 비율)다. 둘 다 **적합 모집단의 정답으로만** 만든다.
    """

    n_fit: int
    majority: frozenset[str]
    freq: dict[str, float]


@dataclass(frozen=True)
class ContentFreeRule:
    """적합이 끝난 무내용 규칙 한 벌."""

    name: str
    kind: str
    """`all_positive` | `constant` | `idq`."""
    k: int | None
    fit_population: str
    classes: tuple[str, ...]
    bins: dict[int, BinRule] = field(default_factory=dict)
    global_freq: dict[str, float] = field(default_factory=dict)
    """적합 모집단 전체의 클래스 빈도. **빈 구간의 폴백**이다(§11-3 보완, 기계적 규칙)."""
    n_fit: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "k": self.k,
            "fit_population": self.fit_population, "n_fit": self.n_fit,
            "n_bins": len(self.bins),
            "global_freq": dict(sorted(self.global_freq.items())),
        }


def _freq_of(ids: Sequence[str], gold: Mapping[str, Iterable[str]],
             classes: Sequence[str]) -> dict[str, float]:
    n = len(ids)
    if n == 0:
        return {c: 0.0 for c in classes}
    scope = frozenset(classes)
    counts = dict.fromkeys(classes, 0)
    for i in ids:
        for c in frozenset(gold.get(i, ())) & scope:
            counts[c] += 1
    return {c: counts[c] / n for c in classes}


def _majority_of(ids: Sequence[str], gold: Mapping[str, Iterable[str]],
                 classes: Sequence[str]) -> frozenset[str]:
    """구간 최빈 정답 코드 집합. 동률은 정렬 역순 튜플이 큰 쪽 — `strata.majority_codeset`
    와 **같은 규칙**이다(두 곳이 갈리면 H 구성과 층화 표가 다른 규칙을 말하게 된다)."""
    scope = frozenset(classes)
    counts: dict[frozenset[str], int] = {}
    for i in ids:
        key = frozenset(gold.get(i, ())) & scope
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return frozenset()
    return max(counts.items(),
               key=lambda kv: (kv[1], tuple(sorted(kv[0], reverse=True))))[0]


def fit_idq(
    k: int,
    fit_ids: Sequence[str],
    gold: Mapping[str, Iterable[str]],
    classes: Sequence[str],
    *,
    snapshot=None,
    fit_population: str = "train+val",
) -> ContentFreeRule:
    """`idq{k}` 규칙을 적합한다. **절단점과 적합 정답 모두 `fit_ids` 에서만 온다.**"""
    fit_bins = bins_for(list(fit_ids), k, snapshot)
    by_bin: dict[int, list[str]] = {}
    for i in fit_ids:
        by_bin.setdefault(fit_bins[i], []).append(i)
    return ContentFreeRule(
        name=f"idq{k}", kind="idq", k=k, fit_population=fit_population,
        classes=tuple(classes), n_fit=len(fit_ids),
        global_freq=_freq_of(list(fit_ids), gold, classes),
        bins={
            b: BinRule(n_fit=len(ids), majority=_majority_of(ids, gold, classes),
                       freq=_freq_of(ids, gold, classes))
            for b, ids in sorted(by_bin.items())
        },
    )


def fit_trivial(
    name: str,
    predicted: Iterable[str],
    fit_ids: Sequence[str],
    gold: Mapping[str, Iterable[str]],
    classes: Sequence[str],
    *,
    fit_population: str = "train+val",
) -> ContentFreeRule:
    """자명 규칙 — 모든 이미지에 같은 코드 집합을 주장한다(구간을 쓰지 않는다).

    `all_positive`(4클래스 전부)·`constant_porosity`(2011 하나)가 여기서 나온다. F 구성의
    점수는 적합 모집단 전체 빈도이며 **모든 이미지에서 같다** — 순위가 없으므로 H 와 F 가
    같은 AP 를 준다(상수 예측기의 성질이고, 그 사실 자체가 표의 정합 검사다).
    """
    pred = frozenset(predicted) & frozenset(classes)
    freq = _freq_of(list(fit_ids), gold, classes)
    return ContentFreeRule(
        name=name, kind="constant" if len(pred) < len(classes) else "all_positive",
        k=None, fit_population=fit_population, classes=tuple(classes),
        n_fit=len(fit_ids), global_freq=freq,
        bins={0: BinRule(n_fit=len(fit_ids), majority=pred, freq=freq)},
    )


def _bin_of(rule: ContentFreeRule, image_ids: Sequence[str], snapshot=None) -> dict[str, int]:
    if rule.kind in ("constant", "all_positive"):
        return dict.fromkeys(image_ids, 0)
    return bins_for(list(image_ids), int(rule.k), snapshot)


def predict_codes(rule: ContentFreeRule, image_ids: Sequence[str],
                  *, snapshot=None) -> dict[str, list[str]]:
    """H 구성의 라벨 집합 예측 — Macro-F1·놓침 축에 그대로 넣는다."""
    b = _bin_of(rule, image_ids, snapshot)
    out: dict[str, list[str]] = {}
    for i in image_ids:
        br = rule.bins.get(b[i])
        out[i] = sorted(br.majority) if br is not None else []
    return out


def scored(rule: ContentFreeRule, image_ids: Sequence[str], construction: str,
           *, snapshot=None) -> tuple[dict[str, list[tuple[str, float]]], dict]:
    """(image_id → [(iso_code, score)], 진단). `image_level_ap` 입력 형식이다.

    - **H**: 예측한 클래스에만 점수 1. 나머지는 부재(0 점이 아니다).
    - **F**: 4클래스 전부에 구간 빈도. 0 점도 순위에 넣는다(§11-2-1).

    빈 구간(적합 표본 0)은 적합 모집단 전체 빈도로 폴백하고 그 건수를 진단에 남긴다 —
    사전등록이 다루지 않은 기계적 경우라 값을 고르지 않고 세어서 보고한다.
    """
    if construction not in (HARD, FREQ):
        raise ValueError(f"점수 구성은 {HARD}(하드) 또는 {FREQ}(빈도)다: {construction!r}")
    b = _bin_of(rule, image_ids, snapshot)
    out: dict[str, list[tuple[str, float]]] = {}
    n_fallback = 0
    for i in image_ids:
        br = rule.bins.get(b[i])
        if br is None:
            n_fallback += 1
        if construction == HARD:
            codes = sorted(br.majority) if br is not None else []
            out[i] = [(c, 1.0) for c in codes]
        else:
            freq = br.freq if br is not None else rule.global_freq
            out[i] = [(c, float(freq.get(c, 0.0))) for c in rule.classes]
    return out, {"construction": construction, "n_images": len(image_ids),
                 "n_bin_fallback": n_fallback}
