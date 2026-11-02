"""엄격 경로의 2 차 보조 지표 — 서술(금지 주장률 · 유보 표기율 · 상태율)과 조항 축(인용 일치율 · 정밀도 · 무근거율).

03 2판 §5a(엄격 경로 · 구간 · 진입점) · §6a 결정 4. 진입점(`scripts/probe/score_unified.py`)이 검증된 묶음의 줄과 1.4 레코드에서
입력을 모으고, `evaluation.strict_main.compute` 가 M1 과 **같은 추첨 행렬 `W`** 를 넘겨 블록을 낸다(인자 `aux_scoring`). M1 · 대조선 · 회복률 칸은 이
모듈이 건드리지 않는다 — `aux_scoring` 을 주지 않으면 엄격 산출은 연결 전과 바이트가 같다(`tests/test_strict_main.py`).

## 언제 내는가

등록이 판 6 이고 그 블록이 있으며, 평가 등록의 보조 지표(`auxiliary`)에 그 이름이 있을 때만이다(03 2판 §5a — "등록 보조 지표에
그 이름이 있으면 계산한다"). 판 5 등록에는 블록이 없으므로 내지 않는다. 진입점이 `wanted_names` 로 그 조건을 맞댄다.

## 무엇을 내는가

| 블록 | 모델마다 | 비교 |
|---|---|---|
| `narrative` | `narrative_metrics.point_estimates` — 설계 가중 · 무가중, '해당 없음' + 적격 건수, 상태율, 보조 둘 | 대응 차(연합 − 중앙, 연합 − 로컬 k, 시드마다)의 보조 구간 — 같은 `W`, 분모 0 추첨 NaN, `min_defined_draws` |
| `clause_axis` | 대표 = 설계 가중(`clause_axis.clause_axis_weighted` — 이미지 가중, M1 · 서술 지표와 같은 `design_weights`), 부표 = 무가중(`clause_axis_report`) — 정답 (이미지, 코드) 쌍 · 이미지 수준 인용(`cited_clauses`) | 등록 대조선의 코드 → 조항(무화소 대조선)을 같은 정답 · 같은 가중으로. 대응 차 두 갈래 — 서술 지표와 같은 모델 짝(시드마다 연합 − 중앙 · 연합 − 로컬 k)과 모델마다 − 대조선, 같은 `W`(설계 가중은 `W`, 무가중은 `W` ÷ 이미지 설계 가중), 분모 0 추첨 NaN, `min_defined_draws`(서술 지표와 같은 규칙 — `CLAUSE_AXIS_INTERVAL`) |

대응 차의 짝은 등록이 정하지 않았다 — 코드가 정한 선택이라 산출의 `unregistered_choices` 에 근거와 함께 싣는다.

등록의 `narrative.exploratory_kinds`(확인 사례에 음성 오탐이 남은 종류)는 금지 주장률 표에서 **탐색적**으로 표시한다 — 그 종류의
`forbidden.<종류>` 와 그 종류를 품는 `forbidden.any`(머리 지표 · 대응 차 포함). 진입점이 무엇을 신원 대조했는지(`identity_checks`)를
서술 블록에 싣고, 신원이 맞지 않으면 블록 대신 `aux_identity` 에 사유를 싣는다(`AuxRefused` — 보조 지표만 내지 않는다).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from evaluation import narrative_metrics as NM
from evaluation.clause_axis import (
    citations_from_codes,
    clause_axis_draws,
    clause_axis_report,
    clause_axis_weighted,
    gold_pairs_without_thickness,
)
from evaluation.narrative_extract import EXTRACTOR_ID, extract

NARRATIVE_NAMES = ("narrative_forbidden_rate", "narrative_reservation_rate")
"""평가 등록 `auxiliary` 의 서술 지표 이름 — 금지 주장률(전체 · 종류별)과 유보 표기율(전체 · 사유별)."""
HEADLINE = {"narrative_forbidden_rate": "forbidden.any", "narrative_reservation_rate": "reservation.all"}
"""대응 차를 내는 머리 지표 — 이름마다 `narrative_metrics.indicators` 의 열쇠 하나."""
UNREGISTERED_PAIRS = {
    "choice": "서술 지표 대응 차의 짝 — 시드마다 연합 − 중앙 · 연합 − 로컬 C1 · C2 · C3",
    "why": ("03 2판 §6a 결정 4 는 '칸 사이 대응 차를 보조로' 까지만 정했다. 짝은 RQ1(연합 대 중앙)과 RQ2(참여자마다 연합 대 로컬)의 "
            "M1 비교와 같게 둔다 — 참여자 평균은 내지 않는다"),
}


EXPLORATORY_KIND = "탐색적 — 확인 사례에 음성 오탐이 남은 종류(등록 narrative.exploratory_kinds)"
EXPLORATORY_TOTAL = "탐색적 종류를 품는다 — 검증 완료 지표로 쓰지 않는다"
REFUSED = "판 6 블록의 신원이 지금 코드 · 검사 인자와 맞지 않아 내지 않았다 — aux_identity 의 사유"
CLAUSE_AXIS_WEIGHTING = ("design_and_unweighted — 대표 = 설계 가중(단위가 속한 이미지의 가중, M1 · 서술 지표와 같은 design_weights), "
                         "부표 = 무가중(score_citation · score_ungrounded 의 정의, 195 M4) — 04 §7-10 ①")
CLAUSE_AXIS_INTERVAL = ("대응 구간 = 같은 W(등록 clause_axis.ci = paired_cluster_bootstrap_aux_cells_and_baseline) — M1 · 서술 지표와 같은 "
                        "추첨 행렬(같은 이미지 순서)에서 대응 차 두 갈래: 서술 지표와 같은 모델 짝(시드마다 연합 − 중앙 · 연합 − 로컬 C1 · C2 · "
                        "C3)과 모델마다 − 무화소 조항 대조선. 설계 가중은 W 그대로, 무가중은 W ÷ 이미지 설계 가중(묶음 중복도). 분모가 0 인 "
                        "추첨은 NaN(미정의) — 유효 · 미정의 추첨 수를 싣고, 유효 추첨이 ci.min_defined_draws 보다 적으면 방향을 말하지 "
                        "않는다. 점추정의 분모가 0 이면 구간을 내지 않는다(해당 없음). 규칙은 서술 지표와 같다"
                        "(narrative_metrics.paired_difference · percentile_rule.defined_interval) — 지시 1023d · 검토 206 · 03 §9-19 열린 질문 14")
UNREGISTERED_CLAUSE_PAIRS = {
    "choice": "조항 축 대응 차의 모델 짝 — 서술 지표와 같은 짝(시드마다 연합 − 중앙 · 연합 − 로컬 C1 · C2 · C3)",
    "why": ("공개 설계 8-3 의 '서술 지표와 같은 추첨과 가중 행렬에서 대응 차'(03 §9-19 열린 질문 14 의 답, 10-23). 등록 칸 clause_axis.ci 는 "
            "두 갈래(모델 짝 · 대조선)를 정하고, 모델 짝의 구성은 서술 지표의 짝과 같은 까닭으로 등록이 정하지 않았다"),
}


class AuxInputError(ValueError):
    """보조 지표의 입력이 엄격 경로의 계약과 맞지 않는다."""


def mark_exploratory(point: dict, kinds: Sequence[str]) -> dict:
    """모델 하나의 점추정에 탐색적 표지를 단다 — 그 종류의 `forbidden.<종류>` 와, 그 종류를 품는 `forbidden.any`."""
    if kinds:
        for k in kinds:
            point[f"forbidden.{k}"]["status"] = EXPLORATORY_KIND
        point["forbidden.any"]["status"] = EXPLORATORY_TOTAL
    return point


@dataclass(frozen=True)
class ModelAux:
    """한 모델 칸의 보조 입력 — 이미지 순서는 모집단(대조선 계수표)의 순서다."""

    tag: str
    seed: int
    items: tuple[NM.ImageItem, ...]
    cited: Mapping[str, tuple[str, ...]]


def material_of_side(side: Mapping, image_ids: Sequence[str], materials: Sequence[str] = ("AL", "ST")) -> dict[str, str]:
    """곁 파일의 층 이름 머리(`AL|…` · `ST|…`)에서 이미지의 재질을 낸다. 모르는 머리면 멈춘다."""
    out = {}
    for i in image_ids:
        head = side[i].stratum.split("|", 1)[0]
        if head not in materials:
            raise AuxInputError(f"층 이름 {side[i].stratum!r} 의 재질 머리가 {list(materials)} 가 아니다")
        out[i] = head
    return out


def model_inputs(tag: str, seed: int, lines: Iterable[str], records, *, image_ids: Sequence[str],
                 material_of: Mapping[str, str], gold_codes: Mapping[str, Iterable[str]],
                 weights: Sequence[float]) -> ModelAux:
    """검증된 묶음의 생성 줄과 같은 줄에서 낸 1.4 레코드 → 이미지마다의 보조 입력.

    서술은 생성 줄의 `text` · `gen_stop` 에서 추출기(`narrative_extract.extract`)가 낸다 — 칸이 없으면 인자를 빼고 부른다. 구조 코드와
    인용은 레코드에서 온다(레코드 실패는 빈 예측이다). 모집단의 이미지마다 줄과 레코드가 정확히 하나여야 한다.
    """
    ids = list(image_ids)
    want = set(ids)
    text_of: dict[str, tuple] = {}
    for raw in lines:
        row = json.loads(raw)
        i = row.get("image_id")
        if i not in want:
            continue
        if i in text_of:
            raise AuxInputError("모집단의 한 이미지에 생성 줄이 둘이다")
        text_of[i] = (row.get("text"), row["gen_stop"]) if "gen_stop" in row else (row.get("text"),)
    rec_of = {}
    for r in records:
        if r.image_id in want:
            if r.image_id in rec_of:
                raise AuxInputError("모집단의 한 이미지에 레코드가 둘이다")
            rec_of[r.image_id] = r
    if set(text_of) != want or set(rec_of) != want:
        raise AuxInputError("모집단의 이미지마다 생성 줄과 레코드가 하나씩 있어야 한다")
    if len(weights) != len(ids):
        raise AuxInputError("설계 가중의 길이가 모집단과 다르다")
    items = []
    for i, w in zip(ids, weights, strict=True):
        ex = extract(*text_of[i])
        codes = tuple(sorted({d.iso_code for d in rec_of[i].defects}))
        items.append(NM.ImageItem(i, ex, material_of[i], tuple(sorted(set(map(str, gold_codes.get(i, ()))))), codes,
                                  float(w)))
    cited = {i: tuple(rec_of[i].cited_clauses) for i in ids}
    return ModelAux(tag, seed, tuple(items), cited)


@dataclass
class AuxScoring:
    """`strict_main.compute` 에 넘기는 보조 지표 묶음 — 진입점이 조건을 맞춘 뒤에만 만든다."""

    models: Mapping[tuple[str, int], ModelAux]
    checker: object
    checker_identity: Mapping
    entries: Sequence
    index_clause_ids: frozenset
    material_of: Mapping[str, str]
    gold_codes: Mapping[str, frozenset]
    baseline_codes: Mapping[str, Sequence[str]]
    wanted: frozenset
    image_weights: Mapping[str, float]
    """이미지마다의 설계 가중 — M1 · 서술 지표와 같은 `design_weights`. 조항 축의 대표표가 쓴다(가중을 1 로 메우지 않는다)."""
    exploratory_kinds: tuple[str, ...]
    """등록의 `narrative.exploratory_kinds` — 신원 대조가 검사기 판의 정본과 맞댄 값."""
    identity_checks: Mapping[str, str]
    """진입점이 신원을 무엇으로 맞댔는가 — 칸마다 `checked` · `not_given`."""

    def computed_names(self) -> list[str]:
        from evaluation.prereg_unified import CLAUSE_AXIS_METRICS

        return [n for n in (*NARRATIVE_NAMES, *CLAUSE_AXIS_METRICS) if n in self.wanted]

    def not_computed_reasons(self) -> dict[str, str]:
        return {}

    def report(self, *, image_ids: Sequence[str], W: np.ndarray | None, alpha: float, min_defined_draws: int,
               seeds: Sequence[int], cell_name, fed: str, central: str, locals_: Sequence[str],
               comparable: bool) -> dict:
        """두 블록. `W` 가 없거나 `comparable` 이 거짓이면(환경 불일치) 모델마다의 값만 싣고 비교는 `산출 불가` 다."""
        from evaluation.prereg_unified import CLAUSE_AXIS_METRICS

        out: dict = {}
        ids = list(image_ids)
        if set(self.image_weights) != set(ids):
            raise AuxInputError("이미지 가중의 이미지가 모집단과 다르다")
        for key, m in self.models.items():
            if [it.image_id for it in m.items] != ids:
                raise AuxInputError(f"{cell_name(*key)}: 보조 입력의 이미지 순서가 모집단과 다르다")
            if any(it.weight != self.image_weights[it.image_id] for it in m.items):
                raise AuxInputError(f"{cell_name(*key)}: 보조 입력의 가중이 이미지 가중과 다르다")
        if any(n in self.wanted for n in NARRATIVE_NAMES):
            rows = {key: NM.build_rows(m.items, self.checker) for key, m in self.models.items()}
            point = {key: mark_exploratory(NM.point_estimates(r), self.exploratory_kinds) for key, r in rows.items()}
            ek = list(self.exploratory_kinds)
            nar: dict = {"checker": dict(self.checker_identity), "extractor_id": EXTRACTOR_ID,
                         "identity_checks": dict(self.identity_checks),
                         "exploratory": {"kinds": ek, "metrics": [*(f"forbidden.{k}" for k in ek), *(["forbidden.any"] if ek else [])],
                                         "rule": "03 2판 §6a 결정 5 · 199 결정 문서 §2 — 음성 오탐이 남은 종류는 탐색적으로 보고"},
                         "names": [n for n in NARRATIVE_NAMES if n in self.wanted],
                         "models": {cell_name(*key): p for key, p in sorted(point.items())},
                         "unregistered_choices": [dict(UNREGISTERED_PAIRS)]}
            if W is None or not comparable:
                nar["paired"] = {"status": "산출 불가", "reason": "env_mismatch" if not comparable else "no_draws"}
            else:
                draws = {key: NM.draws(r, W) for key, r in rows.items()}
                paired: dict = {}
                for s in seeds:
                    for other in (central, *locals_):
                        a, b = (fed, s), (other, s)
                        label = f"{cell_name(*a)} − {cell_name(*b)}"
                        if a not in point or b not in point:
                            paired[label] = {"status": "산출 불가", "missing": [cell_name(*k) for k in (a, b)
                                                                               if k not in point]}
                            continue
                        paired[label] = {
                            name: {**NM.paired_difference(point[a][ind]["weighted"], point[b][ind]["weighted"],
                                                          draws[a][ind], draws[b][ind], alpha=alpha,
                                                          min_defined_draws=min_defined_draws),
                                   **({"status": EXPLORATORY_TOTAL} if ind == "forbidden.any" and ek else {})}
                            for name, ind in HEADLINE.items() if name in self.wanted}
                nar["paired"] = paired
                nar["draws"] = {"n_resamples": int(W.shape[0]), "shared_with_m1": True}
            out["narrative"] = nar
        if any(n in self.wanted for n in CLAUSE_AXIS_METRICS):
            gt = {i: sorted(self.gold_codes.get(i, ())) for i in ids}
            pairs, skipped = gold_pairs_without_thickness(self.entries, self.material_of, gt)
            base_cited = citations_from_codes(self.entries, self.material_of,
                                              {i: list(self.baseline_codes.get(i, ())) for i in ids})
            ca: dict = {
                "names": [n for n in CLAUSE_AXIS_METRICS if n in self.wanted],
                "gold": {"n_pairs": len(pairs), "skipped": dict(skipped),
                         "rule": "허용치 표의 (재질, 정답 코드) RT 행 조항 — 두께 구간과 상관없이(03 2판 §4)"},
                "models": {cell_name(*key): {
                    "design": clause_axis_weighted(dict(m.cited), pairs, self.index_clause_ids, self.image_weights),
                    "unweighted": clause_axis_report(dict(m.cited), pairs, self.index_clause_ids)}
                    for key, m in sorted(self.models.items())},
                "baseline": {"design": clause_axis_weighted(base_cited, pairs, self.index_clause_ids, self.image_weights),
                             "unweighted": clause_axis_report(base_cited, pairs, self.index_clause_ids)},
                "weighting": CLAUSE_AXIS_WEIGHTING,
                "interval": CLAUSE_AXIS_INTERVAL,
            }
            if W is None or not comparable:
                ca["paired"] = {"status": "산출 불가", "reason": "env_mismatch" if not comparable else "no_draws"}
            else:
                ca["paired"] = self._clause_paired(ca, ids, pairs, base_cited, W, alpha=alpha,
                                                   min_defined_draws=min_defined_draws, cell_name=cell_name, seeds=seeds,
                                                   fed=fed, central=central, locals_=locals_)
                ca["unregistered_choices"] = [dict(UNREGISTERED_CLAUSE_PAIRS)]
                ca["draws"] = {"n_resamples": int(np.asarray(W).shape[0]), "shared_with_m1": True,
                               "unweighted": "W ÷ 이미지 설계 가중(묶음 중복도)"}
            out["clause_axis"] = ca
        return out

    def _clause_paired(self, ca: dict, ids: list[str], pairs, base_cited, W, *, alpha: float, min_defined_draws: int,
                       cell_name, seeds: Sequence[int], fed: str, central: str, locals_: Sequence[str]) -> dict:
        """대응 차 두 갈래(설계 가중 · 무가중) — 서술 지표와 같은 모델 짝(시드마다 연합 − 중앙 · 연합 − 로컬 k)과 모델마다 − 무화소
        조항 대조선. 같은 `W` 로 추첨마다 값을 내고, 점추정은 `ca` 에 이미 실은 값을 쓴다. 짝의 한쪽 칸이 없으면 그 짝은 `산출 불가`."""
        from evaluation.prereg_unified import CLAUSE_AXIS_METRICS

        w = np.array([self.image_weights[i] for i in ids], dtype=np.float64)
        if (w <= 0).any() or not np.isfinite(w).all():
            raise AuxInputError("이미지 설계 가중이 양의 유한한 수가 아니다 — 무가중 추첨(W ÷ 가중)을 낼 수 없다")
        W = np.asarray(W, dtype=np.float64)
        views = {"design": W, "unweighted": W / w[None, :]}
        names = [n for n in CLAUSE_AXIS_METRICS if n in self.wanted]
        draws = {key: {v: clause_axis_draws(dict(m.cited), pairs, self.index_clause_ids, ids, Wv) for v, Wv in views.items()}
                 for key, m in self.models.items()}
        base_draws = {v: clause_axis_draws(base_cited, pairs, self.index_clause_ids, ids, Wv) for v, Wv in views.items()}

        def diff(point_a: dict, point_b: dict, draws_a: dict, draws_b: dict) -> dict:
            return {name: {v: NM.paired_difference(point_a[v][name], point_b[v][name], draws_a[v][name], draws_b[v][name],
                                                   alpha=alpha, min_defined_draws=min_defined_draws) for v in views}
                    for name in names}

        out: dict = {}
        for s in seeds:
            for other in (central, *locals_):
                a, b = (fed, s), (other, s)
                label = f"{cell_name(*a)} − {cell_name(*b)}"
                if a not in self.models or b not in self.models:
                    out[label] = {"status": "산출 불가", "missing": [cell_name(*k) for k in (a, b) if k not in self.models]}
                    continue
                out[label] = diff(ca["models"][cell_name(*a)], ca["models"][cell_name(*b)], draws[a], draws[b])
        for key in sorted(self.models):
            out[f"{cell_name(*key)} − 대조선"] = diff(ca["models"][cell_name(*key)], ca["baseline"], draws[key], base_draws)
        return out


@dataclass
class AuxRefused:
    """판 6 블록의 신원이 지금 코드 · 검사 인자와 맞지 않을 때 — 보조 지표만 내지 않고 사유를 산출에 싣는다(지시 1023c 과제 2).

    M1 · 대조선 · 회복률 칸은 그대로 나온다. 원했던 이름은 `not_computed` 에 거부 사유와 함께 남는다.
    """

    wanted: frozenset
    problems: tuple[str, ...]
    identity_checks: Mapping[str, str]

    def computed_names(self) -> list[str]:
        return []

    def not_computed_reasons(self) -> dict[str, str]:
        from evaluation.prereg_unified import CLAUSE_AXIS_METRICS

        return {n: REFUSED for n in (*NARRATIVE_NAMES, *CLAUSE_AXIS_METRICS) if n in self.wanted}

    def report(self, **_) -> dict:
        from evaluation.prereg_unified import CLAUSE_AXIS_METRICS

        return {"aux_identity": {"status": "refused",
                                 "names": [n for n in (*NARRATIVE_NAMES, *CLAUSE_AXIS_METRICS) if n in self.wanted],
                                 "problems": list(self.problems), "checks": dict(self.identity_checks),
                                 "note": "보조 지표를 내지 않았다 — M1 · 대조선 · 회복률 칸은 그대로다"}}


__all__ = ["CLAUSE_AXIS_INTERVAL", "CLAUSE_AXIS_WEIGHTING", "EXPLORATORY_KIND", "EXPLORATORY_TOTAL", "HEADLINE",
           "NARRATIVE_NAMES", "REFUSED", "UNREGISTERED_CLAUSE_PAIRS",
           "UNREGISTERED_PAIRS", "AuxInputError", "AuxRefused", "AuxScoring", "ModelAux", "mark_exploratory",
           "material_of_side", "model_inputs"]
