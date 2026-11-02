"""2 차 조항 축 — 정답 조항을 허용치 표에서 두께 없이 내고, 인용 일치율 · 정밀도 · 무근거 인용률(지어냄 · 무관)을 센다.

03 2판 §4 · §5a · 04 §7-5 ⑦ · 지시 1022b 과제 3 · 195 M4.

## 왜 새 파일인가

03 2판 §5a 는 이 함수들을 `evaluation/metrics/clause.py`(와 정답 조회의 `evaluation/gold.py`)에 두라고 적었다. 두 파일은
`tests/test_scrub_was_comment_only.py` 의 구문 트리 고정(`SCRUB_BASELINE`)에 걸려 있다 — 사전실험 v3 산출물이 싣는 파일별
해시에서 지금 바이트까지 잇는 사슬이다. 그래서 두 파일은 바꾸지 않고(main 과 바이트가 같다) 새 기능을 새 파일에 둔다(D1 의
규칙 — `tests/test_d1_baseline_frozen.py` 의 머리말). 기존 `score_citation` · `score_ungrounded` 의 정의를 그대로 부른다.

## 정답

허용치 표(`corpus.rules.limits_loader.load_limits`)에서 B 의 `derive_gold_clauses` 를 메모리에서 불러(`gold.entries_from_derived`)
(재질, 정답 코드)의 RT 행 조항을 두께 · 품질 구간과 상관없이 모은다. 파생 CSV(`corpus/derived/gold_clauses.csv`)는 읽지 않는다 —
정답이 단일 소스(허용치 CSV) 바이트에서 바로 나와야 등록의 `limits_sha256` 하나로 묶인다(03 2판 §4 의 2). D 가 표를 따로
순회하지 않는다(B 스펙 §1-4 금지 규칙 ④).

## 복수 조항 정책 — 주 조항 하나 + 보조 조항의 분해(07 4판 §13 · §14, 등록 칸 `clause_axis.multi_clause`)

(재질, 결함 코드)마다 표가 조건 없이 대응시킨 조항 하나가 정답(주 조항)이다. 조건이 있을 때만 대응되는 조항(보조 조항 — 종별 구별 곤란)은
정답이 아니며 무관한 인용 안에서 따로 센다. 주 조항이 둘 이상이면 채점을 멈춘다(`MultipleMainClauses`) — 그 쌍을 조용히 빼지 않는다.
이미지 단위의 규칙은 다섯이다: ① 한 이미지에서 어떤 조항이 한 코드의 보조이면서 다른 코드의 주이면 정답 인용이다 — 보조 전용 = 그 이미지의
보조 합집합 − 주 합집합, ② 보조만 인용하면 주 조항 일치는 실패다, ③ 정답 쌍 분모와 인용 분모를 따로 둔다(주가 없는 코드 · AL 만 있는
이미지의 인용도 인용 분모에 남는다), ④ 인용 발생 횟수를 분모로 둔다(분해도 발생 횟수로), ⑤ 서술의 조건부 소개 예외는 구조화 인용으로
옮기지 않는다 — 구조화 인용은 이미지의 주 조항 합집합으로 판정한다. 정책마다 분모가 다른 값을 같은 지표의 우열로 비교하지 않는다.

보조 조항은 생성 쪽 `derive_clause_roles` 의 출력에서 받는다(`roles_from_derived` — 인터페이스 문서 07e §5 의 꼴). 1 판 표(조건 칸 없음)는
`roles_from_entries` 가 주 = 그 코드의 조항 하나, 보조 = 없음으로 낸다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from evaluation.gold import METHOD_ANY, GoldEntry, entries_from_derived
from evaluation.metrics.clause import GoldPair, _gold_map, score_citation, score_ungrounded


def entries_from_limits(table: object) -> tuple[GoldEntry, ...]:
    """허용치 표에서 B 의 `derive_gold_clauses` 를 **메모리에서** 불러 정답 조항 행을 낸다 — `gold_clauses.csv` 파일은 읽지 않는다.

    해석 코드는 B 의 그 함수 하나다 — D 가 표를 따로 순회하지 않는다(B 스펙 §1-4 금지 규칙 ④). 파일을 거치지 않는 까닭은 정답이
    단일 소스(허용치 CSV) 바이트에서 바로 나와야 등록의 `limits_sha256` 하나로 묶이기 때문이다(03 2판 §4 의 2).
    """
    from corpus.rules.limit_eval import derive_gold_clauses

    return entries_from_derived(derive_gold_clauses(table))


class MultipleMainClauses(ValueError):
    """한 (재질, 결함 코드)에 조건 없이 대응되는 조항이 둘 이상 — 채점을 멈춘다(주 조항은 하나다)."""

    def __init__(self, material: str, code: str, clauses: Sequence[str]):
        self.material, self.code, self.clauses = material, code, tuple(clauses)
        super().__init__(f"주 조항이 둘 이상이다 — ({material}, {code}) → {list(self.clauses)}. 그 쌍을 조용히 빼지 않는다")


class RolesFormError(ValueError):
    """생성 쪽 역할 출력의 꼴이 인터페이스 문서 07e §5 와 다르다 — 받지 않는다."""


AUXILIARY_CONDITION = "구별곤란"
"""보조 조항을 내는 조건 종류 — 하나다. 혼재는 영상의 코드 조합에 걸리는 조건이라 보조를 내지 않는다(08 §1-3)."""

ROLE_KEYS = frozenset({"main", "auxiliary", "defect_class", "conditions"})


@dataclass(frozen=True)
class ClauseRole:
    """(재질, 결함 코드)의 역할 — 주 조항(없으면 None)과 보조 조항(조건이 있을 때만 대응되는 조항, 정렬)."""

    main: str | None
    auxiliary: tuple[str, ...] = ()


def roles_from_entries(entries: Sequence[GoldEntry], *, materials: Iterable[str],
                       inspection_method: str = "RT") -> dict[tuple[str, str], ClauseRole]:
    """조건 칸이 없는 표(1 판)의 역할 — 주 = 그 코드의 조항 하나, 보조 = 없음. 조항이 없는 코드는 사전에 없다.

    Raises:
        MultipleMainClauses: 한 (재질, 코드)에 조항이 둘 이상이면.
    """
    out: dict[tuple[str, str], ClauseRole] = {}
    for material in sorted(set(materials)):
        for code, cids in clauses_by_code(entries, material=material, inspection_method=inspection_method).items():
            if len(cids) > 1:
                raise MultipleMainClauses(material, code, cids)
            out[(material, code)] = ClauseRole(cids[0])
    return out


def roles_from_derived(derived: Mapping) -> dict[tuple[str, str], ClauseRole]:
    """생성 쪽 `derive_clause_roles` 출력의 꼴을 맞대어 받는다 — 열쇠 `(재질, 코드)`, 값 `{main, auxiliary, defect_class, conditions}`.

    `main` 은 조항 ID 하나 또는 None, `auxiliary` 는 `(조항 ID, "구별곤란")` 의 목록이다. 평가 쪽은 두 칸의 조항 ID 만 쓴다. 보조가 제 주
    조항과 같거나 겹치거나, 조건 종류가 `구별곤란` 밖이면 받지 않는다.

    Raises:
        RolesFormError: 꼴이 다르면.
    """
    if not isinstance(derived, Mapping):
        raise RolesFormError("역할 출력은 (재질, 코드) → 역할의 사전이어야 한다")
    out: dict[tuple[str, str], ClauseRole] = {}
    for key, role in derived.items():
        if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) and k for k in key)):
            raise RolesFormError(f"열쇠는 (재질, 코드) 두 문자열이다 — {key!r}")
        if not isinstance(role, Mapping) or set(role) != ROLE_KEYS:
            raise RolesFormError(f"{key}: 역할의 칸은 {sorted(ROLE_KEYS)} 이다")
        main = role["main"]
        if main is not None and not (isinstance(main, str) and main):
            raise RolesFormError(f"{key}: main 은 조항 ID 하나 또는 None 이다 — {main!r}")
        aux = role["auxiliary"]
        if not isinstance(aux, (tuple, list)):
            raise RolesFormError(f"{key}: auxiliary 는 목록이다")
        ids: list[str] = []
        for item in aux:
            if not (isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str) and item[0]
                    and item[1] == AUXILIARY_CONDITION):
                raise RolesFormError(f"{key}: auxiliary 의 원소는 (조항 ID, {AUXILIARY_CONDITION!r}) 이다 — {item!r}")
            ids.append(item[0])
        if main is not None and main in ids:
            raise RolesFormError(f"{key}: 보조가 제 주 조항과 같다 — {main!r}")
        if len(set(ids)) != len(ids):
            raise RolesFormError(f"{key}: 보조가 겹친다")
        if role["defect_class"] is not None and not isinstance(role["defect_class"], str):
            raise RolesFormError(f"{key}: defect_class 는 문자열 또는 None 이다")
        if not isinstance(role["conditions"], (tuple, list)) or not all(isinstance(x, str) for x in role["conditions"]):
            raise RolesFormError(f"{key}: conditions 는 토큰 문자열의 목록이다")
        out[key] = ClauseRole(main, tuple(sorted(ids)))
    return out


def gold_pairs_from_roles(
    roles: Mapping[tuple[str, str], ClauseRole],
    material_of: Mapping[str, str],
    gt_codes: Mapping[str, Sequence[str]],
) -> tuple[tuple[GoldPair, ...], dict[str, int]]:
    """(이미지 × 정답 코드) 정답 쌍 — 정답은 주 조항이다. 주가 없는 코드는 보조만 있어도 `정답 조항 없음` 으로 세고 쌍을 만들지 않는다.

    역할이 1 판 표의 것(`roles_from_entries`)이면 `gold_pairs_without_thickness` 와 같은 쌍 · 같은 사유를 낸다.

    Raises:
        KeyError: 정답 코드가 있는 이미지의 재질이 `material_of` 에 없으면.
    """
    pairs: list[GoldPair] = []
    skipped: dict[str, int] = {}
    for img in sorted(gt_codes):
        if img not in material_of:
            raise KeyError(f"재질을 모르는 이미지가 있다 — 정답 쌍을 만들지 않는다(id 생략, {len(gt_codes)} 장 가운데)")
        material = material_of[img]
        for code in sorted({str(c) for c in gt_codes[img]}):
            role = roles.get((material, code))
            if role is None or role.main is None:
                skipped["정답 조항 없음"] = skipped.get("정답 조항 없음", 0) + 1
            else:
                pairs.append(GoldPair(img, code, role.main))
    return tuple(pairs), skipped


def auxiliary_only_by_image(
    roles: Mapping[tuple[str, str], ClauseRole],
    material_of: Mapping[str, str],
    gt_codes: Mapping[str, Sequence[str]],
) -> dict[str, frozenset[str]]:
    """이미지마다 보조 전용 조항 = 그 이미지 정답 코드들의 보조 합집합 − 주 합집합(규칙 ①). 다른 코드의 주인 조항은 보조로 세지 않는다."""
    out: dict[str, frozenset[str]] = {}
    for img in sorted(gt_codes):
        if img not in material_of:
            raise KeyError(f"재질을 모르는 이미지가 있다(id 생략, {len(gt_codes)} 장 가운데)")
        material = material_of[img]
        mains: set[str] = set()
        aux: set[str] = set()
        for code in {str(c) for c in gt_codes[img]}:
            role = roles.get((material, code))
            if role is None:
                continue
            if role.main is not None:
                mains.add(role.main)
            aux.update(role.auxiliary)
        out[img] = frozenset(aux - mains)
    return out


def citations_from_roles(
    roles: Mapping[tuple[str, str], ClauseRole],
    material_of: Mapping[str, str],
    codes_by_image: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """무화소 대조선의 인용 — 코드의 **주 조항만**(정렬 · 한 번씩). 주가 없는 코드는 인용을 만들지 않는다(보조로 메우지 않는다)."""
    out: dict[str, list[str]] = {}
    for img in sorted(codes_by_image):
        material = material_of[img]
        mains = {roles[(material, str(c))].main for c in codes_by_image[img] if (material, str(c)) in roles}
        out[img] = sorted(m for m in mains if m is not None)
    return out


def clauses_by_code(entries: Sequence[GoldEntry], *, material: str,
                    inspection_method: str = "RT") -> dict[str, tuple[str, ...]]:
    """(재질, 검사 방법)에서 결함 코드 → 정답 조항. **두께 · 품질 구간을 묻지 않는다** — 그림에 두께가 없다(03 2판 §4a).

    한 코드의 여러 두께 구간이 같은 조항을 가리키면 그 조항 하나다. 조항이 없는 코드는 사전에 없다.
    """
    out: dict[str, set[str]] = {}
    for e in entries:
        if e.material in (material, METHOD_ANY) and e.matches_method(inspection_method):
            out.setdefault(e.defect_code, set()).add(e.clause_id)
    return {k: tuple(sorted(v)) for k, v in sorted(out.items())}


def gold_pairs_without_thickness(
    entries: Sequence[GoldEntry],
    material_of: Mapping[str, str],
    gt_codes: Mapping[str, Sequence[str]],
    *,
    inspection_method: str = "RT",
) -> tuple[tuple[GoldPair, ...], dict[str, int]]:
    """(이미지 × 정답 코드) 정답 쌍 — 2 차 조항 축의 분모. 정답 코드는 **정답 뷰**에서 온다(예측 코드가 아니다 — 195 M4).

    Returns:
        `(쌍들, 건너뛴 사유별 건수)`. 조항이 없는 (재질, 코드) — AL 전부 · 표에 없는 강재 코드 — 는 `정답 조항 없음` 으로 세고 쌍을
        만들지 않는다. 그래서 일치율의 분모에서 빠지지만, 그 이미지가 낸 인용은 `score_citation` 의 정밀도와 `score_ungrounded` 에
        그대로 들어간다(03 2판 §4 의 3).

    Raises:
        KeyError: 정답 코드가 있는 이미지의 재질이 `material_of` 에 없으면. 재질을 모르는 채로 쌍을 지어내지 않는다.
        MultipleMainClauses: 한 (재질, 코드)에 조건 없이 대응되는 조항이 둘 이상이면 — 예전에는 `복수 조항` 으로 세고 건너뛰었다
            (07 4판 §2-1 · §9 — 분모가 줄고 맞는 인용이 무관이 되는 길). 1 판 표에는 그런 코드가 없다.
    """
    pairs: list[GoldPair] = []
    skipped: dict[str, int] = {}
    tables: dict[str, dict[str, tuple[str, ...]]] = {}
    for img in sorted(gt_codes):
        if img not in material_of:
            raise KeyError(f"재질을 모르는 이미지가 있다 — 정답 쌍을 만들지 않는다(id 생략, {len(gt_codes)} 장 가운데)")
        material = material_of[img]
        if material not in tables:
            tables[material] = clauses_by_code(entries, material=material, inspection_method=inspection_method)
        for code in sorted(set(gt_codes[img])):
            cids = tables[material].get(code, ())
            if not cids:
                skipped["정답 조항 없음"] = skipped.get("정답 조항 없음", 0) + 1
            elif len(cids) > 1:
                raise MultipleMainClauses(material, code, cids)
            else:
                pairs.append(GoldPair(img, code, cids[0]))
    return tuple(pairs), skipped


def citations_from_codes(
    entries: Sequence[GoldEntry],
    material_of: Mapping[str, str],
    codes_by_image: Mapping[str, Sequence[str]],
    *,
    inspection_method: str = "RT",
) -> dict[str, list[str]]:
    """코드 → 조항으로 낸 이미지별 인용 목록 — 무화소 대조선(등록 대조선의 코드 → 조항, 03 2판 §4 의 1)이 쓴다.

    이미지마다 그 코드들이 가리키는 조항을 한 번씩(정렬) 싣는다. 조항이 없는 코드는 인용을 만들지 않는다. 대조선도 주 조항만 인용한다 —
    한 코드에 조건 없는 대응이 둘 이상이면 `MultipleMainClauses` 로 멈춘다(정답 쌍과 같은 식).
    """
    tables: dict[str, dict[str, tuple[str, ...]]] = {}
    out: dict[str, list[str]] = {}
    for img in sorted(codes_by_image):
        material = material_of[img]
        if material not in tables:
            tables[material] = clauses_by_code(entries, material=material, inspection_method=inspection_method)
        for code in codes_by_image[img]:
            got = tables[material].get(str(code), ())
            if len(got) > 1:
                raise MultipleMainClauses(material, str(code), got)
        cids = {c for code in codes_by_image[img] for c in tables[material].get(str(code), ())}
        out[img] = sorted(cids)
    return out


def rate(numerator: int, denominator: int) -> dict:
    """분모가 0 이면 값 대신 `해당 없음` — 0 이나 1 로 메우지 않는다(04 §1-8)."""
    if denominator == 0:
        return {"value": None, "state": "해당 없음", "numerator": numerator, "denominator": 0}
    return {"value": numerator / denominator, "state": "finite", "numerator": numerator, "denominator": denominator}


def clause_axis_report(
    cited_by_image: Mapping[str, Sequence[str]],
    gold: Iterable[GoldPair],
    index_clause_ids: Iterable[str],
    *,
    auxiliary_only: Mapping[str, Iterable[str]] | None = None,
) -> dict:
    """2 차 조항 축 셋 — 인용 일치율 · 인용 정밀도 · 무근거 인용률(지어냄 · 무관)과 코드별 일치율.

    단위는 기존 두 함수의 정의 그대로다(195 M4).
    - 일치율(`score_citation`): 분모 = 정답 (이미지, 코드) 쌍. 쌍의 정답 조항이 그 **이미지의** 인용 목록에 있으면 맞음 — 인용은
      이미지 수준이라 결함마다 어느 인용이 짝인지(귀속)는 보지 않는다.
    - 정밀도(`score_citation`): 분모 = 인용 목록의 원소 전부(같은 조항을 두 번 적으면 두 번 센다), 맞음 = 그 이미지의 정답 조항
      집합 안. 정답 쌍이 없는 이미지(AL · 정상)의 인용은 모두 틀림으로 들어간다.
    - 무근거율(`score_ungrounded`): 분모는 정밀도와 같다. 표에 없는 조항 = 지어냄, 표에 있으나 그 이미지의 정답 밖 = 무관.
    인용 목록은 구조 필드 `cited_clauses` 에서만 온다 — 서술 안의 조항 언급은 여기 더하지 않는다(04 §1-7).

    `auxiliary_only`(이미지 → 보조 전용 조항, `auxiliary_only_by_image`)를 주면 무관 가운데 보조 전용 인용의 수(발생 횟수)를
    `ungrounded_rate.auxiliary_only` 에 싣는다 — 분해 칸이라 점추정만이다. 주지 않으면 지금 바이트 그대로다.
    """
    gold = tuple(gold)
    index = set(index_clause_ids)
    cit = score_citation(cited_by_image, gold)
    ung = score_ungrounded(cited_by_image, index, gold)
    hits = sum(int(v["hit"]) for v in cit.per_code.values())
    allowed: dict[str, set[str]] = {}
    for (img, _), g in _gold_map(gold).items():                 # score_citation 의 정답 집합과 같은 식
        allowed.setdefault(img, set()).add(g.clause_id)
    correct = sum(c in allowed.get(img, set()) for img, cites in cited_by_image.items() for c in cites)
    if ung.n_citations and correct / ung.n_citations != cit.precision:
        raise AssertionError("정밀도의 분자가 score_citation 의 정의와 갈렸다")
    rep = {
        "citation_match_rate": rate(hits, cit.n_pairs),
        "citation_precision": rate(correct, ung.n_citations),
        "ungrounded_rate": {**rate(ung.nonexistent + ung.irrelevant, ung.n_citations),
                            "nonexistent": ung.nonexistent, "irrelevant": ung.irrelevant},
        "per_code": {code: {**rate(int(v["hit"]), int(v["n"]))} for code, v in sorted(cit.per_code.items())},
        "n_gold_pairs": cit.n_pairs,
        "n_citations": ung.n_citations,
        "unit": "일치율 = 정답 (이미지, 코드) 쌍 · 인용 = 이미지 수준 목록(귀속 검증 아님)",
    }
    if auxiliary_only is not None:
        rep["ungrounded_rate"]["auxiliary_only"] = _auxiliary_only_count(cited_by_image, allowed, index, auxiliary_only)
    return rep


def _auxiliary_only_count(cited_by_image: Mapping[str, Sequence[str]], allowed: Mapping[str, set[str]], index: set[str],
                          auxiliary_only: Mapping[str, Iterable[str]], weights: Mapping[str, float] | None = None):
    """무관 가운데 보조 전용 인용의 수(발생 횟수 — `weights` 를 주면 이미지 가중의 합). 보조 전용은 주 합집합과 겹칠 수 없다."""
    aux = {img: set(v) for img, v in auxiliary_only.items()}
    clash = sorted(img for img, s in aux.items() if s & allowed.get(img, set()))
    if clash:
        raise ValueError(f"보조 전용 집합이 그 이미지의 주 조항 합집합과 겹친다({len(clash)} 장) — 보조 전용 = 보조 합집합 − 주 합집합")
    n, w = 0, 0.0
    for img, cites in cited_by_image.items():
        for c in cites:
            if c in index and c not in allowed.get(img, set()) and c in aux.get(img, set()):
                n += 1
                w += 0.0 if weights is None else float(weights[img])
    return n if weights is None else w


def weighted_rate(numerator: float, denominator: float, n: int) -> dict:
    """가중 비율 — 적격 단위가 없으면(`n` = 0) `해당 없음`. 적격 단위가 있는데 가중 합이 0 이면 멈춘다(가중을 지어내지 않는다)."""
    if n == 0:
        return {"value": None, "state": "해당 없음", "numerator": 0.0, "denominator": 0.0, "n": 0}
    if denominator <= 0:
        raise ValueError("적격 단위가 있는데 가중 합이 0 이하다 — 설계 가중을 확인한다")
    return {"value": numerator / denominator, "state": "finite", "numerator": numerator, "denominator": denominator, "n": n}


def clause_axis_weighted(
    cited_by_image: Mapping[str, Sequence[str]],
    gold: Iterable[GoldPair],
    index_clause_ids: Iterable[str],
    weights: Mapping[str, float],
    *,
    auxiliary_only: Mapping[str, Iterable[str]] | None = None,
) -> dict:
    """조항 축 셋의 **설계 가중** — `clause_axis_report` 와 같은 단위에 이미지 가중을 곱한다(04 §7-10 ①, 대표표).

    단위는 그대로다(정의를 바꾸지 않는다). 일치율의 단위 = 정답 (이미지, 코드) 쌍, 정밀도 · 무근거율의 단위 = 인용 원소 — 그 단위가
    속한 이미지의 가중 `weights[image]` 로 분자 · 분모를 더한다. 가중이 모두 1 이면 분자 · 분모가 무가중 셈(`score_citation` ·
    `score_ungrounded`)의 수와 같아야 한다 — 같은 단위를 같은 차례로 세는지 그 자리에서 맞댄다.

    Raises:
        KeyError: 정답 쌍이나 인용이 있는 이미지에 가중이 없으면. 가중을 1 로 메우지 않는다.
    """
    gold = tuple(gold)
    index = set(index_clause_ids)
    gm = _gold_map(gold)
    allowed: dict[str, set[str]] = {}
    for (img, _), g in gm.items():
        allowed.setdefault(img, set()).add(g.clause_id)
    need = {img for img, _ in gm} | {img for img, cites in cited_by_image.items() if cites}
    lost = sorted(need - set(weights))
    if lost:
        raise KeyError(f"가중이 없는 이미지가 {len(lost)} 장이다 — 가중을 메우지 않는다")

    def tally(w_of) -> dict:
        t = {"hit": 0.0, "pairs": 0.0, "correct": 0.0, "cites": 0.0, "nonexistent": 0.0, "irrelevant": 0.0,
             "n_pairs": 0, "n_cites": 0, "per_code": {}}
        for (img, code), g in sorted(gm.items()):
            w = w_of(img)
            ok = g.clause_id in set(cited_by_image.get(img, ()))
            t["hit"] += w * ok
            t["pairs"] += w
            t["n_pairs"] += 1
            slot = t["per_code"].setdefault(code, [0.0, 0.0, 0])
            slot[0] += w * ok
            slot[1] += w
            slot[2] += 1
        for img, cites in sorted(cited_by_image.items()):
            for c in cites:
                w = w_of(img)
                t["cites"] += w
                t["n_cites"] += 1
                t["correct"] += w * (c in allowed.get(img, set()))
                if c not in index:
                    t["nonexistent"] += w
                elif c not in allowed.get(img, set()):
                    t["irrelevant"] += w
        return t

    ones = tally(lambda img: 1.0)
    base = clause_axis_report(cited_by_image, gold, index)
    if (ones["hit"], ones["pairs"], ones["correct"], ones["cites"], ones["nonexistent"], ones["irrelevant"]) != (
            base["citation_match_rate"]["numerator"], base["citation_match_rate"]["denominator"],
            base["citation_precision"]["numerator"], base["citation_precision"]["denominator"],
            base["ungrounded_rate"]["nonexistent"], base["ungrounded_rate"]["irrelevant"]) or {
            code: (v[0], v[1]) for code, v in ones["per_code"].items()} != {
            code: (v["numerator"], v["denominator"]) for code, v in base["per_code"].items()}:
        raise AssertionError("가중 셈의 단위가 무가중 셈(score_citation · score_ungrounded)과 갈렸다")
    t = tally(lambda img: float(weights[img]))
    out = {
        "citation_match_rate": weighted_rate(t["hit"], t["pairs"], t["n_pairs"]),
        "citation_precision": weighted_rate(t["correct"], t["cites"], t["n_cites"]),
        "ungrounded_rate": {**weighted_rate(t["nonexistent"] + t["irrelevant"], t["cites"], t["n_cites"]),
                            "nonexistent": t["nonexistent"], "irrelevant": t["irrelevant"]},
        "per_code": {code: weighted_rate(v[0], v[1], v[2]) for code, v in sorted(t["per_code"].items())},
        "n_gold_pairs": t["n_pairs"],
        "n_citations": t["n_cites"],
        "unit": base["unit"],
        "weighting": "설계 가중 — 단위가 속한 이미지의 가중(M1 · 서술 지표와 같은 design_weights)",
    }
    if auxiliary_only is not None:                    # 분해 칸 — 점추정만(07 4판 §9)
        out["ungrounded_rate"]["auxiliary_only"] = _auxiliary_only_count(cited_by_image, allowed, index, auxiliary_only,
                                                                         weights)
    return out


UNIT_KEYS = ("hit", "pairs", "correct", "cites", "nonexistent", "irrelevant")
"""이미지별 단위의 수 — 맞은 정답 쌍 · 정답 쌍 · 맞은 인용 · 인용 · 지어낸 인용 · 무관 인용."""


def unit_counts(
    cited_by_image: Mapping[str, Sequence[str]],
    gold: Iterable[GoldPair],
    index_clause_ids: Iterable[str],
    image_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """이미지마다 조항 축 단위의 수(열 = `image_ids` 의 차례) — `clause_axis_weighted` 와 같은 단위 · 같은 판정이다.

    일치율의 단위 = 정답 (이미지, 코드) 쌍, 정밀도 · 무근거율의 단위 = 인용 원소(같은 조항을 두 번 적으면 두 번). 모두 더하면 무가중
    셈(`clause_axis_report` — `score_citation` · `score_ungrounded`)의 분자 · 분모와 같아야 한다 — 그 자리에서 맞댄다.

    Raises:
        KeyError: 정답 쌍이나 인용이 있는 이미지가 `image_ids` 밖이면 — 추첨 열에 없는 단위를 세지 않는다.
    """
    gold = tuple(gold)
    index = set(index_clause_ids)
    gm = _gold_map(gold)
    allowed: dict[str, set[str]] = {}
    for (img, _), g in gm.items():
        allowed.setdefault(img, set()).add(g.clause_id)
    pos = {img: k for k, img in enumerate(image_ids)}
    lost = ({img for img, _ in gm} | {img for img, cites in cited_by_image.items() if cites}) - set(pos)
    if lost:
        raise KeyError(f"모집단 밖 이미지가 {len(lost)} 장이다 — 추첨 열에 없는 단위를 세지 않는다")
    u = {k: np.zeros(len(pos), dtype=np.float64) for k in UNIT_KEYS}
    for (img, _), g in gm.items():
        k = pos[img]
        u["pairs"][k] += 1
        u["hit"][k] += g.clause_id in set(cited_by_image.get(img, ()))
    for img, cites in cited_by_image.items():
        for c in cites:
            k = pos[img]
            u["cites"][k] += 1
            u["correct"][k] += c in allowed.get(img, set())
            if c not in index:
                u["nonexistent"][k] += 1
            elif c not in allowed.get(img, set()):
                u["irrelevant"][k] += 1
    base = clause_axis_report(cited_by_image, gold, index)
    if tuple(float(u[k].sum()) for k in UNIT_KEYS) != (
            base["citation_match_rate"]["numerator"], base["citation_match_rate"]["denominator"],
            base["citation_precision"]["numerator"], base["citation_precision"]["denominator"],
            base["ungrounded_rate"]["nonexistent"], base["ungrounded_rate"]["irrelevant"]):
        raise AssertionError("이미지별 단위의 합이 무가중 셈(score_citation · score_ungrounded)과 갈렸다")
    return u


def clause_axis_draw_parts(
    cited_by_image: Mapping[str, Sequence[str]],
    gold: Iterable[GoldPair],
    index_clause_ids: Iterable[str],
    image_ids: Sequence[str],
    W,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """추첨마다 조항 축 셋의 (분자, 분모) — `W`(추첨 × 이미지, 열 = `image_ids` 의 차례)의 가중으로 이미지별 단위 수를 더한다.

    `W` 가 M1 · 서술 지표의 추첨 행렬(`sample_bootstrap.draw_weights` — 묶음 중복도 × 설계 가중)이면 설계 가중의 추첨, 그 행렬을
    이미지 설계 가중으로 나눈 것(묶음 중복도)이면 무가중의 추첨이다.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != len(image_ids):
        raise ValueError(f"가중 행렬의 모양 {W.shape} 가 이미지 수 {len(image_ids)} 와 맞지 않는다")
    if not np.isfinite(W).all() or (W < 0).any():
        raise ValueError("가중 행렬은 유한한 0 이상의 수다")
    u = unit_counts(cited_by_image, gold, index_clause_ids, image_ids)
    return {"citation_match_rate": (W @ u["hit"], W @ u["pairs"]),
            "citation_precision": (W @ u["correct"], W @ u["cites"]),
            "ungrounded_rate": (W @ (u["nonexistent"] + u["irrelevant"]), W @ u["cites"])}


def clause_axis_draws(
    cited_by_image: Mapping[str, Sequence[str]],
    gold: Iterable[GoldPair],
    index_clause_ids: Iterable[str],
    image_ids: Sequence[str],
    W,
) -> dict[str, np.ndarray]:
    """추첨마다 조항 축 셋의 값 — 분모가 0 인 추첨은 `NaN`(미정의). 서술 지표의 `narrative_metrics.draws` 와 같은 규칙이다."""
    out: dict[str, np.ndarray] = {}
    for name, (n, d) in clause_axis_draw_parts(cited_by_image, gold, index_clause_ids, image_ids, W).items():
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)
    return out


__all__ = ["AUXILIARY_CONDITION", "ROLE_KEYS", "UNIT_KEYS", "ClauseRole", "MultipleMainClauses", "RolesFormError",
           "auxiliary_only_by_image", "citations_from_codes", "citations_from_roles", "clause_axis_draw_parts", "clause_axis_draws",
           "clause_axis_report", "clause_axis_weighted", "clauses_by_code", "entries_from_limits", "gold_pairs_from_roles",
           "gold_pairs_without_thickness", "rate", "roles_from_derived", "roles_from_entries", "unit_counts", "weighted_rate"]
