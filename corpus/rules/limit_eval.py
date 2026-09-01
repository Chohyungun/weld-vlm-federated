"""유효 한계 산출 + 판정기 — B 골격 생성기와 D 채점기가 **같은 함수를 import** 한다.

스펙: 11_spec_B_코퍼스합성.md §1-3 (허용치 표현 규약), §1-4 (단일 소스 파생 구조),
§4-2 (함수 시그니처), §4-4 (verdict·margin 산정).

금지 규칙 (§1-4): 판정·한계 산출을 이 모듈 밖에서 재구현하지 않는다.
다결함 이미지 verdict 합성은 aggregate_verdicts 밖에서 재구현하지 않는다 (D 포함).

전 함수 순수 (전역 상태·시각·I/O 없음). 수치는 전부 Decimal — float 는 quantize 가 거부한다.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping, Optional, Sequence, Union

from corpus.rules.clause_text import clause_text
from corpus.rules.schema import (
    FAIL,
    PASS,
    InspectionMethod,
    Judgment,
    LimitOp,
    LimitRow,
    LimitRule,
    LimitsTable,
    LimitType,
    Material,
    Q,
    QualityScheme,
    RatioBasis,
    Scope,
    quantize,
)

__all__ = [
    "Q",
    "quantize",
    "BASIS_ASSUMPTION_S_EQ_T",
    "effective_limit",
    "judge",
    "aggregate_verdicts",
    "applicable_row",
    "rows_for_clause",
    "derive_chunk_meta",
    "derive_gold_clauses",
]

# ratio_basis=s 행을 모재 두께 t 로 평가했다는 표시 (열린 질문 3 — 완전용입 맞대기 가정).
BASIS_ASSUMPTION_S_EQ_T = "assumed_s_eq_t"

# 행 시퀀스 또는 LimitsTable 어느 쪽도 받는다 — rows_for_clause 로 좁힌 후보를 그대로
# applicable_row 에 넘길 수 있어야 §1-4 의 "후보 축소 후 applicable_row 재사용"이 성립한다.
RowSource = Union[LimitsTable, Sequence[LimitRow]]


def _rows_of(source: RowSource) -> tuple[LimitRow, ...]:
    rows = source.rows if isinstance(source, LimitsTable) else tuple(source)
    return tuple(rows)


def effective_limit(row: LimitRow, basis_value: Optional[Decimal] = None) -> Optional[Decimal]:
    """유효 한계 L (§1-3). 양자화 포함.

    const → limit_value / prop_t → factor × basis / prop_t_cap → min(factor × basis, cap)
    none_permitted → None (결함 존재 = 불합격, 수치 한계 없음).

    basis_value: prop 계열의 기준 치수 값 (ratio_basis 가 t 면 모재 두께).
    ratio_basis=s 행에 t 를 넘기는 것은 완전용입 맞대기 가정이며, 그 사실은 judge 가
    Judgment.basis_assumption 에 싣는다 (열린 질문 3).

    ratio_basis=a(목두께) 행은 **평가 보류**다 — 목두께는 라벨·매니페스트 어디에도 없어
    t 를 대입하면 조용히 다른 물리량으로 판정한다. 골격 생성기만 막고 여기를 열어 두면
    같은 행에서 B 는 보류, D 는 판정으로 갈린다 (적대 검증 N6). 가드를 공유 평가기에 둔다.
    """
    rule = row.limit_rule
    if rule is LimitRule.NONE_PERMITTED:
        return None
    if rule is LimitRule.CONST:
        return row.limit_value  # 로드 시점에 이미 양자화됨
    # prop 계열
    if row.ratio_basis is RatioBasis.A:
        raise ValueError(
            f"[limit_eval] ratio_basis=a 행({row.rule_id})은 평가 보류 — 목두께 a 의 값이 "
            "데이터에 없다 (열린 질문 3, 수기 판정 큐 대상). t 대입 금지"
        )
    if basis_value is None:
        raise ValueError(
            f"[limit_eval] limit_rule={rule.value} 행({row.rule_id})은 basis_value 필수"
        )
    b = quantize(basis_value)
    raw = row.limit_factor * b
    if rule is LimitRule.PROP_T_CAP:
        raw = min(raw, row.limit_cap)
    return quantize(raw)


def judge(
    row: LimitRow,
    measured: Optional[Decimal],
    basis_value: Optional[Decimal] = None,
) -> Judgment:
    """B·D 공용 단일 판정기 (§4-4). 재구현 금지.

    등호 규약: le → measured ≤ L 합격 / lt → measured < L 합격. margin = L − measured.
    none_permitted: measured > 0 필수(위반 시 예외 — (c) 경로는 전체 중단, D4 는 호출자가
    quarantine 처리), verdict 불합격, margin None. 비율형(unit=percent)은 동일 로직.

    basis_value 는 prop 계열 행에서만 필요하다 (스펙 §4-2 의 judge(row, measured) 2인자
    형태는 const·none_permitted 행에서 그대로 성립한다). ratio_basis=a 행은
    effective_limit 의 가드가 발동해 평가 보류로 거부된다.
    """
    if measured is None:
        raise ValueError(f"[limit_eval] measured=None ({row.rule_id}) — 평가 불능")
    m = quantize(measured)

    if row.limit_rule is LimitRule.NONE_PERMITTED:
        if m <= 0:
            raise ValueError(
                f"[limit_eval] none_permitted 행({row.rule_id})은 measured > 0 필수 — got {m}"
            )
        return Judgment(verdict=FAIL, margin=None)

    assumption = BASIS_ASSUMPTION_S_EQ_T if row.ratio_basis is RatioBasis.S else None

    if m < 0:
        raise ValueError(f"[limit_eval] measured < 0 ({row.rule_id}): {m}")

    L = effective_limit(row, basis_value)
    assert L is not None  # none_permitted 는 위에서 처리됨
    passed = (m <= L) if row.limit_op is LimitOp.LE else (m < L)
    margin = L - m  # 두 항 모두 0.01 그리드 → 차이도 그리드 위

    # 불변식 (§4-4): le → (margin ≥ 0) ⇔ 합격 / lt → (margin > 0) ⇔ 합격
    if row.limit_op is LimitOp.LE:
        assert (margin >= 0) == passed, f"margin 불변식 붕괴(le): {row.rule_id}"
    else:
        assert (margin > 0) == passed, f"margin 불변식 붕괴(lt): {row.rule_id}"

    return Judgment(
        verdict=PASS if passed else FAIL, margin=margin, basis_assumption=assumption
    )


def aggregate_verdicts(verdicts: Sequence[str]) -> str:
    """이미지 수준 verdict 합성 — 보수 규칙 (게이트 #6 결정 F, §1-4).

    한 결함이라도 불합격이면 불합격. 빈 목록(정상 이미지) ⇒ 합격.
    B 의 assemble_pair 와 D 의 다결함 판정 정합성 재계산이 이 한 함수를 공유한다.
    """
    for v in verdicts:
        if v not in (PASS, FAIL):
            raise ValueError(f"[limit_eval] verdict enum 위반: {v!r} (합격/불합격 2종 외 금지)")
    return FAIL if any(v == FAIL for v in verdicts) else PASS


def applicable_row(
    table: RowSource,
    defect_code: str,
    material: Material | str,
    inspection_method: InspectionMethod | str,
    quality_scheme: QualityScheme | str,
    quality_level: str,
    t: Decimal,
    limit_type: LimitType | str | None = None,
) -> LimitRow:
    """조합·두께로 행 선택 (§4-2). [min, max) 규약. 정확히 1행, fallback 금지.

    scope=active ∧ canonical 행만 대상 (§4-3). material=ALL 행은 모든 재질 질의에 응하고,
    inspection_method=ALL 행은 RT·VT 어느 질의에도 응한다.

    **검사 방법 축은 필수 인자다** (§1-2 5a, 게이트 #13). 기본값을 두지 않는 이유는
    조용한 추정이 바로 이 축을 만든 이유이기 때문이다 — 같은 결함코드의 표면(VT) 기준과
    내부(RT) 기준이 갈릴 때 축 없는 질의는 예외 없이 엉뚱한 행을 집고, 그 결과는 형식상
    정상으로 보인다. 축을 호출자의 뷰 주입 규율에만 맡기면 limit_eval 만 import 하는
    소비처(D 채점기)에는 강제가 없다. 질의 값으로 `ALL` 을 주는 것도 거부한다 — ALL 은
    조항의 속성이지 질의의 값이 아니다.

    같은 조합에 limit_type 이 다른 복수 제약이 정당하게 공존하므로(V3 그룹 키 근거),
    복수 매칭 시에는 limit_type 을 명시해야 한다 — 임의 선택은 fail-closed 로 거부한다.

    table 은 LimitsTable 또는 행 시퀀스다. rows_for_clause 로 조항 후보를 좁힌 결과를
    그대로 넘겨 재선택하는 경로(§1-4 D 판정 정합성 재계산)를 위해서다.
    """
    material = Material(material)
    method = InspectionMethod(inspection_method)
    if method is InspectionMethod.ALL:
        raise ValueError(
            "[applicable_row] 질의 값 ALL 금지 — RT 또는 VT 를 지정하라 "
            "(ALL 은 '검사 방법과 무관한 조항' 표시이지 '전 행'이 아니다)"
        )
    quality_scheme = QualityScheme(quality_scheme)
    lt_filter = None if limit_type is None else LimitType(limit_type)
    tq = quantize(t)

    matches = [
        r
        for r in _rows_of(table)
        if r.scope is Scope.ACTIVE
        and r.canonical
        and r.defect_code == defect_code
        and (r.material is material or r.material is Material.ALL)
        and (r.inspection_method is method or r.inspection_method is InspectionMethod.ALL)
        and r.quality_scheme is quality_scheme
        and r.quality_level == quality_level
        and (lt_filter is None or r.limit_type is lt_filter)
        and r.contains_t(tq)
    ]
    key = (
        f"defect={defect_code}, material={material.value}, method={method.value}, "
        f"scheme={quality_scheme.value}, level={quality_level}, t={tq}"
    )
    if not matches:
        raise LookupError(f"[applicable_row] 해당 행 없음 — fallback 금지 ({key})")
    if len(matches) > 1:
        ids = [r.rule_id for r in matches]
        raise LookupError(
            f"[applicable_row] 복수 행 매칭 {ids} — limit_type 을 명시하라 ({key})"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# D 소비용 파생 함수 3종 (§1-4) — 계약은 여기까지다
#
# D 가 limits.csv 를 직접 순회해 청크 메타·정답 조항 목록을 만들면, 파일 하나를 공유해도
# 해석 코드가 두 벌이 되어 단일 소스가 파생 계층에서 붕괴한다. 세 함수가 그 경로를 대신한다.
# 파생물(corpus/derived/)은 수기 수정 금지이며, 오류는 원천 limits.csv 를 고쳐 재파생한다.
# ---------------------------------------------------------------------------


def _dec_str(v: Optional[Decimal]) -> Optional[str]:
    """직렬화 표기 고정. +∞(공란)는 null 로 나간다 (§1-4)."""
    return None if v is None else str(quantize(v))


def rows_for_clause(table: RowSource, clause_id: str) -> tuple[LimitRow, ...]:
    """조항으로 후보 축소 (§1-4). rule_id 정렬 고정.

    D 판정 정합성 재계산의 행 선택 규칙: 판정문이 인용한 clause_id 로 후보를 좁힌 뒤,
    골격에 내장된 (defect_code, material, inspection_method, quality_level, t) 로
    applicable_row 를 재사용한다. 반환값을 applicable_row 의 첫 인자로 그대로 넘길 수 있다.
    scope·canonical 로 거르지 않는다 — 인용된 조항이 배제 행이라는 사실 자체가
    무근거 인용 판정의 근거이기 때문이며, 실제 행 선택은 applicable_row 가 거른다.
    """
    return tuple(
        sorted((r for r in _rows_of(table) if r.clause_id == clause_id),
               key=lambda r: r.rule_id)
    )


def derive_chunk_meta(
    table: RowSource, defect_names: Optional[Mapping[str, str]] = None
) -> tuple[dict, ...]:
    """`corpus/derived/chunk_meta.jsonl` 내용 (§1-4). D RAG 청크의 메타 필터 축 + 본문.

    clause_id 로 groupby 하고 집계는 합집합이다: defect_codes[], quality_levels[],
    **inspection_methods[]** (게이트 #13 신설 — 이 축이 없으면 D 검색이 표면·내부 조항을
    한 후보 집합에 섞는다). 두께는 [min 합집합, max 합집합) 반열림을 유지하고 +∞ 는
    null 로 직렬화한다.

    scope=excluded 행도 scope 플래그와 함께 **포함**한다 — 빼면 그 조항이 영구 미검색이
    되고, 모델이 그 조항을 인용했을 때 무근거 인용인지 판정할 수 없다.

    `text` 는 조항 본문이다 (`clause_text.clause_text` — 원문 전재가 아니라 구조 필드
    재서술). 본문이 비면 dense 정렬이 걸려도 정렬할 재료가 없어 임베딩 선정이 변별
    불가가 된다 (74번 감사 후속 과제 4). `defect_names` 는 사상표(계약 #1) 유래
    {ISO 코드: 명칭} 이며 없으면 본문이 코드만 쓴다 — 라벨 문자열을 여기에 하드코딩하지
    않는다 (불변조건 8).
    """
    groups: dict[str, list[LimitRow]] = {}
    for r in _rows_of(table):
        groups.setdefault(r.clause_id, []).append(r)

    out: list[dict] = []
    for clause_id in sorted(groups):
        rows = sorted(groups[clause_id], key=lambda r: r.rule_id)
        t_min = min(r.thickness_min for r in rows)
        t_max: Optional[Decimal] = None
        if all(r.thickness_max is not None for r in rows):
            t_max = max(r.thickness_max for r in rows)  # type: ignore[type-var]
        scopes = {r.scope for r in rows}
        out.append({
            "clause_id": clause_id,
            "source_docs": sorted({r.source_doc for r in rows}),
            "defect_codes": sorted({r.defect_code for r in rows}),
            "materials": sorted({r.material.value for r in rows}),
            "inspection_methods": sorted({r.inspection_method.value for r in rows}),
            "quality_schemes": sorted({r.quality_scheme.value for r in rows}),
            "quality_levels": sorted({r.quality_level for r in rows}),
            "thickness_min": _dec_str(t_min),
            "thickness_max": _dec_str(t_max),  # null = +∞
            "scope": scopes.pop().value if len(scopes) == 1 else "mixed",
            "rule_ids": [r.rule_id for r in rows],
            "text": clause_text(rows, dict(defect_names) if defect_names else None),
        })
    return tuple(out)


def derive_gold_clauses(table: RowSource) -> tuple[dict, ...]:
    """`corpus/derived/gold_clauses.csv` 내용 (§1-4). 채점 정답 조항 목록.

    canonical=true ∧ scope=active 행만 싣는다. 키는
    (defect_code, material, inspection_method, 두께구간, quality_scheme:quality_level)
    이며 게이트 #13 로 검사 방법 축이 키에 들어왔다 — 축 없이 키를 만들면 표면 기준
    조항이 내부 검사 정답으로 등재된다.

    같은 키에 limit_type 이 다른 복수 제약은 정당하므로 limit_type 도 키에 포함한다.
    그래도 중복이 남으면 정본 지정 오류이므로 fail-closed 로 거부한다 (G0 V6 의 파생 계층 확인).
    """
    seen: dict[tuple, str] = {}
    out: list[dict] = []
    for r in _rows_of(table):
        if not r.canonical or r.scope is not Scope.ACTIVE:
            continue
        key = (r.defect_code, r.material.value, r.inspection_method.value,
               str(r.thickness_min), _dec_str(r.thickness_max) or "+inf",
               r.quality_scheme.value, r.quality_level, r.limit_type.value)
        if key in seen and seen[key] != r.clause_id:
            raise ValueError(
                f"[derive_gold_clauses] 같은 키에 조항 2개 — {seen[key]} / {r.clause_id} "
                f"(키={key}). 원천 limits.csv 의 canonical 지정을 고쳐 재파생하라"
            )
        seen[key] = r.clause_id
        out.append({
            "defect_code": r.defect_code,
            "material": r.material.value,
            "inspection_method": r.inspection_method.value,
            "thickness_min": _dec_str(r.thickness_min),
            "thickness_max": _dec_str(r.thickness_max),  # null = +∞
            "quality_scheme": r.quality_scheme.value,
            "quality_level": r.quality_level,
            "limit_type": r.limit_type.value,
            "clause_id": r.clause_id,
            "rule_id": r.rule_id,
        })
    return tuple(sorted(out, key=lambda d: tuple(str(v) for v in d.values())))
