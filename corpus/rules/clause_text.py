"""조항 본문 재서술 — `corpus/derived/` 의 `text` 축 정본.

`chunk_meta.jsonl` 의 청크 4개가 전부 `text` 길이 0 이었다(74번 감사 P/과제 4). 메타만
있고 본문이 없으면 dense 정렬이 걸려도 정렬할 재료가 없어 임베딩 선정이 변별 불가가 된다.
이 모듈이 조항 본문을 만든다.

**원문 전재가 아니라 구조 필드에서 만든 재서술이다.** 규약 2-5 는 유료 표준 전문의
색인 반입을 금지하고, 색인 대상은 무료 공개 문서로 한정한다. 원천 KR-RULES-P2 는
`corpus/parse/sources.yaml` 에서 `license_class: OPEN` · `allowed_use: [rag_index]` 라
전재해도 형식 위반은 아니지만, 그 문서가 KS B 0845 계열(유료 표준) 표를 재수록하고
있어 셀 표기를 그대로 옮기면 유료 표준 표현을 색인에 넣는 경로가 생긴다. 그래서
**표기가 실린 필드(`limit_expr` · `source_row_label` · `note`)는 쓰지 않고**, 사실
정보인 구조 필드(결함 코드·재질·검사 방식·두께 구간·한계 규칙과 수치·부등호·단위·
조항 식별자·출처 문서와 쪽)만 우리 템플릿으로 문장화한다.

부등호는 `limit_op` 를 **읽어서** 쓴다. `make_pairs_pilot.clause_basis` 는 이 필드를 한 번도
읽지 않고 "이하"를 f-string 에 박았고, 그 결과 부등식 방향 게이트가 자기참조가 됐다
(74번 M9). 비례 기준의 분모도 `ratio_basis` 를 읽는다 — t·s·a 가 각각 모재 두께·용접부
공칭 두께·목두께로 다른 양이라, "모재 두께의" 를 고정으로 박으면 s·a 행에서 틀린다.

표현 헬퍼(`val`·`num_ko`·`unit_ko`·`method_ko`·`thickness_ko`)의 정본도 여기다.
`corpus.generate.run_cycle_corpus` 는 여기서 가져다 쓴다 — 같은 문구 규칙이 두 벌로
갈리면 한쪽만 개구간 인코딩(+0.01)이나 enum 표기를 노출한다.

순수 함수만 둔다 (I/O·전역 상태·난수 없음). 같은 표면 같은 바이트.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

__all__ = [
    "val",
    "num_ko",
    "unit_ko",
    "method_ko",
    "material_ko",
    "thickness_ko",
    "basis_ko",
    "op_ko",
    "criterion_ko",
    "clause_text",
    "derive_clause_texts",
]


# --------------------------------------------------------------- 표현 헬퍼 (정본)

def val(x) -> str:
    """enum 은 값만, 수치는 뒤따르는 0 을 정리해서 쓴다."""
    v = getattr(x, "value", x)
    if v is None:
        return ""
    txt = str(v)
    if re.fullmatch(r"-?\d+\.\d+", txt):
        txt = txt.rstrip("0").rstrip(".")
    return txt


def unit_ko(u) -> str:
    return {"mm": "mm", "percent": "%"}.get(val(u), val(u) or "mm")


def method_ko(m) -> str:
    return {"RT": "RT(방사선투과)", "VT": "VT(육안)", "ALL": "검사 방식 무관"}.get(
        val(m), val(m))


def num_ko(x) -> str:
    """개구간 인코딩(+0.01)을 사람이 읽는 경계로 되돌린다."""
    v = val(x)
    if not v:
        return ""
    try:
        f = float(v)
    except ValueError:
        return v
    if abs(f - round(f) - 0.01) < 1e-9:      # 10.01 → 10 초과
        return val(round(f - 0.01, 2))
    return val(f)


def thickness_ko(tmin, tmax) -> str:
    """두께 구간을 원문 표기(초과·이하)로 쓴다."""
    lo, hi = val(tmin), val(tmax)
    lo_f = float(lo) if lo else 0.0
    left = "" if lo_f == 0 else f"{num_ko(tmin)} mm 초과"
    if not hi:
        return left or "모든 두께"
    right = f"{num_ko(tmax)} mm 이하"
    return f"{left} {right}".strip() if left else right


def material_ko(m) -> str:
    return {"ST": "강재(ST)", "AL": "알루미늄(AL)", "ALL": "재질 무관"}.get(
        val(m), val(m))


# --------------------------------------------------------------- 기준 서술

def op_ko(op) -> str:
    """부등호 표기. `limit_op` 를 읽는다 — 고정 문자열로 박지 않는다 (74번 M9)."""
    return {"le": "이하", "lt": "미만"}.get(val(op), "이하")


def basis_ko(basis) -> str:
    """비례 기준의 분모. t·s·a 는 서로 다른 양이다."""
    return {"t": "모재 두께", "s": "용접부 공칭 두께", "a": "목두께"}.get(
        val(basis), "모재 두께")


def criterion_ko(row) -> str:
    """행 하나의 한계 서술. `limit_rule` 별 문형 + `limit_op`·`ratio_basis` 반영."""
    rule = val(row.limit_rule)
    if rule == "none_permitted":
        return "크기와 무관하게 허용하지 않는다"
    unit = unit_ko(row.unit)
    op = op_ko(row.limit_op)
    if rule == "const":
        return f"{val(row.limit_value)} {unit} {op}"
    if rule == "prop_t":
        return f"{basis_ko(row.ratio_basis)}의 {val(row.limit_factor)} 배 {op}"
    if rule == "prop_t_cap":
        return (f"{basis_ko(row.ratio_basis)}의 {val(row.limit_factor)} 배 {op}이고"
                f" 최대 {val(row.limit_cap)} {unit}")
    raise ValueError(f"알 수 없는 limit_rule: {rule!r} ({row.rule_id})")


# --------------------------------------------------------------- 조항 본문

def _band_ko(row) -> str:
    """두께 구간 + 그 구간의 기준. 상·하한이 없는 행은 "모든 두께" 하나로 쓴다."""
    band = thickness_ko(row.thickness_min, row.thickness_max)
    head = band if band == "모든 두께" else f"두께 {band}"
    return f"{head}: {criterion_ko(row)}"


def clause_text(rows: Sequence, defect_names: dict[str, str] | None = None) -> str:
    """조항 하나의 본문. 같은 `clause_id` 행 묶음을 받는다.

    `defect_names` 는 사상표(계약 #1) 유래 {ISO 코드: 한국어 명칭}. 없으면 코드만 쓴다 —
    라벨 문자열을 이 모듈에 하드코딩하지 않는다 (불변조건 8).
    """
    if not rows:
        raise ValueError("빈 행 묶음으로 조항 본문을 만들 수 없다")
    rows = sorted(rows, key=lambda r: r.rule_id)
    names = defect_names or {}
    head = rows[0]

    codes = sorted({r.defect_code for r in rows})
    tgt = ", ".join(
        f"{names[c]}(ISO 6520-1 코드 {c})" if c in names else f"ISO 6520-1 코드 {c}"
        for c in codes
    )
    mats = ", ".join(material_ko(m) for m in sorted({val(r.material) for r in rows}))
    methods = ", ".join(method_ko(m)
                        for m in sorted({val(r.inspection_method) for r in rows}))
    types = ", ".join(sorted({val(r.limit_type) for r in rows}))

    parts = [
        f"{head.clause_id} 조항.",
        f"근거 문서 {head.source_doc}"
        + (f" {head.source_page}쪽." if head.source_page is not None else "."),
        f"대상 결함: {tgt}.",
        f"대상 재질: {mats}.",
        f"검사 방식: {methods}.",
        f"한계 종류: {types}.",
    ]
    schemes = sorted({val(r.quality_scheme) for r in rows})
    if schemes != ["none"]:
        levels = sorted({r.quality_level for r in rows})
        parts.append(f"품질 체계: {', '.join(schemes)} 수준 {', '.join(levels)}.")

    # 기준은 결함 코드별로 묶는다. 한 조항이 코드 둘을 덮을 때 행을 그냥 이어 붙이면
    # 같은 구간표가 두 번 나오고(파일럿 T16 실측), 코드마다 구간이 다를 때는 어느 줄이
    # 어느 코드 것인지 사라진다.
    per_code = {c: " / ".join(_band_ko(r) for r in rows if r.defect_code == c)
                for c in codes}
    if len(set(per_code.values())) == 1:
        parts.append(f"기준: {next(iter(per_code.values()))}.")
    else:
        parts.append("기준: " + " ; ".join(
            f"{names.get(c, '코드 ' + c)} {per_code[c]}" for c in codes) + ".")

    scopes = sorted({val(r.scope) for r in rows})
    if scopes != ["active"]:
        parts.append(
            f"적용 상태: {', '.join(scopes)} — 판정 근거로 쓰지 않는다"
            " (무근거 인용 판정을 위해 색인에는 남긴다)."
        )

    # 미표현 단서 부기 (80번 B12·B13, G11-7).
    #
    # 원천 표에는 기계 표현으로 옮기지 못한 항목이 남아 있다 — 대표적으로 KRA27-T16 은
    # **합계 길이** 기준인데 스키마에는 길이 한계로만 실린다. 그대로 "이 조항의 기준은
    # …이다" 로 닫으면 불완전한 규칙을 완전한 규칙으로 가르치고, 그 위에서 판정 근거
    # 신뢰도를 채점하게 된다. 동결본 286건이 그 상태였다.
    #
    # 무엇이 빠졌는지를 여기서 말하려면 원천 표기 필드를 옮겨야 하므로(규약 2-5) 하지
    # 않는다. 대신 **빠진 것이 있다는 사실**을 싣는다. 어떤 항목인지를 기계로 말하려면
    # limits CSV 에 집계 기준 축이 필요하고, 그것은 단일 소스 계약 변경이라 게이트 사항이다
    # (미니스펙 참조).
    if any((getattr(r, "note", None) or "").strip() for r in rows):
        parts.append(
            "이 조항에는 기계 표현으로 옮기지 못한 단서가 있다"
            " (집계 단위·무시 하한·군집 간격 등). 위 기준은 그 단서를 반영하지 않은 값이므로"
            " 단독으로 합부를 결정하지 않는다."
        )
    return " ".join(parts)


def derive_clause_texts(
    rows: Iterable, defect_names: dict[str, str] | None = None
) -> dict[str, str]:
    """조항 단위 본문 사전 {clause_id: text}. `derive_chunk_meta` 와 같은 묶음 기준이다."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.clause_id, []).append(r)
    return {cid: clause_text(groups[cid], defect_names) for cid in sorted(groups)}
