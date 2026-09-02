"""[자료] 직렬화 정본 — 생성 프롬프트와 판정 프롬프트가 **같은 함수**를 쓴다 (80번 G4-1).

체크리스트 6번. 사전실험에서 이렇게 깨졌다.

- 생성 프롬프트는 결함 코드를 "그대로 적으라"고 **의무화**했는데, 판정 프롬프트의 [자료]
  에는 결함 코드가 없었다. 판정 지시는 "자료에 없는 사실은 NG" 였다. **지시대로 쓴 문장이
  구조적으로 NG 조건을 만족한다.** 178건 중 98건(55.1%)이 그 상태였다 (B2·B4).
- 두 프롬프트가 각자 f-string 으로 자료를 폈기 때문에 한쪽만 필드를 빠뜨려도 아무도
  몰랐다.

그래서 자료를 만드는 지점을 하나로 못 박는다. 프롬프트도 판정도 `render_basis()` 가 낸
같은 문자열을 그대로 싣는다. **필드를 빼거나 더하려면 이 파일을 고쳐야 하고, 고치면 양쪽이
같이 움직인다.** 등식은 시험이 아니라 구성으로 보장된다 — 시험은 그것을 확인만 한다.

수치·enum 표기는 `corpus.rules.clause_text` 정본을 쓴다. 내부 표현(`Unit.MM`,
개구간 인코딩 `10.01`, 후행 0 `4.00`)이 자료로 새면 그 문자열이 생성문에 그대로 박히고,
채택 corpus 69건 중 49건이 실제로 그렇게 오염됐다 (B1).

순수 함수만 둔다. I/O·전역 상태·난수 없음.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from corpus.rules.clause_text import (
    material_ko,
    method_ko,
    num_ko,
    op_ko,
    thickness_ko,
    unit_ko,
    val,
)

__all__ = [
    "AXIS_CLAUSE",
    "AXIS_REMEDY",
    "axis_of",
    "basis_fields",
    "render_basis",
    "basis_facts",
    "required_mentions",
]

#: 축 이름. 골격이 `axis` 를 들고 다니지 않으면 필드 구성으로 판별한다.
AXIS_CLAUSE = "조항검색_기준서술"
AXIS_REMEDY = "조치서술"


def axis_of(sk: Mapping[str, Any]) -> str:
    axis = sk.get("axis")
    if axis in (AXIS_CLAUSE, AXIS_REMEDY):
        return str(axis)
    return AXIS_REMEDY if sk.get("remedy_ko") else AXIS_CLAUSE


def _measured(sk: Mapping[str, Any]) -> Optional[str]:
    """실측값 — 길이·직경은 size_mm(mm), 비율은 measured_value(단위 주입)."""
    if sk.get("size_mm") is not None:
        return f"{val(sk['size_mm'])} mm"
    if sk.get("measured_value") is not None:
        return f"{val(sk['measured_value'])} {unit_ko(sk.get('measured_unit'))}"
    return None


def _limit(sk: Mapping[str, Any]) -> Optional[str]:
    """유효 한계 — 부등호는 `limit_op` 를 읽는다. 상수로 박으면 게이트가 자기참조가 된다."""
    if str(sk.get("limit_rule") or "") == "none_permitted":
        return "크기와 무관하게 허용하지 않는다"
    if sk.get("limit_value") is None:
        return None
    return f"{val(sk['limit_value'])} {unit_ko(sk.get('unit'))} {op_ko(sk.get('limit_op'))}"


def _thickness_band(sk: Mapping[str, Any]) -> Optional[str]:
    lo, hi = sk.get("thickness_min"), sk.get("thickness_max")
    if lo is None and hi is None:
        return None
    return thickness_ko(lo, hi)


#: (라벨, 값 추출기). **순서가 곧 자료의 순서다.** 값이 None 이면 그 줄은 실리지 않는다.
#:
#: 자료에 실을 수 있는 것은 **수치 잠금 화이트리스트가 덮는 값**뿐이다. 두께 구간
#: (`10.01`~`25.01`)이나 근거 문서 식별자(`KR-RULES-P2`)를 자료에 실으면, 모델이 그것을
#: 그대로 옮겨 적었을 때 `check_numeric_lock` 이 허용 집합 밖 수치·조항으로 폐기한다 —
#: 지시대로 쓴 문장이 구조적으로 기각되는 B4 와 똑같은 사고가 반대 방향으로 재발한다.
#: 그래서 골격이 화이트리스트로 내주는 값만 싣는다 (모재 두께는 실측 두께 한 값이다).
_CLAUSE_SPEC: tuple[tuple[str, Any], ...] = (
    ("결함 코드(ISO 6520-1)", lambda sk: val(sk.get("defect_code"))),
    ("결함 명칭", lambda sk: sk.get("defect_name")),
    ("재질", lambda sk: material_ko(sk.get("material")) if sk.get("material") else None),
    ("검사 방식", lambda sk: method_ko(sk["inspection_method"])
     if sk.get("inspection_method") else None),
    ("실측 크기", _measured),
    ("모재 두께", lambda sk: f"{val(sk['thickness_mm'])} mm"
     if sk.get("thickness_mm") is not None else None),
    ("적용 조항", lambda sk: sk.get("clause_id")),
    ("한계 종류", lambda sk: val(sk.get("limit_type")) or None),
    ("조항이 정한 기준", _limit),
    ("판정", lambda sk: sk.get("verdict")),
)

_REMEDY_SPEC: tuple[tuple[str, Any], ...] = (
    ("상황", lambda sk: sk.get("topic")),
    ("검사 방식", lambda sk: method_ko(sk["inspection_method"])
     if sk.get("inspection_method") else None),
    ("조치", lambda sk: sk.get("remedy_ko")),
    ("근거 지침", lambda sk: sk.get("source_ref") or sk.get("clause_id")),
    ("근거 원문", lambda sk: sk.get("source_en")),
)


def basis_fields(sk: Mapping[str, Any]) -> dict[str, str]:
    """골격 → [자료] 필드 사전 (라벨 → 사람이 읽는 값). 순서 고정."""
    spec = _REMEDY_SPEC if axis_of(sk) == AXIS_REMEDY else _CLAUSE_SPEC
    out: dict[str, str] = {}
    for label, get in spec:
        v = get(sk)
        if v is None or v == "":
            continue
        out[label] = str(v)
    return out


def render_basis(sk: Mapping[str, Any]) -> str:
    """[자료] 블록. 생성 프롬프트와 판정 프롬프트가 **이 문자열 하나**를 공유한다."""
    return "\n".join(f"- {k}: {v}" for k, v in basis_fields(sk).items())


def basis_facts(sk: Mapping[str, Any]) -> tuple[str, ...]:
    """자료에 실린 사실 값들. 등식 시험이 프롬프트 쪽 사실과 대조한다."""
    return tuple(basis_fields(sk).values())


def required_mentions(sk: Mapping[str, Any]) -> dict[str, str]:
    """stage0 이 **출현을 의무화**하는 요소 (라벨 → 값).

    stage2 가 이것을 처벌하면 지시대로 쓴 문장이 구조적으로 기각된다 (B4). 그래서 의무
    요소는 반드시 [자료]에도 있어야 하고, 그 포함 관계를 시험이 강제한다 (G4-3).
    수치 의무는 `numeric_lock._skeleton_required` 와 같은 §4-4 매트릭스를 따른다 —
    여기서는 사람이 읽는 라벨로 되풀이할 뿐 판정 논리를 재구현하지 않는다.
    """
    out: dict[str, str] = {}
    if axis_of(sk) == AXIS_REMEDY:
        ref = sk.get("source_ref") or sk.get("clause_id")
        if ref:
            out["근거 지침"] = str(ref)
        return out

    if sk.get("clause_id"):
        out["적용 조항"] = str(sk["clause_id"])
    if sk.get("defect_code") is not None:
        out["결함 코드(ISO 6520-1)"] = val(sk["defect_code"])

    rule = str(sk.get("limit_rule") or "")
    meas = _measured(sk)
    if meas is not None:
        out["실측 크기"] = meas
    if sk.get("thickness_mm") is not None:
        out["모재 두께"] = f"{val(sk['thickness_mm'])} mm"
    if rule != "none_permitted":
        lim = _limit(sk)
        if lim is not None:
            out["조항이 정한 기준"] = lim
    if sk.get("verdict"):
        out["판정"] = str(sk["verdict"])
    return out


def band_text(sk: Mapping[str, Any]) -> str:
    """두께 구간 표기 — 개구간 인코딩(+0.01)을 사람이 읽는 경계로 되돌린 값."""
    return _thickness_band(sk) or f"{num_ko(sk.get('thickness_min'))} 이상"
