"""계약 #3 스키마 — LimitRow · LimitsTable · Judgment · enum 전부.

스펙 §1-2 (컬럼 전수 명세),
§1-3 (허용치 표현 규약), §4-4 (verdict·margin), 인터페이스 고정 조항.

- 수치는 전부 decimal.Decimal, Q = Decimal("0.01"), ROUND_HALF_UP 양자화 후 비교 (§1-3).
  양자화 대상은 크기·두께·한계·margin — limit_factor(비례 계수)는 양자화하지 않는다.
- pydantic v2 frozen 모델. LimitRow 필드명은 스펙 §1-2 컬럼명 그대로.
- 널러빌리티(G0 V7 = limit_rule별 필수/공란 규칙)는 모델 validator가 강제한다.
  위반 메시지는 "[V7] ..." 형식으로 검사 ID를 앞에 붙인다 (로더가 행 번호와 함께 집계).
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "Q",
    "quantize",
    "PASS",
    "FAIL",
    "Verdict",
    "Material",
    "InspectionMethod",
    "DEFAULT_INSPECTION_METHODS",
    "QualityScheme",
    "LimitType",
    "LimitRule",
    "LimitOp",
    "Unit",
    "Scope",
    "RatioBasis",
    "VerdictMode",
    "ExtractionPath",
    "Measure",
    "Aggregation",
    "DefectClass",
    "COND_CLUSTER_GAP",
    "COND_PREMISE",
    "COND_MIXTURE",
    "COND_DISTINCTION",
    "CONDITION_PAIR_KINDS",
    "CONDITION_SEP",
    "parse_condition_token",
    "v12_problems",
    "LimitRow",
    "LimitsTable",
    "Judgment",
]

# ---------------------------------------------------------------------------
# Decimal 양자화 (전역 규약, §1-3)
# ---------------------------------------------------------------------------

Q = Decimal("0.01")

PASS: str = "합격"
FAIL: str = "불합격"
Verdict = Literal["합격", "불합격"]


def quantize(x: Decimal | int | str) -> Decimal:
    """0.01 그리드 ROUND_HALF_UP 양자화. float 입력은 거부한다 (float 직접 비교 금지).

    limit_eval.quantize 로 재수출된다 — 정의는 여기 한 곳뿐이다.
    """
    if isinstance(x, float):
        raise TypeError(
            "float 금지 — Decimal(str) 로 만들어 넘겨라 (스펙 §1-3: float 직접 비교 금지)"
        )
    if not isinstance(x, Decimal):
        x = Decimal(x)
    return x.quantize(Q, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# enum (인터페이스 고정 조항의 값 집합 그대로)
# ---------------------------------------------------------------------------


class Material(str, Enum):
    ST = "ST"
    AL = "AL"
    ALL = "ALL"


class InspectionMethod(str, Enum):
    """검사 방법 축 (§1-2 5a, 결정 #13). 표면 검사(VT)와 방사선 내부 검사(RT)는 같은
    결함코드라도 허용치가 다르다 — 축이 없으면 잘못된 행을 집어도 형식상 정상으로 보인다.
    ALL 은 검사 방법과 무관한 조항이라 RT·VT 어느 질의에도 응한다."""

    RT = "RT"
    VT = "VT"
    ALL = "ALL"


# 주 실험이 RT 이므로 행 선택 기본 필터는 {RT, ALL} 이다 (§1-2 5a).
DEFAULT_INSPECTION_METHODS: frozenset[InspectionMethod] = frozenset(
    {InspectionMethod.RT, InspectionMethod.ALL}
)


class QualityScheme(str, Enum):
    ISO5817 = "iso5817"
    ISO10675 = "iso10675"
    IACS = "iacs"
    KS = "ks"
    NONE = "none"


class LimitType(str, Enum):
    LENGTH = "길이"
    DIAMETER = "직경"
    RATIO = "비율"
    NOT_PERMITTED = "불허"


class LimitRule(str, Enum):
    CONST = "const"
    PROP_T = "prop_t"
    PROP_T_CAP = "prop_t_cap"
    NONE_PERMITTED = "none_permitted"


class LimitOp(str, Enum):
    LE = "le"  # measured ≤ L → 합격 (기본, 경계값 합격)
    LT = "lt"  # measured < L → 합격 (원문 "미만")


class Unit(str, Enum):
    MM = "mm"
    PERCENT = "percent"


class Scope(str, Enum):
    ACTIVE = "active"
    EXCLUDED = "excluded"


class RatioBasis(str, Enum):
    T = "t"  # 모재 두께
    S = "s"  # 용접부 공칭 두께
    A = "a"  # 목두께


class VerdictMode(str, Enum):
    """실이미지 합부 모드 (A 소유 3값, §4-7). 현재 D4·정상 페어는 clause_only 고정."""

    FULL = "full"
    CONDITIONAL = "conditional"
    CLAUSE_ONLY = "clause_only"


class ExtractionPath(str, Enum):
    D = "D"  # Docling
    P = "P"  # pdfplumber
    M = "M"  # 수기 전사


# ---------------------------------------------------------------------------
# 2 판(부분 B) 칸 — 1 판 31 칸 **뒤**의 네 칸(C 08 2 판 §1-3, 09 §6 Q1). 1 판 표는 이 칸이 없고 기본값(빈칸)으로 읽힌다.
# ---------------------------------------------------------------------------


class Measure(str, Enum):
    """측정량 — 원천 표의 행 이름표 머리(`source_row_label`)가 가리키는 양. 값 한계 행에만 있다."""

    MAX_SIZE = "최대결함크기"
    INDIVIDUAL_LENGTH = "개별길이"   # 신판 행에는 쓰이지 않는다 — 부분 B 범위 1 의 이름 셋을 닫힌 집합으로 둔다
    TOTAL_LENGTH = "합계길이"


class Aggregation(str, Enum):
    """집계 단위. `합계길이` ⇔ `합계`, 나머지 측정량 ⇒ `개별`(V12)."""

    INDIVIDUAL = "개별"
    TOTAL = "합계"


class DefectClass(str, Enum):
    """원천 표의 종별. 같은 (재질, 검사 방식, 코드)는 종별 하나다(V13)."""

    CLASS_1 = "제1종"
    CLASS_2 = "제2종"
    CLASS_3 = "제3종"


#: `conditions` 칸의 토큰. 짝 토큰(`혼재:<종별>` · `구별곤란:<종별>`)은 뒤에 종별을 든다. 칸은 토큰을 `|` 로 잇고,
#: 토큰은 문자열 정렬 순 · 겹침 없음이다(같은 조건 집합이 한 표기만 갖게).
COND_CLUSTER_GAP = "군집간격"
COND_PREMISE = "무등급전제"
COND_MIXTURE = "혼재"
COND_DISTINCTION = "구별곤란"
CONDITION_PAIR_KINDS: tuple[str, ...] = (COND_MIXTURE, COND_DISTINCTION)
CONDITION_SEP = "|"


def parse_condition_token(token: str) -> tuple[str, str | None]:
    """토큰 하나 → (종류, 짝 종별 또는 None). 닫힌 문법 밖이면 `ValueError`(`[V12]`)."""
    if token in (COND_CLUSTER_GAP, COND_PREMISE):
        return token, None
    kind, sep, cls = token.partition(":")
    if sep and kind in CONDITION_PAIR_KINDS and cls in {c.value for c in DefectClass}:
        return kind, cls
    raise ValueError(f"[V12] 모르는 조건 토큰: {token!r} — 받는 꼴 {COND_CLUSTER_GAP} · {COND_PREMISE} · "
                     f"{COND_MIXTURE}:<종별> · {COND_DISTINCTION}:<종별>")


# iso5817 품질수준 관대 순서: B(엄격) → C → D(관대). V4 단조성 검사의 기준 서열.
ISO5817_LEVEL_ORDER: tuple[str, ...] = ("B", "C", "D")


# ---------------------------------------------------------------------------
# LimitRow — limits.csv 1행 (§1-2 컬럼 전수)
# ---------------------------------------------------------------------------


class LimitRow(BaseModel):
    """limits.csv 한 행. 필드명 = 스펙 §1-2 컬럼명 그대로 (계약 동결)."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    rule_id: str
    canonical: bool
    scope: Scope
    defect_code: str
    material: Material
    inspection_method: InspectionMethod
    thickness_min: Decimal
    thickness_max: Optional[Decimal] = None  # 공란 = +∞ (반열림 [min, max))
    quality_scheme: QualityScheme
    quality_level: str
    limit_type: LimitType
    limit_rule: LimitRule
    limit_value: Optional[Decimal] = None
    limit_factor: Optional[Decimal] = None  # 비례 계수 — 양자화 대상 아님
    limit_cap: Optional[Decimal] = None
    ratio_basis: Optional[RatioBasis] = None
    limit_op: LimitOp
    unit: Optional[Unit] = None
    clause_id: str
    source_doc: str
    source_page: Optional[int] = None
    source_table_id: Optional[str] = None
    source_row_label: Optional[str] = None
    limit_expr: Optional[str] = None
    note: Optional[str] = None
    restated_from: Optional[str] = None
    filled_h: bool
    filled_v: bool
    extraction_path: ExtractionPath
    verified_by: Optional[str] = None
    verified_date: Optional[_dt.date] = None
    # 2 판 칸(C 08 2 판 §1-3). 1 판 행은 받지 않아 기본값이다 — 행 값 · `active_canonical` 이 1 판과 같다.
    measure: Measure | None = None
    aggregation: Aggregation | None = None
    defect_class: DefectClass | None = None
    conditions: tuple[str, ...] = ()

    # -- 2 판 조건 칸의 꼴: 닫힌 토큰 · 정렬 · 겹침 없음 (V12 의 꼴 부분 — 짝 · 필수는 `v12_problems`) ------

    @field_validator("conditions", mode="before")
    @classmethod
    def _conditions_form(cls, v):
        if v is None or v == "":
            return ()
        toks = tuple(v.split(CONDITION_SEP)) if isinstance(v, str) else tuple(v)
        for t in toks:
            if not isinstance(t, str):
                # TypeError 로 내면 pydantic 이 검증 오류로 감싸지 않아 로더의 행 단위 수집을 지나 빠져나간다.
                raise ValueError(f"[V12] 조건 토큰이 문자열이 아니다: {t!r}")  # noqa: TRY004
            parse_condition_token(t)
        if len(set(toks)) != len(toks):
            raise ValueError(f"[V12] 조건 토큰이 겹친다: {list(toks)}")
        if list(toks) != sorted(toks):
            raise ValueError(f"[V12] 조건 토큰이 정렬 순이 아니다: {list(toks)} — {sorted(toks)} 로 적는다")
        return toks

    # -- 양자화: 크기·두께·한계 (§1-3. factor 는 제외) ------------------------

    @field_validator("thickness_min", "thickness_max", "limit_value", "limit_cap", mode="after")
    @classmethod
    def _quantize_grid(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        return None if v is None else quantize(v)

    @field_validator(
        "rule_id", "defect_code", "quality_level", "clause_id", "source_doc", mode="after"
    )
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("[V1] 필수값 공란")
        return v

    # -- V2 · V7 · 단위-유형 교차 · 스킴 스코프 ------------------------------

    @model_validator(mode="after")
    def _validate_contract(self) -> "LimitRow":
        # V2 — 값 규칙
        if self.thickness_min < 0:
            raise ValueError("[V2] thickness_min < 0")
        if self.thickness_max is not None and not (self.thickness_min < self.thickness_max):
            raise ValueError(
                f"[V2] thickness_min({self.thickness_min}) < thickness_max({self.thickness_max}) 위반"
            )
        for name in ("limit_value", "limit_factor", "limit_cap"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"[V2] {name} > 0 위반: {v}")

        # V7 — limit_rule별 필수/공란 (널러빌리티 매트릭스)
        rule = self.limit_rule

        def _require(name: str) -> None:
            if getattr(self, name) is None:
                raise ValueError(f"[V7] limit_rule={rule.value}: {name} 필수인데 공란")

        def _forbid(name: str) -> None:
            if getattr(self, name) is not None:
                raise ValueError(
                    f"[V7] limit_rule={rule.value}: {name} 공란 강제인데 기재됨 "
                    "(파생값 중복 기재가 오염 시점이다)"
                )

        if rule is LimitRule.CONST:
            _require("limit_value")
            _forbid("limit_factor")
            _forbid("limit_cap")
            _forbid("ratio_basis")
        elif rule is LimitRule.PROP_T:
            _require("limit_factor")
            _require("ratio_basis")
            _forbid("limit_value")
            _forbid("limit_cap")
        elif rule is LimitRule.PROP_T_CAP:
            _require("limit_factor")
            _require("limit_cap")
            _require("ratio_basis")
            _forbid("limit_value")
        elif rule is LimitRule.NONE_PERMITTED:
            _forbid("limit_value")
            _forbid("limit_factor")
            _forbid("limit_cap")
            _forbid("ratio_basis")

        # V7 — limit_type ↔ limit_rule 결합
        if self.limit_type is LimitType.RATIO and rule is not LimitRule.CONST:
            raise ValueError("[V7] limit_type=비율 → limit_rule=const 강제")
        if (self.limit_type is LimitType.NOT_PERMITTED) != (rule is LimitRule.NONE_PERMITTED):
            raise ValueError("[V7] limit_type=불허 ⇔ limit_rule=none_permitted 위반")

        # V1 — 단위-유형 교차 (길이·직경↔mm, 비율↔percent, 불허는 공란 허용)
        if self.limit_type in (LimitType.LENGTH, LimitType.DIAMETER):
            if self.unit is not Unit.MM:
                raise ValueError(f"[V1] limit_type={self.limit_type.value} → unit=mm 강제")
        elif self.limit_type is LimitType.RATIO:
            if self.unit is not Unit.PERCENT:
                raise ValueError("[V1] limit_type=비율 → unit=percent 강제")

        # V1 — quality_level 스킴 스코프 (확정 어휘가 있는 스킴만 강제)
        if self.quality_scheme is QualityScheme.ISO5817:
            if self.quality_level not in ISO5817_LEVEL_ORDER:
                raise ValueError(
                    f"[V1] quality_scheme=iso5817 의 quality_level 은 B/C/D — got {self.quality_level!r}"
                )
        elif self.quality_scheme is QualityScheme.NONE:
            if self.quality_level != "ALL":
                raise ValueError(
                    f"[V1] quality_scheme=none 의 quality_level 은 ALL — got {self.quality_level!r}"
                )
        return self

    # -- 편의 -----------------------------------------------------------------

    def contains_t(self, t: Decimal) -> bool:
        """반열림 [min, max) 소속 판정. t 는 호출 전 양자화돼 있어야 한다."""
        if t < self.thickness_min:
            return False
        return self.thickness_max is None or t < self.thickness_max


def v12_problems(row: LimitRow) -> list[str]:
    """V12 — 2 판 행 하나의 필수 · 짝(C 08 2 판 §1-3). **2 판 표에만** 건다 — 1 판 행은 새 칸이 없어 필수를 못 채운다.

    칸의 꼴(열거형 · 토큰 문법 · 정렬 · 겹침)은 모델이 먼저 본다. 여기는 칸 사이의 규칙이다:
    측정량 ↔ 값 한계(`limit_type` 이 불허가 아닌 행) · 측정량 ↔ 집계 · 종별 필수 · 조건 토큰의 앞뒤 조건(군집 ⇒ 합계, 전제 ⇒
    품질 체계 없음, 짝 종별 ≠ 제 종별) · VT 행의 새 칸 넷 빈칸. 행 사이의 규칙(종별 하나 · 짝 대칭 · 조항 안 일관)은 로더의 V13 이다.
    """
    out: list[str] = []
    where = f"[V12] {row.rule_id}:"
    rt_active = (row.inspection_method in (InspectionMethod.RT, InspectionMethod.ALL)
                 and row.scope is Scope.ACTIVE)
    value_limit = row.limit_rule is not LimitRule.NONE_PERMITTED
    if rt_active and value_limit and row.measure is None:
        out.append(f"{where} RT · ALL ∧ active 의 값 한계 행에 measure 가 없다")
    if rt_active and row.defect_class is None:
        out.append(f"{where} RT · ALL ∧ active 행에 defect_class 가 없다")
    if not value_limit and row.measure is not None:
        out.append(f"{where} 불허 행에 measure 가 있다 — 불허 조항은 측정량이 없다")
    if row.inspection_method is InspectionMethod.VT:      # 부분 B 범위 밖 — 넷 가운데 하나라도 차면 거부(09 §10 M1)
        filled = [name for name in ("measure", "aggregation", "defect_class", "conditions") if getattr(row, name)]
        if filled:
            out.append(f"{where} VT 행에 새 칸 {filled} 이 있다 — 부분 B 범위 밖이라 넷 모두 빈칸이다")
    if row.measure is None:
        if row.aggregation is not None:
            out.append(f"{where} measure 가 빈칸인데 aggregation 이 있다")
    elif (row.measure is Measure.TOTAL_LENGTH) != (row.aggregation is Aggregation.TOTAL) or row.aggregation is None:
        out.append(f"{where} measure {row.measure.value} 와 aggregation "
                   f"{getattr(row.aggregation, 'value', None)} 의 짝이 틀리다 — 합계길이 ⇔ 합계, 나머지 ⇒ 개별")
    for tok in row.conditions:
        kind, cls = parse_condition_token(tok)
        if kind == COND_CLUSTER_GAP and row.aggregation is not Aggregation.TOTAL:
            out.append(f"{where} {COND_CLUSTER_GAP} 는 aggregation=합계 행에만 둔다")
        elif kind == COND_PREMISE and row.quality_scheme is not QualityScheme.NONE:
            out.append(f"{where} {COND_PREMISE} 는 quality_scheme=none 행에만 둔다 — 품질 체계가 걸린 행이다")
        elif cls is not None:
            if row.defect_class is None:
                out.append(f"{where} 짝 토큰 {tok} 이 있는데 제 종별(defect_class)이 없다")
            elif cls == row.defect_class.value:
                out.append(f"{where} 짝 토큰 {tok} 의 종별이 제 종별과 같다")
    return out


# ---------------------------------------------------------------------------
# Judgment · LimitsTable
# ---------------------------------------------------------------------------


class Judgment(BaseModel):
    """judge() 의 출력. margin = L − measured (양수 = 여유). none_permitted 는 None.

    `basis_assumption` 은 기준 치수(ratio_basis) 에 가정이 개입한 경우의 흔적이다.
    prop 계열에서 basis 는 호출자가 넘기는데, ratio_basis=s(용접부 공칭 두께) 행에
    모재 두께 t 를 넘기는 것은 완전용입 맞대기 가정(열린 질문 3)이다. 반환값에 이
    표시가 없으면 소비처(D 채점기)가 가정 유래 판정을 확정 판정과 구분하지 못한다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    margin: Optional[Decimal] = None
    basis_assumption: Optional[str] = None


class LimitsTable(BaseModel):
    """G0 통과본. 공식 로더(load_limits) 외 경로로 만들지 않는다 (테스트 픽스처 제외)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[LimitRow, ...]
    sha256: str
    source_path: str
    pilot: bool = False  # True = 파일럿 전용·미검수 (V0·검수 컬럼·V5(sources 부재) 면제됨)
    v4_warnings: tuple[str, ...] = ()  # 단조성 경보 (예외 아님)
    v9_conflicts: tuple[str, ...] = ()  # 교차 문서 충돌 — canonical 로 해소된 기록
    v10_flags: tuple[str, ...] = ()  # 샘플링 실현성 미달 행 (판단 대상)
    checks_skipped: tuple[str, ...] = ()  # pilot 면제 등으로 건너뛴 검사 목록
    #: 표의 판 — 머리에 2 판 칸이 하나라도 있으면 "v2"(C 08 2 판 §2). 1 판 표는 지금 그대로 "v1" 이다.
    edition: Literal["v1", "v2"] = "v1"

    @property
    def active_canonical(self) -> tuple[LimitRow, ...]:
        """골격 생성·채점 정답·RAG 정답 목록이 소비하는 행 (scope=active ∧ canonical)."""
        return tuple(r for r in self.rows if r.scope is Scope.ACTIVE and r.canonical)

    def for_inspection(self, method: "InspectionMethod | str") -> "LimitsTable":
        """검사 방법 축 행 선택 뷰 (§1-2 5a). 행을 {method, ALL} 로 좁힌 같은 테이블을
        돌려준다 — sha256·source_path 는 원천 파일 그대로라 provenance 가 유지된다.

        `applicable_row` 도 같은 축 인자를 필수로 받으므로 이 뷰는 편의(반복 질의 시 후보를
        미리 줄이는 용도)이지 축 강제의 유일한 수단이 아니다. `ALL` 을 질의 값으로 주는 것은
        거부한다 — ALL 은 "검사 방법과 무관한 조항"의 표시이지 "전 행"을 뜻하지 않으며,
        전 행이 필요하면 원 테이블을 그대로 쓰면 된다. 이 구분이 없으면 표면·내부 기준이
        조용히 다시 섞인다.
        """
        m = InspectionMethod(method)
        if m is InspectionMethod.ALL:
            raise ValueError(
                "[for_inspection] 질의 값 ALL 금지 — RT 또는 VT 를 지정하라 "
                "(전 행이 필요하면 원 테이블을 쓴다)"
            )
        rows = tuple(
            r for r in self.rows
            if r.inspection_method is m or r.inspection_method is InspectionMethod.ALL
        )
        return self.model_copy(update={"rows": rows})
