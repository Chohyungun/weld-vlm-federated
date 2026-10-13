"""계약 #4 — 공통 예측 JSON 스키마의 **실행 가능한 정본**. 스펙 §2.

5칸(통합·중앙 / 통합·연합 / 분리·로컬 / 분리·중앙 / 분리·연합)이 전부 이 형식으로
출력하고, 단일 채점기가 이것만 읽는다. 칸마다 채점 코드가 갈라지는 순간 논문의 공정성
주장이 무너지므로, 칸을 구분해도 되는 지점은 어댑터 하나뿐이다.

`prediction.schema.json` 은 이 모듈에서 생성한다 — 손으로 맞추지 않는다. 두 곳을 각각
고치면 C가 보는 계약과 D가 채점하는 계약이 달라진다.

    from evaluation.schema import PredictionRecord, parse_record
    rec = parse_record(line)          # 실패해도 예외가 아니라 parse_ok=False 레코드

**좌표 규약**: `bbox_px` 는 C가 역변환을 끝낸 **원본 이미지 픽셀**이다(게이트 웨이브 #5
§5-1, C=규약 소유자 / D=검증 소유자). D는 좌표를 변환하지 않으며 이 모듈에도 변환 산식이
없다 — 그것이 이중 역변환을 구조적으로 막는 장치다.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = "1.3"
"""계약 #4 판본. v1.1 좌표 필드 삭제 → v1.2 어휘 coord_space 통일 → v1.3 retrieved 를
defects[] 항목으로 내림((이미지 × 결함코드) 쌍 단위 지표, 게이트 #8 결정 ②)."""

Cell = Literal["uni_central", "uni_fed", "sep_local", "sep_central", "sep_fed"]
ClientId = Literal["C1", "C2", "C3"]
Verdict = Literal["합격", "불합격", "판정불가"]
CoordSpace = Literal["NORM_1000", "ABS_RESIZED", "ABS_ORIG"]
SizeBasis = Literal["major_axis", "equiv_diameter"]

ParseError = Literal[
    "no_json",            # 생성문에서 JSON 블록을 찾지 못함
    "json_decode",        # 블록은 있으나 json.loads 실패
    "schema_violation",   # 파싱은 되나 필드 검증 실패
    "unknown_iso_code",   # label_map.yaml 에 없는 코드
    "bbox_invalid",       # 좌표 4개가 아니거나 퇴화·범위 이탈
    "join_missing",       # 분리형에서 한쪽 파일에 레코드 없음
    "truncated",          # max_tokens 도달로 생성이 잘림
]

SEPARATED_CELLS = ("sep_local", "sep_central", "sep_fed")
UNIFIED_CELLS = ("uni_central", "uni_fed")
TOP_K = 3


class Defect(BaseModel):
    """결함 인스턴스 1개. `retrieved` 가 여기 있는 이유는 §2-2에 있다 — 검색 질의가
    결함코드마다 따로 나가므로 조항 검색 지표의 단위가 (이미지 × 결함코드) 쌍이다."""

    model_config = ConfigDict(extra="forbid")

    iso_code: str = Field(min_length=1)
    bbox_px: tuple[float, float, float, float] | None = None
    score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    size_mm: float | None = None
    size_px: float | None = None
    size_basis: SizeBasis | None = None
    retrieved: Annotated[list[str], Field(max_length=TOP_K)] | None = None

    @model_validator(mode="after")
    def _bbox_is_well_formed(self) -> Defect:
        """구조적 유효성만 본다. 이미지 경계 이탈은 W/H 가 필요하므로 어댑터가 검사한다."""
        if self.bbox_px is None:
            return self
        if not all(math.isfinite(v) for v in self.bbox_px):
            raise ValueError("bbox_px 에 NaN·inf 가 있다")
        x1, y1, x2, y2 = self.bbox_px
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"퇴화 bbox: x1<x2, y1<y2 여야 한다 (받은 값 {self.bbox_px})")
        return self


class PredictionRecord(BaseModel):
    """이미지 1장 = 1레코드. jsonl 한 줄에 대응한다."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.3"]
    image_id: str = Field(min_length=1)
    cell: Cell
    client: ClientId | None = None
    seed: int
    defects: list[Defect]
    verdict: Verdict
    cited_clauses: list[str]
    assumed_thickness_mm: float | None = None
    assumed_quality_level: str | None = None
    parse_ok: bool
    parse_error: ParseError | None = None
    raw_output_ref: str | None = None
    coord_space: CoordSpace | None = None
    coord_cfg_hash: str | None = None
    latency_ms: float | None = None

    @model_validator(mode="after")
    def _cross_field_rules(self) -> PredictionRecord:
        # client 는 sep_local 에서만 의미가 있다 — 그 칸만 모델이 3개다(RQ3 분해에 필요).
        if self.cell == "sep_local" and self.client is None:
            raise ValueError("sep_local 은 어느 클라이언트 모델의 출력인지 알아야 한다")
        if self.cell != "sep_local" and self.client is not None:
            raise ValueError(f"{self.cell} 에는 client 를 채우지 않는다")

        # 통합형에는 검색을 붙이지 않는다. 이 비대칭이 판정 근거 신뢰도 비교의 대비축이다.
        if self.cell in UNIFIED_CELLS and any(
            d.retrieved is not None for d in self.defects
        ):
            raise ValueError("통합형은 검색을 붙이지 않으므로 retrieved 가 없어야 한다")

        # parse_ok=False 면 사유를 반드시 남긴다. 실패율은 별도 보고 지표다.
        if not self.parse_ok and self.parse_error is None:
            raise ValueError("parse_ok=False 인데 parse_error 가 비어 있다")
        if self.parse_ok and self.parse_error is not None:
            raise ValueError("parse_ok=True 인데 parse_error 가 채워져 있다")
        return self

    @property
    def iso_codes(self) -> frozenset[str]:
        """이미지 수준 클래스 집합. Macro-F1·Class-Jaccard 가 이 단위로 계산된다(§4-2)."""
        return frozenset(d.iso_code for d in self.defects)

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """(image_id, iso_code) 쌍 — 조항 검색·인용 지표의 단위(§4-8·§4-9)."""
        return tuple((self.image_id, code) for code in sorted(self.iso_codes))


def failed_record(
    image_id: str, cell: Cell, seed: int, error: ParseError, *,
    client: str | None = None, raw_output_ref: str | None = None,
) -> PredictionRecord:
    """파싱 실패를 레코드로 표현한다. 예외로 던지면 그 이미지가 통계에서 사라지고,
    사라진 것은 오답보다 낙관적으로 잡힌다 — 실패는 반드시 세어야 한다."""
    return PredictionRecord(
        schema_version=SCHEMA_VERSION,
        image_id=image_id,
        cell=cell,
        client=client,  # type: ignore[arg-type]
        seed=seed,
        defects=[],
        verdict="판정불가",
        cited_clauses=[],
        parse_ok=False,
        parse_error=error,
        raw_output_ref=raw_output_ref,
    )


def parse_record(payload: str | dict) -> PredictionRecord | tuple[None, ParseError]:
    """한 줄을 레코드로 만든다. 성공하면 레코드, 실패하면 `(None, 사유)`.

    **어떤 필드값도 보정하지 않는다.** 보정은 채점기가 답을 고쳐주는 것이다.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None, "json_decode"
    try:
        return PredictionRecord.model_validate(payload)
    except ValidationError:
        return None, "schema_violation"


def json_schema() -> dict:
    """`prediction.schema.json` 의 내용. 커밋본과의 일치를 테스트가 강제한다."""
    schema = PredictionRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://weld-fl.local/evaluation/prediction.schema.json"
    schema["title"] = "weld-fl 공통 예측 레코드"
    return schema
