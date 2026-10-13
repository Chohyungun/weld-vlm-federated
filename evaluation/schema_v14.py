"""계약 #4 의 **1.4 판** — 통합형 본실험(`uni_local`·`uni_central`·`uni_fed`) 전용. 07번 미니스펙 §12-4 · §13 · §14-2.

**1.3 을 고치지 않고 곁에 선다.** `evaluation/schema.py` 는 사전실험 세 시드의 레코드와 v3 산출물이 기대는 파일이라
바이트를 건드리지 않는다(v3 는 채점 코드의 파일별 해시를 싣는다). 1.4 가 1.3 과 다른 곳은 넷이다.

| | 1.3 | 1.4 |
|---|---|---|
| 칸 | 다섯 칸 | **통합형 세 칸.** 분리형은 1.4 에 들어오지 않는다 — 결합 규칙이라는 개념이 없다 |
| 결합 규칙 | 판정·인용 필드가 틀리면 레코드 전체가 빈 예측(v1) | 박스가 유효하면 채점하고, 판정·인용 오류는 **조항 축에서만** 실패(`decoupled_v2`) |
| 생성 종료 | 기록 없음 | `gen_stop`·`n_new_tokens` **필수** — 길이 종료와 JSON 절단은 다른 사실이다 |
| 좌표 규약 | 선택 | `coord_space`·`coord_cfg_hash` **필수** |

**`1.4 ⇔ decoupled_v2`.** 1.4 레코드는 새 결합 규칙으로만 만들어지고 새 결합 규칙은 1.4 레코드만 만든다.
그래서 판본이 곧 채점 규칙의 표지다 — 판본이 섞인 입력을 CI·집계기가 거부하는 근거가 이것이다.

**1.3 클래스를 상속하지 않는다.** 상속하면 부모 타입으로 선언된 자리에서 직렬화할 때 새 필드가 경고 없이 빠지고
판본 문자열만 `1.4` 로 남는다. `isinstance(rec, PredictionRecord)` 가 참이 되어 판본 혼입 검사도 샌다.
`iso_codes`·`pairs` 는 다시 적고 시험이 1.3 과의 동등을 본다. `Defect` 는 필드가 같아 그대로 쓴다.

**어떤 값도 지어내지 않는다.** 모델의 판정이 어휘 밖이면 `verdict` 는 null 이고 `clause_error` 가 그 사실을 적는다.
레코드 실패도 `verdict` 가 null 이다(1.3 의 `failed_record` 는 `판정불가` 를 채워 넣었다).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator

from evaluation.schema import ClientId, CoordSpace, Defect, PredictionRecord, Verdict

SCHEMA_VERSION_V14 = "1.4"

CellV14 = Literal["uni_local", "uni_central", "uni_fed"]
UNIFIED_CELLS_V14: tuple[str, ...] = ("uni_local", "uni_central", "uni_fed")

GenStop = Literal["eos", "length"]
"""생성이 종료 토큰으로 끝났는가(`eos`), `max_new_tokens` 에 닿아 끝났는가(`length`). 한도에 닿는 자리에서
종료 토큰이 나오면 `eos` 다. 둘 다 아닌 종료는 계약에 없다 — export 가 중단한다(07번 §13-3 의 10)."""

ClauseError = Literal["citations_invalid", "verdict_invalid"]

ParseErrorV14 = Literal["no_json", "json_decode", "truncated", "schema_violation"]
"""상류(C 의 export)가 넘기는 레코드 실패 사유. 1.3 의 `unknown_iso_code`·`bbox_invalid` 는 항목 폐기로 바뀐 지
오래고 `join_missing` 은 분리형의 것이라 1.4 에는 없다."""

CLAUSE_ERRORS: tuple[str, ...] = ("citations_invalid", "verdict_invalid")

_TAGS: dict[str, tuple[str, str | None]] = {
    "uni_local_C1": ("uni_local", "C1"),
    "uni_local_C2": ("uni_local", "C2"),
    "uni_local_C3": ("uni_local", "C3"),
    "uni_central": ("uni_central", None),
    "uni_fed": ("uni_fed", None),
}
UNI_MAIN_TAGS: tuple[str, ...] = tuple(_TAGS)
"""본실험 통합형의 다섯 모델. 파일럿의 `evaluation.cells.UNI_TAGS`(두 칸)와 **다른 이름**이다 — 그 이름은
`score_cells.py`·`stratified_compare.py` 가 쓰고 있어 값을 바꾸면 파일럿 재채점이 죽는다."""


def tag_parts(tag: str) -> tuple[str, str | None]:
    """태그 → (칸, 참여자). **명시 표로만** 푼다. 문자열을 쪼개면 `uni_local_c1`·`uni_central_C1` 같은 것이 새어 든다."""
    try:
        return _TAGS[tag]
    except KeyError:
        raise ValueError(f"본실험 통합형 태그가 아니다: {tag!r} (허용: {', '.join(UNI_MAIN_TAGS)})") from None


class PredictionRecordV14(BaseModel):
    """이미지 1장 = 1레코드. **기본값이 없는 필드는 전부 필수다** — 값이 null 이어도 키는 있어야 한다."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.4"]
    image_id: str = Field(min_length=1)
    cell: CellV14
    client: ClientId | None
    seed: int
    """학습 시드 **값**이다. 파일 이름의 `s{n}` 은 시드 번호이고 둘의 대응은 등록 블록이 정한다."""
    defects: list[Defect]
    verdict: Verdict | None
    cited_clauses: list[str]
    assumed_thickness_mm: float | None = None
    assumed_quality_level: str | None = None
    parse_ok: bool
    parse_error: ParseErrorV14 | None
    clause_error: list[ClauseError]
    gen_stop: GenStop
    n_new_tokens: Annotated[StrictInt, Field(ge=0)]
    raw_output_ref: str | None = None
    coord_space: CoordSpace
    coord_cfg_hash: str = Field(min_length=1)
    latency_ms: float | None = None

    @model_validator(mode="after")
    def _cross_field_rules(self) -> PredictionRecordV14:
        if (self.cell == "uni_local") != (self.client is not None):
            raise ValueError(
                "uni_local 은 어느 참여자 모델의 출력인지 알아야 하고, 그 밖의 칸에는 client 를 채우지 않는다 "
                f"(cell={self.cell}, client={self.client})"
            )
        if any(d.retrieved is not None for d in self.defects):
            raise ValueError("통합형은 검색을 붙이지 않으므로 retrieved 가 없어야 한다")
        if self.clause_error != sorted(set(self.clause_error)):
            raise ValueError(f"clause_error 는 정렬된 중복 없는 목록이다 (받은 값 {self.clause_error})")

        if not self.parse_ok:
            if self.parse_error is None:
                raise ValueError("parse_ok=False 인데 parse_error 가 비어 있다")
            if self.defects or self.clause_error or self.cited_clauses or self.verdict is not None:
                raise ValueError(
                    "레코드 실패는 빈 예측이다 — defects·clause_error·cited_clauses 가 비고 verdict 가 null 이어야 한다. "
                    "레코드 실패와 조항 축 위반은 서로소다(합산 위반율이 두 번 세지 않게)"
                )
        else:
            if self.parse_error is not None:
                raise ValueError("parse_ok=True 인데 parse_error 가 채워져 있다")
            if (self.verdict is None) != ("verdict_invalid" in self.clause_error):
                raise ValueError(
                    "verdict 가 null 인 것과 clause_error 에 verdict_invalid 가 있는 것은 같은 말이어야 한다 — "
                    "판정을 지어내지도, 유효한 판정을 버리지도 않는다"
                )
            if "citations_invalid" in self.clause_error and self.cited_clauses:
                raise ValueError("citations_invalid 면 cited_clauses 는 비어 있어야 한다")

        if self.gen_stop == "eos" and self.n_new_tokens < 1:
            raise ValueError("gen_stop=eos 인데 n_new_tokens 가 0 이다 — 종료 토큰도 생성된 토큰이다")
        return self

    @property
    def iso_codes(self) -> frozenset[str]:
        """이미지 수준 클래스 집합. 1.3 과 같은 정의다."""
        return frozenset(d.iso_code for d in self.defects)

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """(image_id, iso_code) 쌍. 1.3 과 같은 정의다."""
        return tuple((self.image_id, code) for code in sorted(self.iso_codes))


def failed_record_v14(
    image_id: str, *, cell: str, client: str | None, seed: int, error: str,
    gen_stop: str, n_new_tokens: int, coord_space: str, coord_cfg_hash: str,
    latency_ms: float | None = None, raw_output_ref: str | None = None,
) -> PredictionRecordV14:
    """레코드 실패를 1.4 레코드로. **생성자를 거친다** — `model_copy(update=…)` 는 검증 없이 필수 필드에 null 을
    넣을 수 있어 1.4 의 필수 규칙이 실패 레코드에서만 조용히 꺼진다."""
    return PredictionRecordV14(
        schema_version=SCHEMA_VERSION_V14, image_id=image_id,
        cell=cell, client=client, seed=seed,                       # type: ignore[arg-type]
        defects=[], verdict=None, cited_clauses=[],
        parse_ok=False, parse_error=error, clause_error=[],        # type: ignore[arg-type]
        gen_stop=gen_stop, n_new_tokens=n_new_tokens,              # type: ignore[arg-type]
        coord_space=coord_space, coord_cfg_hash=coord_cfg_hash,    # type: ignore[arg-type]
        latency_ms=latency_ms, raw_output_ref=raw_output_ref,
    )


AnyRecord = Annotated[PredictionRecord | PredictionRecordV14, Field(discriminator="schema_version")]
_ANY = TypeAdapter(AnyRecord)


def read_any_records(lines: Iterable[bytes | str]) -> list[PredictionRecord | PredictionRecordV14]:
    """저장된 레코드 줄을 **판본으로 갈라** 읽는다. 1.3 줄은 기존 클래스로, 1.4 줄은 새 클래스로 간다.

    옛 되읽기(`evaluation.adapters.read_records`)는 넓히지 않는다 — 그쪽이 1.4 를 읽게 되면 옛 결합 규칙의 경로가
    1.4 레코드를 조용히 채점한다. 줄은 호출부가 **LF 로만** 나눠 넘긴다(`str.splitlines()` 는 U+2028·U+0085 에서도 쪼갠다).
    빈 줄은 건너뛰지 않고 오류다 — 줄 수가 계약의 일부다.
    """
    out: list[PredictionRecord | PredictionRecordV14] = []
    for n, raw in enumerate(lines, 1):
        if not raw.strip():
            raise ValueError(f"{n}번째 줄이 빈 줄이다")
        out.append(_ANY.validate_json(raw))
    return out


def assert_single_version(records: Sequence[PredictionRecord | PredictionRecordV14]) -> str:
    """한 채점 입력에 판본이 하나뿐임을 확인하고 그 판본을 돌려준다. 판본은 결합 규칙의 표지다."""
    if not records:
        raise ValueError("레코드가 비어 있다")
    versions = sorted({r.schema_version for r in records})
    if len(versions) != 1:
        raise ValueError(f"판본이 섞여 있다: {versions} — 결합 규칙이 다른 레코드를 한 입력으로 채점하지 않는다")
    return versions[0]


def json_schema_v14() -> dict:
    """`prediction.schema.v1_4.json` 의 내용. 커밋본과의 일치를 시험이 강제한다."""
    schema = PredictionRecordV14.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://weld-fl.local/evaluation/prediction.schema.v1_4.json"
    schema["title"] = "weld-fl 공통 예측 레코드 1.4 (통합형 본실험)"
    return schema
