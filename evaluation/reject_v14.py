"""1.4 묶음을 **채점하지 않는 사유**의 정본 표. 07번 미니스펙 §13-3 사 · §14-1 · §16-2.

거부는 "이 파일로는 수치를 낼 수 없다" 는 판정이고, 불일치(`mismatch`)나 낮은 점수와 다르다.
**사유는 언제나 코드 하나로 말한다** — 어느 조건이 걸렸는지 모르면 고칠 수가 없다.

## 이 표가 있는 이유

거부 조건을 함수 안에 흩어 두면 세 가지가 조용히 생긴다.

1. 시험이 "거부됐다" 만 보고 **무엇 때문인지** 안 본다. 그러면 다른 이유로 거부돼도 통과한다.
2. 조건을 더하면서 시험을 안 더한다 — 표가 없으면 빠진 것을 셀 수 없다.
3. 같은 뜻의 코드가 두 개 생긴다.

그래서 코드를 **여기 한 곳에** 두고, 시험이 `set(RejectCode) == 시험이 덮는 집합` 을 단언한다(메타 시험).

## 단계

검사는 순서가 있다. 앞 단계가 실패하면 뒤를 보지 않는다 — 바이트가 깨진 파일의 내용을 해석하면 오류가 원인을 가린다.

| 단계 | 무엇 | 왜 먼저 |
|---|---|---|
| A 존재·스키마 | 파일이 있는가, 곁 파일의 꼴이 맞는가 | 없는 파일의 해시를 셀 수 없다 |
| B 바이트 | 해시·CR·BOM·줄 수 | 내용 해석 전에 파일이 그 파일인지 정한다 |
| C 내용 | id 수열·중복·줄 필드·좌표 규약 | |
| D 등록·원장 | 곁 파일의 실제 값 ↔ 등록값, 학습 산출물·원장 | 값이 맞아도 **약속한 조건으로 돌린 것인지**는 별개다 |
| E 카나리아 | 문자 일치·축 배율 | 위가 다 서야 좌표를 말할 수 있다 |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    EXISTS = "A"
    BYTES = "B"
    CONTENT = "C"
    REGISTRATION = "D"
    CANARY = "E"


class RejectCode(str, Enum):
    """거부 사유. **값은 산출물과 보고에 그대로 나간다** — 이름을 바꾸면 과거 보고와 이어지지 않는다."""

    # ── A 존재·스키마
    SIDECAR_MISSING = "sidecar_missing"
    SIDECAR_SCHEMA = "sidecar_schema"
    START_STAMP_MISSING = "start_stamp_missing"
    ATTEMPTS_MISSING = "attempts_missing"
    RECEIPT_MISSING = "receipt_missing"
    TOKENS_FILE_MISMATCH = "tokens_file_mismatch"
    """`tokens_sha256` 과 파일의 있고 없음이 어긋난다. 선택 파일이라 **둘 다 없는 것만** 정상이다."""
    START_STAMP_SCHEMA = "start_stamp_schema"
    """시작 도장을 읽을 수 없거나 계약이 요구하는 키가 없다(§14-1 나·다 · §13-3 바-15).
    **해시가 맞는 것과 내용이 있는 것은 다르다** — 존재·해시만 보면 빈 도장도 통과한다."""
    ATTEMPTS_SCHEMA = "attempts_schema"
    """시도 기록의 행을 읽을 수 없거나 행에 주인·구간·사유·시각이 없다(§14-1 다 · §13-3 바-17)."""

    # ── B 바이트
    GENERATIONS_HASH_MISMATCH = "generations_hash_mismatch"
    TOKENS_HASH_MISMATCH = "tokens_hash_mismatch"
    START_STAMP_HASH_MISMATCH = "start_stamp_hash_mismatch"
    CR_IN_FILE = "cr_in_file"
    BOM_IN_FILE = "bom_in_file"
    NOT_UTF8 = "not_utf8"
    BLANK_LINE = "blank_line"
    TORN_TAIL = "torn_tail"
    """끝 개행이 없다 — 쓰다 만 파일이다."""
    LINE_COUNT_MISMATCH = "line_count_mismatch"
    """실제 줄 수 · 곁 파일의 `n_lines` · 생성 목록 길이 **셋을 한 번에** 맞댄다."""

    # ── C 내용
    LINE_SCHEMA = "line_schema"
    ID_SEQUENCE_MISMATCH = "id_sequence_mismatch"
    DUPLICATE_IMAGE_ID = "duplicate_image_id"
    POPULATION_NOT_SUBSET = "population_not_subset"
    """채점 모집단이 생성 목록의 부분집합이 아니다(§13-2 가)."""
    POPULATION_GROUP_INCOMPLETE = "population_group_incomplete"
    LIST_NOT_CANONICAL = "list_not_canonical"
    COORD_SPACE_MISMATCH = "coord_space_mismatch"
    MODEL_INPUT_WH_MISMATCH = "model_input_wh_mismatch"
    """줄 사이 또는 곁 파일과 다르다. **단일 크기 모집단의 전제 위에 있다**(§16-2 m-1)."""
    MODEL_INPUT_GRID_MISMATCH = "model_input_grid_mismatch"
    """`model_input_wh` 가 관측 격자 × 패치 × 병합에서 나온 값이 아니다 — 문자 일치의 동어반복을 막는 닻."""
    IMAGE_BYTES_MISMATCH = "image_bytes_mismatch"
    TOKENS_CONTENT_MISMATCH = "tokens_content_mismatch"
    """토큰 파일의 줄 수 · id 수열이 생성 파일과 다르거나, 줄의 `token_ids` · `token_logprobs` 길이가 그 줄의
    `n_new_tokens` 와 다르다(진입점 미니스펙 3판 1-8). 상세의 `reason` 이 셋 가운데 어느 것인지 말한다."""
    STOP_UNDETERMINED = "stop_undetermined"
    """`gen_stop` 을 정할 수 없다고 적힌 줄이 있다. 중단이 아니라 거부다 — 고친 뒤 그 파일만 다시 만든다(§16-2 m-2)."""

    # ── D 등록·원장
    GENERATION_FINGERPRINT_MISMATCH = "generation_fingerprint_mismatch"
    REGISTRATION_ITEM_MISMATCH = "registration_item_mismatch"
    SEED_NOT_REGISTERED = "seed_not_registered"
    SEED_MAPPING_MISMATCH = "seed_mapping_mismatch"
    """파일 이름의 시드 번호와 곁 파일의 번호·값이 등록 시드 목록의 자리와 어긋난다."""
    TAG_MISMATCH = "tag_mismatch"
    ADAPTER_DUPLICATE = "adapter_duplicate"
    ADAPTER_NOT_FINAL = "adapter_not_final"
    """`scored_adapter_step` 이 학습 원장의 그 run 마지막 행과 다르다 — 중간 체크포인트를 채점하는 것을 막는다(§16-1)."""
    LEDGER_MISMATCH = "ledger_mismatch"
    ARTIFACT_META_MISSING = "artifact_meta_missing"
    """어댑터 sha 와 스텝이 사는 자리다. 원장에는 그 열이 없다(§16-1)."""
    ENV_MISMATCH = "env_mismatch"
    """다섯 모델 × 시드 사이에 프로세서 설정·관측 격자·생성 접두·라이브러리 판이 다르다."""
    FAULT_EVENTS_PRESENT = "fault_events_present"
    """본실험 · 진단 묶음의 곁 파일 `fault_events` 가 빈 목록이 아니다(07번 §30-4). 고의 중단 장치는 리허설의 것이다."""
    TIME_ORDER = "time_order"
    TIME_NAIVE = "time_naive"
    MODE_MISMATCH = "mode_mismatch"
    """`mode` 가 파일 이름과 필드에서 다르거나, 채점이 요구하는 모드가 아니다."""
    ECHO_LIST_MISMATCH = "echo_list_mismatch"
    """에코 목록이 등록된 에코 목록(val 의 부분집합)과 다르다(§16-2 I-3)."""
    ECHO_ADAPTER_PRESENT = "echo_adapter_present"
    """에코는 모델을 부르지 않으므로 채점 대상 어댑터가 **없어야** 한다(§16-2 I-3).
    곁 파일과 **시작 도장** 양쪽에서 선다."""
    START_STAMP_MISMATCH = "start_stamp_mismatch"
    """시작 도장의 내용이 곁 파일·등록값과 다르다. 다른 묶음의 도장을 놓고 그 해시만 곁
    파일에 적으면 바이트 해시 검사는 통과한다 — 그 길을 여기서 막는다."""
    ATTEMPTS_ROW_MISMATCH = "attempts_row_mismatch"
    """이 묶음 몫으로 고른 시도 줄의 수가 곁 파일의 `attempts` 와 다르거나, 줄의 구간·시각·장비가
    계약을 어긴다. 시도 기록은 폴더당 하나를 열다섯 묶음이 같이 쓴다(§14-1 다)."""
    ATTEMPTS_ACCOUNTING_MISMATCH = "attempts_accounting_mismatch"
    """시도가 남긴 줄 수의 회계가 서지 않는다(§28-1). 자리를 목록 위치로 세는데 그 차가 음수거나,
    끝 줄이 적은 수와 다르거나, 합이 곁 파일의 줄 수와 다르다.

    **합이 보는 것은 둘이다** — 첫 시도가 목록 처음에서 시작했는가, 시작 자리가 뒤로 가지 않는가.
    자리 차를 다 더하면 그 둘만 남는다. 그래서 사유의 상세에 어느 쪽인지를 적는다."""

    # ── E 카나리아
    LITERAL_MATCH_FAILED = "literal_match_failed"
    """`text` 를 참조 파서로 다시 읽은 값이 `bbox_px` 와 다르다."""
    PARSER_DISAGREEMENT = "parser_disagreement"
    """참조 파서와 상류 파서의 **유지·폐기 판정**이 다르다(항목 수·순서·폐기 수)."""


STAGE_OF: dict[RejectCode, Stage] = {
    RejectCode.SIDECAR_MISSING: Stage.EXISTS,
    RejectCode.SIDECAR_SCHEMA: Stage.EXISTS,
    RejectCode.START_STAMP_MISSING: Stage.EXISTS,
    RejectCode.ATTEMPTS_MISSING: Stage.EXISTS,
    RejectCode.RECEIPT_MISSING: Stage.EXISTS,
    RejectCode.TOKENS_FILE_MISMATCH: Stage.EXISTS,
    # 도장·시도 기록의 스키마 검사는 **D 단계에서 돈다**(내용을 등록·곁 파일과 맞대는 자리다).
    # 앞 판은 이 둘을 A 로 적어 보고의 단계 표기가 검사 위치와 어긋났다(19번 Minor 4-1).
    RejectCode.START_STAMP_SCHEMA: Stage.REGISTRATION,
    RejectCode.ATTEMPTS_SCHEMA: Stage.REGISTRATION,
    RejectCode.START_STAMP_MISMATCH: Stage.REGISTRATION,
    RejectCode.ATTEMPTS_ROW_MISMATCH: Stage.REGISTRATION,
    RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH: Stage.REGISTRATION,
    RejectCode.GENERATIONS_HASH_MISMATCH: Stage.BYTES,
    RejectCode.TOKENS_HASH_MISMATCH: Stage.BYTES,
    RejectCode.START_STAMP_HASH_MISMATCH: Stage.BYTES,
    RejectCode.CR_IN_FILE: Stage.BYTES,
    RejectCode.BOM_IN_FILE: Stage.BYTES,
    RejectCode.NOT_UTF8: Stage.BYTES,
    RejectCode.BLANK_LINE: Stage.BYTES,
    RejectCode.TORN_TAIL: Stage.BYTES,
    RejectCode.LINE_COUNT_MISMATCH: Stage.BYTES,
    RejectCode.LINE_SCHEMA: Stage.CONTENT,
    RejectCode.ID_SEQUENCE_MISMATCH: Stage.CONTENT,
    RejectCode.DUPLICATE_IMAGE_ID: Stage.CONTENT,
    RejectCode.POPULATION_NOT_SUBSET: Stage.CONTENT,
    RejectCode.POPULATION_GROUP_INCOMPLETE: Stage.CONTENT,
    RejectCode.LIST_NOT_CANONICAL: Stage.CONTENT,
    RejectCode.COORD_SPACE_MISMATCH: Stage.CONTENT,
    RejectCode.MODEL_INPUT_WH_MISMATCH: Stage.CONTENT,
    RejectCode.MODEL_INPUT_GRID_MISMATCH: Stage.CONTENT,
    RejectCode.IMAGE_BYTES_MISMATCH: Stage.CONTENT,
    RejectCode.TOKENS_CONTENT_MISMATCH: Stage.CONTENT,
    RejectCode.STOP_UNDETERMINED: Stage.CONTENT,
    RejectCode.GENERATION_FINGERPRINT_MISMATCH: Stage.REGISTRATION,
    RejectCode.REGISTRATION_ITEM_MISMATCH: Stage.REGISTRATION,
    RejectCode.SEED_NOT_REGISTERED: Stage.REGISTRATION,
    RejectCode.SEED_MAPPING_MISMATCH: Stage.REGISTRATION,
    RejectCode.TAG_MISMATCH: Stage.REGISTRATION,
    RejectCode.ADAPTER_DUPLICATE: Stage.REGISTRATION,
    RejectCode.ADAPTER_NOT_FINAL: Stage.REGISTRATION,
    RejectCode.LEDGER_MISMATCH: Stage.REGISTRATION,
    RejectCode.ARTIFACT_META_MISSING: Stage.REGISTRATION,
    RejectCode.ENV_MISMATCH: Stage.REGISTRATION,
    RejectCode.FAULT_EVENTS_PRESENT: Stage.REGISTRATION,
    RejectCode.TIME_ORDER: Stage.REGISTRATION,
    RejectCode.TIME_NAIVE: Stage.REGISTRATION,
    RejectCode.MODE_MISMATCH: Stage.REGISTRATION,
    RejectCode.ECHO_LIST_MISMATCH: Stage.REGISTRATION,
    RejectCode.ECHO_ADAPTER_PRESENT: Stage.REGISTRATION,
    RejectCode.LITERAL_MATCH_FAILED: Stage.CANARY,
    RejectCode.PARSER_DISAGREEMENT: Stage.CANARY,
}
"""코드 → 단계. **모든 코드가 여기 있어야 한다** — 시험이 그것을 단언한다."""

ECHO_EXEMPT: frozenset[RejectCode] = frozenset({
    RejectCode.ADAPTER_DUPLICATE,
    RejectCode.ADAPTER_NOT_FINAL,
    RejectCode.ARTIFACT_META_MISSING,
    RejectCode.LEDGER_MISMATCH,
    RejectCode.POPULATION_NOT_SUBSET,
    RejectCode.POPULATION_GROUP_INCOMPLETE,
})
"""에코 묶음에 적용하지 않는 검사(§16-2 I-3).

에코는 모델을 부르지 않아 채점 대상 어댑터가 없고, **val 이미지**를 쓰므로 동결 eval 모집단 검사가 성립하지 않는다.
대신 `ECHO_LIST_MISMATCH`·`ECHO_ADAPTER_PRESENT`·`MODE_MISMATCH` 가 에코 전용으로 선다.
"""


@dataclass(frozen=True)
class Rejection:
    """거부 한 건. **어느 조건이 걸렸는지와 무엇을 봤는지를 같이 들고 다닌다.**"""

    code: RejectCode
    where: str
    """파일 이름이나 `<파일>:<줄 번호>`. **식별자 값을 넣지 않는다** — 보고서로 새어 나간다."""
    detail: dict = field(default_factory=dict)

    @property
    def stage(self) -> Stage:
        return STAGE_OF[self.code]

    def as_dict(self) -> dict:
        return {"code": self.code.value, "stage": self.stage.value, "where": self.where,
                **({"detail": self.detail} if self.detail else {})}

    def __str__(self) -> str:
        extra = f" {self.detail}" if self.detail else ""
        return f"[{self.stage.value}/{self.code.value}] {self.where}{extra}"


class BundleRejected(Exception):
    """묶음을 채점하지 않는다. **거부 목록을 들고 있다** — 첫 건만 알려 주면 두 번 돌게 된다."""

    def __init__(self, rejections: list[Rejection]):
        self.rejections = list(rejections)
        super().__init__("; ".join(str(r) for r in self.rejections) or "거부 사유가 비어 있다")

    def codes(self) -> set[RejectCode]:
        return {r.code for r in self.rejections}
