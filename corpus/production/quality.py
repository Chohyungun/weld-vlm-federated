"""생성 · 판정 출력의 전량 품질 검사 — `audit_run(run_dir, config, output_dir)`.

작업 행 · 릴리스 행의 꼴은 생산 파이프라인의 공통 계약을 따르고, 공통 계약이 정하지 않은 AUDIT · 검수 양식 칸은
릴리스(`release.py`)와 맞춘 공유 계약을 따른다. 샤드는 `infer.read_stage_rows` 로만 읽는다(입력 · 행 순서 · 샤드
지문 검사를 거친 행). 작업 명부는 RUN.json 의 `tasks_sha256` 과 바이트를 대조해 읽는다.

검사하는 것(행마다):
  스키마 · 미완료(생성/판정 행 없음, 길이 절단) · 언어 혼입(한자 · 가나, 영어 문장, 한글 비율) ·
  지시문 유출(시스템 지시 · 작업 지시 조각, 역할 표지) · 수치 추적(본문의 모든 수가 `locked_target` 이나
  근거 구절에 있어야 한다 — 유료 표준 수치는 이 규칙으로 막는다) · 근거 ID · 판정기 결과(릴리스와 같은 엄격 파싱) ·
  명부 계약(이미지 지문, 학습 질문 틀, 원문 문서 지문) ·
  판정 출력의 꼴은 판정 출처가 정한다: 실행 설정의 첫 판정은 옛 네 키(`_judge`), 따로 둔 판정은 묶인 정책의 출력 계약
  (`judge_structured_v7` → `parse_judge_v7`, 지난 판 `judge_structured_v6` → `parse_judge_v6` ·
  `judge_structured_v5` → `parse_judge_v5` — 행 종류별 구조화 물음의 답을 엄격히 읽는다). 한 꼴로 못 읽으면 오류이고
  다른 꼴로 다시 읽지 않는다. v7 은 정상 행에 판정 모델을 부르지 않는다 — 그 행은 판정 신원이 남긴 결정적 영수증
  (생성문 · 정책 지문에 묶이고 집계가 다시 계산해 맞댄다)으로만 가르고 판정 칸은 '적용 안 됨' 이다 ·
  결정적 정책(`image_report` 생성문만): 정책의 `deterministic_reasons` 는 기각 사유(`policy:<사유>`),
  `deterministic_flags` 는 관찰 표지(기각하지 않는다)다. 문법은 정책 쪽 몫이고 여기서는 정책의 꼴과 지문만 본다 —
  따로 둔 판정이면 정책 지문이 판정 신원의 근거에 묶인 것과 같아야 한다. 정책이 예외를 내면 통과가 아니라 오류다.
전체로 보는 것:
  분할 · 계보 · 같은 이미지 · 중복 · 구절당 상한 · 원문 QA 부족량 · 참여자 × 결함 커버리지 · 시험용(`fake`) 백엔드 행.

산출(새 디렉터리, 덮어쓰지 않는다):
  `dataset.jsonl`        자동 검사를 모두 통과한 행만, 계획 순서대로, 릴리스 행 꼴(messages · provenance).
  `rows_status.jsonl`    계획한 모든 작업의 판정(accepted · rejected · error · incomplete)과 사유. 판정기의 원문 지문과
                         엄격 파싱한 판정(구조화 판정이면 원 답 · 세 판정값 · 모델 표지), 결정적 정책의 사유 · 표지 · 오류를
                         따로 싣는다(미판정 행도 정책 결과는 싣는다).
  `REVIEW_TEMPLATE.json` 사람 검수 양식 — 릴리스가 읽는 검수 파일과 같은 꼴, 판정 · 검수자 · 시각 칸은 비어 있다.
  `AUDIT.json`           위 세 파일의 sha256, 실행 신원, 계획 수열, 수, 사유별 건수, 커버리지, 부족량, 검수 표본, 막는 사유. 마지막에 쓴다.

**부분 통과로 준비 완료를 만들지 않는다.** 미완료 · 오류 · 기각 · 부족량 · 시험용 백엔드가 하나라도 있으면
`status = blocked` 다. 막는 사유가 없어도 `ready_for_training` 은 늘 false 다 — 판정기의
grounded · scope_ok · language_ok 가 모두 true 여도 사람 승인이 아니다. 학습 릴리스는 사람 검수 기록과
자료 이용 확인을 받는 `release_run` 이 정한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .common import canonical_bytes, parse_json, write_json_new
from .infer import GENERATE_SYSTEM, _family, read_stage_rows

AUDIT_SCHEMA_VERSION = 1
AUDIT_VERSION = "quality_v9"
JUDGE_KEYS = ("grounded", "language_ok", "reason", "scope_ok")
#: 옛 판정(실행 설정의 판정 단계)의 꼴 — 네 키를 `_judge` 가 읽는다
JUDGE_LEGACY_SCHEMA = "legacy_four_keys"
#: 구조화 판정의 출력 계약 — 판정 정책의 출력 계약 이름과 같아야 따로 둔 판정을 `parse_judge_v5` 로 읽는다
JUDGE_V5_SCHEMA_VERSION = "judge_structured_v5"
#: 행 종류별로 구조화 판정이 내야 하는 키(정렬, 정확히 이 집합). `reason` 밖은 모두 참 · 거짓. 원문 QA 는 옛 네 키 그대로
JUDGE_V5_KEYS = {
    "defect_free": ("asserts_verdict", "claims_defect", "has_limitation", "language_ok", "reason",
                    "states_no_marked_defect"),
    "defect": ("asserts_verdict_dimension_or_cause", "fields_match", "has_limitation", "language_ok", "reason"),
    "source_qa": ("grounded", "language_ok", "reason", "scope_ok"),
}
#: 구조화 판정 파서의 판. 스키마 판(`JUDGE_V5_SCHEMA_VERSION`)은 그대로 두고 뜻을 바꾸지 않는 형식 정규화 둘을 더했다
JUDGE_V5_PARSER_VERSION = "parse_judge_v5_1"
#: 파서가 적용할 수 있는 형식 정규화 — 적용한 것만 이 순서로 `format_normalized` 에 남는다.
#: `code_fence` 출력 전체를 감싼 코드 울타리 하나를 벗겼다 · `string_boolean` 물음 답의 "true" · "false" 글을 참 · 거짓으로 읽었다
JUDGE_V5_FORMAT_NORMALIZATIONS = ("code_fence", "string_boolean")
#: 물음 답으로 받는 참 · 거짓 글 — 소문자로 맞춘 뒤 이 둘만. "yes" · "참" · 1 · 문장 · 앞뒤 공백이 붙은 글은 받지 않는다
JUDGE_V5_BOOL_STRINGS = {"true": True, "false": False}
#: 구조화 판정 v6 의 출력 계약 — 판정 정책의 출력 계약 이름과 같아야 따로 둔 판정을 `parse_judge_v6` 로 읽는다.
#: v6 는 v5 를 대체하지 않는다 — v5 의 원문 · 영수증은 v5 파서로만, v6 의 것은 v6 파서로만 읽는다
JUDGE_V6_SCHEMA_VERSION = "judge_structured_v6"
#: 행 종류별 v6 키(정렬, 정확히 이 집합). 결함 행의 `fields_match` 를 이름 · 개수 · 위치 셋으로 나눴다. 정상 행 · 원문 QA 는 v5 와 같다
JUDGE_V6_KEYS = {
    "defect_free": ("asserts_verdict", "claims_defect", "has_limitation", "language_ok", "reason",
                    "states_no_marked_defect"),
    "defect": ("asserts_verdict_dimension_or_cause", "counts_match", "has_limitation", "language_ok", "names_match",
               "positions_match", "reason"),
    "source_qa": ("grounded", "language_ok", "reason", "scope_ok"),
}
#: v6 파서의 판
JUDGE_V6_PARSER_VERSION = "parse_judge_v6"
#: v6 의 형식 정규화 — v5_1 보다 좁다. 적용한 것만 이 순서로 `format_normalized` 에 남는다
#: (`[]` · `[code_fence]` · `[string_boolean]` · `[code_fence, string_boolean]` 넷 가운데 하나)
JUDGE_V6_FORMAT_NORMALIZATIONS = ("code_fence", "string_boolean")
#: v6 의 코드 울타리 — 소문자 `json` 표시가 붙은 여는 줄과 제 줄의 닫는 울타리만. 표시 없는 울타리 · 대문자는 받지 않는다
JUDGE_V6_FENCE_OPEN = "```json\n"
JUDGE_V6_FENCE_CLOSE = "\n```"
#: v6 가 받는 참 · 거짓 글 — 정확히 이 둘만(대소문자를 맞추지 않는다). "True" · " true" · "yes" 는 받지 않는다
JUDGE_V6_BOOL_STRINGS = {"true": True, "false": False}
#: 구조화 판정 v7 의 출력 계약 — 판정 정책의 출력 계약 이름과 같아야 따로 둔 판정을 `parse_judge_v7` 로 읽는다.
#: v7 은 v5 · v6 를 대체하지 않는다 — 각 판의 원문 · 영수증은 그 판의 파서로만 읽는다
JUDGE_V7_SCHEMA_VERSION = "judge_structured_v7"
#: 행 종류별 v7 키(정렬, 정확히 이 집합). 결함 행은 물음 넷 + `reason` 이고 넷 모두 "참 = 문제 없음" 이다. 정상 행은 판정
#: 모델을 부르지 않으므로 키가 없다(`parse_judge_v7` 이 `judge_not_applicable` 로 받지 않는다). 원문 QA 는 옛 네 키 그대로
JUDGE_V7_KEYS = {
    "defect": ("has_limitation", "language_ok", "no_unannotated_attributes", "no_unsupported_relation", "reason"),
    "source_qa": ("grounded", "language_ok", "reason", "scope_ok"),
}
#: v7 파서의 판
JUDGE_V7_PARSER_VERSION = "parse_judge_v7"
#: v7 의 형식 정규화 — v6 의 좁은 둘 그대로. 적용한 것만 이 순서로 `format_normalized` 에 남는다
JUDGE_V7_FORMAT_NORMALIZATIONS = ("code_fence", "string_boolean")
#: v7 의 코드 울타리 — v6 와 같은 글자. v7 의 폐쇄가 v6 의 상수 · 함수에 닿지 않게 따로 둔다
JUDGE_V7_FENCE_OPEN = "```json\n"
JUDGE_V7_FENCE_CLOSE = "\n```"
#: v7 이 받는 참 · 거짓 글 — v6 와 같이 정확히 이 둘만
JUDGE_V7_BOOL_STRINGS = {"true": True, "false": False}
#: 생성이 끝까지 간 것으로 받는 종료 — 모델의 끝 표지(`eos`)와, 첫 JSON 객체가 닫혀 정지 조건이 멈춘 것(`json_closed`).
#: 그 밖(`length` 등)은 잘림이다
JUDGE_V7_FINISH_REASONS = ("eos", "json_closed")
#: `reason` 이 이 글자 수를 넘으면 표지 `reason_long`(기각이 아니다). 글자 수는 받은 글 그대로의 `len`(코드 포인트 —
#: 앞뒤 공백을 떼거나 유니코드 정규화를 하지 않는다)
JUDGE_V7_REASON_MAX_CHARS = 40
#: 판정 출력의 꼴 → 그 꼴을 읽은 파서의 판. 옛 네 키는 파서 판이 없다. 품질 집계가 읽는 구조화 판정의 꼴도 이 표다.
#: v7 은 정상 행에 판정 행이 없다 — 집계는 판정 신원의 결정적 영수증으로 정상 행을 가른다(`_v7_receipts`)
JUDGE_PARSER_VERSIONS = {JUDGE_V5_SCHEMA_VERSION: JUDGE_V5_PARSER_VERSION, JUDGE_V6_SCHEMA_VERSION: JUDGE_V6_PARSER_VERSION,
                         JUDGE_V7_SCHEMA_VERSION: JUDGE_V7_PARSER_VERSION}
#: 판정 신원 읽기를 부른 호출자의 외부 핀 — 다시 읽을 때 원값(None 포함)을 그대로 넘긴다
JUDGE_SOURCE_PIN_KEYS = ("expected_judge_identity_sha256", "expected_probe_approval_sha256")
#: 결정적 정책을 거는 작업 종류 — 원문 QA 에는 걸지 않는다(영상 규칙 · 단위 검사가 원문의 허용치를 떨어뜨린다)
POLICY_KINDS = frozenset({"image_report"})
#: 정책이 내야 하는 것 — 문법 · 판 이름에 묶지 않고 이 셋만 부른다
POLICY_API = ("deterministic_reasons", "deterministic_flags", "policy_identity")
#: v7 판정 출처에 거는 정책이 내야 하는 것 — 행마다의 결정적 판단과 판정 신원에 묶인 정책 지문
POLICY_API_V7 = ("evaluate_v7", "policy_identity_v7")
#: v7 판정 신원 읽기가 영수증마다 붙이는 출처 칸 — 영수증 본문(다시 계산해 맞대는 것)에 들지 않는다
V7_RECEIPT_SOURCE_KEYS = ("_judge_identity_sha256", "_shard_sha256")
#: 집계가 영수증에서 읽는 칸 — 다시 계산한 값과 통째로 맞댄 뒤 이 칸의 꼴을 본다
V7_RECEIPT_READ_KEYS = ("deterministic_codes", "deterministic_pass", "deterministic_reasons", "flags",
                        "generation_contract_applied", "model_applicable", "row_class")
QA_KEYS = ("answer", "question", "source_ids")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _ranges(*pairs: tuple[int, int]) -> re.Pattern:
    return re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in pairs) + "]")


# 언어 — 한국어 산출에 섞이면 안 되는 문자와, 영어 문장으로 볼 연속 낱말 수.
_HAN = _ranges((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))
_KANA = _ranges((0x3040, 0x30FF))
_HANGUL = _ranges((0xAC00, 0xD7A3))
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_EN_WORD = r"[A-Za-z][A-Za-z'’\-]*"
EN_SENTENCE_WORDS = 6
_EN_SENTENCE = re.compile(rf"\b{_EN_WORD}(?:[\s,;:]+{_EN_WORD}){{{EN_SENTENCE_WORDS - 1},}}")
MIN_HANGUL_SHARE = 0.6

# 지시문 유출 — 역할 · 형식 표지와, 시스템 지시 · 작업 지시의 조각(공백을 뺀 글자 수).
_LEAK_MARKERS = ("<think>", "</think>", "<|", "|>", "```", "system:", "user:", "assistant:",
                 "시스템 프롬프트", "이전 지시", "지시를 무시")
SYSTEM_WINDOW = 12
TEMPLATE_WINDOW = 16

# 학습 질문 틀(영상 설명의 user message)에 들면 안 되는 꼴 — 숫자 · 괄호(코드 · bbox · 조항 id · JSON 골격). 릴리스와 같은 규칙.
_TEMPLATE_LEAK = re.compile(r"[0-9{}\[\]]")

# 수 — 글자에 붙은 번호(C1 · ISO5817 의 앞 글자 뒤)는 수로 보지 않는다. 1,000 꼴은 하나로 읽는다.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")


class AuditRefused(ValueError):
    """명부 · 설정 · 출력 디렉터리 계약 위반. 검사를 시작하지 않는다."""


# --------------------------------------------------------------------------- 작은 도구


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    return _sha_bytes(canonical_bytes(value))


def ids_sha256(ids: Sequence[str]) -> str:
    """id 수열의 지문 — 각 id 를 LF 로 이어 붙인 UTF-8 바이트(끝 LF 없음)의 sha256. 릴리스와 같은 산식."""
    return _sha_bytes("\n".join(ids).encode("utf-8"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _squash(text: str) -> str:
    """중복 판정용 — NFC, 소문자, 공백 · 문장부호 제거."""
    t = unicodedata.normalize("NFC", text).casefold()
    return "".join(ch for ch in t if not (ch.isspace() or unicodedata.category(ch).startswith("P")))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield json.dumps(value)


def numbers_in(text: str) -> list[Decimal]:
    """본문의 수. 4 · 4.0 · 4.00 은 같고 4 와 40 은 다르다(옛 `rstrip('0.')` 사고를 되풀이하지 않는다)."""
    out = []
    for m in _NUMBER.finditer(text):
        whole = m.group(1).replace(",", "")
        frac = m.group(2)
        try:
            out.append(Decimal(f"{whole}.{frac}" if frac else whole))
        except InvalidOperation:  # pragma: no cover — 정규식이 숫자만 잡는다
            continue
    return out


#: 근거(`evidence`) 가운데 수치 추적의 원천으로 읽는 칸 — 구절 본문뿐이다. 쪽 번호 · 문서 지문 · 출처 칸의 수는 원천이 아니다.
PASSAGE_EVIDENCE_KEYS = frozenset({"passage", "passages", "text", "required_context", "context_passages"})


def _passage_strings(evidence: Any) -> Iterable[str]:
    if isinstance(evidence, Mapping):
        for k, v in evidence.items():
            if k in PASSAGE_EVIDENCE_KEYS:
                yield from (x for x in _strings(v) if x != k)
            elif isinstance(v, Mapping):
                yield from _passage_strings(v)


def allowed_numbers(task: Mapping[str, Any]) -> set[Decimal]:
    """추적 원천: `locked_target` 의 값, 근거의 구절 본문 칸(`PASSAGE_EVIDENCE_KEYS`)의 수, 잠긴 결함 목록의 길이(결함 수),
    영상 행의 구역별 소계(`_zone_subtotals`).

    다른 목록의 길이(bbox 의 4 등) · 쪽 번호 · 지문(64자리 16진)은 원천이 아니다 — 아무 수나 통과시키지 않게.
    """
    allowed: set[Decimal] = set()
    target = task.get("locked_target")
    for s in [*_strings(target), *_passage_strings(task.get("evidence"))]:
        if not _HEX64.fullmatch(s):
            allowed.update(numbers_in(s))
    if isinstance(target, Mapping) and isinstance(target.get("defects"), list):
        allowed.add(Decimal(len(target["defects"])))
    allowed.update(_zone_subtotals(task))
    return allowed


def _zone_subtotals(task: Mapping[str, Any]) -> set[Decimal]:
    """결함 행의 (이름, 구역)별 수 — 판정 정책이 결정적 검사에 쓰는 잠긴 칸 투영(`grouped_locked_fields`: 근거의 이름별 수 ·
    위치가 잠긴 결함과 같은지 대조한다)에서 그대로 읽는다. '위 왼쪽에 1개, 아래 오른쪽에 4개' 의 1 · 4 가 여기서 온다.

    구역 묶음의 합 · 여러 이름을 합친 구역 수 · `0..N` 은 넣지 않는다. 대조에 실패한 행(근거와 잠긴 결함이 어긋남)은
    소계를 하나도 내지 않는다 — 원천을 넓히지 않는 쪽으로 닫는다. 정상 행 · 원문 QA 는 대상이 아니다.
    """
    target = task.get("locked_target")
    if task.get("kind") != "image_report" or not isinstance(target, Mapping) or not target.get("defects"):
        return set()
    from .judge_policy import grouped_locked_fields
    try:
        zones = grouped_locked_fields(task)["defects_by_zone"]
    except (KeyError, TypeError, ValueError):
        return set()
    return {Decimal(k) for by_name in zones.values() for k in by_name.values()}


def language_problems(text: str) -> list[str]:
    reasons = []
    if _HAN.search(text) or _KANA.search(text):
        reasons.append("language_mixed_cjk")
    if _EN_SENTENCE.search(text):
        reasons.append("language_mixed_english_sentence")
    hangul = len(_HANGUL.findall(text))
    letters = hangul + len(_LATIN_LETTER.findall(text)) + len(_HAN.findall(text))
    if letters == 0 or hangul / letters < MIN_HANGUL_SHARE:
        reasons.append("language_not_korean")
    return reasons


def _windows(text: str, size: int) -> set[str]:
    t = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    return {t[i:i + size] for i in range(len(t) - size + 1)}


_SYSTEM_WINDOWS = _windows(GENERATE_SYSTEM, SYSTEM_WINDOW)


def instruction_template(task: Mapping[str, Any]) -> str:
    """작업 지시에서 근거(구절 · 주석) 문자열을 뺀 나머지 — 지시문 틀."""
    prompt = task.get("prompt") or ""
    pieces = sorted({s for s in _strings(task.get("evidence")) if len(s) >= 8}
                    | {s for s in _strings(task.get("locked_target")) if len(s) >= 8}
                    | {canonical_bytes(task.get("locked_target")).decode("utf-8")}, key=len, reverse=True)
    for p in pieces:
        prompt = prompt.replace(p, " ")
    return prompt


def leak_problems(text: str, task: Mapping[str, Any]) -> list[str]:
    reasons = []
    low = text.casefold()
    if any(m.casefold() in low for m in _LEAK_MARKERS):
        reasons.append("instruction_leak_marker")
    if _windows(text, SYSTEM_WINDOW) & _SYSTEM_WINDOWS:
        reasons.append("instruction_leak_system")
    if _windows(text, TEMPLATE_WINDOW) & _windows(instruction_template(task), TEMPLATE_WINDOW):
        reasons.append("instruction_leak_prompt")
    return reasons


def _parse_object(text: str) -> tuple[dict | None, bool]:
    """생성 QA 의 JSON 객체 하나. 그 객체만 담은 ```json 울타리 하나는 받되 표시한다. 그 밖은 None."""
    s = text.strip()
    fenced = False
    m = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL)
    if m:
        s, fenced = m.group(1), True
    try:
        value = parse_json(s)
    except ValueError:
        return None, fenced
    return (value if isinstance(value, dict) else None), fenced


def defect_codes(locked_target: Any) -> list[str]:
    """커버리지용 결함 코드. `defects` 목록 항목의 iso_code · code · type 가운데 처음 있는 것."""
    if not isinstance(locked_target, Mapping):
        return []
    out = []
    for d in locked_target.get("defects") or []:
        if isinstance(d, Mapping):
            code = next((d[k] for k in ("iso_code", "code", "type") if d.get(k) not in (None, "")), None)
            if code is not None:
                out.append(str(code))
    return out


def _label_names() -> list[str]:
    from data.label_map import load_label_map

    return sorted({dt.name_ko for dt in load_label_map().defect_types.values()})


# --------------------------------------------------------------------------- 입력


def _quality_config(config: Mapping[str, Any]) -> dict:
    q = config.get("quality")
    if not isinstance(q, Mapping):
        raise AuditRefused("config.quality 가 없다")
    need_int = ("minimum_source_qa_train", "target_source_qa_train", "minimum_distinct_train_passages",
                "maximum_questions_per_passage", "minimum_review_per_stratum")
    for k in need_int:
        if type(q.get(k)) is not int or q[k] < 0:
            raise AuditRefused(f"config.quality.{k} 는 0 이상의 정수여야 한다")
    if q["maximum_questions_per_passage"] < 1 or q["minimum_review_per_stratum"] < 1:
        raise AuditRefused("구절당 상한 · 층별 최소 검수 수는 1 이상이어야 한다")
    frac = q.get("review_fraction")
    if not isinstance(frac, (int, float)) or isinstance(frac, bool) or not 0 < frac <= 1:
        raise AuditRefused("config.quality.review_fraction 은 (0, 1] 이어야 한다")
    if q.get("human_review_required") is not True or q.get("no_partial_release") is not True:
        raise AuditRefused("human_review_required · no_partial_release 는 true 여야 한다")
    return dict(q)


def _read_tasks(run_dir: Path) -> tuple[dict, list[dict], str]:
    run = parse_json((run_dir / "RUN.json").read_bytes())
    raw = (run_dir / "tasks.jsonl").read_bytes()
    sha = _sha_bytes(raw)
    if not isinstance(run, dict) or run.get("tasks_sha256") != sha:
        raise AuditRefused("tasks.jsonl 이 RUN.json 의 tasks_sha256 과 다르다")
    tasks = [parse_json(line) for line in raw.splitlines()]
    return run, tasks, sha


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
            raise AuditRefused(f"출력 디렉터리는 새것이거나 비어 있어야 한다: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _write_new_bytes(path: Path, raw: bytes) -> str:
    """덮어쓰지 않는 원자 발행(common.write_json_new 와 같은 방식). sha256 을 돌려준다."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()
    return _sha_bytes(raw)


def _jsonl(rows: Iterable[Any]) -> bytes:
    return b"".join(canonical_bytes(r) + b"\n" for r in rows)


def _run_dir_ref(run_dir: Path, output_dir: Path) -> str:
    """AUDIT 의 `run_dir` — 출력 디렉터리 기준 상대 경로(POSIX). 드라이브가 달라 상대로 못 쓰면 절대 경로."""
    try:
        return Path(os.path.relpath(run_dir.resolve(), output_dir.resolve())).as_posix()
    except ValueError:
        return run_dir.resolve().as_posix()


# --------------------------------------------------------------------------- 행 판정


def _task_contract(task: Mapping[str, Any], label_names: Sequence[str]) -> list[str]:
    """명부 행이 계약을 지키는가. infer 가 이미 보는 것 위에 이미지 지문 · 학습 질문 틀 · 원문 문서 지문을 본다."""
    bad = []
    if task.get("kind") == "image_report":
        if not _HEX64.fullmatch(str(task.get("image_sha256", ""))):
            bad.append("task_image_sha256_invalid")
        text = task.get("train_user_text")
        if not isinstance(text, str) or not text.strip():
            bad.append("task_train_user_text_missing")
        elif (text == task.get("prompt") or _TEMPLATE_LEAK.search(text)
              or any(n in text for n in label_names)):
            bad.append("task_train_user_text_leaks_target")
    else:
        ev = task.get("evidence") or {}
        if not (isinstance(ev.get("source_id"), str) and ev["source_id"]
                and _HEX64.fullmatch(str(ev.get("source_sha256", "")))):
            bad.append("task_source_document_missing")
    return bad


def _judge(row: Mapping[str, Any] | None) -> tuple[list[str], list[str], dict | None]:
    """(오류 사유, 기각 사유, 판정 객체). 릴리스와 같이 행 글 전체를 JSON 하나로 읽는다 — 울타리는 받지 않는다."""
    if row is None:
        return [], [], None
    if row.get("finish_reason") != "eos":
        return ["judge_truncated"], [], None
    try:
        verdict = parse_json(row.get("text", ""))
    except ValueError:
        return ["judge_unparsable"], [], None
    if not isinstance(verdict, dict) or tuple(sorted(verdict)) != JUDGE_KEYS:
        return ["judge_unparsable"], [], None
    if not all(isinstance(verdict[k], bool) for k in ("grounded", "scope_ok", "language_ok")):
        return ["judge_unparsable"], [], None
    if not isinstance(verdict["reason"], str) or not verdict["reason"].strip():
        return ["judge_reason_missing"], [], None
    rejected = [f"judge_{k}_false" for k in ("grounded", "scope_ok", "language_ok") if verdict[k] is not True]
    return [], rejected, verdict


# 구조화 판정(v5 · v6 · v7)의 판독. 판정 정책이 각 판의 파서와 그것이 닿는 함수 · `JUDGE_V5_*` / `JUDGE_V6_*` /
# `JUDGE_V7_*` 상수 · `common.parse_json` 의 폐쇄를 판정 신원에 묶는다 — v6 의 폐쇄에는 v5 파서가 들어가지 않는다
# (지문기가 거부한다). v7 의 폐쇄에도 v5 · v6 의 파서 · 울타리 함수 · 상수가 들어가지 않게 따로 둔다.
# 이 경계 안에서는 다른 모듈을 불러오지 않고(정규식도 쓰지 않는다), 파일 · 시각 · 난수를 읽지 않는다
# (내장 함수 · 글 메서드와 `parse_json` 만 쓴다). 상수는 지문기가 읽을 수 있게 글자 그대로의 값이어야 한다.


def _judge_row_class(task: Any) -> str | None:
    """구조화 판정의 행 종류 — 신뢰된 작업 행(명부)에서만 고른다. 판정 출력은 보지 않는다. 고를 수 없으면 None.

    원문 QA → `source_qa`, 영상 설명 → 잠긴 결함 목록이 비었으면 `defect_free`, 있으면 `defect`. 판정 지시의 렌더러가
    물음 · 기대 키를 고르는 규칙과 같다(시험이 대조한다).
    """
    if not isinstance(task, dict):
        return None
    if task.get("kind") == "source_qa":
        return "source_qa"
    target = task.get("locked_target")
    if task.get("kind") != "image_report" or not isinstance(target, dict) or not isinstance(target.get("defects"), list):
        return None
    return "defect" if target["defects"] else "defect_free"


def _unfence_judge_text(text: str) -> str | None:
    """출력 전체가 코드 울타리 하나로만 감싸졌으면 그 안의 글, 아니면 None.

    앞뒤 공백을 뗀 글이 여는 줄(```` ``` ```` 또는 ```` ```json ```` — `json` 은 대소문자 무관, 같은 줄에 다른 글 없음)로
    시작하고, 닫는 줄(```` ``` ```` 만, 제 줄에)로 끝나야 한다. 울타리 앞 · 뒤에 다른 글이 있거나 안에 ```` ``` ```` 가
    또 있으면(중첩 · 둘째 울타리) 벗기지 않는다 — 그 글은 원래대로 JSON 으로 읽혀 못 읽음이 된다.
    """
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return None
    head, newline, rest = stripped[3:].partition("\n")
    if not newline or head.strip().lower() not in ("", "json") or not rest.endswith("```"):
        return None
    body = rest[:-3]
    if "```" in body or not body.rstrip(" \t").endswith("\n"):
        return None
    return body


def parse_judge_v5(task: Any, row: Any) -> dict:
    """구조화 판정 한 행의 엄격 판독 — `{schema, parser, row_class, errors, rejected, verdict, flags, answers,
    format_normalized}`.

    키 집합은 작업 행으로만 고른다(`JUDGE_V5_KEYS[_judge_row_class(task)]`). 행 글 전체가 JSON 객체 하나여야 하고
    (덧붙인 글 · 잘린 JSON · 중복 키 · NaN 은 못 읽음), 키는 정확히 그 집합, `reason` 밖의 값은 참 · 거짓
    (1 · 0 · 문장은 안 된다), `reason` 은 비지 않은 글이어야 한다. 어기면 `errors` 에 사유 하나가 남고 `verdict` ·
    `answers` 는 None 이다 — 옛 네 키로 다시 읽지 않는다.

    뜻을 바꾸지 않는 형식 정규화 둘만 한다(`parse_judge_v5_1`). 적용한 것은 `format_normalized` 에 남는다.
    ① 출력 전체가 코드 울타리 하나로 감싸졌으면 벗긴다(`_unfence_judge_text`). ② 키 집합이 맞은 뒤, `reason` 밖의
    값이 "true" · "false" 글(대소문자 무관, 그 밖의 글자 없이)이면 참 · 거짓으로 읽는다. 그 밖의 글 · 수 · null 은
    그대로 `judge_value_type` 이다. `answers` 는 정규화한 뒤의 답이고, 판정 원문은 판정 샤드에 그대로 있다.

    세 판정값(`verdict`, 옛 판정과 같은 네 키) — 정상: grounded = states_no_marked_defect ∧ ¬claims_defect,
    scope_ok = ¬asserts_verdict. 결함: grounded = fields_match, scope_ok = has_limitation ∧
    ¬asserts_verdict_dimension_or_cause. 원문 QA: 네 키 그대로. language_ok · reason 은 답 그대로.
    `rejected` 는 거짓인 판정값(`judge_<값>_false`), `flags` 는 관찰 표지로 기각하지 않는다 — 정상 행의 ¬has_limitation 은
    `n5_absent`.
    """
    cls = _judge_row_class(task)
    normalized: list[str] = []
    out = {"schema": JUDGE_V5_SCHEMA_VERSION, "parser": JUDGE_V5_PARSER_VERSION, "row_class": cls, "errors": [],
           "rejected": [], "verdict": None, "flags": [], "answers": None, "format_normalized": normalized}
    if cls is None:
        out["errors"] = ["judge_task_unsupported"]
        return out
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        out["errors"] = ["judge_row_invalid"]
        return out
    if row.get("finish_reason") != "eos":
        out["errors"] = ["judge_truncated"]
        return out
    text = row["text"]
    unfenced = _unfence_judge_text(text)
    if unfenced is not None:
        text = unfenced
        normalized.append(JUDGE_V5_FORMAT_NORMALIZATIONS[0])
    try:
        answers = parse_json(text)
    except (ValueError, RecursionError):
        out["errors"] = ["judge_unparsable"]
        return out
    if not isinstance(answers, dict):
        out["errors"] = ["judge_not_object"]
        return out
    keys = JUDGE_V5_KEYS[cls]
    if tuple(sorted(answers)) != keys:
        out["errors"] = ["judge_keys"]
        return out
    worded = [k for k in keys if k != "reason" and type(answers[k]) is str and answers[k].isascii()
              and answers[k].lower() in JUDGE_V5_BOOL_STRINGS]
    if worded:
        answers = {k: (JUDGE_V5_BOOL_STRINGS[v.lower()] if k in worded else v) for k, v in answers.items()}
        normalized.append(JUDGE_V5_FORMAT_NORMALIZATIONS[1])
    if not all(type(answers[k]) is bool for k in keys if k != "reason"):
        out["errors"] = ["judge_value_type"]
        return out
    if not isinstance(answers["reason"], str) or not answers["reason"].strip():
        out["errors"] = ["judge_reason_missing"]
        return out
    if cls == "defect_free":
        verdict = {"grounded": answers["states_no_marked_defect"] and not answers["claims_defect"],
                   "scope_ok": not answers["asserts_verdict"]}
        flags = [] if answers["has_limitation"] else ["n5_absent"]
    elif cls == "defect":
        verdict = {"grounded": answers["fields_match"],
                   "scope_ok": answers["has_limitation"] and not answers["asserts_verdict_dimension_or_cause"]}
        flags = []
    else:
        verdict = {"grounded": answers["grounded"], "scope_ok": answers["scope_ok"]}
        flags = []
    verdict["language_ok"] = answers["language_ok"]
    verdict["reason"] = answers["reason"]
    out["verdict"], out["flags"], out["answers"] = verdict, flags, answers
    out["rejected"] = [f"judge_{k}_false" for k in ("grounded", "scope_ok", "language_ok") if verdict[k] is not True]
    return out


def _unfence_judge_text_v6(text: str) -> str | None:
    """v6 — 출력 전체가 소문자 ```` ```json ```` 울타리 하나로만 감싸졌으면 그 안의 글, 아니면 None.

    JSON 의 공백(공백 · 탭 · 줄바꿈 · 복귀)만 앞뒤로 뗀 글이 ```` ```json ```` 과 줄바꿈으로 시작하고(여는 줄에 다른 글
    없음), 줄바꿈과 ```` ``` ```` 로 끝나야 한다(닫는 울타리는 제 줄에). 표시 없는 울타리 · `JSON` · 여는 줄 끝의 공백 ·
    울타리 앞뒤의 글 · 안의 ```` ``` ````(중첩 · 둘째 울타리) · `~~~` 는 벗기지 않는다 — 그 글은 원래대로 JSON 으로
    읽혀 못 읽음이 된다.
    """
    stripped = text.strip(" \t\n\r")
    if not stripped.startswith(JUDGE_V6_FENCE_OPEN) or not stripped.endswith(JUDGE_V6_FENCE_CLOSE):
        return None
    body = stripped[len(JUDGE_V6_FENCE_OPEN):]
    if not body.endswith(JUDGE_V6_FENCE_CLOSE):         # 여는 줄과 닫는 줄이 줄바꿈 하나를 나눠 쓴 꼴(```json\n```)
        return None
    body = body[:-len(JUDGE_V6_FENCE_CLOSE)]
    if "```" in body:
        return None
    return body


def parse_judge_v6(task: Any, row: Any) -> dict:
    """구조화 판정 v6 한 행의 엄격 판독 — `{schema, parser, row_class, errors, rejected, verdict, flags, answers,
    format_normalized}`. v5 파서를 부르지 않는다.

    키 집합은 작업 행으로만 고른다(`JUDGE_V6_KEYS[_judge_row_class(task)]`) — 결함 행은 일곱 키(`names_match` ·
    `counts_match` · `positions_match` · `has_limitation` · `asserts_verdict_dimension_or_cause` · `language_ok` ·
    `reason`), 정상 행 · 원문 QA 는 v5 와 같다. 행 글 전체가 JSON 객체 하나여야 하고(덧붙인 글 · 잘린 JSON · 중복 키 ·
    NaN 은 못 읽음), 키는 정확히 그 집합, `reason` 밖의 값은 참 · 거짓, `reason` 은 비지 않은 글이어야 한다. 어기면
    `errors` 에 사유 하나가 남고 `verdict` · `answers` 는 None 이다. 사유와 그 순서는 v5 와 같다.

    형식 정규화는 표기만 바꾸는 좁은 둘뿐이다. 적용한 것은 이 순서로 `format_normalized` 에 남는다(오류 행에도).
    ① `code_fence`: 출력 전체가 소문자 ```` ```json ```` 울타리 하나로 감싸졌으면 벗긴다(`_unfence_judge_text_v6`).
    ② `string_boolean`: 키 집합이 맞은 뒤, `reason` 밖의 값이 정확히 "true" · "false" 글이면 참 · 거짓으로 읽는다.
    "True" · " true" · "yes" · 물음 문장 · 수 · null 은 그대로 `judge_value_type` 이다 — 뜻을 짐작하지 않는다.
    `answers` 는 정규화한 뒤의 답이고, 판정 원문은 판정 샤드에 그대로 있다.

    세 판정값(`verdict`, 옛 판정과 같은 네 키) — 결함: grounded = names_match ∧ counts_match ∧ positions_match,
    scope_ok = has_limitation ∧ ¬asserts_verdict_dimension_or_cause. 정상: grounded = states_no_marked_defect ∧
    ¬claims_defect, scope_ok = ¬asserts_verdict. 원문 QA: 네 키 그대로. language_ok · reason 은 답 그대로.
    `rejected` 는 거짓인 판정값(`judge_<값>_false`), `flags` 는 관찰 표지로 기각하지 않는다 — 정상 행의 ¬has_limitation 은
    `n5_absent`.
    """
    cls = _judge_row_class(task)
    normalized: list[str] = []
    out = {"schema": JUDGE_V6_SCHEMA_VERSION, "parser": JUDGE_V6_PARSER_VERSION, "row_class": cls, "errors": [],
           "rejected": [], "verdict": None, "flags": [], "answers": None, "format_normalized": normalized}
    if cls is None:
        out["errors"] = ["judge_task_unsupported"]
        return out
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        out["errors"] = ["judge_row_invalid"]
        return out
    if row.get("finish_reason") != "eos":
        out["errors"] = ["judge_truncated"]
        return out
    text = row["text"]
    unfenced = _unfence_judge_text_v6(text)
    if unfenced is not None:
        text = unfenced
        normalized.append(JUDGE_V6_FORMAT_NORMALIZATIONS[0])
    try:
        answers = parse_json(text)
    except (ValueError, RecursionError):
        out["errors"] = ["judge_unparsable"]
        return out
    if not isinstance(answers, dict):
        out["errors"] = ["judge_not_object"]
        return out
    keys = JUDGE_V6_KEYS[cls]
    if tuple(sorted(answers)) != keys:
        out["errors"] = ["judge_keys"]
        return out
    worded = [k for k in keys if k != "reason" and type(answers[k]) is str and answers[k] in JUDGE_V6_BOOL_STRINGS]
    if worded:
        answers = {k: (JUDGE_V6_BOOL_STRINGS[v] if k in worded else v) for k, v in answers.items()}
        normalized.append(JUDGE_V6_FORMAT_NORMALIZATIONS[1])
    if not all(type(answers[k]) is bool for k in keys if k != "reason"):
        out["errors"] = ["judge_value_type"]
        return out
    if not isinstance(answers["reason"], str) or not answers["reason"].strip():
        out["errors"] = ["judge_reason_missing"]
        return out
    if cls == "defect_free":
        verdict = {"grounded": answers["states_no_marked_defect"] and not answers["claims_defect"],
                   "scope_ok": not answers["asserts_verdict"]}
        flags = [] if answers["has_limitation"] else ["n5_absent"]
    elif cls == "defect":
        verdict = {"grounded": answers["names_match"] and answers["counts_match"] and answers["positions_match"],
                   "scope_ok": answers["has_limitation"] and not answers["asserts_verdict_dimension_or_cause"]}
        flags = []
    else:
        verdict = {"grounded": answers["grounded"], "scope_ok": answers["scope_ok"]}
        flags = []
    verdict["language_ok"] = answers["language_ok"]
    verdict["reason"] = answers["reason"]
    out["verdict"], out["flags"], out["answers"] = verdict, flags, answers
    out["rejected"] = [f"judge_{k}_false" for k in ("grounded", "scope_ok", "language_ok") if verdict[k] is not True]
    return out


def _unfence_judge_text_v7(text: str) -> str | None:
    """v7 — `_unfence_judge_text_v6` 와 같은 규칙을 v7 상수로 따로 둔다(v7 의 폐쇄가 v6 의 것에 닿지 않게).

    JSON 의 공백(공백 · 탭 · 줄바꿈 · 복귀)만 앞뒤로 뗀 글이 ```` ```json ```` 과 줄바꿈으로 시작하고, 줄바꿈과
    ```` ``` ```` 로 끝나야 한다. 그 밖(표시 없는 울타리 · `JSON` · 울타리 앞뒤의 글 · 중첩 · 닫는 울타리 없음)은 None.
    """
    stripped = text.strip(" \t\n\r")
    if not stripped.startswith(JUDGE_V7_FENCE_OPEN) or not stripped.endswith(JUDGE_V7_FENCE_CLOSE):
        return None
    body = stripped[len(JUDGE_V7_FENCE_OPEN):]
    if not body.endswith(JUDGE_V7_FENCE_CLOSE):         # 여는 줄과 닫는 줄이 줄바꿈 하나를 나눠 쓴 꼴(```json\n```)
        return None
    body = body[:-len(JUDGE_V7_FENCE_CLOSE)]
    if "```" in body:
        return None
    return body


def parse_judge_v7(task: Any, row: Any) -> dict:
    """구조화 판정 v7 한 행의 엄격 판독 — `{schema, parser, row_class, errors, rejected, verdict, flags, answers,
    format_normalized}`(v6 와 같은 꼴). v5 · v6 파서를 부르지 않는다.

    판정 모델은 결함 행과 원문 QA 에만 부른다. 키 집합은 작업 행으로만 고른다(`JUDGE_V7_KEYS[_judge_row_class(task)]`).
    - 결함 행: 다섯 키 `has_limitation` · `language_ok` · `no_unannotated_attributes` · `no_unsupported_relation` ·
      `reason`. 물음 넷은 모두 "참 = 문제 없음" 이다.
    - 정상 행: 판정 모델을 부르지 않는 행이다 — 어떤 행이 와도 읽지 않고 `judge_not_applicable` 오류다.
    - 원문 QA: 옛 네 키 그대로(`grounded` · `scope_ok` · `language_ok` · `reason`).

    생성의 종료는 `eos` 나 `json_closed`(첫 JSON 객체가 닫혀 정지 조건이 멈춤)여야 한다 — 그 밖은 `judge_truncated`.
    행 글 전체가 JSON 객체 하나여야 하고(덧붙인 글 · 잘린 JSON · 중복 키 · NaN 은 못 읽음), 키는 정확히 그 집합(빠진 키 ·
    여분 키는 `judge_keys`), `reason` 밖의 값은 참 · 거짓, `reason` 은 비지 않은 글이어야 한다. 어기면 `errors` 에 사유
    하나가 남고 `verdict` · `answers` 는 None 이다. 사유와 그 순서는 v6 와 같고 `judge_not_applicable` 이 더해졌다.

    형식 정규화는 v6 의 좁은 둘 그대로다(`code_fence` · `string_boolean`, 적용한 것만 이 순서로, 오류 행에도 남는다).
    닫는 울타리가 없는 글(정지 조건이 첫 객체에서 멈춘 ```` ```json ```` 출력)은 벗기지 않는다 — 못 읽음이다.

    판정값(`verdict`) — 결함 행은 물음 넷 그대로 + `reason`, 원문 QA 는 옛 네 키 그대로. `rejected` 는 거짓인 물음
    (`judge_<키>_false`)이다. v7 의 결함 행에는 grounded · scope_ok 가 없다 — grounded 는 결정적 정책이 가르고 판정
    모델에 묻지 않는다. `flags` 는 관찰 표지로 기각하지 않는다 — `reason` 이 `JUDGE_V7_REASON_MAX_CHARS` 자를 넘으면
    `reason_long`.
    """
    cls = _judge_row_class(task)
    normalized: list[str] = []
    out = {"schema": JUDGE_V7_SCHEMA_VERSION, "parser": JUDGE_V7_PARSER_VERSION, "row_class": cls, "errors": [],
           "rejected": [], "verdict": None, "flags": [], "answers": None, "format_normalized": normalized}
    if cls is None:
        out["errors"] = ["judge_task_unsupported"]
        return out
    if cls not in JUDGE_V7_KEYS:
        out["errors"] = ["judge_not_applicable"]
        return out
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        out["errors"] = ["judge_row_invalid"]
        return out
    if row.get("finish_reason") not in JUDGE_V7_FINISH_REASONS:
        out["errors"] = ["judge_truncated"]
        return out
    text = row["text"]
    unfenced = _unfence_judge_text_v7(text)
    if unfenced is not None:
        text = unfenced
        normalized.append(JUDGE_V7_FORMAT_NORMALIZATIONS[0])
    try:
        answers = parse_json(text)
    except (ValueError, RecursionError):
        out["errors"] = ["judge_unparsable"]
        return out
    if not isinstance(answers, dict):
        out["errors"] = ["judge_not_object"]
        return out
    keys = JUDGE_V7_KEYS[cls]
    if tuple(sorted(answers)) != keys:
        out["errors"] = ["judge_keys"]
        return out
    worded = [k for k in keys if k != "reason" and type(answers[k]) is str and answers[k] in JUDGE_V7_BOOL_STRINGS]
    if worded:
        answers = {k: (JUDGE_V7_BOOL_STRINGS[v] if k in worded else v) for k, v in answers.items()}
        normalized.append(JUDGE_V7_FORMAT_NORMALIZATIONS[1])
    if not all(type(answers[k]) is bool for k in keys if k != "reason"):
        out["errors"] = ["judge_value_type"]
        return out
    if not isinstance(answers["reason"], str) or not answers["reason"].strip():
        out["errors"] = ["judge_reason_missing"]
        return out
    if cls == "defect":
        verdict = {k: answers[k] for k in keys if k != "reason"}
    else:
        verdict = {"grounded": answers["grounded"], "scope_ok": answers["scope_ok"], "language_ok": answers["language_ok"]}
    verdict["reason"] = answers["reason"]
    out["verdict"], out["answers"] = verdict, answers
    out["flags"] = ["reason_long"] if len(answers["reason"]) > JUDGE_V7_REASON_MAX_CHARS else []
    out["rejected"] = [f"judge_{k}_false" for k in verdict if k != "reason" and verdict[k] is not True]
    return out


def _read_judge(task: Mapping[str, Any], row: Mapping[str, Any] | None,
                schema: str) -> tuple[list[str], list[str], dict | None, list[str], dict | None, list[str]]:
    """(오류 사유, 기각 사유, 세 판정값, 모델 표지, 구조화 답, 형식 정규화). 판정 출처가 정한 꼴 하나로만 읽는다 —
    v6 출력을 v5 로, v5 출력을 v6 로 다시 읽지 않는다. 꼴 → 파서는 하나씩 짝짓는다 — 짝이 없는 꼴(판 표에 새로 든 꼴
    포함)은 다른 판의 파서로 읽지 않고 집계를 거부한다."""
    if row is None:
        return [], [], None, [], None, []
    if schema == JUDGE_LEGACY_SCHEMA:
        errors, rejected, verdict = _judge(row)
        return errors, rejected, verdict, [], None, []
    if schema == JUDGE_V7_SCHEMA_VERSION:
        parsed = parse_judge_v7(task, row)
    elif schema == JUDGE_V6_SCHEMA_VERSION:
        parsed = parse_judge_v6(task, row)
    elif schema == JUDGE_V5_SCHEMA_VERSION:
        parsed = parse_judge_v5(task, row)
    else:
        raise AuditRefused(f"judge_schema_unsupported: 판정 출력의 꼴 {schema!r} 를 읽는 파서가 짝지어져 있지 않다")
    return (parsed["errors"], parsed["rejected"], parsed["verdict"], parsed["flags"], parsed["answers"],
            parsed["format_normalized"])


def _resolve_policy(policy: Any) -> tuple[Any, dict]:
    """결정적 정책과 그 지문. `None` 이면 지금의 판정 정책 모듈(`judge_policy`).

    문법 · 판 이름은 보지 않는다 — 정책이 `POLICY_API` 셋을 내는지와 지문의 꼴만 본다. 지문은 정규 JSON 으로 한 번
    오가게 해서, 판정 신원의 근거(JSON 파일에서 읽은 값)와 같은 꼴로 비교한다.
    """
    if policy is None:
        from . import judge_policy as policy
    missing = [name for name in POLICY_API if not callable(getattr(policy, name, None))]
    if missing:
        raise AuditRefused(f"policy_interface: 정책에 {missing} 가 없다")
    try:
        identity = policy.policy_identity()
        identity = json.loads(canonical_bytes(identity)) if isinstance(identity, Mapping) else None
    except Exception as exc:  # 지문을 못 내는 정책으로는 집계하지 않는다
        raise AuditRefused(f"policy_identity: 정책 지문을 낼 수 없다 — {type(exc).__name__}: {exc}") from exc
    if not identity:
        raise AuditRefused("policy_identity: 정책 지문은 비지 않은 객체여야 한다")
    return policy, identity


def _declares_v7(judge_source: Any) -> bool:
    """주어진 판정 출처가 v7 이라고 적었는가 — 맨 위 칸이든 신원 근거든 한 곳이라도. 둘이 엇갈리면 `_judge_input` 이 거부한다."""
    if not isinstance(judge_source, Mapping):
        return False
    basis = judge_source.get("identity_basis")
    return JUDGE_V7_SCHEMA_VERSION in (judge_source.get("output_schema"),
                                       basis.get("output_schema") if isinstance(basis, Mapping) else None)


def _resolve_policy_v7(policy: Any) -> tuple[Any, dict]:
    """v7 판정 출처에 거는 정책과 그 지문(`policy_identity_v7`). `None` 이면 지금의 판정 정책 모듈. 지문은 판정 신원의
    근거에 묶인 것과 같아야 하고(`_judge_input`), 행마다의 결정적 판단은 영수증을 다시 계산하는 데만 쓴다(`_v7_receipts`)."""
    if policy is None:
        from . import judge_policy as policy
    missing = [name for name in POLICY_API_V7 if not callable(getattr(policy, name, None))]
    if missing:
        raise AuditRefused(f"policy_interface: v7 판정 출처에 거는 정책에 {missing} 가 없다")
    try:
        identity = policy.policy_identity_v7()
        identity = json.loads(canonical_bytes(identity)) if isinstance(identity, Mapping) else None
    except Exception as exc:  # 지문을 못 내는 정책으로는 집계하지 않는다
        raise AuditRefused(f"policy_identity: v7 정책 지문을 낼 수 없다 — {type(exc).__name__}: {exc}") from exc
    if not identity or identity.get("output_schema") != JUDGE_V7_SCHEMA_VERSION:
        raise AuditRefused(f"policy_identity: v7 정책 지문의 출력 계약이 {JUDGE_V7_SCHEMA_VERSION!r} 가 아니다")
    return policy, identity


def _v7_receipts(tasks: Sequence[dict], gen: Mapping[str, dict], judge_rows: Sequence[dict],
                 judge_info: Mapping[str, Any], policy: Any, policy_id: Mapping[str, Any]) -> dict[str, dict]:
    """v7 판정 신원의 결정적 영수증 — 작업 id → 영수증. 정상 행은 판정 모델을 부르지 않으므로 이것이 그 행의 유일한 판단이다.

    판정 신원 읽기는 판정을 마친 샤드(명부의 앞부분)의 영수증만 낸다. 영수증을 그대로 믿지 않는다 — 행마다 생성문 지문 ·
    생성 종료 · 정책 지문 · 판정 신원을 이 집계가 읽은 값과 맞대고, 묶인 정책으로 다시 계산한 판단과 통째로 같아야 한다
    (생성 계약을 건 운영 판단만 — `generation_contract_applied` 가 참). 모델 대상은 명부의 행 종류(`_judge_row_class`)가
    정상이 아닌 행이고, 판정 행은 모델 대상에만 있어야 한다. 어긋나면 집계하지 않고 `AuditRefused`."""
    src = judge_info["v7"]
    ident = judge_info["identity_sha256"]
    by_id = {t["id"]: t for t in tasks}
    task_ids = [t["id"] for t in tasks]
    done = src["all_task_ids"]
    if done != task_ids[:len(done)]:
        raise AuditRefused("judge_receipts: 판정을 마친 작업 id 가 작업 명부의 앞부분과 같은 수열이 아니다")
    if src["model_task_ids"] != [tid for tid in done if _judge_row_class(by_id[tid]) != "defect_free"]:
        raise AuditRefused("judge_model_task_ids: 모델 대상 id 가 판정을 마친 작업 가운데 정상이 아닌 행과 다르다")
    stray = sorted({r["id"] for r in judge_rows} - set(src["model_task_ids"]))
    if stray:
        raise AuditRefused(f"judge_row_not_applicable: 판정 모델을 부르지 않는 행(또는 판정 전 행)의 판정 행 {stray[:3]}")
    receipts = src["deterministic_receipts"]
    if [r.get("id") if isinstance(r, Mapping) else None for r in receipts] != done:
        raise AuditRefused("judge_receipts: 영수증 id 가 판정을 마친 작업 id 와 하나씩 같은 수열이 아니다")
    policy_sha = _digest(policy_id)
    out: dict[str, dict] = {}
    for r in receipts:
        tid = r["id"]
        t, g = by_id[tid], gen[tid]
        if r.get("_judge_identity_sha256") != ident or not _HEX64.fullmatch(str(r.get("_shard_sha256", ""))):
            raise AuditRefused(f"judge_receipt_identity: {tid} 의 영수증이 이 판정 신원 · 샤드에 묶이지 않았다")
        try:
            evaluated = policy.evaluate_v7(copy.deepcopy(dict(t)), g["text"], generation_contract=True)
            expected = _digest({"id": tid, "candidate_sha256": _sha_bytes(g["text"].encode("utf-8")),
                                "candidate_finish_reason": g["finish_reason"], "policy_identity_sha256": policy_sha,
                                **evaluated})
            stored = _digest({k: v for k, v in r.items() if k not in V7_RECEIPT_SOURCE_KEYS})
        except Exception as exc:  # noqa: BLE001 — 다시 계산하지 못한 영수증은 받지 않는다
            raise AuditRefused(f"judge_receipt_policy: {tid} 의 결정적 판단을 다시 계산하지 못했다 — "
                               f"{type(exc).__name__}: {exc}") from exc
        if stored != expected:
            raise AuditRefused(f"judge_receipt_mismatch: {tid} 의 영수증이 생성문 · 정책 지문 · 다시 계산한 판단과 다르다")
        reasons, flags, codes = r.get("deterministic_reasons"), r.get("flags"), r.get("deterministic_codes")
        if not (all(k in r for k in V7_RECEIPT_READ_KEYS)
                and r["model_applicable"] is (_judge_row_class(t) != "defect_free")
                and r["generation_contract_applied"] is True
                and isinstance(reasons, list) and all(isinstance(x, str) and x for x in reasons)
                and isinstance(flags, list) and all(isinstance(x, str) and x for x in flags)
                and r["deterministic_pass"] is (not reasons)
                and isinstance(codes, Mapping) and all(isinstance(v, list) for v in codes.values())):
            raise AuditRefused(f"judge_receipt_schema: {tid} 의 영수증 꼴이 v7 결정적 판단이 아니다")
        out[tid] = dict(r)
    return out


def _receipt_summary(rows: Sequence[dict], receipts: Mapping[str, dict], status: Mapping[str, str]) -> dict:
    """영수증의 수 · 사유 코드 분포 — `rows` 는 작업 행(영수증이 없는 판정 전 행 포함)."""
    got = [receipts[t["id"]] for t in rows if t["id"] in receipts]
    codes: dict[str, Counter] = defaultdict(Counter)
    for r in got:
        for group, values in r["deterministic_codes"].items():
            codes[group].update(values)
    return {"planned": len(rows), "receipts": len(got),
            "deterministic_pass": sum(r["deterministic_pass"] for r in got),
            "deterministic_reject": sum(not r["deterministic_pass"] for r in got),
            "status": dict(sorted(Counter(status[t["id"]] for t in rows).items())),
            "reasons": dict(sorted(Counter(x for r in got for x in r["deterministic_reasons"]).items())),
            "codes": {g: dict(sorted(c.items())) for g, c in sorted(codes.items())},
            "flags": dict(sorted(Counter(x for r in got for x in r["flags"]).items()))}


def _policy_check(policy: Any, task: Mapping[str, Any], text: str) -> tuple[list[str] | None, list[str] | None, str | None]:
    """(기각 사유, 관찰 표지, 오류). 정책이 예외를 내거나 꼴이 틀린 값을 내면 사유 · 표지는 None 이고 오류가 남는다 —
    그 행은 통과로 세지 않는다. 정책에는 작업 행의 사본을 준다(정책이 명부를 바꾸지 못하게)."""
    try:
        reasons = policy.deterministic_reasons(copy.deepcopy(dict(task)), text)
        flags = policy.deterministic_flags(copy.deepcopy(dict(task)), text)
    except Exception as exc:  # noqa: BLE001 — 정책이 어떤 예외를 내든 통과가 아니라 그 행의 오류다
        return None, None, f"{type(exc).__name__}: {exc}"
    for value in (reasons, flags):
        if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
            return None, None, "policy_return: 사유 · 표지는 비지 않은 문자열의 목록이어야 한다"
    return list(reasons), list(flags), None


def _row_class(task: Mapping[str, Any]) -> str:
    """집계용 행의 종류 — 잠긴 결함 목록이 비었으면 defect_free. 목록이 없으면 unknown(정책이 오류를 낸다)."""
    target = task.get("locked_target")
    if not isinstance(target, Mapping) or not isinstance(target.get("defects"), list):
        return "unknown"
    return "defect" if target["defects"] else "defect_free"


def _content(task: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[dict | None, list[str], bool]:
    """생성 행 → (본문 dict, 기각 사유, 울타리 여부). image_report 는 {'text'}, source_qa 는 {question, answer, source_ids}.

    본문은 앞뒤 공백만 떼고 그대로 둔다 — 릴리스가 생성문이 그대로 들어 있는지 샤드와 대조한다.
    """
    text = row.get("text", "")
    if row.get("finish_reason") != "eos":
        return None, ["generation_truncated"], False
    if task["kind"] == "source_qa":
        qa, fenced = _parse_object(text)
        if qa is None or tuple(sorted(qa)) != QA_KEYS:
            return None, ["qa_schema"], fenced
        if not all(isinstance(qa[k], str) and qa[k].strip() for k in ("question", "answer")):
            return None, ["qa_schema"], fenced
        ids = qa["source_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
            return None, ["qa_schema"], fenced
        return {"question": qa["question"].strip(), "answer": qa["answer"].strip(), "source_ids": ids}, [], fenced
    if not text.strip():
        return None, ["description_empty"], False
    obj, _ = _parse_object(text)
    if obj is not None:
        return None, ["description_is_json"], False
    return {"text": text.strip()}, [], False


def _content_problems(task: Mapping[str, Any], body: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if task["kind"] == "source_qa":
        parts = (body["question"], body["answer"])
        if not set(body["source_ids"]) <= set(task["source_ids"]):
            reasons.append("source_ids_outside_task")
        if len(set(body["source_ids"])) != len(body["source_ids"]):
            reasons.append("source_ids_duplicated")
    else:
        parts = (body["text"],)
    allowed = allowed_numbers(task)
    for part in parts:
        for r in language_problems(part) + leak_problems(part, task):
            if r not in reasons:
                reasons.append(r)
        if any(n not in allowed for n in numbers_in(part)) and "untraceable_number" not in reasons:
            reasons.append("untraceable_number")
    return reasons


def _provenance(gen: Mapping[str, Any], judge: Mapping[str, Any], verdict: dict, config: Mapping[str, Any],
                judge_info: Mapping[str, Any], policy: Mapping[str, Any] | None,
                structured: tuple[list[str], dict] | None = None) -> dict:
    """`policy` — 결정적 정책을 건 행(영상 설명)이면 {identity_sha256, flags}. 원문 QA 는 칸이 없다.
    `structured` — 구조화 판정이면 (모델 표지, 답, 형식 정규화). `verdict` 는 그 답에서 만든 세 판정값이고 파서 판 · 답 ·
    표지 · 형식 정규화를 옆에 싣는다. `judge` 가 None 이면 판정 모델을 부르지 않은 행(v7 정상 행)이다 — 판정 칸은
    '적용 안 됨' 두 칸뿐이다(백엔드 · 샤드 · 원문 지문 · 판정값이 없다)."""
    out = {}
    for side, row, spec in (("generate", gen, config.get("generation") or {}), ("judge", judge, judge_info["model"])):
        if row is None:
            continue
        out[side] = {"backend": row["_backend"], "family": spec.get("family"), "model_id": spec.get("model_id"),
                     "shard_sha256": row["_shard_sha256"], "text_sha256": _sha_bytes(row["text"].encode("utf-8"))}
    if judge is None:
        out["judge"] = {"applicable": False, "identity_sha256": judge_info["identity_sha256"]}
        if policy is not None:
            out["policy"] = dict(policy)
        return out
    out["judge"]["identity_sha256"] = judge_info["identity_sha256"]
    out["judge"]["verdict"] = verdict
    if structured is not None:
        out["judge"].update(schema=judge_info["parser_schema"], parser=JUDGE_PARSER_VERSIONS[judge_info["parser_schema"]],
                            flags=list(structured[0]), answers=structured[1], format_normalized=list(structured[2]))
    if policy is not None:
        out["policy"] = dict(policy)
    return out


def _judge_input(run_dir: Path, config: Mapping[str, Any], run: Mapping[str, Any], tasks: Sequence[dict],
                 gen_rows: Sequence[dict], judge_source: Mapping[str, Any] | None,
                 policy_identity: Mapping[str, Any]) -> tuple[list[dict], dict]:
    """판정 행과 판정 출처. 출처는 둘 가운데 하나다 — 섞지 않고, 한쪽이 모자라도 다른 쪽으로 채우지 않는다.

    - `judge_source=None`: 실행 설정(RUN.json)의 판정 칸으로 돈 첫 판정(`run_config`). 생성이 전량일 때만 읽는다.
    - `judge_source` 가 주어지면(`separate`): 그 행만 쓴다. 실행 디렉터리의 옛 판정 폴더는 읽지 않는다.
      `identity_sha256` = sha256(정규 JSON(`identity_basis`)), 근거는 실행 신원 · 생성 샤드 지문 목록 · 판정 모델 ·
      정책 지문(`policy_identity` — 이 집계가 거는 정책과 같아야 한다)을 담고, 행마다 `_judge_identity_sha256` 이
      그 신원이어야 한다. 생성이 전량이 아니면 미완료가 아니라 거부다(판정 신원이 생성 전량에 묶이므로).
      주어진 객체를 그대로 믿지 않는다 — `judge_dir` 를 판정 신원 읽기(`infer.read_judge_source`)로, 호출자가 처음
      준 외부 핀(`caller_expected_pins`, None 포함)을 그대로 넘겨 다시 읽고, `judge_dir` 밖의 모든 칸이 바이트 그대로
      같아야 한다(샤드 지문 · 렌더 · 실제 판정 모델 계열 · 핀 대조는 그 읽기가 본다). 판정 출력의 꼴은 묶인 정책의
      출력 계약이 정한다 — `judge_structured_v7` · `judge_structured_v6` · `judge_structured_v5` 밖이면 읽지 않는다.
      v7 이면 다시 읽은 출처가 v7 이라고 적고 모델 대상 · 결정적 영수증 칸을 내야 한다(`v7` 칸 — 대조는
      `_v7_receipts`). 출처가 적은 꼴과 묶인 출력 계약이 엇갈리면 거부한다. 어긋나면 집계하지 않고 `AuditRefused`.

    판정 출처 칸: source · identity_sha256 · identity_basis · model · parser_schema(판정 출력을 읽는 꼴) ·
    caller_expected_pins · pin_provenance · probe_approval_sha256 · gpu_model_verified(따로 둔 판정만, 판정 신원 읽기의 값).
    """
    if judge_source is None:
        rows = read_stage_rows(run_dir, "judge", dict(config)) if len(gen_rows) == len(tasks) else []
        spec = config.get("judge") or {}
        return rows, {"source": "run_config", "identity_sha256": None, "identity_basis": None,
                      "model": {"model_id": spec.get("model_id"), "family": spec.get("family")},
                      "parser_schema": JUDGE_LEGACY_SCHEMA, "caller_expected_pins": None, "pin_provenance": None,
                      "probe_approval_sha256": None, "gpu_model_verified": None, "v7": None}
    if not isinstance(judge_source, Mapping):
        raise AuditRefused("judge_source 는 객체여야 한다")
    ident, basis, rows = judge_source.get("identity_sha256"), judge_source.get("identity_basis"), judge_source.get("rows")
    if not (isinstance(ident, str) and _HEX64.fullmatch(ident)) or not isinstance(basis, Mapping) or not isinstance(rows, list):
        raise AuditRefused("judge_source: identity_sha256(64 자리) · identity_basis(객체) · rows(목록)가 필요하다")
    if _digest(basis) != ident:
        raise AuditRefused("judge_identity_basis: identity_sha256 이 identity_basis 의 정규 JSON 지문이 아니다")
    if basis.get("run_identity_sha256") != run.get("identity_sha256"):
        raise AuditRefused("judge_identity_run: 판정 신원이 이 실행 신원에 묶이지 않았다")
    if len(gen_rows) != len(tasks):
        raise AuditRefused("judge_identity_generate: 따로 둔 판정은 생성 전량 위에서만 읽는다")
    if basis.get("generate_shard_sha256") != sorted({r["_shard_sha256"] for r in gen_rows}):
        raise AuditRefused("judge_identity_generate: 판정 신원의 생성 샤드 지문 목록이 읽은 생성 샤드와 다르다")
    model = basis.get("judge_model")
    if not (isinstance(model, Mapping) and isinstance(model.get("model_id"), str) and isinstance(model.get("family"), str)):
        raise AuditRefused("judge_identity_model: 판정 신원에 judge_model{model_id, family} 가 없다")
    if _family(model["family"]) == _family((config.get("generation") or {}).get("family")):
        raise AuditRefused("judge_identity_model: 생성과 같은 계열의 판정이다")
    if basis.get("policy_identity") != policy_identity:
        raise AuditRefused("judge_policy_mismatch: 판정 신원에 묶인 정책 지문이 이 집계가 거는 정책과 다르다(또는 없다)")
    schema = policy_identity.get("output_schema")
    if schema not in JUDGE_PARSER_VERSIONS:
        raise AuditRefused(f"judge_schema_unsupported: 판정 신원에 묶인 출력 계약 {schema!r} 를 읽는 파서가 없다")
    # 출처가 적은 꼴(맨 위 · 신원 근거) — 아래의 다시 읽기가 주어진 객체와 바이트로 같음을 보므로 다시 읽은 것의 꼴과 같다
    declared = {v for v in (judge_source.get("output_schema"), basis.get("output_schema")) if isinstance(v, str)}
    if (schema == JUDGE_V7_SCHEMA_VERSION) != (JUDGE_V7_SCHEMA_VERSION in declared):
        raise AuditRefused(f"judge_schema_mixed: 판정 출처가 적은 꼴 {sorted(declared)} 와 묶인 출력 계약 {schema!r} 가 다르다")
    ids = {t["id"] for t in tasks}
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, Mapping) or r.get("_judge_identity_sha256") != ident:
            raise AuditRefused("judge_identity_mixed: 다른 판정 신원(또는 신원 없음)의 행이 섞였다")
        if r.get("id") not in ids or r["id"] in seen:
            raise AuditRefused(f"judge_rows: 작업에 없거나 겹친 판정 행 {r.get('id')!r}")
        seen.add(r["id"])
    verified = _reread_judge_source(run_dir, config, judge_source)
    keys = sorted(set(judge_source) - {"judge_dir"})
    try:
        same = keys == sorted(set(verified) - {"judge_dir"}) and _digest(
            {k: judge_source[k] for k in keys}) == _digest({k: verified[k] for k in keys})
    except (TypeError, ValueError):            # 정규 JSON 이 아닌 값 — 판정 신원 읽기가 낸 객체가 아니다
        same = False
    if not same:
        raise AuditRefused("judge_source_unverified: 주어진 판정 객체가 판정 경로를 다시 읽은 것과 다르다")
    v7 = None
    if schema == JUDGE_V7_SCHEMA_VERSION:
        # v7 — 정상 행은 판정 행 없이 결정적 영수증만 있다. 칸의 꼴만 여기서 보고 대조는 `_v7_receipts` 가 한다
        v7 = {k: verified.get(k) for k in ("all_task_ids", "model_task_ids", "deterministic_receipts")}
        if (verified.get("output_schema") != JUDGE_V7_SCHEMA_VERSION or basis.get("output_schema") != JUDGE_V7_SCHEMA_VERSION
                or verified.get("policy_version") != "v7"
                or not all(isinstance(v7[k], list) for k in v7)
                or not all(isinstance(x, str) for k in ("all_task_ids", "model_task_ids") for x in v7[k])):
            raise AuditRefused("judge_source_v7: v7 판정 출처에 policy_version · all_task_ids · model_task_ids · "
                               "deterministic_receipts 가 없거나 꼴이 다르다")
    return list(verified["rows"]), {"source": "separate", "identity_sha256": ident, "identity_basis": dict(basis),
                                    "model": {"model_id": model["model_id"], "family": model["family"]},
                                    "parser_schema": schema,
                                    "caller_expected_pins": dict(verified["caller_expected_pins"]),
                                    "pin_provenance": verified.get("pin_provenance"),
                                    "probe_approval_sha256": verified.get("probe_approval_sha256"),
                                    "gpu_model_verified": verified.get("gpu_model_verified"), "v7": v7}


def _reread_judge_source(run_dir: Path, config: Mapping[str, Any], judge_source: Mapping[str, Any]) -> dict:
    """주어진 판정 객체의 `judge_dir` 를 판정 신원 읽기로 다시 읽는다 — 경로는 신원이 아니라 찾는 자리다.

    외부 핀은 주어진 객체의 `caller_expected_pins`(판정 신원 읽기를 처음 부른 호출자의 값, None 포함) 그대로 넘긴다 —
    돌려받은 신원이나 판정 폴더 안의 영수증으로 핀을 새로 만들지 않는다. 핀 값의 꼴과 대조는 판정 신원 읽기가 한다.
    """
    from .infer import read_judge_source

    judge_dir = judge_source.get("judge_dir")
    if not isinstance(judge_dir, str) or not judge_dir:
        raise AuditRefused("judge_source_unverified: judge_dir 가 없다 — 호출자가 준 판정 행을 그대로 믿지 않는다")
    pins = judge_source.get("caller_expected_pins")
    if not isinstance(pins, Mapping) or sorted(pins) != sorted(JUDGE_SOURCE_PIN_KEYS):
        raise AuditRefused(f"judge_source_unverified: caller_expected_pins{list(JUDGE_SOURCE_PIN_KEYS)} 가 없다 — "
                           "판정 신원 읽기를 처음 부른 호출자의 핀을 그대로 넘겨야 한다")
    try:
        verified = read_judge_source(run_dir, json.loads(canonical_bytes(config)), judge_dir,
                                     **{k: pins[k] for k in JUDGE_SOURCE_PIN_KEYS})
    except (ValueError, OSError) as exc:     # StageRefused 는 ValueError 다
        raise AuditRefused(f"judge_source_unverified: 판정 경로를 다시 읽지 못했다 — {exc}") from exc
    if (not isinstance(verified, Mapping) or not isinstance(verified.get("rows"), list)
            or verified.get("caller_expected_pins") != dict(pins)):
        raise AuditRefused("judge_source_unverified: 판정 신원 읽기가 행 목록 · 받은 핀을 그대로 내지 않았다")
    return dict(verified)


def _dataset_row(task: Mapping[str, Any], body: Mapping[str, Any], provenance: dict) -> dict:
    if task["kind"] == "source_qa":
        return {"id": task["id"], "kind": "source_qa", "split": task["split"], "client": "public",
                "lineage_group_id": task["lineage_group_id"], "image_path": None,
                "source_ids": list(task["source_ids"]),
                "messages": [{"role": "user", "content": [{"type": "text", "text": body["question"]}]},
                             {"role": "assistant", "content": [{"type": "text", "text": body["answer"]}]}],
                "provenance": provenance}
    return {"id": task["id"], "kind": "image_report", "split": task["split"], "client": task["client"],
            "lineage_group_id": task["lineage_group_id"], "image_path": task["image_path"],
            "image_sha256": task["image_sha256"], "source_ids": list(task["source_ids"]),
            "messages": [{"role": "user", "content": [{"type": "image", "image": task["image_path"]},
                                                       {"type": "text", "text": task["train_user_text"]}]},
                         {"role": "assistant", "content": [{"type": "text", "text": body["text"]}]}],
            "provenance": provenance}


def counts_of(rows: Sequence[Mapping[str, Any]]) -> dict:
    """데이터셋 행의 수 — 릴리스가 같은 산식으로 다시 세어 대조한다."""
    c = Counter((r["kind"], r["split"]) for r in rows)
    return {"rows": len(rows),
            "image_report": {"train": c[("image_report", "train")], "val": c[("image_report", "val")]},
            "source_qa": {"train": c[("source_qa", "train")], "val": c[("source_qa", "val")]},
            "by_client": dict(sorted(Counter(r["client"] for r in rows).items()))}


# --------------------------------------------------------------------------- 검수 표본


#: 같은 계보 안의 같은 바이트 영상인데 잠긴 타깃이 엇갈린 행 — 전수 검수하는 층
CONFLICT_STRATUM = "image_report|identical_bytes_conflicting_target"


def stratum_of(task: Mapping[str, Any]) -> str:
    if task["kind"] == "source_qa":
        return f"source_qa|{task['split']}|{task['lineage_group_id']}"
    codes = sorted(set(defect_codes(task.get("locked_target"))))
    return f"image_report|{task['split']}|{task['client']}|{'+'.join(codes) if codes else 'none'}"


def review_sample(rows: Sequence[dict], strata: Mapping[str, str], dataset_sha: str,
                  minimum: int, fraction: float, full: frozenset[str] = frozenset({CONFLICT_STRATUM}),
                  ) -> tuple[dict[str, list[str]], dict[str, int]]:
    """층마다 max(min(minimum, n), ceil(fraction·n)) 건과 층 크기. 순서는 sha256(dataset_sha + 줄바꿈 + id) — 데이터가 정하고 숨길 값이 없다."""
    by: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by[strata[r["id"]]].append(r["id"])
    sample, sizes = {}, {}
    for s, ids in sorted(by.items()):
        n = len(ids) if s in full else max(min(minimum, len(ids)), math.ceil(fraction * len(ids)))
        ordered = sorted(ids, key=lambda i: _sha_bytes(f"{dataset_sha}\n{i}".encode()))
        sample[s] = sorted(ordered[:n])
        sizes[s] = len(ids)
    return sample, sizes


# --------------------------------------------------------------------------- 본체


def audit_run(run_dir, config: Mapping[str, Any], output_dir, *, judge_source: Mapping[str, Any] | None = None,
              policy: Any = None) -> dict:
    """`judge_source` — 따로 둔 판정 신원의 행(`_judge_input`). 없으면 실행 설정의 첫 판정을 읽는다.
    `policy` — 결정적 정책(`POLICY_API`). 없으면 지금의 판정 정책 모듈. 어느 판정 출처든 늘 건다.
    v7 판정 출처(`judge_structured_v7`)면 정책은 `POLICY_API_V7` 이고, 행마다의 결정적 판단은 판정 신원의 영수증을
    다시 계산해 맞댄 값이다. 정상 행(모델 대상 밖)은 판정 행 없이 그 영수증으로만 가르고, `judge_missing` 은 모델 대상
    행에만 붙는다."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    q = _quality_config(config)
    is_v7 = _declares_v7(judge_source)
    policy, policy_id = _resolve_policy_v7(policy) if is_v7 else _resolve_policy(policy)
    policy_sha = _digest(policy_id)
    gen_rows = read_stage_rows(run_dir, "generate", dict(config))
    run, tasks, tasks_sha = _read_tasks(run_dir)
    judge_rows, judge_info = _judge_input(run_dir, config, run, tasks, gen_rows, judge_source, policy_id)
    if is_v7 != (judge_info["v7"] is not None):
        raise AuditRefused("judge_schema_mixed: v7 정책과 v7 판정 출처는 함께만 건다")
    gen = {r["id"]: r for r in gen_rows}
    receipts = _v7_receipts(tasks, gen, judge_rows, judge_info, policy, policy_id) if is_v7 else {}
    #: 판정 모델을 부르는 행 — v7 은 명부의 행 종류가 정상이 아닌 행(판정 전 샤드 포함), 그 밖의 꼴은 모든 행
    model_target = {t["id"] for t in tasks if not is_v7 or _judge_row_class(t) != "defect_free"}
    _prepare_output(output_dir)
    label_names = _label_names() if any(t["kind"] == "image_report" for t in tasks) else []

    judged = {r["id"]: r for r in judge_rows}
    backends = {"generate": Counter(r["_backend"] for r in gen_rows),
                "judge": Counter(r["_backend"] for r in judge_rows)}

    status: dict[str, str] = {}
    reasons: dict[str, list[str]] = {}
    bodies: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    #: 결정적 정책의 (사유, 표지, 오류) — 생성문이 있는 영상 설명 행마다. 판정 유무 · 행 상태와 상관없이 남긴다
    policy_out: dict[str, tuple[list[str] | None, list[str] | None, str | None]] = {}
    #: 판정 행의 (오류 사유, 엄격 파싱 판정(세 판정값), 모델 표지, 구조화 원 답) — 원문은 손대지 않고 지문만 싣는다
    judge_out: dict[str, tuple[list[str], dict | None, list[str], dict | None, list[str]]] = {}
    schema = judge_info["parser_schema"]
    fenced = Counter()
    run_errors: list[str] = []

    # 분할 · 계보 — 같은 계보가 train 과 val 에 걸치면 안 된다(kind 를 가리지 않는다).
    lineage_split: dict[str, set[str]] = defaultdict(set)
    for t in tasks:
        lineage_split[t["lineage_group_id"]].add(t["split"])
    crossing = sorted(k for k, v in lineage_split.items() if len(v) > 1)
    if crossing:
        run_errors.append("lineage_crosses_split")

    # 같은 바이트의 영상. 같은 계보 안이면 연속 촬영의 화소 쌍둥이 — 주석에서 필연적인 반복으로 세기만 한다.
    # 계보가 다르면 같은 개체가 두 계보에 든 것이라 오류다. 같은 계보 안에서 잠긴 타깃이 엇갈리면 같은 영상을
    # 서로 다른 답으로 가르치게 되므로, 막지는 않되 그 행들을 사람 검수 표본에 전부 넣는다.
    by_image: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        if t["kind"] == "image_report" and _HEX64.fullmatch(str(t.get("image_sha256", ""))):
            by_image[t["image_sha256"]].append(t)
    same_image = sorted(k for k, ts in by_image.items()
                        if len(ts) > 1 and len({x["lineage_group_id"] for x in ts}) > 1)
    twins = {k: ts for k, ts in by_image.items()
             if len(ts) > 1 and len({x["lineage_group_id"] for x in ts}) == 1}
    conflicting = {x["id"] for ts in twins.values()
                   if len({_digest(x.get("locked_target")) for x in ts}) > 1 for x in ts}
    if same_image:
        run_errors.append("duplicate_image_entity")

    # 학습 질문 틀은 모든 image_report 에 하나다. 이미지마다 다르면 그 문장으로 정답이 샐 수 있다.
    user_texts = {t.get("train_user_text") for t in tasks if t["kind"] == "image_report"}
    if len(user_texts) > 1:
        run_errors.append("train_user_text_varies")

    # 구절당 계획 질문 수.
    planned_per_passage = Counter(s for t in tasks if t["kind"] == "source_qa" for s in t["source_ids"])
    over_planned = sorted(s for s, n in planned_per_passage.items() if n > q["maximum_questions_per_passage"])
    if over_planned:
        run_errors.append("questions_per_passage_planned_over_limit")

    for t in tasks:
        tid = t["id"]
        errs = _task_contract(t, label_names)
        g = gen.get(tid)
        j = judged.get(tid)
        receipt = receipts.get(tid)
        if is_v7:                                           # 영수증 = 다시 계산해 맞댄 결정적 판단(정책 오류는 거부로 끝났다)
            if t["kind"] in POLICY_KINDS and receipt is not None:
                policy_out[tid] = (list(receipt["deterministic_reasons"]), list(receipt["flags"]), None)
        elif t["kind"] in POLICY_KINDS and g is not None:   # CPU 검사는 판정이 없어도 돈다 — 상태는 아래에서 정한다
            policy_out[tid] = _policy_check(policy, t, g["text"])
        j_err, j_rej, verdict, j_flags, j_answers, j_norm = _read_judge(t, j, schema)
        if j is not None:
            judge_out[tid] = (j_err, verdict, j_flags, j_answers, j_norm)
        if errs:
            status[tid], reasons[tid] = "error", errs
            continue
        if g is None or (tid in model_target and j is None) or (is_v7 and receipt is None):
            status[tid] = "incomplete"
            reasons[tid] = ["generation_missing" if g is None else
                            "judge_missing" if tid in model_target else "deterministic_receipt_missing"]
            continue
        body, rej, was_fenced = _content(t, g)
        fenced["generate"] += was_fenced
        if body is not None:
            rej += _content_problems(t, body)
            bodies[tid] = body
        if is_v7:                                           # 원문 QA 도 영수증의 판단을 따른다(영상 규칙은 원문 QA 에 사유를 내지 않는다)
            p_reasons, p_error = receipt["deterministic_reasons"], None
        else:
            p_reasons, _, p_error = policy_out.get(tid, ([], [], None))
        rej += [f"policy:{r}" for r in p_reasons or []]
        rej += j_rej
        errors = j_err + (["policy_error"] if p_error else [])
        if verdict is not None:
            verdicts[tid] = verdict
        if errors:
            status[tid], reasons[tid] = "error", errors + rej
        elif rej:
            status[tid], reasons[tid] = "rejected", rej
        else:
            status[tid], reasons[tid] = "accepted", []

    # 중복 — 계획 순서에서 먼저 나온 것을 남기고 뒤의 것을 기각한다.
    seen_q: dict[str, str] = {}
    seen_a: dict[str, str] = {}
    seen_desc: dict[str, tuple[str, str]] = {}
    annotation_repeat = 0
    for t in tasks:
        tid = t["id"]
        if status.get(tid) != "accepted":
            continue
        body = bodies[tid]
        dup = []
        if t["kind"] == "source_qa":
            kq, ka = _squash(body["question"]), _squash(body["answer"])
            if kq in seen_q:
                dup.append("duplicate_question")
            if ka in seen_a:
                dup.append("duplicate_answer")
            seen_q.setdefault(kq, tid)
            seen_a.setdefault(ka, tid)
        else:
            kd = _squash(body["text"])
            target = _digest(t.get("locked_target"))
            if kd in seen_desc:
                if seen_desc[kd][1] == target:
                    annotation_repeat += 1          # 같은 주석 → 같은 설명. 주석에서 필연적인 반복이라 기각하지 않는다
                else:
                    dup.append("description_collapse")  # 다른 주석에 같은 설명 — 주석을 반영하지 않은 것이다
            else:
                seen_desc[kd] = (tid, target)
        if dup:
            status[tid], reasons[tid] = "rejected", dup

    # 구절당 채택 질문 수.
    accepted_per_passage = Counter(s for t in tasks if t["kind"] == "source_qa" and status[t["id"]] == "accepted"
                                   for s in t["source_ids"])
    over_accepted = sorted(s for s, n in accepted_per_passage.items() if n > q["maximum_questions_per_passage"])
    if over_accepted:
        run_errors.append("questions_per_passage_accepted_over_limit")

    accepted_tasks = [t for t in tasks if status[t["id"]] == "accepted"]
    # 판정 모델을 부르지 않은 행(v7 정상 행)은 판정 칸이 '적용 안 됨' 이다 — 그 밖의 행은 판정 행이 반드시 있다
    dataset_rows = [_dataset_row(t, bodies[t["id"]], _provenance(
        gen[t["id"]], judged[t["id"]] if t["id"] in model_target else None, verdicts.get(t["id"]), config, judge_info,
        {"identity_sha256": policy_sha, "flags": policy_out[t["id"]][1]} if t["id"] in policy_out else None,
        judge_out[t["id"]][2:] if schema in JUDGE_PARSER_VERSIONS and t["id"] in model_target else None))
                    for t in accepted_tasks]
    dataset_raw = _jsonl(dataset_rows)
    dataset_sha = _sha_bytes(dataset_raw)

    # 원문 QA 부족량 — 숫자 그대로 적는다. 반복으로 채우지 않는다.
    qa_train = [t for t in accepted_tasks if t["kind"] == "source_qa" and t["split"] == "train"]
    distinct_train_passages = len({s for t in qa_train for s in t["source_ids"]})
    source_qa = {
        "planned": {s: sum(t["kind"] == "source_qa" and t["split"] == s for t in tasks) for s in ("train", "val")},
        "accepted": {s: sum(t["kind"] == "source_qa" and t["split"] == s for t in accepted_tasks)
                     for s in ("train", "val")},
        "distinct_train_passages": distinct_train_passages,
        "minimum_source_qa_train": q["minimum_source_qa_train"],
        "target_source_qa_train": q["target_source_qa_train"],
        "minimum_distinct_train_passages": q["minimum_distinct_train_passages"],
        "maximum_questions_per_passage": q["maximum_questions_per_passage"],
        "short_of_minimum": max(0, q["minimum_source_qa_train"] - len(qa_train)),
        "short_of_target": max(0, q["target_source_qa_train"] - len(qa_train)),
        "distinct_passages_short": max(0, q["minimum_distinct_train_passages"] - distinct_train_passages),
        "passages_over_limit": over_accepted,
        "passages_planned_over_limit": over_planned,
    }

    # 커버리지 — 참여자 × 결함(이미지), 계보(원문 QA). 계획 대 채택.
    coverage: dict[str, dict] = {"image_report": {}, "source_qa": {}}
    for t in tasks:
        if t["kind"] == "image_report":
            codes = sorted(set(defect_codes(t.get("locked_target")))) or ["none"]
            for c in codes:
                cell = coverage["image_report"].setdefault(t["client"], {}).setdefault(c, {"planned": 0, "accepted": 0})
                cell["planned"] += 1
                cell["accepted"] += status[t["id"]] == "accepted"
        else:
            cell = coverage["source_qa"].setdefault(t["lineage_group_id"], {"planned": 0, "accepted": 0})
            cell["planned"] += 1
            cell["accepted"] += status[t["id"]] == "accepted"

    strata = {t["id"]: (CONFLICT_STRATUM if t["id"] in conflicting else stratum_of(t)) for t in tasks}
    sample, sizes = review_sample(dataset_rows, strata, dataset_sha, q["minimum_review_per_stratum"],
                                  q["review_fraction"])
    row_sha = {r["id"]: _digest(r) for r in dataset_rows}
    review_template = {
        "schema_version": 1, "reviewer": None, "reviewed_at": None, "target_dataset_sha256": dataset_sha,
        "approved": False,
        "samples": [{"id": sid, "stratum": s, "row_sha256": row_sha[sid], "decision": None, "comment": None,
                     "reviewer": None, "reviewed_at": None}
                    for s, ids in sample.items() for sid in ids],
    }

    none3, none5 = (None, None, None), (None, None, None, None, None)
    status_rows =[{"id": t["id"], "kind": t["kind"], "split": t["split"], "status": status[t["id"]],
                    "reasons": reasons[t["id"]],
                    "policy_reasons": policy_out.get(t["id"], none3)[0],
                    "policy_flags": policy_out.get(t["id"], none3)[1],
                    "policy_error": policy_out.get(t["id"], none3)[2],
                    "judge_text_sha256": (_sha_bytes(judged[t["id"]]["text"].encode("utf-8"))
                                          if t["id"] in judged else None),
                    "judge_verdict": judge_out.get(t["id"], none5)[1],
                    "judge_flags": judge_out.get(t["id"], none5)[2],
                    "judge_answers": judge_out.get(t["id"], none5)[3],
                    "judge_format_normalized": judge_out.get(t["id"], none5)[4]} for t in tasks]
    if is_v7:   # 판정 모델을 불렀는가 · 결정적 영수증(판정 전 행은 None) — v7 의 행에만 싣는다
        for r in status_rows:
            got = receipts.get(r["id"])
            r.update(judge_applicable=r["id"] in model_target,
                     deterministic_pass=None if got is None else got["deterministic_pass"],
                     deterministic_reasons=None if got is None else got["deterministic_reasons"],
                     deterministic_codes=None if got is None else got["deterministic_codes"])
    policy_summary = _policy_summary(tasks, policy_out, judge_out, judged,
                                     frozenset(t["id"] for t in tasks) - model_target)
    judge_flags = Counter(f for v in judge_out.values() for f in v[2])
    judge_normalized = Counter(f for v in judge_out.values() for f in v[4])

    counts = Counter(status.values())
    n_errors = counts["error"] + len(run_errors)
    blockers = []
    if counts["incomplete"]:
        blockers.append("incomplete_rows")
    if n_errors:
        blockers.append("errors")
    if counts["rejected"]:
        blockers.append("rejected_rows")
    non_tf = {s: sum(n for b, n in c.items() if b != "transformers") for s, c in backends.items()}
    if any(non_tf.values()):
        blockers.append("non_transformers_backend")
    if source_qa["short_of_minimum"] or source_qa["distinct_passages_short"]:
        blockers.append("source_qa_below_minimum")

    files = {}
    files["dataset.jsonl"] = {"sha256": _write_new_bytes(output_dir / "dataset.jsonl", dataset_raw),
                              "rows": len(dataset_rows)}
    files["rows_status.jsonl"] = {"sha256": _write_new_bytes(output_dir / "rows_status.jsonl", _jsonl(status_rows)),
                                  "rows": len(status_rows)}
    files["REVIEW_TEMPLATE.json"] = {
        "sha256": _write_new_bytes(output_dir / "REVIEW_TEMPLATE.json", canonical_bytes(review_template) + b"\n"),
        "samples": len(review_template["samples"])}

    task_ids = [t["id"] for t in tasks]
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_version": AUDIT_VERSION,
        # 구조화 판정을 읽은 파서의 판 — 스키마 판(judge_identity.parser_schema)과 따로. 옛 네 키면 None
        "judge_parser_version": JUDGE_PARSER_VERSIONS.get(schema),
        # 릴리스가 읽는 칸(공유 계약)
        # v7 은 판정 행이 모델 대상 행 수만큼이고 결정적 영수증이 모든 행에 있어야 완료다
        "complete": counts["incomplete"] == 0 and len(judge_rows) == len(model_target)
                    and (not is_v7 or len(receipts) == len(tasks)),
        "errors": n_errors,
        "rejected": counts["rejected"],
        "incomplete": counts["incomplete"],
        "run_dir": _run_dir_ref(run_dir, output_dir),
        "run_identity_sha256": run.get("identity_sha256"),
        "run_identity_basis": run.get("identity_basis"),
        # parser_schema — 판정 출력을 읽은 꼴(옛 네 키 · 구조화 v5 · v6 · v7). 핀 · 영수증 칸은 따로 둔 판정에서만 값이 있다
        "judge_identity": {k: judge_info[k] for k in (
            "source", "identity_sha256", "identity_basis", "model", "parser_schema", "caller_expected_pins",
            "pin_provenance", "probe_approval_sha256", "gpu_model_verified")},
        "policy": {"identity": policy_id, "identity_sha256": policy_sha,
                   # separate 면 판정 신원의 근거에 같은 정책 지문이 묶였음을 _judge_input 이 확인했다
                   "bound_to_judge_identity": judge_info["source"] == "separate",
                   "applies_to": sorted(POLICY_KINDS), "flags_reject": False, **policy_summary},
        "tasks_sha256": tasks_sha,
        "config_sha256": run.get("config_sha256"),
        "config": run.get("config"),
        "dataset_sha256": dataset_sha,
        "planned": {"count": len(task_ids), "ids_sha256": ids_sha256(task_ids)},
        "accepted": {"count": len(accepted_tasks), "ids_sha256": ids_sha256([t["id"] for t in accepted_tasks])},
        "user_templates": {"image_report": sorted(x for x in user_texts if isinstance(x, str))},
        "source_documents": [{"source_id": s, "source_sha256": h} for s, h in sorted(
            {(t["evidence"].get("source_id"), t["evidence"].get("source_sha256"))
             for t in tasks if t["kind"] == "source_qa" and isinstance(t.get("evidence"), Mapping)})],
        "review": {"strata": sample, "stratum_sizes": sizes,
                   "minimum_per_stratum": q["minimum_review_per_stratum"], "fraction": q["review_fraction"],
                   "order": "sha256(dataset_sha256 + newline + id)"},
        "counts": {**counts_of(dataset_rows),
                   "planned": len(tasks), "accepted": counts["accepted"], "status_rejected": counts["rejected"],
                   "status_error": counts["error"], "status_incomplete": counts["incomplete"],
                   "by_kind_split": {k: dict(sorted(v.items())) for k, v in sorted(
                       _nested_count((t["kind"], t["split"], status[t["id"]]) for t in tasks).items())},
                   "reasons": dict(sorted(Counter(r for rs in reasons.values() for r in rs).items())),
                   # 판정기가 낸 관찰 표지(구조화 판정의 n5_absent 등) — 판정 행 전부, 기각이 아니다
                   "judge_model_flags": dict(sorted(judge_flags.items())),
                   # 파서가 판정 원문에 한 형식 정규화(울타리 벗김 · 참거짓 글) — 판정 행 전부, 행마다 rows_status 에
                   "judge_format_normalized": dict(sorted(judge_normalized.items())),
                   "run_errors": run_errors,
                   "fenced_json": dict(sorted(fenced.items())),
                   "annotation_repeat_descriptions": annotation_repeat},
        # 집계 쪽 진단 칸
        "stages": {s: {"rows": sum(c.values()), "backends": dict(sorted(c.items())),
                       "shard_sha256": sorted({r["_shard_sha256"] for r in rows})}
                   for s, c, rows in (("generate", backends["generate"], gen_rows),
                                      ("judge", backends["judge"], judge_rows))},
        "lineage_crossing_split": crossing,
        "duplicate_image_sha256": same_image,
        "identical_images_same_lineage": {
            "images": len(twins), "rows": sum(len(ts) for ts in twins.values()),
            "conflicting_target_images": sum(len({_digest(x.get("locked_target")) for x in ts}) > 1
                                             for ts in twins.values()),
            "conflicting_target_rows": len(conflicting), "review": "conflicting rows go to a fully reviewed stratum"},
        "coverage": coverage,
        "source_qa": source_qa,
        # v7 판정 신원의 결정적 영수증 — 정상 행은 판정 모델 없이 이것으로만 갈린다. 다른 꼴이면 None
        "deterministic_receipts": None if not is_v7 else {
            "policy_identity_sha256": policy_sha, "rows": len(receipts), "planned": len(tasks),
            "model_task_ids": {"count": len(model_target), "ids_sha256": ids_sha256(
                [t["id"] for t in tasks if t["id"] in model_target])},
            "shard_sha256": sorted({r["_shard_sha256"] for r in receipts.values()}),
            "acceptance": "normal rows: content checks and the deterministic receipt pass, no judge model; "
                          "model rows: content checks, the deterministic receipt pass and every judge question true",
            "normal": _receipt_summary([t for t in tasks if t["id"] not in model_target], receipts, status),
            "model": _receipt_summary([t for t in tasks if t["id"] in model_target], receipts, status)},
        "files": files,
        "status": "blocked" if blockers else "ready_for_human_review",
        "blockers": blockers,
        "judge_is_not_human_approval": True,
        "human_review_required": True,
        "ready_for_training": False,
    }
    write_json_new(output_dir / "AUDIT.json", audit)
    return {"status": audit["status"], "blockers": blockers, "dataset_sha256": dataset_sha,
            "counts": audit["counts"], "exit_code": 0 if not blockers else 6}


_AGREEMENT_CELLS = ("both_pass", "model_pass_policy_reject", "model_reject_policy_pass", "both_reject")


def _policy_summary(tasks: Sequence[dict], policy_out: Mapping[str, tuple], judge_out: Mapping[str, tuple],
                    judged: Mapping[str, Any], no_model: frozenset[str] = frozenset()) -> dict:
    """정책의 수와 판정기 · 정책의 맞대기.

    정책의 수: 생성문이 있는 영상 설명 행 전부(행 상태 · 판정 유무와 상관없이). 표지는 기각이 아니다. v7 은 결정적
    영수증이 있는 행이다.
    맞대기: 분모는 판정 출력을 엄격 파싱으로 읽은 영상 설명 행 가운데 정책이 오류 없이 돈 행뿐이다 — 판정이 없거나
    읽을 수 없는 행 · 정책 오류 행은 분모에 넣지 않고 따로 센다. 판정 모델을 부르지 않는 행(`no_model` — v7 정상 행)은
    `judge_not_applicable` 로 센다. 모델 통과는 판정값의 `reason` 밖 물음이 모두 참이다. 원문 QA 는 어디에도 세지 않는다.
    """
    def blank() -> dict:
        return {"denominator": 0, **dict.fromkeys(_AGREEMENT_CELLS, 0)}

    checked, errors = Counter(), Counter()
    reasons_by: dict[str, Counter] = defaultdict(Counter)
    flags_by: dict[str, Counter] = defaultdict(Counter)
    agree, agree_by = blank(), defaultdict(blank)
    missed, excluded = Counter(), Counter()
    for t in tasks:
        if t["kind"] not in POLICY_KINDS:
            continue
        tid, cls = t["id"], _row_class(t)
        p = policy_out.get(tid)
        if p is not None and p[2] is None:
            checked[cls] += 1
            reasons_by[cls].update(p[0])
            flags_by[cls].update(p[1])
        elif p is not None:
            errors[cls] += 1
        if tid in no_model:
            excluded["judge_not_applicable"] += 1
            continue
        if tid not in judged:
            excluded["judge_missing"] += 1
            continue
        verdict = judge_out[tid][1]
        if verdict is None:
            excluded["judge_unparsed"] += 1
            continue
        if p is None:
            excluded["generation_missing"] += 1
            continue
        if p[2] is not None:
            excluded["policy_error"] += 1
            continue
        model_pass = all(v is True for k, v in verdict.items() if k != "reason")
        policy_pass = not p[0]
        cell = _AGREEMENT_CELLS[2 * (not model_pass) + (not policy_pass)]
        for box in (agree, agree_by[cls]):
            box["denominator"] += 1
            box[cell] += 1
        if model_pass and not policy_pass:
            missed.update(p[0])
    classes = sorted(set(checked) | set(errors))
    return {
        "rows_checked": sum(checked.values()), "rows_error": sum(errors.values()),
        "reasons": dict(sorted(sum(reasons_by.values(), Counter()).items())),
        "flags": dict(sorted(sum(flags_by.values(), Counter()).items())),
        "by_class": {c: {"rows_checked": checked[c], "rows_error": errors[c],
                         "reasons": dict(sorted(reasons_by[c].items())), "flags": dict(sorted(flags_by[c].items()))}
                     for c in classes},
        "model_agreement": {**agree, "by_class": {c: agree_by[c] for c in sorted(agree_by)},
                            "policy_reasons_when_model_passed": dict(sorted(missed.items())),
                            "excluded": dict(sorted(excluded.items())),
                            "denominator_rule": "image_report rows with a strictly parsed judge verdict "
                                                "and a policy result without error"},
    }


def _nested_count(items: Iterable[tuple[str, str, str]]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for kind, split, st in items:
        out[f"{kind}|{split}"][st] += 1
    return out
