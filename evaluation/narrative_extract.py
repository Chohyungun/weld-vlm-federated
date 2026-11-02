"""서술 추출 · 상태 — 게이트 판정 04 §1(공통 계약) + §7-2 ~ §7-5 · 03 2판 §1-4b 의 R1 · P1 ~ P5. 채점 쪽(D) 구현이다.

학습 타깃은 JSON 한 줄 · 줄바꿈 하나 · 서술(계약 v2 `json_line + separator + basis`)이지만, 모델이 실제로 낸 글은 그 꼴을 벗어날 수
있다. 그래서 채점의 서술은 **첫 JSON 객체 밖의 글 전부(앞 + 뒤)** 다. 학습 쪽 점검(`vlm/narrative.py`, C 구현)과 이 모듈은 같은
벡터 파일(`evaluation/vectors/narrative_extract.v1.json`)에서 같은 추출문 · 같은 상태를 내야 한다 — 두 구현의 대조는
`evaluation.narrative_conformance` 가 한다.

## 상태(우선순위 순)

| 상태 | 조건 |
|---|---|
| `validation_error` ⓪ | 종료 사유가 없다(칸 없음 · `None`)거나 `eos` · `length` 밖이다(`stop_undetermined` 포함) — R1: 1.4 계약에서 `gen_stop` 은 필수이고 줄 판독이 해독보다 먼저 본다 |
| `record_fail` ① | 상류 파싱 오류 — 참조 파서(`evaluation.adapters_v14.reference_parse`)의 판정 그대로(닫히지 않음 · 해독 불가 · 스키마 위반). 뒤의 다른 JSON 을 찾아 살리지 않는다 |
| `missing` ② | 구조 파싱 유효 · 추출문이 비었다(유효 JSON 직후 끝 · 줄바꿈 · 표지 줄만 남음 포함). 종료 사유는 `stop_kept` 로 보존 |
| `truncated` ③ | 추출문 있음 · `length` |
| `complete` ④ | 추출문 있음 · `eos` — 생성 종료 상태이지 의미적 완전성이 아니다 |

## 추출(P1 ~ P5)

1. 첫 객체의 범위는 참조 파서가 해독에 쓰는 같은 식으로 낸다 — 생성문을 `str.strip()` 으로 벗기고 첫 `{` 부터
   `json.JSONDecoder().raw_decode`. 참조 파서는 끝 위치를 돌려주지 않고 이 모듈은 파서를 바꾸지 않으므로, 끝 위치만 같은 식으로 다시
   내고 해독한 객체가 참조 파서의 것과 같은지 매번 맞댄다(어긋나면 멈춘다).
2. 앞 글 · 뒤 글을 각각 `\\n` 으로 나눈 조각에서 **표지 줄**을 뺀다 — 앞뒤 공백을 벗긴 조각 전체가 역따옴표 셋 이상 + 언어 표시
   (`[A-Za-z0-9_+-]*`)뿐인 줄(04 §7-4). 남은 조각을 `\\n` 으로 다시 잇는다(빈 줄을 남기지 않는다). 물결표 펜스는 표지가 아니다.
3. 각 조각의 앞뒤 공백(`str.strip()` — 유니코드 공백)을 벗기고, 둘 다 비지 않았으면 공백 하나로 잇는다(P3).
4. 안쪽의 줄바꿈 · 공백 · 유니코드 꼴은 바꾸지 않는다(P5) — 정규화는 검사기의 일이다.
5. `front_text` = ② ~ ④ 에서 3 을 거친 앞 글이 비지 않음(P4). ⓪ · ① 에서는 `None`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import get_args

from evaluation.adapters_v14 import OUTCOME_OK, decode_line, reference_parse
from evaluation.schema_v14 import GenStop

EXTRACTOR_ID = "narrative_extract_v1"
"""추출기 식별자 — 등록 판 6 `narrative.extractor_id` 의 값. 규칙을 바꾸면 올린다."""

VALIDATION_ERROR = "validation_error"
RECORD_FAIL = "record_fail"
MISSING = "missing"
TRUNCATED = "truncated"
COMPLETE = "complete"
STATES = (VALIDATION_ERROR, RECORD_FAIL, MISSING, TRUNCATED, COMPLETE)
"""상태 — 우선순위 순서다(04 §1-3 · §7-4)."""

GEN_STOPS: tuple[str, ...] = get_args(GenStop)
"""계약 안의 종료 사유 — 1.4 레코드의 `GenStop` 에서 읽는다(eos · length)."""

_NO_FIELD = object()
"""`gen_stop` 칸이 없는 줄(벡터 B5b)을 부르는 꼴 — 인자를 빼고 부른다. `None` 과 구별한다."""

_MARKER = re.compile(r"`{3,}[A-Za-z0-9_+\-]*")
"""표지 줄 — 앞뒤 공백을 벗긴 조각 **전체**에 맞댄다(`fullmatch`)."""


@dataclass(frozen=True)
class NarrativeExtract:
    """생성문 하나의 추출 결과 — 적합성 벡터가 맞대는 다섯 값."""

    state: str
    narrative: str | None
    """`missing` 이면 빈 문자열, `truncated` · `complete` 면 비지 않은 문자열, ⓪ · ① 이면 `None`."""
    front_text: bool | None
    stop_kept: str | None
    """② ~ ④ 에서 보존한 종료 사유. 그 밖은 `None`."""
    upstream_reason: str | None
    """① 일 때 참조 파서의 사유(1.4 어휘). 그 밖은 `None`."""

    def as_dict(self) -> dict:
        return {"state": self.state, "narrative": self.narrative, "front_text": self.front_text,
                "stop_kept": self.stop_kept, "upstream_reason": self.upstream_reason}


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _outside(text: str) -> tuple[str, str]:
    """첫 JSON 객체의 앞 글 · 뒤 글. 참조 파서가 `ok` 를 낸 생성문에만 부른다."""
    s = text.strip()
    i = s.find("{")
    obj, end = json.JSONDecoder().raw_decode(s[i:])
    ref_obj, why = decode_line(text)
    if why is not None or _canonical(obj) != _canonical(ref_obj):
        raise RuntimeError("첫 객체의 범위가 참조 파서의 해독과 갈렸다 — 참조 파서의 식이 바뀌었는지 본다")
    return s[:i], s[i + end:]


def _clean(fragment: str) -> str:
    kept = [ln for ln in fragment.split("\n") if not _MARKER.fullmatch(ln.strip())]
    return "\n".join(kept).strip()


def extract(text: str, gen_stop=_NO_FIELD) -> NarrativeExtract:
    """생성문(1.4 생성 줄의 `text`)과 종료 사유(`gen_stop`)에서 추출문 · 상태를 낸다.

    `gen_stop` 칸이 없는 줄은 인자를 빼고 부른다(`extract(text)`). `None` 은 칸은 있고 값이 null 인 줄이다 — 둘 다 ⓪ 이다.
    """
    if not isinstance(text, str):
        raise TypeError("text 는 문자열이다 — 1.4 생성 줄의 text 칸")
    if gen_stop is _NO_FIELD or gen_stop not in GEN_STOPS:
        return NarrativeExtract(VALIDATION_ERROR, None, None, None, None)
    ref = reference_parse(text, known_iso_codes=(), scoring_iso_codes=())   # 판정 · 사유만 쓴다 — 코드 목록과 무관하다
    if ref.outcome != OUTCOME_OK:
        return NarrativeExtract(RECORD_FAIL, None, None, None, ref.reason)
    front, back = (_clean(part) for part in _outside(text))
    narrative = f"{front} {back}" if front and back else front or back
    if not narrative:
        return NarrativeExtract(MISSING, "", False, gen_stop, None)
    return NarrativeExtract(TRUNCATED if gen_stop == "length" else COMPLETE, narrative, bool(front), gen_stop, None)


__all__ = ["COMPLETE", "EXTRACTOR_ID", "GEN_STOPS", "MISSING", "RECORD_FAIL", "STATES", "TRUNCATED",
           "VALIDATION_ERROR", "NarrativeExtract", "extract"]
