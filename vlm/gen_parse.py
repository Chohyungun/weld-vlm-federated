"""생성문 파싱 — 쓰는 쪽(export)의 파서. **역변환 전 결과를 내고, 사유와 결과의 짝을 언제나 지킨다**(리허설 2판 §2-4).

평가 쪽 참조 파서(`evaluation.adapters_v14.reference_parse`)를 가져오지 않고 다시 쓴 구현이다. 두 파서가 같은 적합성 벡터
(`evaluation.conformance_v14`)를 지나야 본 export 를 시작한다(평가 계약 §14-1 아).
**코드는 따로지만 규칙은 하나다** — 해독 규칙(첫 `{` 부터 `raw_decode`, 잘림 판정식, 크기 상한)과 항목의 좌표 검사(길이 4 ·
수 · 유한 · 퇴화)는 참조 파서를 읽은 뒤 같은 규칙으로 옮겨 적었다. 그래서 벡터 통과는 옮김이 맞다는 증거이고, 규칙 자체가
틀렸을 때 그것을 잡는 증거는 아니다.

## 두 단계로 나눈다

1. `parse_generation(text)` — 글자에서 **역변환 전** 값까지. JSON 을 꺼내고, `defects` 가 목록인지 보고, 항목마다 좌표를 읽는다.
   레코드 실패면 사유 하나(`no_json` · `json_decode` · `truncated` · `schema_violation`)이고 결과는 없다. 모델 출력의 어떤 자료형에도
   예외를 내지 않는다 — 깊은 중첩 · 수천 자리 정수 · 거대한 입력도 분류로 떨어뜨린다(§14-1 마).
2. `line_fields(parsed, geom, coord_cfg)` — 줄에 싣는 `bbox_px_parsed` · `parse_error` · `n_bad_items_dropped`. 좌표는
   `vlm.coords.to_px` **한 번**으로 원본 픽셀로 돌리고 `round(v, 3)` 만 한다 — 정수화 · 클리핑 없음(계약 §13-3 다-10).
   `verdict` · `cited_clauses` 는 **고치지 않고** 넘기고, 키가 없으면 `null` 이다(§14-1 바).

적합성 벡터가 맞대는 일곱 값(`conformance_tuple`)은 1 의 결과에 **공통 폐기 정책**(`evaluation.policy.filter_defect_items` — 다섯 칸이
같은 규칙으로 버리게 하려고 둔 공용 코드)과 조항 축 판정을 더한 것이다. 폐기 정책은 두 파서가 같은 코드를 쓴다 — 두 구현이 따로 서는
자리는 글자 → 항목까지다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_TEXT_BYTES",
    "PARSE_ERRORS",
    "ParsedGeneration",
    "conformance_tuple",
    "line_fields",
    "pair_ok",
    "parse_generation",
]

#: 레코드 실패의 사유 — 계약 #4 의 1.4 어휘.
PARSE_ERRORS = ("no_json", "json_decode", "truncated", "schema_violation")
#: 이보다 큰 생성문은 해독하지 않고 `json_decode` 로 분류한다 — 파서가 메모리로 죽지 않게.
MAX_TEXT_BYTES = 4 << 20


@dataclass(frozen=True)
class ParsedGeneration:
    """생성문 하나를 읽은 결과 — **역변환 전**이다."""

    parse_error: str | None
    items: tuple[tuple[Any, float, float, float, float], ...] = ()
    """좌표가 선 항목 `(iso_code 원값, x1, y1, x2, y2)` — 모델이 낸 순서 그대로, 모델 좌표 그대로."""
    n_bad: int = 0
    """좌표가 깨져 버린 항목 수(길이 · 수치 · 유한 · 퇴화)."""
    verdict: Any = None
    cited: Any = None
    """`verdict` · `cited_clauses` 의 원값. 키가 없으면 `None` — 빈 목록으로 메우지 않는다."""


def _first_object(text: str) -> tuple[Any, str | None]:
    """첫 `{` 부터 JSON 값 하나를 꺼낸다. 앞의 산문 · 코드 펜스와 뒤의 잔여는 버린다. 예외를 내지 않는다."""
    s = text.strip()
    if len(s.encode("utf-8", "ignore")) > MAX_TEXT_BYTES:
        return None, "json_decode"
    start = s.find("{")
    if start < 0:
        return None, "no_json"
    body = s[start:]
    try:
        obj, _end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as exc:
        # 글이 끝나서 못 읽었으면 잘린 것이고, 도중에 틀렸으면 해독 실패다.
        cut = exc.pos >= len(body) or exc.msg.startswith("Unterminated string")
        return None, "truncated" if cut else "json_decode"
    except (RecursionError, ValueError):
        return None, "json_decode"
    return obj, None


def _box(value: Any) -> tuple[float, float, float, float] | None:
    """`bbox_2d` 를 float 넷으로. 길이 4 의 목록이 아니거나 · 수로 읽히지 않거나 · 유한하지 않거나 · 퇴화하면 None."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def parse_generation(text: str) -> ParsedGeneration:
    """글자 → 역변환 전 결과. 레코드 실패면 사유만 있고 항목 · 조항 값은 없다."""
    obj, why = _first_object(text if isinstance(text, str) else "")
    if why is not None:
        return ParsedGeneration(why)
    defects = obj.get("defects") if isinstance(obj, Mapping) else None
    if not isinstance(defects, list):
        return ParsedGeneration("schema_violation")
    items: list[tuple[Any, float, float, float, float]] = []
    n_bad = 0
    for d in defects:
        box = _box(d.get("bbox_2d")) if isinstance(d, Mapping) else None
        if box is None:
            n_bad += 1
            continue
        items.append((d.get("iso_code", ""), *box))
    verdict = obj.get("verdict", None)
    cited = obj.get("cited_clauses", None)
    return ParsedGeneration(None, tuple(items), n_bad, verdict, cited)


def line_fields(parsed: ParsedGeneration, geom, coord_cfg) -> dict[str, Any]:
    """줄에 싣는 셋 — `bbox_px_parsed` · `parse_error` · `n_bad_items_dropped`. 역변환은 `to_px` 한 번이다."""
    from vlm.coords import to_px

    if parsed.parse_error is not None:
        return {"bbox_px_parsed": None, "parse_error": parsed.parse_error, "n_bad_items_dropped": 0}
    defects = []
    for code, *box in parsed.items:
        px = to_px(box, geom, coord_cfg)
        defects.append({"iso_code": code, "bbox_px": [round(float(v), 3) for v in px]})
    return {"bbox_px_parsed": {"defects": defects, "verdict": parsed.verdict, "cited_clauses": parsed.cited},
            "parse_error": None, "n_bad_items_dropped": int(parsed.n_bad)}


def pair_ok(bbox_px_parsed: Any, parse_error: Any) -> bool:
    """짝 규칙 — 사유가 없으면 결과는 `defects` 가 목록인 객체이고, 사유가 있으면(1.4 어휘) 결과는 없다."""
    if parse_error is None:
        return isinstance(bbox_px_parsed, Mapping) and isinstance(bbox_px_parsed.get("defects"), list)
    return parse_error in PARSE_ERRORS and bbox_px_parsed is None


def _verdicts() -> frozenset[str]:
    from typing import get_args

    from evaluation.schema import Verdict

    return frozenset(get_args(Verdict))


def conformance_tuple(text: str, *, known_iso_codes: Iterable[str], scoring_iso_codes: Sequence[str]) -> tuple:
    """적합성 벡터가 맞대는 일곱 값 `(outcome, kept, dropped, verdict, cited, clause_error, reason)`.

    `kept` 는 공통 폐기 정책을 지난 항목 — 역변환 전 좌표다. `dropped` 는 좌표가 깨진 항목과 정책이 버린 항목의 합이다.
    조항 축은 판정이 어휘 안인지 · 인용이 문자열 목록인지만 보고 값을 고치지 않는다.
    """
    from evaluation.policy import filter_defect_items

    p = parse_generation(text)
    if p.parse_error is not None:
        return ("record_fail", (), 0, None, (), (), p.parse_error)
    filt = filter_defect_items([{"iso_code": c, "bbox_px": list(b)} for c, *b in p.items],
                               known_codes=set(known_iso_codes), scoring_codes=list(scoring_iso_codes))
    errors = []
    verdict = p.verdict if isinstance(p.verdict, str) and p.verdict in _verdicts() else None
    if verdict is None:
        errors.append("verdict_invalid")
    cited = p.cited if isinstance(p.cited, list) and all(isinstance(c, str) for c in p.cited) else None
    if cited is None:
        errors.append("citations_invalid")
        cited = []
    kept = tuple((d["iso_code"], *d["bbox_px"]) for d in filt.kept)
    return ("ok", kept, p.n_bad + filt.n_dropped, verdict, tuple(cited), tuple(sorted(errors)), None)
