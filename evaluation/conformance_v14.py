"""생성문 파싱의 **적합성 벡터** — 내보내기 쪽 파서와 채점 쪽 참조 파서가 같은 파일을 읽는다.
07번 미니스펙 §14-1 아 · §13-3 다.

## 왜 벡터를 공유하나

문자 일치 카나리아가 어긋나는 원인은 거의 언제나 **두 파서의 관대함 차이**다. 한쪽은 지수 표기를 받고
다른 쪽은 안 받는다든지, 폐기한 항목의 자리를 한쪽만 당긴다든지.

그 차이가 **내보내기 뒤의 파일 거부**로 드러나면 복구 경로가 없다 — greedy 는 다시 돌려도 같은 글을 내므로
재추론은 무의미하고, 남는 유인은 곁 파일을 손으로 고치는 것뿐이다.
그래서 차이가 **내보내기 전의 시험 실패**로 드러나야 한다. 두 파서가 이 벡터를 통과해야 본 내보내기를 시작한다.

## 벡터가 정하는 것

각 항목은 생성문 한 줄과 **그것을 읽은 결과의 기대**다.

| | 무엇 |
|---|---|
| `outcome` | `ok`(레코드가 선다) · `record_fail`(그 사유로 레코드 실패) |
| `reason` | `record_fail` 의 사유(1.4 어휘). `ok` 면 null. **사유별 실패율이 이 값에 기댄다** |
| `kept` | 유지되는 결함 항목 — `(iso_code, x1, y1, x2, y2)` 를 **원시 순서대로**, 모델이 낸 좌표 그대로(역변환 전) |
| `dropped` | 폐기한 항목 수 — 좌표가 깨져 버린 것과 공통 정책이 버린 것(미지 코드·채점 밖 코드)의 **합** |
| `verdict` · `cited` · `clause_error` | 조항 축. 판정·인용을 **고치지 않고** 넘긴 값과 위반 표시 |

**유지 목록을 순서까지 본다.** 박스만 맞대면 코드와 박스가 폐기 뒤에 한 칸 밀려 묶이는 버그를 못 본다 —
클래스 인식 지표만 조용히 무너지고 위치 지표는 멀쩡해서 눈에 띄지 않는다.

**기대는 모두 명시한다.** 앞 판은 생성문 도우미가 판정 `합격`·인용 `A-1` 을 기본으로 넣는데 기대의 기본값은
빈 판정·빈 인용이어서, 기대를 적지 않은 `ok` 사례 스물넷이 입력과 모순이었다. 그 벡터를 실제 파서에 태운
시험이 없어서 드러나지 않았다. 이 판은 `ok` 사례를 만드는 도우미가 입력과 기대를 **같은 값에서** 만들고,
시험이 벡터 전량을 참조 파서에 태운다(`tests/test_conformance_v14.py`).

## 판정이 필요한 자리 하나

좌표가 문자열(`"100"`)이거나 불리언인 항목, 코드가 수인 항목은 **공통 폐기 정책이 받아 준다**(`evaluation/policy.py` 가
`float(v)` 와 `str(code)` 로 읽는다). 프롬프트는 정수 좌표와 문자열 코드를 요구하므로 이것은 구제 파싱에 가깝지만,
그 정책은 다섯 칸이 같은 규칙으로 버리게 하려고 둔 것이고 v3 가 고정한 파일이다. **지금 판은 현행 동작을 사실로 적는다** —
엄격화는 그 파일을 건드리는 일이라 결정을 받아야 한다. 아래 `LENIENT_BY_SHARED_POLICY` 가 그 항목들이다.

## 코드는 사상표에서 읽는다

벡터에 쓰는 결함 코드는 모듈에 적지 않고 사상표와 채점 클래스 목록에서 읽는다(불변조건 1-8). 그래서 사상표가 바뀌면
**코드 수정 없이 벡터의 바이트가 바뀐다.** 판 번호는 손으로 올리므로, 벡터 바이트의 지문을 `VECTOR_SHA256` 에 박고
시험이 맞댄다 — 바이트가 바뀌었는데 판을 올리지 않으면 시험이 떨어진다.
필요한 코드(채점 밖이지만 사상표에 있는 코드, 사상표에 없는 코드)를 만들 수 없으면 **떨어지지 않고 멈춘다.**
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from data.label_map import load_label_map
from evaluation.params import CLASS_NAMES

VECTOR_VERSION = "2"
"""벡터 판. 항목을 더하거나 기대를 바꾸면 올리고, 쓰는 쪽과 읽는 쪽이 같은 판을 쓴다.

판 2(2026-10-01): 입력과 모순이던 기대 스물넷을 고쳤다 · `코드가 수` 의 기대를 현행 정책대로 고쳤다 ·
사유(`reason`)를 맞대는 값에 넣었다."""

VECTOR_SHA256 = "6b857b8e33b9fee33fec13517763f9890652d1eec8674c8cdc1498d42b8566c2"
"""`as_jsonl()` 바이트의 sha256. 판을 올릴 때 함께 새로 적는다 — 시험이 이 값과 맞댄다."""

OK = "ok"
RECORD_FAIL = "record_fail"


@dataclass(frozen=True)
class Case:
    """벡터 한 줄."""

    name: str
    text: str
    """모델이 냈다고 가정하는 생성문."""
    outcome: str
    kept: tuple[tuple, ...] = ()
    dropped: int = 0
    reason: str | None = None
    """`record_fail` 일 때의 사유(1.4 어휘). `ok` 면 `None`."""
    clause_error: tuple[str, ...] = ()
    verdict: str | None = None
    cited: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in (OK, RECORD_FAIL):
            raise ValueError(f"{self.name}: outcome 이 {self.outcome!r} 다")
        if (self.outcome == RECORD_FAIL) != (self.reason is not None):
            raise ValueError(f"{self.name}: 레코드 실패에만 사유가 있고, 레코드 실패에는 사유가 있어야 한다")
        if self.outcome == RECORD_FAIL and (self.kept or self.dropped or self.verdict is not None
                                           or self.cited or self.clause_error):
            raise ValueError(f"{self.name}: 레코드 실패는 빈 예측이다")

    def expected(self) -> tuple:
        """`check_case` 가 맞대는 일곱 값 — `(outcome, kept, dropped, verdict, cited, clause_error, reason)`."""
        return (self.outcome, tuple(self.kept), self.dropped, self.verdict,
                tuple(self.cited), tuple(self.clause_error), self.reason)

    def as_dict(self) -> dict:
        return {"name": self.name, "text": self.text, "outcome": self.outcome,
                "kept": [list(k) for k in self.kept], "dropped": self.dropped,
                "reason": self.reason, "clause_error": list(self.clause_error),
                "verdict": self.verdict, "cited": list(self.cited), "note": self.note}


def _gen(defects: str, *, verdict: str = '"합격"', cited: str = '["A-1"]', before: str = "",
         after: str = "") -> str:
    return f'{before}{{"defects": {defects}, "verdict": {verdict}, "cited_clauses": {cited}}}{after}'


_DEFAULT_VERDICT = "합격"
_DEFAULT_CITED = ("A-1",)


def _ok(name: str, defects: str, kept: tuple = (), dropped: int = 0, *, before: str = "",
        after: str = "", note: str = "") -> Case:
    """판정·인용이 멀쩡한 `ok` 사례. **입력과 기대를 같은 값에서** 만든다 — 앞 판의 모순이 이 자리였다."""
    return Case(name, _gen(defects, before=before, after=after), OK, kept, dropped,
                verdict=_DEFAULT_VERDICT, cited=_DEFAULT_CITED, note=note)


_LM = load_label_map()
KNOWN_CODES: frozenset[str] = _LM.iso_codes(include_alt=True)
"""사상표 전체 코드(별칭 포함). 참조 파서가 공통 정책에 넘기는 값이다."""
SCORING_CODES: tuple[str, ...] = tuple(_LM.iso_code(n) for n in CLASS_NAMES)
"""채점 4클래스. 채점기(`evaluation.cells`)와 같은 규칙으로 만든다 — 클래스 이름 목록을 사상표로 옮긴 것."""

PRIMARY = _LM.iso_code("porosity")
"""벡터가 쓰는 대표 코드. **모듈에 코드 문자열을 적지 않는다**(불변조건 1-8) — 사상표에서 읽는다."""
SECOND = _LM.iso_code("slag_inclusion")
_OUT_OF_SCOPE = sorted(KNOWN_CODES - set(SCORING_CODES))
if not _OUT_OF_SCOPE:
    raise RuntimeError("사상표에 채점 밖 코드가 없다 — `채점 밖 코드` 사례를 만들 수 없다. 벡터를 다시 설계한다")
ALT = _OUT_OF_SCOPE[0]
"""사상표에 있으나 채점 4클래스 밖인 코드. **없으면 멈춘다** — 대표 코드로 떨어지면 그 사례가 틀린 채 선다."""
UNKNOWN_CODE = "9999"
"""사상표에 **없는** 코드. 환각 신호를 흉내 내는 자리라 값 자체가 라벨이 아니다."""
if UNKNOWN_CODE in KNOWN_CODES:
    raise RuntimeError("`UNKNOWN_CODE` 가 사상표에 있다 — `미지 코드` 사례가 틀린 채 선다. 다른 값을 고른다")

B = f'{{"iso_code": "{PRIMARY}", "bbox_2d": [10, 20, 40, 55]}}'
KEPT_B = (PRIMARY, 10.0, 20.0, 40.0, 55.0)

CASES: tuple[Case, ...] = (
    # ── 읽힌다 ────────────────────────────────────────────────────────────
    _ok("정상", f"[{B}]", (KEPT_B,)),
    _ok("빈 결함", "[]", note="정상 이미지. 빈 예측은 레코드 실패가 아니다"),
    _ok("지수 표기", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [1e1, 2e1, 4e1, 5.5e1]}}]', (KEPT_B,),
        note="1e1 은 10.0 이다 — 자료형이 아니라 값으로 읽는다"),
    _ok("정수와 실수 표기", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [10.0, 20, 40.0, 55]}}]', (KEPT_B,)),
    _ok("음수 좌표", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [-5, 20, 40, 55]}}]',
        ((PRIMARY, -5.0, 20.0, 40.0, 55.0),),
        note="경계 밖은 폐기하지 않는다 — 클리핑은 지표를 올리는 쪽으로만 작동한다"),
    _ok("경계 밖 좌표", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [10, 20, 9999, 55]}}]',
        ((PRIMARY, 10.0, 20.0, 9999.0, 55.0),), note="세기만 하고 버리지 않는다"),
    _ok("앞에 산문", f"[{B}]", (KEPT_B,), before="다음과 같이 판단합니다.\n",
        note="첫 여는 중괄호부터 읽는다"),
    _ok("코드 펜스", f"[{B}]", (KEPT_B,), before="```json\n", after="\n```"),
    _ok("뒤에 잔여", f"[{B}]", (KEPT_B,), after=' 그리고 추가 설명입니다. {"another": 1}',
        note="첫 객체만 읽는다"),
    Case("문자열 속 중괄호", _gen(f"[{B}]", cited='["A-{1}"]'), OK, (KEPT_B,), verdict="합격",
         cited=("A-{1}",), note="중괄호를 세는 방식이면 여기서 깨진다"),
    Case("중복 키", '{"defects": [], "defects": ' + f"[{B}]" + ', "verdict": "합격", "cited_clauses": []}',
         OK, (KEPT_B,), verdict="합격", note="뒤가 이긴다 — 표준 JSON 의 거동"),
    _ok("반올림 경계", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [100.0625, 0.0005, 200, 719.9995]}}]',
        ((PRIMARY, 100.0625, 0.0005, 200.0, 719.9995),),
        note="모델이 낸 값 그대로다. 상류는 역변환 뒤 round(v,3) 을 한다 — 참조 대조의 허용오차가 여기서 정해진다"),
    _ok("결함 여럿", f'[{B}, {{"iso_code": "{SECOND}", "bbox_2d": [1, 2, 3, 4]}}]',
        (KEPT_B, (SECOND, 1.0, 2.0, 3.0, 4.0)), note="원시 순서 그대로"),

    # ── 항목만 폐기 ───────────────────────────────────────────────────────
    _ok("좌표 셋", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [1, 2, 3]}}]', (KEPT_B,), 1),
    _ok("좌표 다섯", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [1, 2, 3, 4, 5]}}]', (KEPT_B,), 1),
    _ok("퇴화 상자", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [10, 20, 10, 55]}}]', (KEPT_B,), 1,
        note="x1 >= x2"),
    _ok("NaN", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [NaN, 2, 3, 4]}}]', (KEPT_B,), 1),
    _ok("무한대", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [Infinity, 2, 3, 4]}}]', (KEPT_B,), 1),
    _ok("1e400", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [1e400, 2, 3, 4]}}]', (KEPT_B,), 1,
        note="파이썬이 inf 로 읽는다"),
    _ok("중첩 목록", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox_2d": [[1, 2], [3, 4]]}}]', (KEPT_B,), 1),
    _ok("키 이름 다름", f'[{B}, {{"iso_code": "{PRIMARY}", "bbox": [1, 2, 3, 4]}}]', (KEPT_B,), 1,
        note="bbox_2d 가 아니면 좌표가 없는 것이다"),
    _ok("항목이 객체가 아님", f'[{B}, "기공"]', (KEPT_B,), 1),
    _ok("미지 코드", f'[{B}, {{"iso_code": "{UNKNOWN_CODE}", "bbox_2d": [1, 2, 3, 4]}}]', (KEPT_B,), 1,
        note="사상표 밖 — 환각 신호로 따로 센다"),
    _ok("채점 밖 코드", f'[{B}, {{"iso_code": "{ALT}", "bbox_2d": [1, 2, 3, 4]}}]', (KEPT_B,), 1,
        note="사상표에는 있으나 채점 4클래스 밖 — 버리되 센다"),
    _ok("폐기가 순서를 당기지 않는다",
        f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [1, 2, 3]}}, {B}, {{"iso_code": "{SECOND}", "bbox_2d": [1, 2, 3, 4]}}]',
        (KEPT_B, (SECOND, 1.0, 2.0, 3.0, 4.0)), 1,
        note="가운데가 아니라 앞이 폐기돼도 남는 둘의 코드·박스 짝이 유지되는가"),

    # ── 레코드 실패 ───────────────────────────────────────────────────────
    Case("JSON 없음", "결함이 없습니다.", RECORD_FAIL, reason="no_json"),
    Case("잘림", f'{{"defects": [{{"iso_code": "{PRIMARY}", "bbox_2d": [10, 20,', RECORD_FAIL, reason="truncated"),
    Case("문자열이 안 닫힘", '{"defects": [], "verdict": "합격', RECORD_FAIL, reason="truncated"),
    Case("해독 불가", '{"defects": [,]}', RECORD_FAIL, reason="json_decode"),
    Case("defects 가 목록이 아님", '{"defects": "없음", "verdict": "합격", "cited_clauses": []}',
         RECORD_FAIL, reason="schema_violation"),
    Case("깊은 중첩", '{"defects":' + "[" * 3000, RECORD_FAIL, reason="json_decode",
         note="RecursionError 다. 파서가 죽으면 그 이미지를 영원히 못 넘긴다"),
    Case("수천 자리 정수", f'{{"defects": [{{"iso_code": "{PRIMARY}", "bbox_2d": [' + "9" * 5000 + ', 1, 2, 3]}}]}}',
         RECORD_FAIL, reason="json_decode", note="ValueError 다 — JSONDecodeError 가 아니다"),

    # ── 조항 축만 실패 (박스는 채점) ──────────────────────────────────────
    Case("판정이 어휘 밖", _gen(f"[{B}]", verdict='"통과"'), OK, (KEPT_B,),
         clause_error=("verdict_invalid",), verdict=None, cited=("A-1",),
         note="박스는 채점한다. 판정을 지어내지 않는다"),
    Case("판정에 공백", _gen(f"[{B}]", verdict='"합격 "'), OK, (KEPT_B,),
         clause_error=("verdict_invalid",), verdict=None, cited=("A-1",),
         note="정규화하지 않는다 — 고쳐 주면 모델이 하지 않은 말이 된다"),
    Case("판정이 목록", _gen(f"[{B}]", verdict='["합격"]'), OK, (KEPT_B,),
         clause_error=("verdict_invalid",), verdict=None, cited=("A-1",),
         note="`in` 검사가 TypeError 를 내는 자리다"),
    Case("판정이 수", _gen(f"[{B}]", verdict="1"), OK, (KEPT_B,),
         clause_error=("verdict_invalid",), verdict=None, cited=("A-1",)),
    Case("판정 키 누락", '{"defects": ' + f"[{B}]" + ', "cited_clauses": ["A-1"]}', OK, (KEPT_B,),
         clause_error=("verdict_invalid",), verdict=None, cited=("A-1",)),
    Case("인용이 문자열", _gen(f"[{B}]", cited='"A-1"'), OK, (KEPT_B,),
         clause_error=("citations_invalid",), verdict="합격",
         note="목록이 아니면 빈 목록으로 두고 위반으로 센다"),
    Case("인용에 수", _gen(f"[{B}]", cited="[1]"), OK, (KEPT_B,),
         clause_error=("citations_invalid",), verdict="합격"),
    Case("인용 키 누락", '{"defects": ' + f"[{B}]" + ', "verdict": "합격"}', OK, (KEPT_B,),
         clause_error=("citations_invalid",), verdict="합격",
         note="상류가 키 누락을 null 로 넘긴다 — 빈 목록을 낸 것과 다른 사실이다"),
    Case("둘 다 깨짐", '{"defects": ' + f"[{B}]" + ', "verdict": 1, "cited_clauses": 2}', OK, (KEPT_B,),
         clause_error=("citations_invalid", "verdict_invalid"), verdict=None,
         note="둘 다 적는다 — 단일 값이면 하나를 잃는다"),
    Case("인용이 빈 목록", _gen(f"[{B}]", cited="[]"), OK, (KEPT_B,), verdict="합격",
         note="위반이 아니다. 키 누락과 가르는 자리"),
)

LENIENT_BY_SHARED_POLICY: tuple[Case, ...] = (
    _ok("좌표가 문자열", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": ["10", "20", "40", "55"]}}]', (KEPT_B,),
        note="공통 폐기 정책이 float() 로 읽어 통과한다. 프롬프트는 정수를 요구하므로 구제 파싱에 가깝다 — 판정 대기"),
    _ok("좌표가 불리언", f'[{{"iso_code": "{PRIMARY}", "bbox_2d": [false, true, 40, 55]}}]',
        ((PRIMARY, 0.0, 1.0, 40.0, 55.0),),
        note="같은 이유로 통과한다 — 판정 대기"),
    _ok("코드가 수", f'[{{"iso_code": {PRIMARY}, "bbox_2d": [10, 20, 40, 55]}}]', (KEPT_B,),
        note="공통 정책이 str(iso_code) 로 문자열화한 뒤 사상표와 맞대므로 유지된다. "
             "프롬프트는 문자열 코드를 요구한다 — 판정 대기"),
)
"""**현행 동작을 사실로 적은 것**이지 규약으로 정한 것이 아니다.

이 셋은 공통 폐기 정책(`evaluation/policy.py`)의 관대함에서 나온다. 엄격화하려면 그 파일을 고쳐야 하고,
그 파일은 다섯 칸이 같은 규칙으로 버리게 하려고 둔 것이며 v3 산출물이 해시로 고정한다. 결정을 받은 뒤 이 표를 옮긴다.
"""


def all_cases(*, include_lenient: bool = True) -> tuple[Case, ...]:
    return CASES + (LENIENT_BY_SHARED_POLICY if include_lenient else ())


def as_jsonl(cases=None) -> str:
    """벡터를 파일로 내보낸다. 내보내기 쪽 시험이 이 파일을 읽는다 — 쓰는 쪽과 읽는 쪽이 같은 바이트를 본다."""
    rows = all_cases() if cases is None else cases
    head = json.dumps({"vector_version": VECTOR_VERSION, "n_cases": len(rows)}, ensure_ascii=False)
    return head + "\n" + "".join(json.dumps(c.as_dict(), ensure_ascii=False) + "\n" for c in rows)


def vector_sha256() -> str:
    """`as_jsonl()` 바이트(UTF-8)의 sha256. 등록의 `conformance_vector_sha256` 이 이 값이다."""
    return hashlib.sha256(as_jsonl().encode("utf-8")).hexdigest()


@dataclass
class CheckResult:
    """벡터 한 줄을 실제 파서에 태운 결과."""

    name: str
    passed: bool
    detail: dict = field(default_factory=dict)


def check_case(case: Case, *, parse) -> CheckResult:
    """`parse(text)` 가 돌려준 일곱 값을 기대와 맞댄다.

    `parse(text) -> (outcome, kept, dropped, verdict, cited, clause_error, reason)`.
    **사유까지 맞댄다** — 깊은 중첩을 `truncated` 로 분류하는 파서는 사유별 실패율을 틀리게 낸다.

    **파서를 인자로 받는다** — 내보내기 쪽 파서와 채점 쪽 참조 파서가 같은 함수를 통과해야 하기 때문이다.
    """
    got = tuple(parse(case.text))
    got = (got[0], tuple(tuple(k) for k in got[1]), *got[2:4], tuple(got[4]), tuple(got[5]), *got[6:])
    want = case.expected()
    ok = got == want
    return CheckResult(case.name, ok, {} if ok else {"want": want, "got": got})
