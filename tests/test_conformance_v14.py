"""적합성 벡터를 **실제 파서에 태운다** — 07번 미니스펙 §14-1 아.

앞 판의 벡터는 한 번도 실행되지 않았다. 그래서 기대가 입력과 모순인 사례 스물넷이 그대로 서 있었다 —
그 벡터로는 계약대로 판정·인용을 넘기는 옳은 파서가 관문에서 막힌다. 여기서 지키는 것 넷.

1. **벡터 전량이 참조 파서를 지난다.** 지나지 못하는 사례는 벡터가 틀렸거나 파서가 틀렸다.
2. **대조에 이가 있다.** 사유·유지 목록·판정 가운데 하나만 틀린 파서도 떨어진다.
3. **벡터의 바이트가 바뀌면 판을 올린다.** 코드가 사상표에서 오므로 코드 수정 없이 바이트가 바뀔 수 있다.
4. **벡터가 스스로 모순인 사례를 만들 수 없다.**
"""

from __future__ import annotations

import json

import pytest

from evaluation.adapters_v14 import reference_parse
from evaluation.conformance_v14 import (
    CASES,
    KNOWN_CODES,
    LENIENT_BY_SHARED_POLICY,
    OK,
    RECORD_FAIL,
    SCORING_CODES,
    UNKNOWN_CODE,
    VECTOR_SHA256,
    VECTOR_VERSION,
    Case,
    all_cases,
    as_jsonl,
    check_case,
    vector_sha256,
)


def parse(text: str) -> tuple:
    return reference_parse(text, known_iso_codes=KNOWN_CODES,
                           scoring_iso_codes=SCORING_CODES).as_tuple()


# ---------------------------------------------------------------- 벡터 전량

@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_벡터가_참조_파서를_지난다(case: Case) -> None:
    got = check_case(case, parse=parse)
    assert got.passed, got.detail


def test_벡터의_수가_계약의_목록을_덮는다() -> None:
    """07번 §14-1 아 끝의 목록 — 죽이는 입력 둘과 판정 대기 셋이 빠지면 안 된다."""
    names = {c.name for c in all_cases()}
    for must in ("깊은 중첩", "수천 자리 정수", "좌표가 문자열", "좌표가 불리언", "코드가 수",
                 "지수 표기", "중복 키", "반올림 경계", "문자열 속 중괄호"):
        assert must in names
    assert len(CASES) + len(LENIENT_BY_SHARED_POLICY) == len(all_cases())


def test_레코드_실패_사유가_셋_다_나온다() -> None:
    """사유별 실패율의 칸이 전부 시험에 선다."""
    reasons = {c.reason for c in CASES if c.outcome == RECORD_FAIL}
    assert reasons == {"no_json", "truncated", "json_decode", "schema_violation"}


# ---------------------------------------------------------------- 대조에 이가 있는가

def _broken(fn):
    """참조 파서의 결과 하나를 비트는 파서."""
    def run(text: str) -> tuple:
        return fn(list(parse(text)))
    return run


def _by_name(name: str) -> Case:
    (c,) = [c for c in all_cases() if c.name == name]
    return c


def test_사유만_틀려도_떨어진다() -> None:
    """깊은 중첩을 `truncated` 로 분류하는 파서 — 앞 판의 대조는 이것을 통과시켰다."""
    def wrong(t):
        t[6] = "truncated"
        return tuple(t)
    assert not check_case(_by_name("깊은 중첩"), parse=_broken(wrong)).passed


def test_유지_목록의_짝이_밀리면_떨어진다() -> None:
    def shift(t):
        kept = t[1]
        codes = [k[0] for k in kept]
        t[1] = tuple((codes[(i + 1) % len(codes)], *k[1:]) for i, k in enumerate(kept))
        return tuple(t)
    assert not check_case(_by_name("폐기가 순서를 당기지 않는다"), parse=_broken(shift)).passed


def test_판정을_지어내면_떨어진다() -> None:
    def invent(t):
        t[3] = "판정불가"
        return tuple(t)
    assert not check_case(_by_name("판정이 어휘 밖"), parse=_broken(invent)).passed


def test_인용을_빈_목록으로_메우면_떨어진다() -> None:
    """키 누락과 빈 목록을 같게 보는 파서."""
    def fill(t):
        t[5] = ()
        return tuple(t)
    assert not check_case(_by_name("인용 키 누락"), parse=_broken(fill)).passed


def test_앞_판의_모순된_기대로는_옳은_파서가_떨어진다() -> None:
    """앞 판이 기대를 적지 않아 판정·인용이 비어 있던 꼴을 다시 만들어 본다 — 그 벡터가 틀렸다는 증거다."""
    c = _by_name("지수 표기")
    stale = Case(c.name, c.text, OK, c.kept, c.dropped)
    assert not check_case(stale, parse=parse).passed


# ---------------------------------------------------------------- 판과 바이트

def test_벡터_바이트가_바뀌면_판을_올린다() -> None:
    """사상표가 바뀌면 코드 수정 없이 벡터 바이트가 바뀐다. 이 시험이 떨어지면 판을 올리고 지문을 새로 적는다."""
    assert vector_sha256() == VECTOR_SHA256, (
        f"벡터 바이트가 판 {VECTOR_VERSION} 의 지문과 다르다 — VECTOR_VERSION 을 올리고 VECTOR_SHA256 을 새로 적는다")


def test_파일의_머리에_판과_수가_있다() -> None:
    head = json.loads(as_jsonl().split("\n", 1)[0])
    assert head == {"vector_version": VECTOR_VERSION, "n_cases": len(all_cases())}


def test_파일의_줄이_사례와_같은_순서다() -> None:
    rows = [json.loads(x) for x in as_jsonl().rstrip("\n").split("\n")[1:]]
    assert [r["name"] for r in rows] == [c.name for c in all_cases()]
    assert all("reason" in r for r in rows)


# ---------------------------------------------------------------- 스스로 모순인 사례

@pytest.mark.parametrize("kw", [
    {"outcome": RECORD_FAIL},                                   # 사유 없는 실패
    {"outcome": OK, "reason": "no_json"},                       # 사유 있는 통과
    {"outcome": RECORD_FAIL, "reason": "no_json", "dropped": 1},
    {"outcome": RECORD_FAIL, "reason": "no_json", "verdict": "합격"},
    {"outcome": "maybe"},
])
def test_모순된_사례는_만들_수_없다(kw) -> None:
    with pytest.raises(ValueError):
        Case("x", "t", **kw)


def test_미지_코드는_사상표에_없다() -> None:
    assert UNKNOWN_CODE not in KNOWN_CODES


def test_채점_밖_사례의_코드는_사상표에_있고_채점_밖이다() -> None:
    from evaluation.conformance_v14 import ALT
    assert ALT in KNOWN_CODES and ALT not in SCORING_CODES
