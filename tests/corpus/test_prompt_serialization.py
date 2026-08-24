"""프롬프트 직렬화 회귀 방지.

골격의 내부 표현(enum, 개구간 +0.01 인코딩)이 프롬프트로 새면 그 문자열이 판정문에
그대로 박힌다. 파일럿 실측에서 기각 사유의 다수가 이것이었다.
"""

from __future__ import annotations

import re

import pytest

from corpus.generate import run_cycle_corpus as M


def test_val_strips_enum_and_trailing_zeros():
    class E:
        value = "mm"

    assert M.val(E()) == "mm"
    assert M.val("4.00") == "4"
    assert M.val("0.25") == "0.25"
    assert M.val(None) == ""


def test_unit_and_method_render_in_korean_units():
    class U:
        value = "mm"

    class P:
        value = "percent"

    class RT:
        value = "RT"

    assert M.unit_ko(U()) == "mm"
    assert M.unit_ko(P()) == "%"
    assert "RT" in M.method_ko(RT())
    assert "InspectionMethod" not in M.method_ko(RT())


def test_open_interval_encoding_is_decoded():
    # CSV 는 원문 (10, 25] 를 [10.01, 25.01) 로 담는다. 사람이 읽는 형태로 되돌린다.
    assert M.num_ko("10.01") == "10"
    assert M.num_ko("25.01") == "25"
    assert M.num_ko("4.00") == "4"
    assert M.thickness_ko("10.01", "25.01") == "10 mm 초과 25 mm 이하"
    assert M.thickness_ko("0.00", "10.01") == "10 mm 이하"
    assert M.thickness_ko("0.00", None) == "모든 두께"


def test_prompt_has_no_internal_representation():
    sks = M.build_clause_skeletons(12)
    assert sks, "골격이 비었다"
    for sk in sks:
        p = M.prompt_clause(sk)
        assert "Unit." not in p, f"단위 enum 노출: {sk['rule_id']}"
        assert "InspectionMethod." not in p, f"검사 방식 enum 노출: {sk['rule_id']}"
        # 개구간 인코딩(정수+.01)이 그대로 노출되면 안 된다
        assert not re.search(r"\b\d+\.01\b", p), f"개구간 인코딩 노출: {sk['rule_id']}"


def test_rejection_reason_is_recorded_or_marked():
    # 사유 없는 기각은 감사가 안 된다. 빈 문자열이 아니라 표시가 남아야 한다.
    assert M.extract_reason("", False) == "사유 미기재"
    assert M.extract_reason("NG", False) == "사유 미기재"
    assert M.extract_reason("NG\n조항 번호가 자료와 다르다", False).startswith("조항")
    assert M.extract_reason("OK", True) == ""
