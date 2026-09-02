"""프롬프트 직렬화 회귀 방지.

골격의 내부 표현(enum, 개구간 +0.01 인코딩)이 프롬프트로 새면 그 문자열이 판정문에
그대로 박힌다. 파일럿 실측에서 기각 사유의 다수가 이것이었다.
"""

from __future__ import annotations

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
    """자료에 내부 표현이 새면 그 문자열이 생성문에 박힌다 (80번 B1: 채택 69건 중 49건).

    골격이 정본 생성기 산출로 바뀌었으므로 자료도 정본 표기(`clause_text`)를 지난다.
    """
    from corpus.generate import basis as B

    recs = M.build_clause_records(12, "full") + M.build_remedy_records(4)
    assert recs
    for rec in recs:
        p = M.prompt_generate(rec)
        assert "Unit." not in p, rec["sample_id"]
        assert "InspectionMethod." not in p, rec["sample_id"]
        assert "None" not in B.render_basis(rec), rec["sample_id"]
        # 개구간 인코딩은 표집 실측값과 모양이 같을 수 있어 자료 기준으로 본다
        from corpus.generate.numeric_lock import find_artifacts
        b = B.render_basis(rec)
        assert find_artifacts(b, basis=b) == (), rec["sample_id"]


def test_rejection_reason_is_recorded_or_marked():
    """사유 없는 기각은 감사가 안 된다. 되풀이도 사유가 아니다 (B17)."""
    cfg = M.load_config()
    src = "KRA27-T15 조항에 따르면 기공의 크기는 4 mm 이하로 제한된다"
    assert M.clean_reason("", src, cfg) == ("사유 미기재", True)
    assert M.clean_reason("NG", src, cfg) == ("사유 미기재", True)
    txt, echo = M.clean_reason("NG\n조항 번호가 자료와 다르다", src, cfg)
    assert not echo and txt.startswith("조항")
