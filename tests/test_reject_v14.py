"""거부 사유 표 — 07번 미니스펙 §13-3 사 · §16-2.

이 표가 있는 이유는 "거부됐다"만 보는 시험을 막기 위해서다. 다른 이유로 거부돼도 통과하는
시험은 조건이 사라진 것을 못 잡는다. 그래서 여기서 표 자체의 성질을 건다 —
**모든 코드에 단계가 있고, 코드 값이 겹치지 않고, 면제 목록이 실재하는 코드를 가리킨다.**

거부는 들고 다니는 정보가 반이다. 어느 조건이 어디서 걸렸는지 없으면 고칠 수가 없고,
`where` 에 식별자 값을 넣으면 보고서로 새어 나간다.
"""

from __future__ import annotations

import pytest

from evaluation.reject_v14 import (
    ECHO_EXEMPT,
    STAGE_OF,
    BundleRejected,
    RejectCode,
    Rejection,
    Stage,
)

# ---------------------------------------------------------------- 표 자체

def test_모든_코드에_단계가_있다() -> None:
    assert set(STAGE_OF) == set(RejectCode)


# 반대 방향(표에 지워진 코드가 남는 경우)은 시험으로 걸 수 없다. 표의 키가 `RejectCode` 멤버로만
# 적히므로, 지운 코드가 표에 남으면 **모듈을 읽는 순간** `AttributeError` 로 죽는다. 위의 같음
# 단언이 나머지를 덮는다. 앞 판에 있던 `<=` 단언은 늘 참이라 지웠다.


def test_코드_값이_겹치지_않는다() -> None:
    """값은 산출물에 그대로 나간다. 겹치면 과거 보고와 이어지지 않는다."""
    values = [c.value for c in RejectCode]
    assert len(set(values)) == len(values)


def test_코드_값은_이름을_소문자로_옮긴_것이다() -> None:
    for code in RejectCode:
        assert code.value == code.name.lower(), f"{code.name} 의 값이 이름과 다르다"


def test_다섯_단계가_모두_쓰인다() -> None:
    assert {s for s in STAGE_OF.values()} == set(Stage)


def test_에코_전용_검사는_면제되지_않는다() -> None:
    """면제하면 에코 묶음에 아무 검사도 남지 않는다."""
    for code in (RejectCode.ECHO_LIST_MISMATCH, RejectCode.ECHO_ADAPTER_PRESENT,
                 RejectCode.MODE_MISMATCH):
        assert code not in ECHO_EXEMPT


def test_면제는_어댑터와_모집단에_한정된다() -> None:
    """에코는 모델을 부르지 않고 val 이미지를 쓴다. 그 두 가지 말고는 면제할 이유가 없다."""
    assert ECHO_EXEMPT == {
        RejectCode.ADAPTER_DUPLICATE, RejectCode.ADAPTER_NOT_FINAL,
        RejectCode.ARTIFACT_META_MISSING, RejectCode.LEDGER_MISMATCH,
        RejectCode.POPULATION_NOT_SUBSET, RejectCode.POPULATION_GROUP_INCOMPLETE,
    }


@pytest.mark.parametrize(("code", "stage"), [
    (RejectCode.SIDECAR_MISSING, Stage.EXISTS),
    (RejectCode.CR_IN_FILE, Stage.BYTES),
    (RejectCode.LIST_NOT_CANONICAL, Stage.CONTENT),
    (RejectCode.ADAPTER_NOT_FINAL, Stage.REGISTRATION),
    (RejectCode.LITERAL_MATCH_FAILED, Stage.CANARY),
])
def test_단계_배정_표본(code: RejectCode, stage: Stage) -> None:
    assert STAGE_OF[code] is stage


# "앞 단계가 실패하면 뒤를 보지 않는다" 는 표의 성질이 아니라 **검증의 거동**이다. 글자 순서를
# 보는 단언은 검증이 그 순서로 도는지와 이어지지 않아 지웠다. 그 거동은 묶음 검증의 시험이
# 실제 묶음으로 건다(`tests/test_bundle_v14.py` 의 `test_*_단계가_걸리면_뒤를_보지_않는다`).


# ---------------------------------------------------------------- 거부 한 건

def test_거부는_단계를_스스로_안다() -> None:
    assert Rejection(RejectCode.BOM_IN_FILE, "generations.jsonl").stage is Stage.BYTES


def test_사전으로_바꾸면_코드와_단계가_값으로_나간다() -> None:
    got = Rejection(RejectCode.TORN_TAIL, "generations.jsonl", {"line": 7}).as_dict()
    assert got == {"code": "torn_tail", "stage": "B", "where": "generations.jsonl",
                   "detail": {"line": 7}}


def test_세부가_없으면_열_자체를_넣지_않는다() -> None:
    assert "detail" not in Rejection(RejectCode.TORN_TAIL, "f").as_dict()


def test_문자열에_단계와_코드와_자리가_다_있다() -> None:
    s = str(Rejection(RejectCode.BLANK_LINE, "generations.jsonl:12", {"n": 1}))
    assert "[B/blank_line]" in s
    assert "generations.jsonl:12" in s
    assert "{'n': 1}" in s


def test_세부는_기본이_빈_사전이고_공유되지_않는다() -> None:
    a, b = Rejection(RejectCode.TORN_TAIL, "f"), Rejection(RejectCode.TORN_TAIL, "g")
    a.detail["x"] = 1
    assert b.detail == {}, "기본값을 공유하면 한 거부의 세부가 다른 거부에 새어 든다"


# ---------------------------------------------------------------- 묶음 거부

def test_거부_목록을_통째로_들고_있다() -> None:
    """첫 건만 알려 주면 고치고 다시 돌리기를 되풀이하게 된다."""
    exc = BundleRejected([Rejection(RejectCode.CR_IN_FILE, "a"),
                          Rejection(RejectCode.BLANK_LINE, "b", {"line": 3})])
    assert exc.codes() == {RejectCode.CR_IN_FILE, RejectCode.BLANK_LINE}
    assert len(exc.rejections) == 2


def test_메시지에_모든_사유가_들어간다() -> None:
    exc = BundleRejected([Rejection(RejectCode.CR_IN_FILE, "a"),
                          Rejection(RejectCode.NOT_UTF8, "b")])
    assert "cr_in_file" in str(exc)
    assert "not_utf8" in str(exc)


def test_사유가_비면_그렇다고_말한다() -> None:
    """빈 거부는 조용한 통과와 구분되지 않는다."""
    assert "거부 사유가 비어 있다" in str(BundleRejected([]))


def test_준_목록을_복사해_들고_있다() -> None:
    src = [Rejection(RejectCode.CR_IN_FILE, "a")]
    exc = BundleRejected(src)
    src.clear()
    assert len(exc.rejections) == 1
