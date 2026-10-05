"""평가 목록의 정규형 — 07번 미니스펙 §14-1 자.

목록은 계약의 일부다. 쓰는 쪽과 읽는 쪽이 각자 읽으면 BOM·CRLF·끝 개행·공백 하나로 갈리고,
그 차이가 해시를 바꾸고, 바뀐 해시가 **정상 묶음을 거부한다**. 그래서 갈릴 수 있는 것을
하나씩 세워 둔다.

두 해시를 따로 거는 이유도 여기 있다. 파일 해시만 두면 "같은 집합을 다른 순서로 돌린 것"과
"다른 집합"이 구분되지 않는다.
"""

from __future__ import annotations

import hashlib

import pytest

from evaluation.eval_list import (
    MAX_ID_LEN,
    EvalList,
    canonical_bytes,
    set_digest,
    validate_eval_list,
)
from evaluation.reject_v14 import BundleRejected, RejectCode, Stage

GOOD = b"aihub000003\naihub000001\naihub000002\n"


def reason_of(exc: BundleRejected) -> str:
    assert exc.codes() == {RejectCode.LIST_NOT_CANONICAL}
    return exc.rejections[0].detail["reason"]


def rejects(raw: bytes) -> BundleRejected:
    with pytest.raises(BundleRejected) as exc:
        validate_eval_list(raw)
    return exc.value


# ---------------------------------------------------------------- 정상

def test_순서를_그대로_둔다() -> None:
    """**정렬하지 않는다** — 배치 구획이 목록 순서를 따르므로 순서 자체가 계약이다."""
    got = validate_eval_list(GOOD)
    assert got.ids == ("aihub000003", "aihub000001", "aihub000002")
    assert len(got) == 3


def test_파일_해시는_받은_바이트_그대로다() -> None:
    assert validate_eval_list(GOOD).file_sha256 == hashlib.sha256(GOOD).hexdigest()


def test_집합_해시는_순서에_흔들리지_않는다() -> None:
    a = validate_eval_list(b"x\ny\nz\n")
    b = validate_eval_list(b"z\nx\ny\n")
    assert a.set_sha256 == b.set_sha256
    assert a.file_sha256 != b.file_sha256, "순서가 다르면 파일 해시는 달라야 한다"


def test_집합_해시는_끝_개행을_붙이지_않는다() -> None:
    """파일 해시와 같은 값이 나오면 둘을 섞어 써도 드러나지 않는다."""
    ids = ["x", "y"]
    assert set_digest(ids) != hashlib.sha256(b"x\ny\n").hexdigest()
    assert set_digest(ids) == hashlib.sha256(b"x\ny").hexdigest()


def test_집합_해시는_중복을_접는다() -> None:
    assert set_digest(["a", "b", "a"]) == set_digest(["b", "a"])


def test_id_집합을_준다() -> None:
    assert validate_eval_list(GOOD).id_set == {"aihub000001", "aihub000002", "aihub000003"}


def test_상한_길이까지는_받는다() -> None:
    long_id = "a" * MAX_ID_LEN
    assert validate_eval_list(long_id.encode() + b"\n").ids == (long_id,)


# ---------------------------------------------------------------- 거부

def test_BOM() -> None:
    assert reason_of(rejects(b"\xef\xbb\xbfa\n")) == "BOM"


def test_CR_은_줄_안에_있어도_걸린다() -> None:
    assert reason_of(rejects(b"a\r\nb\n")) == "CR"
    assert reason_of(rejects(b"a\rb\n")) == "CR"


def test_빈_파일() -> None:
    assert reason_of(rejects(b"")) == "빈 파일"


def test_끝_개행이_없으면_쓰다_만_파일로_본다() -> None:
    assert reason_of(rejects(b"a\nb")) == "끝 개행 없음"


def test_UTF8_이_아니면_어디서_깨졌는지_말한다() -> None:
    exc = rejects(b"a\n\xff\xfe\n")
    assert reason_of(exc) == "UTF-8 아님"
    assert "at" in exc.rejections[0].detail


def test_빈_줄() -> None:
    exc = rejects(b"a\n\nb\n")
    assert reason_of(exc) == "빈 줄"
    assert exc.rejections[0].detail["line"] == 2


def test_개행만_있는_파일도_빈_줄이다() -> None:
    assert reason_of(rejects(b"\n")) == "빈 줄"


def test_앞뒤_공백() -> None:
    assert reason_of(rejects(b"a\n b\n")) == "앞뒤 공백"
    assert reason_of(rejects(b"a\nb \n")) == "앞뒤 공백"
    assert reason_of(rejects(b"a\n\tb\n")) == "앞뒤 공백"


def test_너무_긴_id_는_목록_파일이_아니라고_본다() -> None:
    exc = rejects(("a" * (MAX_ID_LEN + 1)).encode() + b"\n")
    assert reason_of(exc) == "id 가 너무 길다"
    assert exc.rejections[0].detail["length"] == MAX_ID_LEN + 1


def test_중복은_줄_번호와_함께_거부한다() -> None:
    exc = rejects(b"a\nb\na\n")
    assert reason_of(exc) == "중복"
    assert exc.rejections[0].detail["line"] == 3


def test_거부는_내용_단계다() -> None:
    assert rejects(b"").rejections[0].stage is Stage.CONTENT


def test_어디서_걸렸는지를_호출부가_정한다() -> None:
    with pytest.raises(BundleRejected) as exc:
        validate_eval_list(b"", where="subsample_list")
    assert exc.value.rejections[0].where == "subsample_list"


# ---------------------------------------------------------------- 정규형 만들기

def test_만든_바이트는_그대로_다시_읽힌다() -> None:
    ids = ["c", "a", "b"]
    got = validate_eval_list(canonical_bytes(ids))
    assert list(got.ids) == ids, "만들 때도 정렬하지 않는다"


def test_중복이_있으면_만들지_않는다() -> None:
    with pytest.raises(ValueError, match="중복"):
        canonical_bytes(["a", "a"])


@pytest.mark.parametrize("bad", ["", " a", "a ", "a\nb", "a\rb"])
def test_정규형에_맞지_않는_id_는_만들지_않는다(bad: str) -> None:
    with pytest.raises(ValueError, match="정규형"):
        canonical_bytes(["ok", bad])


# ---------------------------------------------------------------- 만드는 쪽과 검사하는 쪽이 같은 규칙인가
#
# 앞 판은 만드는 함수가 낸 바이트를 검사기가 거부하는 경우가 셋 있었다(빈 목록 · 상한을 넘는 id ·
# BOM 으로 시작하는 첫 id). 평범한 id 로만 왕복을 시험해서 드러나지 않았다.

BOM = chr(0xFEFF)
LINE_SEP = chr(0x2028)


def naive_bytes(ids) -> bytes:
    """규칙을 하나도 보지 않고 잇기만 한 바이트 — 검사기에 그대로 넣어 본다."""
    return ("\n".join(ids) + "\n").encode("utf-8")


ADVERSARIAL = [
    ["ok"],
    ["c", "a", "b"],
    ["a" * MAX_ID_LEN],
    ["a", BOM + "b"],               # 첫 줄이 아닌 자리의 U+FEFF 는 파일의 BOM 이 아니다
    ["a" + LINE_SEP + "b"],         # 줄 구분 문자로 쪼개지지 않는다
    [],                             # 빈 목록
    ["a" * (MAX_ID_LEN + 1)],       # 상한 초과
    [BOM + "a", "b"],               # 파일이 BOM 으로 시작하게 된다
    ["a", ""],
    [" a"],
    ["a "],
    ["a", "\tb"],
    ["a", "a"],
    ["a", "b", "a"],
]


@pytest.mark.parametrize("ids", ADVERSARIAL)
def test_만드는_쪽과_검사하는_쪽의_판정이_같다(ids) -> None:
    """한쪽만 받는 목록이 있으면 만든 목록이 검사에서 떨어진다 — 정상 묶음의 거부로 이어진다."""
    try:
        made = canonical_bytes(ids)
    except ValueError:
        made = None
    try:
        validate_eval_list(naive_bytes(ids))
        accepted = True
    except BundleRejected:
        accepted = False
    assert (made is not None) == accepted
    if made is not None:
        assert made == naive_bytes(ids)
        assert list(validate_eval_list(made).ids) == ids


def test_빈_목록은_만들지_않는다() -> None:
    with pytest.raises(ValueError, match="빈 목록"):
        canonical_bytes([])


def test_상한을_넘는_id_는_만들지_않는다() -> None:
    with pytest.raises(ValueError, match="너무 길다"):
        canonical_bytes(["a" * (MAX_ID_LEN + 1)])


def test_BOM_으로_시작하는_첫_id_는_만들지_않는다() -> None:
    with pytest.raises(ValueError, match="BOM"):
        canonical_bytes([BOM + "a"])


def test_만들지_못한_까닭에_id_값을_싣지_않는다() -> None:
    """식별자가 오류 문구를 타고 보고서로 새어 나가지 않게 — 몇 번째인지만 말한다."""
    with pytest.raises(ValueError) as exc:
        canonical_bytes(["aihub000001", " aihub000002"])
    assert "aihub" not in str(exc.value)
    assert "2번째" in str(exc.value)


# ---------------------------------------------------------------- 검사 없이 만들 수 없다

def test_해시를_지어내서는_목록_객체를_만들_수_없다() -> None:
    """묶음 검증은 이 객체의 해시를 곁 파일과 맞댄다. 아무 값이나 넣을 수 있으면 그 대조가 빈말이 된다."""
    good = validate_eval_list(GOOD)
    with pytest.raises(ValueError, match="file_sha256"):
        EvalList(ids=good.ids, file_sha256="0" * 64, set_sha256=good.set_sha256)
    with pytest.raises(ValueError, match="set_sha256"):
        EvalList(ids=good.ids, file_sha256=good.file_sha256, set_sha256="0" * 64)


@pytest.mark.parametrize("bad_ids", [(), ("a", "a"), ("a", " b"), ("a" * (MAX_ID_LEN + 1),)])
def test_정규형이_아닌_id_로는_목록_객체를_만들_수_없다(bad_ids) -> None:
    good = validate_eval_list(GOOD)
    with pytest.raises(ValueError):
        EvalList(ids=bad_ids, file_sha256=good.file_sha256, set_sha256=good.set_sha256)


def test_ids_가_튜플이_아니면_만들지_않는다() -> None:
    good = validate_eval_list(GOOD)
    with pytest.raises(ValueError, match="튜플"):
        EvalList(ids=list(good.ids), file_sha256=good.file_sha256,    # type: ignore[arg-type]
                 set_sha256=good.set_sha256)


def test_검사기가_낸_값으로는_그대로_다시_만들어진다() -> None:
    good = validate_eval_list(GOOD)
    again = EvalList(ids=good.ids, file_sha256=good.file_sha256, set_sha256=good.set_sha256)
    assert again == good
