"""2차 자동 판독의 기록 계약.

여기서 지키는 것 넷.

1. **사람 검수로 읽힐 수 없다.** 표시를 끌 수 있으면 언젠가 꺼진다 — 상수로 두고 시험으로 건다.
2. **판독자가 기준을 정하지 않는다.** 어휘 밖 판정은 만들 수 없다.
3. **순서가 봉인된다.** 1차를 본 뒤 고친 판독으로는 일치율을 내지 못한다.
4. **일치율을 정답률로 부르지 않는다.** 같은 기준을 읽은 두 구현이라 오류가 상관될 수 있다.
"""

from __future__ import annotations

import pytest

from evaluation.second_read import (
    HUMAN_REVIEW,
    READ_TYPE,
    REQUIRED_LIMIT,
    Criteria,
    SealAnchor,
    NotSecondRead,
    Read,
    SealBroken,
    SecondRead,
    agreement,
    agreement_from_payload,
    contract_digest,
    load_pairs,
    rebuild,
    require_second_read,
)

V = ("duplicate", "not_duplicate", "undecidable")
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def anchor(seal: str, **over) -> SealAnchor:
    """봉인 값과 그것을 남긴 자리. 시험 안에서는 방금 계산한 봉인을 넣지만 **자리는 따로 적는다.**"""
    kw = {"sha256": seal, "commit": COMMIT, "path": "docs/reads/seal.md"}
    kw.update(over)
    return SealAnchor(**kw)


def crit(**over) -> Criteria:
    kw = {"criteria_id": "neardup_rules_v2", "criteria_sha256": "a" * 64,
          "allowed_verdicts": V, "source": "1차 판독자(A)"}
    kw.update(over)
    return Criteria(**kw)


def sr(verdicts=("duplicate", "not_duplicate", "undecidable"), **over) -> SecondRead:
    reads = tuple(Read(f"p{n}", v) for n, v in enumerate(verdicts, 1))
    return SecondRead(criteria=over.pop("criteria", crit()), reads=reads, **over)


# ---------------------------------------------------------------- 이름을 헐겁게 쓰지 않는다

def test_사람_검수가_아니라고_산출물에_박힌다() -> None:
    d = sr().as_dict()
    assert d["read_type"] == READ_TYPE == "automated_second_read"
    assert d["human_review"] is False
    assert "사람 검수가 아니다" in d["what_this_is_not"]


def test_사람_판독_미측정이_한계에_항상_있다() -> None:
    assert REQUIRED_LIMIT in sr().as_dict()["limits"]


def test_다른_한계를_줘도_그_줄은_남는다() -> None:
    d = sr(limits=("쌍이 60개다",)).as_dict()
    assert REQUIRED_LIMIT in d["limits"] and "쌍이 60개다" in d["limits"]


def test_사람_검수_표시를_켤_수_있는_자리가_없다() -> None:
    """필드면 언젠가 켜진다. 상수라 켤 자리가 없다."""
    assert HUMAN_REVIEW is False
    assert not any(f.name == "human_review" for f in SecondRead.__dataclass_fields__.values())


def test_일치율_산출물에도_같은_표시가_붙는다() -> None:
    s = sr()
    got = agreement(s, {"p1": "duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                    sealed=anchor(s.seal()))
    assert got["read_type"] == READ_TYPE and got["human_review"] is False
    assert REQUIRED_LIMIT in got["limits"]
    assert "정답률이 아니다" in got["comparison"]
    assert "상관될 수 있다" in got["caveat"]


# ---------------------------------------------------------------- 기준은 받는 것이다

def test_기준_밖_어휘로_판독할_수_없다() -> None:
    with pytest.raises(ValueError, match="기준 밖 어휘"):
        sr(verdicts=("duplicate", "아마도", "undecidable"))


def test_기준의_판이_없으면_거부한다() -> None:
    with pytest.raises(ValueError, match="기준의 id"):
        crit(criteria_id="")


def test_어휘가_하나면_판독이_아니다() -> None:
    with pytest.raises(ValueError, match="둘 미만"):
        crit(allowed_verdicts=("duplicate",))


def test_기준을_누가_줬는지_산출물에_남는다() -> None:
    d = sr().as_dict()["criteria"]
    assert d["source"] == "1차 판독자(A)"
    assert "2차 판독자가 정하지 않았다" in d["note"]


def test_같은_쌍을_두_번_판독할_수_없다() -> None:
    with pytest.raises(ValueError, match="두 번 판독"):
        SecondRead(criteria=crit(), reads=(Read("p1", "duplicate"), Read("p1", "not_duplicate")))


# ---------------------------------------------------------------- 봉인

def test_봉인은_판정에만_달렸다() -> None:
    """사유를 고치는 것은 판독을 고치는 것이 아니다."""
    a = SecondRead(criteria=crit(), reads=(Read("p1", "duplicate", "왼쪽이 밝다"),))
    b = SecondRead(criteria=crit(), reads=(Read("p1", "duplicate", "다시 보니 각도"),))
    assert a.seal() == b.seal()


def test_판정을_고치면_봉인이_깨진다() -> None:
    before = sr().seal()
    with pytest.raises(SealBroken, match="본 뒤 판독이 바뀌었다"):
        agreement(sr(verdicts=("not_duplicate", "not_duplicate", "undecidable")),
                  {"p1": "duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                  sealed=anchor(before))


def test_기준_판이_바뀌어도_봉인이_깨진다() -> None:
    before = sr().seal()
    with pytest.raises(SealBroken):
        agreement(sr(criteria=crit(criteria_sha256="b" * 64)),
                  {"p1": "duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                  sealed=anchor(before))


# ---------------------------------------------------------------- 일치율

def test_전부_같으면_1_이다() -> None:
    s = sr()
    got = agreement(s, {"p1": "duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                    sealed=anchor(s.seal()))
    assert got["agreement_rate"] == 1.0 and got["disagreed_pairs"] == []


def test_어긋난_쌍을_이름으로_준다() -> None:
    s = sr()
    got = agreement(s, {"p1": "not_duplicate", "p2": "not_duplicate", "p3": "duplicate"},
                    sealed=anchor(s.seal()))
    assert got["n_agreed"] == 1
    assert got["agreement_rate"] == pytest.approx(1 / 3)
    assert got["disagreed_pairs"] == ["p1", "p3"]


def test_1차_판정별로_나눠_센다() -> None:
    """전체 일치율만 보면 어느 판정에서 갈리는지 사라진다."""
    s = sr()
    got = agreement(s, {"p1": "duplicate", "p2": "duplicate", "p3": "undecidable"},
                    sealed=anchor(s.seal()))
    assert got["by_first_verdict"]["duplicate"] == {"n_first": 2, "n_agreed": 1, "rate": 0.5}
    assert got["by_first_verdict"]["not_duplicate"]["rate"] is None, "1차에 없는 판정은 미정의"


def test_혼동표가_방향을_보존한다() -> None:
    s = sr()
    got = agreement(s, {"p1": "not_duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                    sealed=anchor(s.seal()))
    assert got["confusion"]["not_duplicate->duplicate"] == 1


def test_쌍_집합이_다르면_거부한다() -> None:
    s = sr()
    with pytest.raises(ValueError, match="쌍 집합이 다르다"):
        agreement(s, {"p1": "duplicate", "p2": "not_duplicate"}, sealed=anchor(s.seal()))


def test_1차가_기준_밖_어휘를_쓰면_거부한다() -> None:
    s = sr()
    with pytest.raises(ValueError, match="기준 밖 어휘"):
        agreement(s, {"p1": "maybe", "p2": "not_duplicate", "p3": "undecidable"},
                  sealed=anchor(s.seal()))


# ---------------------------------------------------------------- 쌍 목록

def test_쌍_목록만_받는다() -> None:
    assert load_pairs("# 주석\np1\n\np2\n", allowed=V) == ("p1", "p2")


def test_판정이_섞여_오면_거부한다() -> None:
    """1차 판정이 같은 파일에 섞여 오면 독립이 깨진다."""
    with pytest.raises(ValueError, match="판정처럼 보이는"):
        load_pairs("p1 duplicate\np2 not_duplicate\n", allowed=V)


def test_쌍_목록의_중복을_거부한다() -> None:
    with pytest.raises(ValueError, match="중복"):
        load_pairs("p1\np1\n", allowed=V)


@pytest.mark.parametrize("line", ["p1,duplicate", "p1;not_duplicate", "p1|undecidable", "p1\tduplicate"])
def test_쉼표로_붙은_판정도_거부한다(line: str) -> None:
    """앞 판은 공백으로만 나눠 `p1,duplicate` 가 id 에 붙은 채 들어왔다."""
    with pytest.raises(ValueError, match="판정처럼 보이는"):
        load_pairs(line + "\n", allowed=V)


def test_줄은_LF_로만_나눈다() -> None:
    """U+2028 에서 쪼개면 id 하나가 둘이 된다."""
    ls = chr(0x2028)
    assert load_pairs(f"p1{ls}x\np2\r\n", allowed=V) == (f"p1{ls}x", "p2")


# ================================================================ 읽는 쪽 관문
#
# 표시를 박기만 하면 아무것도 막지 못한다. **요구하는 자리**가 있어야 뜻이 생긴다.


def payload(**over) -> dict:
    d = sr().as_dict()
    for k, v in over.items():
        if v is _DROP:
            d.pop(k, None)
        else:
            d[k] = v
    return d


_DROP = object()


def test_제대로_만든_산출물은_지난다() -> None:
    got = require_second_read(payload())
    assert got["read_type"] == READ_TYPE


def test_표시가_없으면_거부한다() -> None:
    """손으로 만든 사전이 아무 표시 없이 소비처에 들어가던 자리다."""
    with pytest.raises(NotSecondRead) as exc:
        require_second_read({"reads": [{"pair_id": "p1", "verdict": "duplicate"}]})
    assert len(exc.value.reasons) >= 3, "사유를 전부 낸다"


def test_다른_판독_종류는_거부한다() -> None:
    with pytest.raises(NotSecondRead, match="read_type"):
        require_second_read(payload(read_type="human_review"))


@pytest.mark.parametrize("bad", [True, 0, "", None, "false"])
def test_human_review_는_False_그_자체여야_한다(bad) -> None:
    """`0`·`""` 을 통과시키면 거짓값 아무거나로 표시를 우회할 수 있다."""
    with pytest.raises(NotSecondRead, match="human_review"):
        require_second_read(payload(human_review=bad))


def test_한계에서_사람_판독_미측정을_빼면_거부한다() -> None:
    with pytest.raises(NotSecondRead, match="한계 목록"):
        require_second_read(payload(limits=["쌍이 60개다"]))


def test_계약_지문이_없으면_거부한다() -> None:
    with pytest.raises(NotSecondRead, match="계약 판"):
        require_second_read(payload(contract=_DROP))


def test_옛_계약_판은_거부한다() -> None:
    with pytest.raises(NotSecondRead, match="계약 판"):
        require_second_read(payload(contract={"version": "second_read/0", "sha256": "x" * 64}))


def test_지문이_표시와_맞지_않으면_거부한다() -> None:
    """as_dict() 를 거치지 않았거나 저장 뒤 고쳐진 경우다."""
    d = payload()
    d["reader"] = "E"                       # 지문은 그대로 둔 채 표시만 바꾼다
    with pytest.raises(NotSecondRead, match="계약 지문"):
        require_second_read(d)


def test_표시를_전부_올바르게_담은_사전은_지난다() -> None:
    """지문은 비밀이 아니라 **구조 요구**다. 맞췄다는 것은 계약을 지켰다는 뜻이다."""
    d = payload()
    hand = {k: v for k, v in d.items()}     # as_dict 가 아니라 손으로 다시 만든 사전
    assert require_second_read(hand)["seal_sha256"] == d["seal_sha256"]


def test_사유를_한꺼번에_낸다() -> None:
    d = payload(read_type="사람검수", human_review=True, limits=[])
    with pytest.raises(NotSecondRead) as exc:
        require_second_read(d)
    assert len(exc.value.reasons) == 4, exc.value.reasons


# ---------------------------------------------------------------- 되살리기

def test_되살리면_같은_판독이다() -> None:
    s = sr()
    back = rebuild(s.as_dict())
    assert back.seal() == s.seal()
    assert [r.verdict for r in back.reads] == [r.verdict for r in s.reads]


def test_저장_뒤_판정을_고치면_관문이_잡는다() -> None:
    """앞 판은 되살리기(`rebuild`)만 잡고 관문은 지났다 — 관문만 부르는 소비자는 고친 판독을 받았다."""
    d = sr().as_dict()
    d["reads"][0]["verdict"] = "not_duplicate"
    d["contract"]["sha256"] = contract_digest(d)      # 지문까지 다시 맞춰도
    with pytest.raises(NotSecondRead, match="봉인"):
        require_second_read(d)
    with pytest.raises(NotSecondRead, match="봉인"):
        rebuild(d)


def test_판독을_지우고_수를_그대로_두면_관문이_잡는다() -> None:
    d = sr().as_dict()
    d["reads"].pop()
    d["contract"]["sha256"] = contract_digest(d)
    with pytest.raises(NotSecondRead) as exc:
        require_second_read(d)
    assert any("판독 수" in r for r in exc.value.reasons)


@pytest.mark.parametrize("reads", [None, "p1", [{"pair_id": "p1"}], [{"pair_id": 1, "verdict": "duplicate"}]])
def test_판독_목록의_꼴이_틀리면_관문이_잡는다(reads) -> None:
    d = sr().as_dict()
    d["reads"] = reads
    with pytest.raises(NotSecondRead, match="판독 목록의 꼴"):
        require_second_read(d)


def test_저장된_산출물로_일치율을_낸다() -> None:
    s = sr()
    got = agreement_from_payload(
        s.as_dict(), {"p1": "duplicate", "p2": "not_duplicate", "p3": "duplicate"},
        sealed=anchor(s.seal()), first_source="A 1차")
    assert got["n_agreed"] == 2
    assert got["disagreed_pairs"] == ["p3"]


def test_일치율_산출에_봉인을_남긴_자리가_실린다() -> None:
    """봉인 값만 받으면 그 값이 1차 판정보다 앞섰는지를 산출만 보고 확인할 길이 없다."""
    s = sr()
    got = agreement(s, {"p1": "duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                    sealed=anchor(s.seal()))
    assert got["seal"]["sha256"] == s.seal()
    assert got["seal"]["commit"] == COMMIT and got["seal"]["path"] == "docs/reads/seal.md"
    assert "스스로 증명하지 않는다" in got["seal"]["note"]


def test_봉인_값만_넘기면_받지_않는다() -> None:
    s = sr()
    with pytest.raises(TypeError, match="SealAnchor"):
        agreement(s, {"p1": "duplicate", "p2": "not_duplicate", "p3": "undecidable"},
                  sealed=s.seal())                                      # type: ignore[arg-type]


@pytest.mark.parametrize(("key", "bad"), [("sha256", "x"), ("commit", "abc"), ("path", "/abs"),
                                          ("path", "../x"), ("path", "")])
def test_봉인의_자리는_꼴이_맞아야_한다(key: str, bad: str) -> None:
    with pytest.raises(ValueError):
        anchor("a" * 64, **{key: bad})


def test_관문을_못_지난_사전으로는_일치율을_못_낸다() -> None:
    s = sr()
    d = s.as_dict()
    d["human_review"] = True
    with pytest.raises(NotSecondRead):
        agreement_from_payload(d, {"p1": "duplicate", "p2": "not_duplicate",
                                   "p3": "undecidable"}, sealed=anchor(s.seal()))


# ---------------------------------------------------------------- M-4 · M-5

@pytest.mark.parametrize("bad", ["x", "A" * 64, "g" * 64, "a" * 63, ""])
def test_기준_해시는_sha256_형식이어야_한다(bad: str) -> None:
    """한 글자짜리도 통과하면 '해시를 적었다' 가 아무 뜻이 없다."""
    with pytest.raises(ValueError, match="sha256"):
        crit(criteria_sha256=bad)


def test_올바른_해시는_받는다() -> None:
    assert crit(criteria_sha256="0123456789abcdef" * 4).criteria_sha256.startswith("0123")


@pytest.mark.parametrize("bad", ["사람 검수단", "human reviewer", "", "D 사람", "tool:사람"])
def test_판독자_어휘를_자유_문자열로_두지_않는다(bad: str) -> None:
    """자유 문자열이면 read_type 을 우회해 사람 검수로 읽히는 길이 열린다."""
    with pytest.raises(ValueError, match="판독자는"):
        sr(reader=bad)


@pytest.mark.parametrize("ok", ["D", "A", "tool:neardup_reader", "tool:box-iou.v2"])
def test_한_글자_표지나_도구_id_는_받는다(ok: str) -> None:
    assert sr(reader=ok).as_dict()["reader"] == ok


def test_판독자가_지문에_들어간다() -> None:
    """판독자를 바꾸면 계약 지문이 갈린다 — 조용히 바꿔 끼울 수 없다."""
    assert sr(reader="D").as_dict()["contract"]["sha256"] != \
        sr(reader="A").as_dict()["contract"]["sha256"]
