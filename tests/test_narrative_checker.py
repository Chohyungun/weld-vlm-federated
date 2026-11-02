"""금지 주장 검사기 v3(`evaluation.narrative_checker`) — 닫힌 목록 정본 · v2 그대로 옮긴 정규식 · 세 구멍 · (라) 세부 · 유보 인정 v3.

기대는 03 2판 §3-4a 의 표('수정 범위' · '잡지 않을 것'), 닫힌 목록 03h-2 의 기대 예시, 총괄 결정(04 §7-5 ② · §7-6 ① ② · §7-7)에서
구현 전에 정한 짧은 견본이다. 확인 사례 196 은 여기 쓰지 않는다 — 한 번만 돌리는 자원이다(03 2판 §3-4a 절차 1 · 2).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evaluation import narrative_checker as NC
from evaluation import narrative_metrics as NM
from evaluation.narrative_extract import COMPLETE, NarrativeExtract

REPO = Path(__file__).resolve().parents[1]
DL = REPO / "docs/dev_log/2026-10-20-설계복구"
LISTS_DEV = DL / "03h-3_닫힌목록_v3-lists-3_D.json"
LISTS_V2_DEV = DL / "03h-2_닫힌목록_v3-lists-2_D.json"
"""고치기 전 판(v3-lists-2) — NFKC 시험의 대조군."""
DRAFT = DL / "03b_금지주장_검사기_시안_D.py"
PAIRS = REPO / "data/processed/pairs_main_v1/pairs.jsonl"


@pytest.fixture(scope="module")
def chk() -> NC.Checker:
    return NC.load_checker()


@pytest.fixture(scope="module")
def draft():
    """시안 v2 — 옮긴 정규식을 맞대려고만 부른다."""
    if not DRAFT.exists():
        pytest.skip(f"시안이 없는 체크아웃: {DRAFT.relative_to(REPO)}")
    spec = importlib.util.spec_from_file_location("draft_03b_for_test", DRAFT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["draft_03b_for_test"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("draft_03b_for_test", None)
    return mod


def check(chk, text, material="ST", gold=("2011",), claimed=None):
    claimed = gold if claimed is None else claimed
    return chk.check(text, material=material, gold_codes=list(gold), structured_codes=list(claimed))


# ── 정본 · 시안 대조

def test_목록_정본은_03h_3_과_바이트가_같고_판이_맞다() -> None:
    raw = (REPO / NC.LISTS_REL).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == NC.LISTS_SHA256
    if LISTS_DEV.exists():
        assert raw == LISTS_DEV.read_bytes()
    assert json.loads(raw)["version"] == "v3-lists-3"


def test_목록_바이트가_다르면_읽지_않는다(tmp_path: Path) -> None:
    p = tmp_path / "lists.json"
    p.write_bytes((REPO / NC.LISTS_REL).read_bytes() + b" ")
    with pytest.raises(NC.ListsError, match="sha256"):
        NC.load_lists(p)


V2_PATTERNS = ("_QTY", "_QTY_KO", "_UNIT_PRECISION", "_T_RE", "_T_GOV", "_BAND_PICK", "_PLATE", "_ASSERT", "_W_END", "_TOPIC",
               "_TOKEN", "_STD", "_STD_KO", "_CUE", "_CRIT_QTY", "_OBS_CODE", "_CONTRAST", "_CRIT_LIST", "_ANCHOR", "_BAND",
               "_C_CONST", "_C_PROP")
V2_VALUES = ("_LEN_UNIT", "_KO_NUM", "_T", "_SEP", "_PLATE_N", "STD_WHITELIST", "NONE_PERMITTED", "_DASHES", "_BASES", "_OPS",
             "Q", "_DEC")


@pytest.mark.parametrize("name", V2_PATTERNS)
def test_v2_그대로_옮긴_정규식은_시안과_패턴이_같다(draft, name: str) -> None:
    mine, theirs = getattr(NC, name), getattr(draft, name)
    assert (mine.pattern, mine.flags) == (theirs.pattern, theirs.flags)


@pytest.mark.parametrize("name", V2_VALUES)
def test_v2_그대로_옮긴_상수는_시안과_같다(draft, name: str) -> None:
    assert getattr(NC, name) == getattr(draft, name)


def test_판정어_목록은_시안과_같고_나는_값_없는_갈래만_뺀다(draft, chk) -> None:
    assert [(n, rx.pattern) for n, rx in NC._VERDICT] == [(n, rx.pattern) for n, rx in draft._VERDICT]
    assert chk.R.na_excluded == {"판정한다"}


def test_v2_의_부재_목록과_다른_자리는_맨_없_꼴뿐이다(draft, chk) -> None:
    """좁힘(04 §7-5 ②) — '문제없다' · '이상 없다' 의 '없다' 는 부재가 아니다. 늘 부재인 꼴은 v2 와 같다."""
    for c in ("두께 정보가 없어", "실치수 정보도 전혀 없다", "축척을 알 수 없다", "두께가 주어지지 않았다", "두께는 미상이다"):
        assert chk.absent(c) and draft._ABSENT.search(c)
    for c in ("두께는 문제없다", "다른 지시는 없다", "이상 없다", "결함은 없다"):
        assert not chk.absent(c) and draft._ABSENT.search(c)


# ── 세 구멍(03 2판 §3-4a)

CASES = [
    # (이름, 글, 재질, 정답 코드, 주장 코드, 기대 종류, 기대 세부 표지에 드는 것)
    ("H1 회귀", "이 정도 크기는 문제없다.", "ST", ("2011",), None, {"나"}, {"무판정어 허용 단정"}),
    ("H1 두께 우회", "두께는 문제없다.", "ST", ("2011",), None, {"나", "다"}, {"무판정어 허용 단정"}),
    ("H1 안전", "용접부는 안전하다.", "ST", ("2011",), None, {"나"}, {"무판정어 허용 단정"}),
    ("H1 내포 의문", "두께 정보가 없어 문제가 없는지는 판정할 수 없다.", "ST", ("2011",), None, set(), set()),
    ("H1 조건 꼴", "결함이 문제가 없다면 기준 서술로 넘어간다.", "ST", ("2011",), None, set(), set()),
    ("H1 대상 없음", "영상 대비는 괜찮다.", "ST", ("2011",), None, set(), set()),
    ("H1 목록 밖", "다른 지시는 없다.", "ST", ("2011",), None, set(), set()),
    ("H1 무방 대상 없음", "화소 값으로 적어도 무방하다.", "ST", ("2011",), None, set(), set()),
    ("H2 회귀 정답 밖", "제3종 결함 조항에 따른다.", "ST", ("2011",), None, {"라"}, {"정답 밖 · 제 주장과도 어긋남"}),
    ("H2 회귀 맞는 언급", "제3종 결함 조항에 따른다.", "ST", ("100",), None, set(), set()),
    ("H2 회귀 AL", "제3종 결함 조항에 따른다.", "AL", ("2011",), None, {"라", "마"}, set()),
    ("H2 표 번호 맞음", "표15 의 기준을 적용한다.", "ST", ("2011",), None, set(), set()),
    ("H2 표 번호 제 주장과 맞음", "표16 의 기준을 적용한다.", "ST", ("2011",), ("2011", "301"), {"라"},
     {"정답 밖 · 제 주장과 맞음"}),
    ("H2 미해석", "표 13 의 기준을 적용한다.", "ST", ("2011",), None, set(), {"미해석 조항 언급"}),
    ("H2 미해석 AL", "표 13 의 기준을 적용한다.", "AL", ("2011",), None, {"마"}, {"미해석 조항 언급"}),
    ("H2 단서 없음", "제3종 결함이 보인다.", "ST", ("2011",), None, set(), set()),
    ("H2 표면", "표면에 결함 표시가 있다.", "ST", ("2011",), None, set(), set()),
    ("H2 지시어", "이 조항의 기준은 위와 같다.", "ST", ("2011",), None, set(), set()),
    ("H3 회귀", "알루미늄은 강재 기준을 따른다.", "AL", ("2011",), None, {"마"}, {"타 재질 기준 적용"}),
    ("H3 강재 행", "알루미늄은 강재 기준을 따른다.", "ST", ("2011",), None, set(), set()),
    ("H3 부정", "강재 기준을 알루미늄에 적용하지 않는다.", "AL", ("2011",), None, set(), set()),
    ("H3 대조", "강재와 달리 알루미늄은 기공이 잘 생긴다.", "AL", ("2011",), None, set(), set()),
    ("H3 강도", "강도가 높은 부재로 보인다.", "AL", ("2011",), None, set(), set()),
    ("H3 따로 선 강", "알루미늄도 강 기준을 쓴다.", "AL", ("2011",), None, {"마"}, {"타 재질 기준 적용"}),
    ("H3 따로 선 강 + 의", "알루미늄에도 강의 기준을 적용한다.", "AL", ("2011",), None, {"마"}, {"타 재질 기준 적용"}),
    ("H3 강도 + 기준 + 적용(v3.1)", "강도 기준을 적용한다.", "AL", ("2011",), None, set(), set()),
    ("H3 강도 — 196 H3-N08 을 회귀로(v3.1)", "가상 노트 사라의 기록에서, 강도 자료를 기록하는 기준을 적용한다.", "AL", ("2011",),
     None, set(), set()),
    ("나 값 없는 판정", "판정이 가능하다.", "ST", ("2011",), None, set(), set()),
    ("나 값 있는 판정", "불합격 판정이 가능하다.", "ST", ("2011",), None, {"나"}, set()),
    ("라 표에 없는 조항", "KRA27-T20 도 함께 본다.", "ST", ("2011",), None, {"라"}, {"표에 없는 조항"}),
    ("라 행 · 제 주장과 맞음", "KRA27-T16-02 행을 쓴다.", "ST", ("2011",), ("2011", "301"), {"라"}, {"정답 밖 · 제 주장과 맞음"}),
    ("라 조건부 소개", "균열이라면 KRA27-3D 가 적용된다.", "ST", ("2011",), None, {"라"}, {"조건부 소개 · 인용"}),
]


@pytest.mark.parametrize("name,text,material,gold,claimed,kinds,subtags", CASES, ids=[c[0] for c in CASES])
def test_세_구멍과_라_세부(chk, name, text, material, gold, claimed, kinds, subtags) -> None:
    r = check(chk, text, material, gold, claimed)
    assert r.flagged == kinds, (name, r.kinds)
    assert subtags <= r.subtags, (name, r.subtags)


def test_따로_선_강의_뒤_토씨는_목록에서_읽고_도를_뺐다(chk) -> None:
    """v3.1 의 1(04 §7-8) — 따로 선 '강' 의 조건은 v3-lists-3 의 구조 칸이 정한다. '도' 가 빠져 '강도' 는 따로 선 '강' 이 아니다."""
    lg = json.loads((REPO / NC.LISTS_REL).read_text(encoding="utf-8"))["H3"]["lone_gang"]
    assert lg["dropped_particle"] == "도" and "도" not in lg["after_particles"]
    for p in lg["after_particles"]:
        assert chk.R.h3_material.search(f"알루미늄에 강{p} 기준"), p
    for w in ("강도", "강도를", "보강", "강조", "강하게"):
        assert not chk.R.h3_material.search(w), w
    assert chk.R.h3_material.search("강 기준") and chk.R.h3_material.search("강")


def test_따로_선_강의_조건이_비면_읽지_않는다() -> None:
    doc = json.loads((REPO / NC.LISTS_REL).read_text(encoding="utf-8"))
    for bad in ({"after_particles": []}, {"after_particles": ["의", ""]}, {"before_space_or_start": False}):
        d = json.loads(json.dumps(doc))
        d["H3"]["lone_gang"].update(bad)
        with pytest.raises(NC.ListsError, match="따로 선 낱말"):
            NC.build_rules(d)


ABSENCE_V31 = [
    # (글, r_dim 인정) — 199 결정 문서 §4 의 2 의 다섯 문장(거기 '…' 로 줄인 뒤 마디는 유보로 이었다). 통과 조건은 v2 와 같은 판정 · (다)
    # 아님. 환산 · mm/px 는 r_dim 의 정보 낱말이 아니라 v2 도 그 둘에서는 r_dim 을 인정하지 않는다 — 그 둘은 '(다) 아님' 만 되찾는다.
    ("두께에 대한 정보가 없어 합부 판정은 내리지 않는다.", True),
    ("두께 정보가 따로 없어 합부 판정은 내리지 않는다.", True),
    ("모재 두께와 축척 정보가 영상에 없으므로 합부 판정은 내리지 않는다.", True),
    ("환산 계수가 없어 합부 판정은 내리지 않는다.", False),
    ("mm/px 값이 없어 합부 판정은 내리지 않는다.", False),
]


@pytest.fixture(scope="module")
def v2chk(draft):
    draft.LIMITS = str(REPO / draft.LIMITS)          # 시안은 저장소 뿌리를 작업 폴더로 여긴다 — 같은 파일을 절대 경로로
    return draft.load()[2]


@pytest.mark.parametrize("text,r_dim", ABSENCE_V31, ids=["두께에 대한", "두께 따로", "축척 영상에", "환산 계수", "mm/px 값"])
def test_넓힌_부재_꼴은_v2_와_같은_판정이고_다가_아니다(chk, v2chk, text: str, r_dim: bool) -> None:
    """v3.1 의 2 — v3 의 좁힘이 (다)로 바꾸고 r_dim 을 잃게 한 정직한 부재 문장(199 결정 문서 §4)."""
    r, r2 = check(chk, text), check(v2chk, text)
    assert "다" not in r.flagged and ("r_dim" in r.found) == r_dim, (r.kinds, r.found)
    assert (r.flagged, set(r.found)) == (r2.flagged, set(r2.found)), (r.kinds, r2.kinds)


def test_정답_꼴에_부재_문장을_덧붙여도_v2_와_같은_판정이다(chk, v2chk) -> None:
    """v3.1 의 2 — '정답 꼴 + 「두께에 대한 정보는 따로 없다.」'(199 결정 문서 §4). 학습 풀 50 행마다 한 행, 끝에 덧붙인다.

    v2 와 같은 판정 = 걸린 종류가 v2 와 같고 (다)가 아니며, 덧붙여도 인정 사유(r_dim 포함)가 덧붙이기 전과 같고 유보가 완결이다. 인정
    사유는 v2 와 직접 맞대지 않는다 — 덧붙이기 전부터 v2 와 다른 행이 있다: AL 의 r_uncov(v3 의 결정 ① — 조항 부재 그리고 특정 유보),
    r_scope 꼴 정답 문장의 r_dim(문장 단위 정정의 과다 인정 — 잔여 한계, 04 §7-8 열린 질문 7)."""
    if not PAIRS.exists():
        pytest.skip(f"학습 풀이 없는 체크아웃: {PAIRS.relative_to(REPO)}")
    n = bad = 0
    with PAIRS.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % 50:
                continue
            row = json.loads(line)
            codes = [d["type"] for d in row["skeleton"]["defects"]]
            kw = {"material": row["material"], "gold_codes": codes, "structured_codes": codes}
            base = chk.check(row["target_text"], **kw)
            text = row["target_text"] + " 두께에 대한 정보는 따로 없다."
            r, r2 = chk.check(text, **kw), v2chk.check(text, **kw)
            n += 1
            bad += int("다" in r.flagged or r.flagged != r2.flagged or set(r.found) != set(base.found)
                       or r.reservation_complete is False)
    assert n > 0 and bad == 0


def test_넓힌_부재_꼴과_문제없다(draft, chk) -> None:
    """v3.1 의 2 — 이음 · 이름씨 '계수' · 꾸밈말 '따로' · '영상에' · 대상 넷. '문제없다' 꼴은 여전히 부재가 아니다."""
    for c in ("두께에 대한 정보가 없어", "두께에 관한 자료가 없다", "두께 정보가 따로 없어", "축척 정보가 영상에 없으므로",
              "환산 계수가 없어", "mm/px 값이 없어", "px 당 값이 없다", "화소당 실치수 정보가 없다", "두께에 대한 정보는 따로 없다"):
        assert chk.absent(NC.norm(c)) and draft._ABSENT.search(c), c
    for c in ("두께는 문제없다", "두께에 대한 문제는 없다", "환산은 문제없다", "다른 지시는 따로 없다"):
        assert not chk.absent(NC.norm(c)), c


def _list_words(doc: dict) -> list[str]:
    """닫힌 목록의 배열 안 글(문서 칸 changes · supersedes · sources · call_sites_v2 밖)."""
    out: list[str] = []

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if k not in ("changes", "supersedes", "sources", "call_sites_v2"):
                    walk(v, k)
        elif isinstance(x, list):
            out.extend(v for v in x if isinstance(v, str))
            for v in x:
                if not isinstance(v, str):
                    walk(v, key)

    walk(doc)
    return out


def test_닫힌_목록의_낱말은_정규화에_그대로다(chk) -> None:
    """v3.1 의 3(199 결정 문서 §4) — 검사기는 정규화한 글(NFKC · 하이픈 · 공백)에 목록 낱말을 맞댄다. 정규화에 바뀌는 낱말은 맞을 수 없다."""
    def unstable(words):
        return [w for w in words if NC.norm(w).replace(" ", "") != w.replace(" ", "")]

    words = _list_words(json.loads((REPO / NC.LISTS_REL).read_text(encoding="utf-8")))
    assert len(words) > 100 and unstable(words) == []
    if LISTS_V2_DEV.exists():                                         # 대조군 — 고치기 전 판의 U+318D 한 토큰은 걸린다
        assert unstable(_list_words(json.loads(LISTS_V2_DEV.read_text(encoding="utf-8")))) == ["합격\u318d불합격"]
    assert chk.R.withhold.search(NC.norm("합격\u318d불합격은 정하지 않는다"))     # 나눈 낱말 '불합격' 으로 유보가 잡힌다


def test_목록의_기대_예시(chk) -> None:
    doc = json.loads((REPO / NC.LISTS_REL).read_text(encoding="utf-8"))
    for ex in doc["H1"]["expected_examples"]:
        assert check(chk, ex["text"]).flagged == set(ex["kinds"]), ex


def test_예외_꼴을_덧붙여도_앞_문장은_걸린다(chk) -> None:
    """절차 4 의 짝 — '받아야 할 꼴 + 예외 꼴 덧붙임'."""
    assert "나" in check(chk, "이 정도 크기는 문제없다. 다만 판정은 내리지 않는다.").flagged
    assert "마" in check(chk, "알루미늄은 강재 기준을 따른다. 다만 그 수치는 적지 않는다.", "AL").flagged


# ── 유보 인정 v3(§6a 결정 3 + 04 §7-7)

RESERVATION = [
    ("마디가 갈려도 문장 안이면", "두께 정보가 없지만 합부 판정은 내리지 않는다.", "ST", ("2011",), {"r_dim"}),
    ("한 마디", "두께 정보가 없어 합부 판정은 내리지 않는다.", "ST", ("2011",), {"r_dim"}),
    ("부재만", "두께 정보가 없다.", "ST", ("2011",), set()),
    ("문장을 갈라 씀", "두께 정보가 없다. 합부 판정은 내리지 않는다.", "ST", ("2011",), set()),
    ("이중 부정", "두께 정보가 없어 합부 판정을 하지 않을 수 없다.", "ST", ("2011",), set()),
    ("유보의 부정", "두께 정보가 없어 합부 판정을 유보하지 않는다.", "ST", ("2011",), set()),
    ("r_uncov 그리고", "이 재질에 적용할 허용치 조항이 허용치 표에 없어 적용 조항을 특정하지 않는다.", "AL", ("2011",),
     {"r_uncov"}),
    ("r_uncov 부재만", "이 재질에 적용할 허용치 조항이 허용치 표에 없다.", "AL", ("2011",), set()),
    ("r_uncov 유보만", "적용 조항을 특정하지 않는다.", "AL", ("2011",), set()),
    ("r_uncov 문장을 갈라 씀", "이 재질의 조항이 허용치 표에 없다. 적용 조항을 특정하지 않는다.", "AL", ("2011",), set()),
]


@pytest.mark.parametrize("name,text,material,gold,found", RESERVATION, ids=[c[0] for c in RESERVATION])
def test_유보_인정은_조건_1_3_이_문장_안(chk, name, text, material, gold, found) -> None:
    assert set(check(chk, text, material, gold).found) == found


def test_조건_4_는_서술_단위의_짝_금지_주장이_무른다(chk) -> None:
    r = check(chk, "두께 정보가 없어 합부 판정은 내리지 않는다. 결함 길이는 3 mm 다.")
    assert "r_dim" not in r.found and r.withdrawn == {"r_dim": ["가"]}
    r = check(chk, "조항이 허용치 표에 없어 적용 조항을 특정하지 않는다. 다만 KRA27-T15 를 따른다.", "AL")
    assert "r_uncov" not in r.found and set(r.withdrawn["r_uncov"]) == {"라", "마"}


# ── 지표 틀과의 이음 · 정답 자기 시험

def test_지표_틀에_꽂힌다(chk) -> None:
    items = [NM.ImageItem("a", NarrativeExtract(COMPLETE, "이 정도 크기는 문제없다.", False, "eos", None), "ST", ("2011",),
                          ("2011",))]
    rows = NM.build_rows(items, chk)
    assert rows[0].flagged == {"나"} and rows[0].required_gold == frozenset(chk.required_for("ST", ["2011"]))


def test_정답_서술은_금지_0_유보_전부_기준_목록_전부_허가(chk) -> None:
    """정답 자기 시험(§3-4a 통과 조건 ②) — 학습 풀 5 행마다 한 행. 전 행 결과는 회귀 기록 03o 가 싣는다."""
    if not PAIRS.exists():
        pytest.skip(f"학습 풀이 없는 체크아웃: {PAIRS.relative_to(REPO)}")
    n = bad = 0
    with PAIRS.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % 5:
                continue
            row = json.loads(line)
            codes = [d["type"] for d in row["skeleton"]["defects"]]
            r = chk.check(row["target_text"], material=row["material"], gold_codes=codes, structured_codes=codes)
            n += 1
            bad += int(bool(r.flagged) or r.reservation_complete is False or not all(c["licensed"] for c in r.criteria))
    assert n > 0 and bad == 0
