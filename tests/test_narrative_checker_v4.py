"""검사기 v4(미니스펙 10 7판 §11) — 목록 · 핀 · 섞임 거부 · 2 판 문장 계약 · 목록 경계 · 별칭 표 · B 회귀 · 쓰임 탐침 v2 · 정답 자기 시험.

확인 사례(외부 검토 236 의 `narrative_v4_confirm_cases.v1.json`)는 여기 쓰지 않는다 — 한 번만 돌리는 확인 배치의 자원이다(10 §10 의 4 · 6).
196 · 199 · 230 예문 · B 회귀의 다섯 자리 · 2 판 정답 전량은 저장소 밖 자원도 읽으므로 회귀 기록(13b)이 돈다. 여기는 그 가운데 저장소 안의
자원만으로 서는 것이다(b1′ 견본은 산출 폴더에 있을 때만).
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest

from evaluation import narrative_checker as NC
from evaluation import narrative_checker_v4 as V4

REPO = Path(__file__).resolve().parents[1]
PROBES = REPO / "evaluation/vectors/narrative_v4_list_probes.v2.json"
EXPECT = REPO / "evaluation/vectors/narrative_v4_expectation_changes.v3.json"
BREG = REPO / "evaluation/vectors/narrative_b_regression.v1.json"
B1P = REPO / "outputs/pairs_main_v2_tmp/b1p/pairs.jsonl"
B1P_STEP = 25
"""정답 자기 시험의 견본 간격 — b1′ 의 25 행마다 한 행(전량은 회귀 기록 13b)."""
FOUR = {"적용 전제 단정", "군집 · 간격 단정", "조건부 대응의 단정", V4.MEASURE_SUBTAG}
"""B 회귀 v1 의 세부 표지 넷(07 1판 §3-2) — `sublabels` 는 이 넷의 집합이다."""


@pytest.fixture(scope="module")
def chk() -> V4.Checker:
    return V4.load_checker(REPO)


def with_band(chk: V4.Checker, cid: str, code: str, band: str) -> V4.Checker:
    """합성 띠를 (조항, 코드)의 기준으로 주입한 검사기 — B 회귀 · 확인 배치의 기준 목록은 합성 수치라 실제 표의 수를 쓰지 않는다."""
    import copy

    crits = NC.parse_body(band)
    assert crits is not None
    t = copy.copy(chk.t)
    t.crits = {**chk.t.crits, (cid, code): crits}
    return V4.Checker(t, {v: k for k, v in chk.code_of_name.items()}, chk.L)


@pytest.fixture(scope="module")
def chk_breg(chk) -> V4.Checker:
    """B 회귀의 합성 띠(`synthetic.band`)를 기공(2011)의 주 조항 T15 의 기준으로 주입한다 — 사례의 앵커가 그 조항이다."""
    doc = json.loads(BREG.read_text(encoding="utf-8"))
    inc = next(c for c in doc["cases"] if c["id"] == "measure.include")
    assert "KRA27-T15" in inc["text"] and inc["context"]["codes"] == ["2011"]
    return with_band(chk, "KRA27-T15", "2011", doc["synthetic"]["band"])


def run(chk: V4.Checker, text: str, material: str = "ST", codes=("2011",)) -> V4.Result:
    return chk.check(text, material=material, gold_codes=list(codes), structured_codes=list(codes))


# ── 목록 · 핀 · 섞임 거부

def test_목록_정본은_v4_lists_2_이고_sha_를_맞댄다() -> None:
    doc = V4.load_lists()
    assert doc["version"] == V4.LISTS_VERSION == "v4-lists-2"
    assert doc["sources"]["limits_csv"]["path"] == V4.LIMITS_REL and doc["sources"]["limits_csv"]["sha256"] == V4.LIMITS_SHA256


def test_목록_바이트가_다르면_읽지_않는다(tmp_path: Path) -> None:
    bad = tmp_path / "l.json"
    bad.write_bytes((REPO / V4.LISTS_REL).read_bytes() + b"\n")
    with pytest.raises(V4.ListsError):
        V4.load_lists(bad)
    with pytest.raises(V4.ListsError):
        V4.load_lists(REPO / NC.LISTS_REL)


def test_1판_표와_v3_1_표로는_v4_가_재지_않는다() -> None:
    with pytest.raises(V4.ListsError):
        V4.Table([], {}, edition="v1")
    with pytest.raises(V4.ListsError):
        V4.Checker(NC.Table([]), {}, V4.load_lists())


def test_2판_문장은_생성_쪽_상수와_같은_글자다() -> None:
    """§11 목록 ↔ 문장 계약 — 검사기는 생성기를 import 하지 않는다. 시험이 생성 쪽 상수와 맞댄다(1 판 꼴은 v3.1 의 앵커 · 머리)."""
    from corpus.generate import pairs_text_v2 as T2

    assert V4.T_CORRESPOND == T2.CORRESPOND and V4.T_HEAD_MEASURE == T2.CRIT_HEAD_MEASURE and V4.T_HEAD == T2.CRIT_HEAD
    assert NC._ANCHOR.search(V4.T_ANCHOR_V1.format(name="기공", cid="KRA27-T15"))
    assert NC._CRIT_LIST.match(V4.T_HEAD + "모든 두께: 크기와 무관하게 허용하지 않는다.")


def test_세_목록은_겹치지_않는다() -> None:
    doc = V4.load_lists()
    groups = {"조건 표지": doc["H2"]["exempt_conditional"], "적용 술어": doc["ra_apply"],
              "단정의 부정": doc["ra_not_assertion"]["after_da"] + doc["ra_not_assertion"]["after_hae"]}
    ns = V4._ns
    for an, aw in groups.items():
        for bn, bw in groups.items():
            if an < bn:
                assert not [(p, q) for p in aw for q in bw if ns(p) in ns(q) or ns(q) in ns(p)], (an, bn)


def test_별칭_표의_종별_조항은_2판_defect_class_로_고른_조항과_같다() -> None:
    from corpus.rules import limits_loader

    doc = V4.load_lists()
    t2 = limits_loader.load_limits(REPO / V4.LIMITS_REL, pilot=True)
    by_class = collections.defaultdict(set)
    for r in t2.rows:
        if r.defect_class is not None and r.material.value == "ST":
            by_class[r.defect_class.value].add(r.clause_id)
    class_alias = {a["alias"]: a["clause_id"] for a in doc["derived"]["alias_table"] if re.fullmatch(r"제\d종", a["alias"])}
    assert class_alias and all(by_class.get(k) == {v} for k, v in class_alias.items())


def test_정답_조항은_주_조항이고_보조_전용은_보조_빼기_주다(chk) -> None:
    assert chk.t.gold_clauses("ST", ["2011"]) == ["KRA27-T15"]
    assert chk.t.auxiliary_only("ST", ["2011"]) == {"KRA27-T16"}
    assert chk.t.auxiliary_only("ST", ["2011", "301"]) == set()
    assert chk.t.gold_clauses("AL", ["2011"]) == []


# ── B 회귀 v1(기대 변경 표의 둘은 그 표의 기대로 — v3 의 B 회귀 항목은 v2 그대로다)

def _breg():
    doc = json.loads(BREG.read_text(encoding="utf-8"))
    changes = {c["id"]: c for c in json.loads(EXPECT.read_text(encoding="utf-8"))["changes"] if c["source"] == "breg"}
    out = []
    for c in doc["cases"]:
        exp = dict(c["expect"])
        if c["id"] in changes:
            exp.update({k: v for k, v in changes[c["id"]]["new"].items() if k in ("kinds", "sublabels")})
        out.append((c, exp))
    return out


@pytest.mark.parametrize("case,exp", _breg(), ids=[c["id"] for c, _ in _breg()])
def test_B_회귀(chk_breg, case, exp) -> None:
    ctx = case["context"]
    r = run(chk_breg, case["text"], ctx["material"], ctx["codes"])
    assert sorted(r.flagged, key=V4.KINDS.index) == exp["kinds"]
    assert r.subtags & FOUR == set(exp["sublabels"])
    if "anchor" in exp:
        assert bool(r.criteria) and all(c["anchored"] for c in r.criteria) == exp["anchor"]
    if "list_allowed" in exp:
        assert all(c["licensed"] for c in r.criteria) == exp["list_allowed"]


# ── 쓰임 탐침 v2 — 기대 글을 판정 칸으로 읽는다

EX_HEAD = "분자 밖(면제)"
RA = {"(라) · '조건부 대응의 단정'": "조건부 대응의 단정", "(라) · '조건부 소개 · 인용'": "조건부 소개 · 인용", "(라) · 꼴 표지 차례 4": None}


def probe_checks(expect: str) -> list[tuple[str, object]]:
    """탐침의 기대 글 → (칸, 값). 모르는 글은 빈 목록이 아니라 ('모름', 글)로 돌려 시험이 멈추게 한다."""
    head = expect.split(" — ", 1)[0]
    out: list[tuple[str, object]] = []
    if head.startswith(EX_HEAD):
        out += [("라", False), ("diag", "보조 조건부 소개")]
    elif head.startswith("(라) '정답 밖' · '조건부 소개 · 인용'"):
        out += [("라", True), ("mode", "조건부 소개 · 인용")]
    elif any(head.startswith(k) for k in RA):
        key = next(k for k in RA if head.startswith(k))
        out += [("라", True), ("mode", RA[key])]
        if "진단 '보조 면제 막힘'" in head:
            out.append(("diag", "보조 면제 막힘(다른 작은 마디의 단정)"))
    elif head.startswith("(다) 없음"):
        out += [("다", False)]
    elif head.startswith(("(다) '적용 전제 단정'", "(다) '군집 · 간격 단정'")):
        out += [("dsub", head.split("'")[1])]
    elif head.startswith("r_note 인정 아님"):
        out += [("r_note", False)]
    elif head.startswith("r_note 인정"):
        out += [("r_note", True)]
    elif head.startswith("걸림 없음"):
        out += [("none", True)]
    else:
        m = re.match(r"작은 마디 (\d)", head)
        rest = head
        if m:
            out.append(("small", int(m.group(1))))
            rest = head[m.end():].lstrip(" ·")
        if rest.startswith("(나) H1"):
            out.append(("나", True))
        elif rest.startswith("(나) 없음"):
            out.append(("나", False))
        elif rest and not m:
            out.append(("모름", expect))
    return out or [("모름", expect)]


KNOWN_LIST_GAPS = {
    ("reservation.reasons.r_note.presence.reject_quote", "단서가 있다고 보기 어려워 판정을 유보한다."):
        ("목록 v4-2 의 빈틈 — ra_not_assertion.after_da 의 '고 보기 어렵' 은 '어려워'(ㅂ 불규칙 활용)에 걸리지 않아 거부 꼴이 서지 않는다. "
         "탐침의 기대는 규칙의 뜻이고, 검사기는 목록대로다(목록 · 기대는 이 판에서 고치지 않는다 — 보고 · 다음 판의 몫)"),
}
"""목록대로 구현하면 기대와 어긋나는 탐침 — 엄격한 xfail 로 고정한다(고쳐지면 시험이 알린다)."""


def _probes():
    doc = json.loads(PROBES.read_text(encoding="utf-8"))
    out = []
    for i, p in enumerate(doc["probes"]):
        gap = KNOWN_LIST_GAPS.get((p["list"], p["text"]))
        out.append(pytest.param(i, p, marks=pytest.mark.xfail(strict=True, reason=gap)) if gap else (i, p))
    return out


def _probe_id(x) -> str:
    i, p = x.values if hasattr(x, "values") else x
    return f"{i}-{p['list']}-{p['word']}-{p['kind']}"


def _small_count(chk: V4.Checker, text: str) -> int:
    t = NC.norm(text)
    return sum(len(chk.smalls(s[a:b])) for s in NC.sentences(t) for a, b in V4.contrast_spans(s))


@pytest.mark.parametrize("i,probe", _probes(), ids=[_probe_id(x) for x in _probes()])
def test_쓰임_탐침_v2(chk, i, probe) -> None:
    text = probe["text"].replace("{보조}", "KRA27-T16").replace("{주}", "KRA27-T15")
    r = run(chk, text)
    for key, want in probe_checks(probe["expect"]):
        assert key != "모름", f"기대 글을 읽지 못했다: {want}"
        if key == "라":
            assert ("라" in r.flagged) == want
        elif key == "mode":
            modes = r.subtags_by_kind["라"] & {"조건부 대응의 단정", "조건부 소개 · 인용"}
            assert modes == ({want} if want else set())
        elif key == "diag":
            assert want in r.diagnostic_tags
        elif key == "다":
            assert ("다" in r.flagged) == want
        elif key == "dsub":
            assert want in r.subtags_by_kind["다"]
        elif key == "r_note":
            assert ("r_note" in r.found) == want
        elif key == "none":
            assert not r.flagged
        elif key == "small":
            assert _small_count(chk, text) == want
        elif key == "나":
            assert ("나" in r.flagged) == want


# ── 정답 자기 시험(§11) — v4 × 2 판 정답 서술(b1′ 견본)

@pytest.mark.skipif(not B1P.is_file(), reason="b1′ 산출 폴더가 없다 — 전량은 회귀 기록 13b 가 잰다")
def test_정답_서술은_금지_0_유보_전부_기준_목록_전부_허가_라_0(chk) -> None:
    n = bad = 0
    for k, line in enumerate(B1P.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or k % B1P_STEP:
            continue
        rec = json.loads(line)
        codes = [d["type"] for d in rec["skeleton"]["defects"]]
        r = chk.check(rec["target_text"], material=rec["material"], gold_codes=codes, structured_codes=codes)
        n += 1
        if r.flagged or r.reservation_complete is False or not all(c["licensed"] for c in r.criteria) or r.ra_items:
            bad += 1
    assert n > 0 and bad == 0
