"""D4 페어 2 판 서술 — 생성기(`pairs_text_v2`)와 게이트 ③(`pair_criteria_gate` 2 판)(C 08 2 판 §4 · §7 의 시험 8 · 9).

- 생성기의 문장은 §4 의 표 글자 그대로, 붙는 규칙(있음 · 없음 · 자리 · 한 번)대로다. JSON 줄 · 골격은 1 판과 같은 바이트다(T4).
- 게이트는 생성기를 import 하지 않고 같은 문장을 따로 옮겼다 — 두 곳의 글자가 같은지, 그리고 평가 쪽 기대 사례의 계약 문장 · 문장 수와
  같은지 맞댄다(두 구현 대조).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from corpus.generate import make_pairs_pilot as P
from corpus.generate import pairs_text_v2 as T2
from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules.limits_loader import load_limits
from corpus.rules.skeleton_gen import load_defect_lexicon
from corpus.validate import pair_criteria_gate as G

REPO = Path(__file__).resolve().parents[2]
NAMES = defect_names()
LEX = load_defect_lexicon()
T1 = load_limits(REPO / "corpus/rules/limits_v0_pilot.csv", pilot=True)
T2_TABLE = load_limits(REPO / "corpus/rules/limits_v2_pilot.csv", pilot=True)
R1 = G.PairRules(T1.rows, NAMES)
R2 = G.PairRules(T2_TABLE.rows, NAMES, edition="v2")


def rec(material: str, codes: list[str], edition: str = "v2", table=None) -> dict:
    """합성 레코드 — 빌더의 조립 함수 그대로(재질 · 코드만 정한다). 상자는 경계 안의 같은 값이다."""
    table = table or (T2_TABLE if edition == "v2" else T1)
    anns = [{"ann_id": f"x#{i}", "geom_valid": True, "iso_code": c, "bbox_x1_px": 1, "bbox_y1_px": 1, "bbox_x2_px": 5,
             "bbox_y2_px": 5, "major_axis_px": 3.0, "equiv_diameter_px": 2.0} for i, c in enumerate(codes)]
    row = {"image_id": "x", "rel_path": "a/b.png", "client": "C1", "split": "train", "material": material,
           "width_px": 10, "height_px": 10, "has_defect": bool(codes)}
    base = P.clause_basis(table, material)
    kw = {}
    if edition == "v2":
        base, kw = T2.augment_basis(base, table, material), {"edition": "v2"}
    r, _ = P.assemble_record(row, anns, NAMES, base, **kw)
    return r


#: §4 표의 문장 꼴 — 조각('. ' 로 나눈 것)의 종류를 가른다. 띠 목록은 머리 뒤의 조각이다.
_FIXED = {T2.AGGREGATION: "C5", T2.MIXTURE: "C6", T2.DISTINCTION: "C7", T2.PREMISE: "C8", T2.NO_CLAUSE_MATERIAL: "C9",
          T2.NOTE_UNEXPRESSED: "C4", T2.CLOSE_DIMENSIONAL: "C11", T2.CLOSE_SIZE_INDEPENDENT: "C12",
          T2.CRIT_HEAD.strip(): "C3b"}


def kinds(text: str) -> list[str]:
    if text == T2.NORMAL_TEXT:
        return ["C13"]
    out = []
    for piece in re.split(r"(?<=\.) ", text):
        if piece in _FIXED:
            out.append(_FIXED[piece])
        elif piece.startswith("방사선투과 영상에서 ") and piece.endswith("가 관찰된다."):
            out.append("C1")
        elif "에 현 허용치 표에서 대응되는 조항은 " in piece:
            out.append("C2")
        elif piece.startswith("이 조항의 측정량은 "):
            out.append("C3a")
        elif piece.endswith(T2.NO_CLAUSE_CODE_TAIL):
            out.append("C10")
        elif out and out[-1] in ("C3a", "C3b"):
            out.append("BAND")
        else:
            out.append("?")
    return out


# ---------------------------------------------------------------- 시험 8 — 생성기

def test_JSON_줄과_골격은_1판과_같은_바이트다():
    for mat, codes in (("ST", ["2011", "301"]), ("ST", ["100"]), ("AL", ["2011"]), ("ST", ["402"]), ("ST", [])):
        a, b = rec(mat, codes, "v1"), rec(mat, codes, "v2")
        assert {k: v for k, v in a.items() if k != "target_text"} == {k: v for k, v in b.items() if k != "target_text"}
        assert json.dumps(a["skeleton"], ensure_ascii=False) == json.dumps(b["skeleton"], ensure_ascii=False)


def test_대상_조합의_문장_차례는_계약_표대로다():
    """강재 기공 + 슬래그혼입 — §4 예문 2 의 꼴(관찰 → 덩이 둘 → C5 · C6 · C7 · C8 → 치수 맺음)."""
    k = kinds(rec("ST", ["2011", "301"])["target_text"])
    assert k == ["C1", "C2", "C3a", "BAND", "C4", "C2", "C3a", "BAND", "C4", "C5", "C6", "C7", "C8", "C11"]


def test_균열만이면_불허_머리와_전제_뒤에_불허_맺음이다():
    k = kinds(rec("ST", ["100"])["target_text"])
    assert k == ["C1", "C2", "C3b", "BAND", "C8", "C12"]


def test_측정량은_값_한계_조항의_기준_목록_머리에_한_번이다():
    text = rec("ST", ["2011", "100"])["target_text"]
    assert text.count("이 조항의 측정량은 최대결함크기이고, 기준은 다음과 같다. ") == 1
    assert text.count(T2.CRIT_HEAD) == 1                                # 불허 조항(균열)의 C3b 머리 하나뿐
    assert "합계길이이고" not in text and rec("ST", ["301"])["target_text"].count("측정량은 합계길이이고") == 1


@pytest.mark.parametrize("codes, mixture", [
    (["2011", "301"], 1), (["2011", "301", "401"], 1),                 # 대상 — 제1종 + 제2종
    (["301", "401"], 0), (["2011", "100"], 0), (["301", "100"], 0), (["2011", "402"], 0), (["2011"], 0)])
def test_혼재_문장은_원천의_종_조합에만_한_번(codes, mixture):
    assert kinds(rec("ST", codes)["target_text"]).count("C6") == mixture


@pytest.mark.parametrize("material, codes, want", [
    ("ST", ["301"], {"C5": 1, "C7": 1, "C8": 1}), ("ST", ["2011"], {"C5": 0, "C7": 1, "C8": 1}),
    ("ST", ["100"], {"C5": 0, "C7": 0, "C8": 1}), ("ST", ["402"], {"C5": 0, "C7": 0, "C8": 0}),
    ("AL", ["2011"], {"C5": 0, "C7": 0, "C8": 0})])
def test_조건_문장은_붙는_규칙대로_있거나_없고_한_번이다(material, codes, want):
    k = kinds(rec(material, codes)["target_text"])
    assert {c: k.count(c) for c in want} == want


def test_미대응_두_꼴과_정상은_계약_문장이다():
    assert kinds(rec("AL", ["2011"])["target_text"]) == ["C1", "C9", "C11"]
    assert kinds(rec("ST", ["2011", "402"])["target_text"])[-2:] == ["C10", "C11"]
    assert rec("ST", [])["target_text"] == P.NORMAL_TEXT == T2.NORMAL_TEXT


def test_군집_조건_조항이_한_영상에_둘이면_멈춘다(tmp_path):
    """합성 표 — 기공 조항도 합계 · 군집 조건을 든다. 영상 수준 문장 하나로는 어느 조항의 조건인지 말할 수 없다(§4 의 C5)."""
    import csv

    with open(REPO / "corpus/rules/limits_v2_pilot.csv", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        head, rows = r.fieldnames, list(r)
    for row in rows:
        if row["clause_id"] == "KRA27-T15":
            row.update(measure="합계길이", aggregation="합계", conditions="구별곤란:제2종|군집간격|무등급전제|혼재:제2종")
    p = tmp_path / "two_cluster.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    table = load_limits(p, pilot=True)
    rec("ST", ["2011"], table=table)                                    # 한 조항뿐이면 받는다
    with pytest.raises(SystemExit, match="군집 조건 조항"):
        rec("ST", ["2011", "301"], table=table)


def test_2판_서술은_2판_표로만_짓는다():
    with pytest.raises(SystemExit, match="2 판 표"):
        T2.augment_basis(P.clause_basis(T1, "ST"), T1, "ST")
    with pytest.raises(P.PairInputError, match="표의 판"):
        P.build_pairs(None, None, T1, NAMES, LEX, edition="v2")


# ---------------------------------------------------------------- 평가 쪽 기대 사례와의 대조(두 구현)

def _vector(name: str, sha: str) -> dict | None:
    rel = f"evaluation/vectors/{name}"
    p = REPO / rel
    raw = p.read_bytes() if p.is_file() else None
    if raw is None:
        out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"91e83979:{rel}"], capture_output=True, check=False)
        raw = out.stdout if out.returncode == 0 else None
    if raw is None:
        return None
    assert hashlib.sha256(raw).hexdigest() == sha, f"{name} 의 바이트가 고정값과 다르다"
    return json.loads(raw)


B_REGRESSION = ("narrative_b_regression.v1.json", "d75253aedf68747dd478d02fe7ba7cd8d15fc4dc446b17aaaa02d309593e04c6")


def test_계약_문장은_평가_쪽_기대_사례의_글자와_같다():
    doc = _vector(*B_REGRESSION)
    if doc is None:
        pytest.skip("평가 쪽 기대 사례를 읽지 못했다")
    s = doc["contract_sentences"]
    assert s["C2"].replace("{이름}", "{name}").replace("{조항}", "{cid}") == T2.CORRESPOND
    assert s["C3a"].split("{띠}")[0].replace("{측정량}", "{measure}") == T2.CRIT_HEAD_MEASURE
    assert s["C3b"].split("{띠}")[0] == T2.CRIT_HEAD
    exact = {"C4": T2.NOTE_UNEXPRESSED, "C5": T2.AGGREGATION, "C6": T2.MIXTURE, "C7": T2.DISTINCTION, "C8": T2.PREMISE,
             "C9": T2.NO_CLAUSE_MATERIAL, "C11": T2.CLOSE_DIMENSIONAL, "C12": T2.CLOSE_SIZE_INDEPENDENT, "C13": T2.NORMAL_TEXT}
    assert {k: s[k] for k in exact} == exact
    assert s["C10"].split("[, …]")[1] == T2.NO_CLAUSE_CODE_TAIL


def test_레코드마다_문장_수가_평가_쪽_기대와_같다():
    """평가 쪽 기대 사례의 `n_items` — 2 판 정답 서술의 문장 수(§4 의 붙는 규칙에서 낸 기대)를 생성기 산출에서 센다."""
    doc = _vector(*B_REGRESSION)
    if doc is None:
        pytest.skip("평가 쪽 기대 사례를 읽지 못했다")
    for item in doc["n_items"]:
        r = item["record"]
        k = kinds(rec(r["material"], r["codes"])["target_text"])
        got = {c: k.count(c) for c in item["v2_sentence_counts"]}
        assert got == item["v2_sentence_counts"], r


# ---------------------------------------------------------------- 시험 9 — 게이트 ③(2 판)

CASES = [("ST", ["2011"]), ("ST", ["301"]), ("ST", ["2011", "301"]), ("ST", ["301", "401"]), ("ST", ["2011", "100"]),
         ("ST", ["100"]), ("ST", ["2011", "402"]), ("ST", ["402"]), ("AL", ["2011"]), ("ST", []),
         ("ST", ["2011", "301", "401"])]


@pytest.mark.parametrize("material, codes", CASES)
def test_2판_서술은_2판_규칙을_지나고_판을_바꾸면_읽지_못한다(material, codes):
    a, b = rec(material, codes, "v1"), rec(material, codes, "v2")
    assert P.check_pair(b, NAMES, R2, LEX, wh=(10, 10)) == []
    assert P.check_pair(a, NAMES, R1, LEX, wh=(10, 10)) == []
    if codes:
        assert P.check_pair(a, NAMES, R2, LEX, wh=(10, 10)) == ["text_unparsable"]      # 1 판 서술 × 2 판 규칙
        assert P.check_pair(b, NAMES, R1, LEX, wh=(10, 10)) == ["text_unparsable"]      # 2 판 서술 × 1 판 규칙


def _with(r: dict, text: str) -> dict:
    return {**r, "target_text": text}


@pytest.mark.parametrize("sentence, reason", [(T2.AGGREGATION, "aggregation_mismatch"), (T2.MIXTURE, "mixture_mismatch"),
                                              (T2.DISTINCTION, "distinction_mismatch"), (T2.PREMISE, "premise_mismatch")])
def test_조건_문장이_빠지면_그_사유다(sentence, reason):
    r = rec("ST", ["2011", "301"])
    assert sentence in r["target_text"]
    assert P.check_pair(_with(r, r["target_text"].replace(" " + sentence, "")), NAMES, R2, LEX, wh=(10, 10)) == [reason]


def test_조건_문장이_없어야_할_레코드에_붙으면_그_사유다():
    r = rec("ST", ["301", "401"])                                     # 혼재 대상 밖
    text = r["target_text"].replace(T2.DISTINCTION, T2.MIXTURE + " " + T2.DISTINCTION)
    assert P.check_pair(_with(r, text), NAMES, R2, LEX, wh=(10, 10)) == ["mixture_mismatch"]


def test_측정량이_틀리거나_빠지면_측정량_사유다():
    r = rec("ST", ["301"])
    wrong = r["target_text"].replace("측정량은 합계길이이고", "측정량은 최대결함크기이고")
    assert P.check_pair(_with(r, wrong), NAMES, R2, LEX, wh=(10, 10)) == ["measure_mismatch"]
    gone = r["target_text"].replace("이 조항의 측정량은 합계길이이고, 기준은 다음과 같다. ", T2.CRIT_HEAD)
    assert P.check_pair(_with(r, gone), NAMES, R2, LEX, wh=(10, 10)) == ["measure_mismatch"]


@pytest.mark.parametrize("edit", ["dup", "swap"])
def test_조건_문장이_겹치거나_자리가_바뀌면_읽지_못한다(edit):
    r = rec("ST", ["2011", "301"])
    t = r["target_text"]
    t = (t.replace(T2.PREMISE, T2.PREMISE + " " + T2.PREMISE) if edit == "dup" else
         t.replace(T2.MIXTURE + " " + T2.DISTINCTION, T2.DISTINCTION + " " + T2.MIXTURE))
    assert P.check_pair(_with(r, t), NAMES, R2, LEX, wh=(10, 10)) == ["text_unparsable"]


def test_1판_대응_문구는_2판_게이트를_지나지_못한다():
    r = rec("ST", ["2011"])
    t = r["target_text"].replace("에 현 허용치 표에서 대응되는 조항은 ", "에 적용되는 조항은 ")
    assert P.check_pair(_with(r, t), NAMES, R2, LEX, wh=(10, 10)) == ["text_unparsable"]


def test_게이트의_2판_문장은_생성기의_문장과_글자가_같다():
    assert G.V2_CORRESPOND_TAIL in T2.CORRESPOND and G.V2_CRIT_HEAD_MEASURE == tuple(T2.CRIT_HEAD_MEASURE.split("{measure}"))
    pairs = {G.V2_AGGREGATION: T2.AGGREGATION, G.V2_MIXTURE: T2.MIXTURE, G.V2_DISTINCTION: T2.DISTINCTION,
             G.V2_PREMISE: T2.PREMISE, G.V2_NO_CLAUSE_MATERIAL: T2.NO_CLAUSE_MATERIAL,
             G.V2_NO_CLAUSE_CODE_TAIL: T2.NO_CLAUSE_CODE_TAIL, G.V2_CLOSE_SIZE_INDEPENDENT: T2.CLOSE_SIZE_INDEPENDENT}
    assert all(a == b for a, b in pairs.items())
    assert G.CRIT_HEAD == T2.CRIT_HEAD and G.NOTE_UNEXPRESSED == T2.NOTE_UNEXPRESSED


def test_게이트의_새_사유는_빌더의_사유_어휘에_든다():
    assert {"measure_mismatch", "aggregation_mismatch", "mixture_mismatch", "distinction_mismatch",
            "premise_mismatch"} <= P.DISCARD_REASONS


def test_게이트는_생성기를_import_하지_않는다():
    import ast

    tree = ast.parse(Path(G.__file__).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in mods if m and m.startswith(("corpus.generate", "corpus.rules"))}


def test_2판_규칙은_1판_표를_받지_않는다():
    with pytest.raises(ValueError, match="2 판 칸"):
        G.PairRules(T1.rows, NAMES, edition="v2")
