"""구절 은행 v3 — 행 꼴 · 등록부 결속 · 소비 경계.

소비 경계의 반례는 합성 행으로 만든다. 본문 은행은 추적하지 않으므로 환경변수
`PASSAGE_BANK_FULL` 이 그 파일을 가리킬 때만 실물로 본다.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

from corpus.parse import build_passage_bank as B
from corpus.parse.passage_bank import (
    check_row_shape,
    gate_reasons,
    load_jsonl,
    paid_sentence_reason,
    pointer_row,
    select_for_generation,
    text_sha256,
)

REPO = Path(__file__).resolve().parents[2]
POINTER = REPO / "corpus/parse/passage_bank_v3.jsonl"
REGISTRY = REPO / "corpus/parse/sources.yaml"
FULL = Path(os.environ["PASSAGE_BANK_FULL"]) if os.environ.get("PASSAGE_BANK_FULL") else None

SHA_A = "a" * 64
TEXT = "Porosity is caused by inadequate shielding. Clean the joint before welding to prevent it."


def _registry() -> dict:
    return {"documents": [{"doc_id": "DOC-A", "sha256": SHA_A}]}


def _row(**over) -> dict:
    r = {
        "schema_version": 2, "passage_id": "DOC-A:p001:1:00", "doc_id": "DOC-A", "edition": "2020",
        "source_file_sha256": SHA_A,
        "locator": {"pdf_page_start": 1, "pdf_page_end": 1, "section_id": "1"},
        "text_kind": "verbatim_extract", "text": TEXT, "text_sha256": text_sha256(TEXT),
        "facets": ["cause", "prevention"], "defect_classes": ["porosity"],
        "lineage_group_id": "LG-A", "scope": "test scope", "rights_status": "pending", "flags": [],
    }
    r.update(over)
    return r


def _rights(**over) -> dict:
    r = {"doc_id": "DOC-A", "source_file_sha256": SHA_A, "rights_status": "approved",
         "approved_uses": ["qa_generation"], "decided_by": "user", "decided_at": "2026-10-08"}
    r.update(over)
    return r


def _review(**over) -> dict:
    r = {"passage_id": "DOC-A:p001:1:00", "text_sha256": text_sha256(TEXT), "decision": "approved",
         "reviewer": "reviewer-1", "reviewed_at": "2026-10-08"}
    r.update(over)
    return r


def _gate(full, *, pointer=None, rights=(), review=(), registry=None, use="qa_generation"):
    pointer = [pointer_row(r) for r in full] if pointer is None else pointer
    return select_for_generation(full, pointer_rows=pointer, registry=registry or _registry(),
                                 rights_records=rights, review_records=review, use=use)


# --------------------------------------------------------------------------- 소비 경계


def test_권리_결정과_사람_검수가_모두_있어야_넘긴다():
    sel, why = _gate([_row()], rights=[_rights()], review=[_review()])
    assert [r["passage_id"] for r in sel] == ["DOC-A:p001:1:00"]
    assert why["n_rejected"] == 0


def test_기록이_없으면_넘기지_않는다():
    sel, why = _gate([_row()])
    assert sel == []
    assert why["rights:no_decision"] == 1 and why["review:none"] == 1


def test_은행_행의_approved_로는_권리를_건너뛰지_못한다():
    sel, why = _gate([_row(rights_status="approved")], review=[_review()])
    assert sel == [] and why["rights:no_decision"] == 1


@pytest.mark.parametrize("over, reason", [
    ({"rights_status": "pending"}, "rights:pending"),
    ({"rights_status": "denied"}, "rights:denied"),
    ({"approved_uses": ["rag_index"]}, "rights:use_not_approved:qa_generation"),
    ({"decided_by": ""}, "rights:no_decider"),
])
def test_권리_기록의_반례(over, reason):
    sel, why = _gate([_row()], rights=[_rights(**over)], review=[_review()])
    assert sel == [] and why[reason] == 1


@pytest.mark.parametrize("over, reason", [
    ({"decision": "rejected"}, "review:rejected"),
    ({"text_sha256": "0" * 64}, "review:text_sha256_mismatch"),
    ({"reviewer": None}, "review:no_reviewer"),
])
def test_사람_검수_기록의_반례(over, reason):
    sel, why = _gate([_row()], rights=[_rights()], review=[_review(**over)])
    assert sel == [] and why[reason] == 1


def test_본문을_고치면_지문에서_걸린다():
    bad = _row(text=TEXT + " Extra sentence added later.")
    sel, why = _gate([bad], pointer=[pointer_row(_row())], rights=[_rights()], review=[_review()])
    assert sel == [] and why["text_sha256_mismatch"] == 1


def test_포인터_은행과_어긋나면_걸린다():
    sel, why = _gate([_row()], pointer=[pointer_row(_row(edition="2019"))],
                     rights=[_rights()], review=[_review()])
    assert sel == [] and why["pointer_mismatch:edition"] == 1
    sel, why = _gate([_row()], pointer=[], rights=[_rights()], review=[_review()])
    assert sel == [] and why["no_pointer_row"] == 1


def test_등록부에_없거나_지문이_다르면_걸린다():
    sel, why = _gate([_row()], rights=[_rights()], review=[_review()],
                     registry={"documents": [{"doc_id": "DOC-B", "sha256": SHA_A}]})
    assert sel == [] and why["registry:unregistered"] == 1
    sel, why = _gate([_row()], rights=[_rights()], review=[_review()],
                     registry={"documents": [{"doc_id": "DOC-A", "sha256": "b" * 64}]})
    assert sel == [] and why["registry:sha256_mismatch"] == 1


def test_재서술_본문은_넘기지_않는다():
    sel, why = _gate([_row(text_kind="paraphrase")], rights=[_rights()], review=[_review()])
    assert sel == [] and why["text_kind_not_verbatim"] == 1


def test_유료_표준_수치_문장은_승인이_있어도_넘기지_않는다():
    text = TEXT + " AWS D1.5 limits undercut to 1/32 in. for primary members."
    row = _row(text=text, text_sha256=text_sha256(text))
    sel, why = _gate([row], rights=[_rights()], review=[_review(text_sha256=text_sha256(text))])
    assert sel == [] and why["paid_standard_numeric"] == 1


def test_같은_열쇠의_기록이_둘이면_거부한다():
    with pytest.raises(ValueError):
        _gate([_row()], rights=[_rights(), _rights()], review=[_review()])
    with pytest.raises(ValueError):
        _gate([_row()], rights=[_rights()], review=[_review(), _review()])
    with pytest.raises(ValueError):
        _gate([_row(), _row()], rights=[_rights()], review=[_review()])


def test_모르는_용도는_거부한다():
    with pytest.raises(ValueError):
        _gate([_row()], use="fine_tune_everything")


@pytest.mark.parametrize("sentence, reason", [
    ("Undercut is limited to 1/32 in. by AWS D1.5 clause 3.6.", "paid_standard_numeric"),
    ("ISO 5817 level B allows 0.5 mm.", "paid_standard_numeric"),
    ("Porosity appears as rounded voids (AWS, 2010).", "third_party_quote_aws"),
    ("The Bridge Welding Code governs steel bridges.", None),
    ("Undercut deeper than 0.5 mm shall be repaired.", None),
])
def test_유료_표준_문장_규칙(sentence, reason):
    assert paid_sentence_reason(sentence) == reason


def test_FHWA_의_조항_번호는_D1_5_참조로_센다():
    """FHWA 물리 25쪽: "(clause x)" 는 Code(AWS D1.5)의 조항 번호다. 이름 없이도 수치와 함께면 덜어 낸다."""
    s = "Undercut parallel to the tensile stress may be 1/32 inch deep (clause 6.26.1.5(2))."
    assert paid_sentence_reason(s) is None
    assert paid_sentence_reason(s, "FHWA-HIF-19-088") == "paid_standard_numeric"
    assert paid_sentence_reason("Repairs follow clause 3.7.", "FHWA-HIF-19-088") is None
    assert paid_sentence_reason("See Table 8.5 for 0.5 mm undercut.", "IACS47") is None
    assert paid_sentence_reason("See table 6.1 for 0.5 mm undercut.", "FHWA-HIF-19-088") == "paid_standard_numeric"


def test_계보_미확인_블록은_권리_기록이_그_표지를_풀어야_넘긴다():
    row = _row(flags=["L2_lineage_unverified"])
    sel, why = _gate([row], rights=[_rights()], review=[_review()])
    assert sel == [] and why["rights:lineage_not_cleared:L2_lineage_unverified"] == 1
    sel, _ = _gate([row], rights=[_rights(cleared_lineage_flags=["L2_lineage_unverified"])], review=[_review()])
    assert len(sel) == 1


def test_위험_표지는_사람_검수가_확인했다고_적어야_넘긴다():
    row = _row(flags=["pua_unmapped:3", "table_text_layer"])
    sel, why = _gate([row], rights=[_rights()], review=[_review(flags_checked=["table_text_layer"])])
    assert sel == [] and why["review:flag_unchecked:pua_unmapped"] == 1
    sel, _ = _gate([row], rights=[_rights()], review=[_review(flags_checked=["pua_unmapped", "table_text_layer"])])
    assert len(sel) == 1


@pytest.mark.parametrize("uses", ["NOT_qa_generation_denied", {"qa_generation": False}, ["qa_generation", 1],
                                  ["qa_generation", "everything"]])
def test_승인_용도의_꼴이_틀리면_넘기지_않는다(uses):
    sel, why = _gate([_row()], rights=[_rights(approved_uses=uses)], review=[_review()])
    assert sel == [] and why["rights:approved_uses_malformed"] == 1


@pytest.mark.parametrize("field,value", [("lineage_group_id", "LG-B"), ("scope", "다른 범위"),
                                         ("flags", ["table_text_layer"]), ("facets", ["definition"])])
def test_포인터와_본문_은행은_본문_말고_모든_칸이_같아야_한다(field, value):
    sel, why = _gate([_row()], pointer=[pointer_row(_row(**{field: value}))], rights=[_rights()], review=[_review()])
    assert sel == [] and why[f"pointer_mismatch:{field}"] == 1


def test_같은_키가_두_번_있는_줄은_읽지_않는다(tmp_path):
    f = tmp_path / "dup.jsonl"
    f.write_text('{"decision": "rejected", "decision": "approved"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(f)


@pytest.mark.parametrize("over,bad", [
    ({"locator": {"pdf_page_start": True, "pdf_page_end": True}}, "locator:pdf_page"),
    ({"passage_id": 1}, "passage_id:type"),
    ({"flags": ["historical_only"], "facets": ["cause"]}, "facet:historical_row_with_modern_facet"),
    ({"schema_version": 1}, "schema_version"),
])
def test_행_꼴의_반례(over, bad):
    assert bad in check_row_shape(_row(**over))


def test_경계를_직접_불러도_모르는_용도는_거부한다():
    with pytest.raises(ValueError):
        gate_reasons(_row(), pointer=pointer_row(_row()), registry_entry={"sha256": SHA_A}, rights=None,
                     review=None, use="anything")


@pytest.mark.parametrize("sentence,doc", [
    ("Per D1.5, a minimum 7-inch thickness is needed.", None),
    ("The Bridge Welding Code allows skews up to 77 degrees.", None),
    ("AWS D1.5 requires a ¾-inch specimen.", None),
    ("The Code requires 77 inches in every 77 feet.", "FHWA-HIF-19-088"),
    ("At least 77 percent is required (clause C-9.99.9).", "FHWA-HIF-19-088"),
    ("The AASHTO design limit is 5t for inside bends.", None),
])
def test_유료_표준_수치의_빈틈_반례(sentence, doc):
    assert paid_sentence_reason(sentence, doc) == "paid_standard_numeric"


@pytest.mark.parametrize("sentence", [
    "Weld metal toughness follows (AASHTO/AWS, 2015) guidance.",
    "Porosity is shown in figure 3 (AWS, 2010c.).",
    "Undercut, as defined by AWS 3.0, is a groove melted into the base metal.",
])
def test_AWS_출처_인용의_변형(sentence):
    assert paid_sentence_reason(sentence) == "third_party_quote_aws"


def test_옮긴_구절의_텍스트_층_대조():
    page = "Blowhole.-A hole or cavity formed in the weld by trapped gases escaping from the hot metal."
    assert B.text_layer_check("A hole or cavity formed in the weld", page) == "exact"
    assert B.text_layer_check("A hole, or cavity formed in the weld", page) == "alnum"
    assert B.text_layer_check("A crack formed in the weld", page) == "failed"


@pytest.mark.parametrize("over,reason", [
    ({"source_policy_holds": ["paid_standard_numeric"]}, "hold:source_policy:paid_standard_numeric"),
    ({"context_dependency_policy_holds": ["X:p03"]}, "hold:context:X:p03"),
    ({"source_policy_holds": "paid_standard_numeric"}, "hold:source_policy:malformed"),
])
def test_보류_칸은_권리_검수_기록이_있어도_넘기지_않는다(over, reason):
    sel, why = _gate([_row(**over)], rights=[_rights()], review=[_review()])
    assert sel == [] and why[reason] == 1


# --------------------------------------------------------------------------- 빌더 규칙


def test_사용자_영역_글자와_줄끝_하이픈():
    raw = "D " + chr(0xF0A3) + " 0.5 mm " + chr(0xE042) + " weld" + chr(0xFFFE)
    out, c = B.clean_chars(raw)
    assert "≤" in out and B.UNMAPPED_MARK in out and out.endswith(B.JOIN)
    assert c["pua_symbol_mapped"] == 1 and c["pua_unmapped"] == 1 and c["soft_hyphen_joined"] == 1


def test_일렉트로슬래그는_슬래그_혼입이_아니다():
    _, defects, _ = B.classify("Electro-slag welding (72x) and electroslag welding are processes.")
    assert "slag_inclusion" not in defects
    _, defects, _ = B.classify("Slag inclusions are trapped between passes.")
    assert "slag_inclusion" in defects


def test_학술지_쪽_머리글은_쪽을_넘는_구절_대조에서만_뺀다():
    page = "the weld pool did not\nMetals 2020, 10, 1121 6 of 14\nhave time to escape"
    assert "Metals 2020" not in B.strip_running_heads(page)
    assert B.text_layer_check("the weld pool did not have time to escape", B.strip_running_heads(page)) == "exact"


def test_JATS_본문만_대조에_쓴다(tmp_path):
    x = tmp_path / "a.xml"
    x.write_text("<article><front><p>Front matter only</p></front><body><sec><p>Hydrogen-assisted "
                 "<italic>cracking</italic> occurred.</p></sec></body></article>", encoding="utf-8")
    body = B.xml_body_text(x)
    assert "Front matter" not in body and B.text_layer_check("Hydrogen-assisted cracking occurred.", body) == "exact"


def test_긴_문장은_공백에서_자르고_표시한다():
    s = " ".join(["word"] * 400)
    pieces = B.hard_split(s)
    assert all(len(p) <= B.MAX_CHARS for p in pieces) and " ".join(pieces) == s
    wins = B.windows([(s, 1, 2)])
    assert all(cut for _, cut in wins)
    # 문장 중간에서 자른 조각도 그 문장이 걸친 쪽 전체(1~2)를 물려받는다 — 쪽을 좁게 적지 않는다.
    assert all(u[1:] == (1, 2) for win, _ in wins for u in win)


# --------------------------------------------------------------------------- 실물 은행 · 등록부


@pytest.fixture(scope="module")
def pointer_rows() -> list[dict]:
    return load_jsonl(POINTER)


@pytest.fixture(scope="module")
def registry() -> dict:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return {e["doc_id"]: e for e in data["documents"]}


def test_포인터_은행의_행_꼴과_본문_부재(pointer_rows):
    assert pointer_rows, "포인터 은행이 비었다"
    ids = [r["passage_id"] for r in pointer_rows]
    assert len(ids) == len(set(ids))
    for r in pointer_rows:
        assert check_row_shape(r) == [], r["passage_id"]
        assert r["text"] is None, "포인터 은행에 본문이 들어갔다 — 원문 전재가 추적된다"
        assert r["rights_status"] == "pending" and r["human_review"] is None
        assert r["text_kind"] == "verbatim_extract"
        assert r["scope"], r["passage_id"]


def test_포인터_은행의_문서가_등록부와_결속된다(pointer_rows, registry):
    specs = {**B.DOCS, **B.CURATED}
    for doc_id in Counter(r["doc_id"] for r in pointer_rows):
        assert doc_id in registry, doc_id
        if doc_id in specs:
            assert registry[doc_id]["sha256"] == specs[doc_id].sha256
    for r in pointer_rows:
        assert r["source_file_sha256"] == registry[r["doc_id"]]["sha256"]
        assert r["registry"] == "corpus/parse/sources.yaml"
        if r["doc_id"] in specs:
            assert r["lineage_group_id"] == specs[r["doc_id"]].lineage_group_id


def test_새로_등록한_문서는_결정_대기다(registry):
    for doc_id in ("ABS-PUB87-2007", "IACS-URW33-R1C1", "FHWA-HIF-19-088", "US-WAR-DEPT-TM-1-430-1942",
                   "MDPI-APPLSCI-2020-10-1878", "mdpi_met11060904_2021", "mdpi_met11020293_2021",
                   "MDPI-CRYSTALS-2022-12-832", "MDPI-METALS-2024-14-857", "MDPI-METALS-2020-10-1121"):
        e = registry[doc_id]
        assert e["license_class"] == "PENDING_DECISION" and e["allowed_use"] == []


def test_빌드_보고와_용량_보고가_은행과_맞는다(pointer_rows):
    raw = POINTER.read_bytes()
    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    rep = json.loads((REPO / "corpus/parse/passage_bank_v3.build_report.json").read_text(encoding="utf-8"))
    cap = json.loads((REPO / "corpus/parse/passage_bank_v3.capacity.json").read_text(encoding="utf-8"))
    assert cap["full_bank_sha256"] == rep["outputs"]["full_bank_sha256"]
    assert rep["outputs"]["pointer_bank_sha256"] == sha and rep["outputs"]["rows"] == len(pointer_rows)
    assert cap["bank_sha256"] == sha
    assert cap["qa"]["all_tiers"]["pool"]["passages"] == len(pointer_rows)
    assert cap["qa"]["all_tiers"]["pool"]["qa_upper"] == 3 * len(pointer_rows)
    assert cap["passing_generation_gate_now"] == 0


def test_역사_자료_구절은_현대_면을_달지_않고_범위를_적는다(pointer_rows):
    hist = [r for r in pointer_rows if "historical_only" in r["flags"]]
    assert len(hist) == 6
    for r in hist:
        assert not ({"cause", "prevention", "repair", "criteria"} & set(r["facets"]))
        assert r["doc_id"] == "US-WAR-DEPT-TM-1-430-1942" and r["use_boundary"]


def test_계보_미확인_블록_표지가_쪽_범위와_맞는다(pointer_rows):
    for r in pointer_rows:
        if r["doc_id"] != "KR-RULES-P2":
            continue
        pages = set(range(r["locator"]["pdf_page_start"], r["locator"]["pdf_page_end"] + 1))
        assert ("L2_lineage_unverified" in r["flags"]) == bool(pages & {327, 328})
        assert ("UT_lineage_unverified" in r["flags"]) == bool(pages & {328, 329})


def test_합친_원문_보강분의_문맥_보류_표지가_보존된다(pointer_rows):
    ids = {r["passage_id"] for r in pointer_rows}
    merged = [r for r in pointer_rows if r.get("merged_from")]
    assert len(merged) == 58
    for r in merged:
        assert set(r.get("required_context_passage_ids") or []) <= ids
        assert r["extraction"]["bank_v3_source_check"].split("_", 1)[1] in ("exact", "alnum")
    held = {r["passage_id"] for r in merged if r.get("source_policy_holds")}
    dependents = {r["passage_id"] for r in merged if r.get("context_dependency_policy_holds")}
    assert held == {"mdpi_met11060904_2021_p03"} and len(dependents) == 11


@pytest.mark.skipif(FULL is None or not FULL.exists(), reason="본문 은행(PASSAGE_BANK_FULL)이 없다")
def test_실물_본문_은행은_지금_하나도_넘어가지_않는다(pointer_rows):
    full = load_jsonl(FULL)
    assert [r["passage_id"] for r in full] == [r["passage_id"] for r in pointer_rows]
    sel, why = select_for_generation(full, pointer_rows=pointer_rows, registry=REGISTRY)
    assert sel == []
    assert why["rights:no_decision"] == len(full) and why["review:none"] == len(full)
    # 지문 · 포인터 · 등록 결속에서는 걸리는 행이 없어야 한다 — 걸리면 은행이 깨진 것이다.
    broken = {k: v for k, v in why.items()
              if k.startswith(("text_sha256", "pointer_", "no_pointer", "registry:", "shape:", "no_text"))}
    assert broken == {}


# --------------------------------------------------------------------------- 필수 문맥 · 단일 진입점

T2 = "Clean the joint before welding."
T3 = "Shielding gas flow must be steady."


def _chain_rows(**c_over):
    """P1 ← P2 ← P3 사슬(P2 는 P1 을, P3 은 P2 를 필수 문맥으로). 셋 다 같은 문서 · 계보."""
    p1 = _row()
    p2 = _row(passage_id="DOC-A:p002:2:00", text=T2, text_sha256=text_sha256(T2),
              locator={"pdf_page_start": 2, "pdf_page_end": 2}, required_context_passage_ids=["DOC-A:p001:1:00"])
    p3 = _row(passage_id="DOC-A:p003:3:00", text=T3, text_sha256=text_sha256(T3),
              locator={"pdf_page_start": 3, "pdf_page_end": 3}, required_context_passage_ids=["DOC-A:p002:2:00"])
    p3.update(c_over)
    return [p1, p2, p3]


def _reviews(rows):
    return [_review(passage_id=r["passage_id"], text_sha256=r["text_sha256"]) for r in rows]


def test_필수_문맥이_모두_넘어가면_의존_구절도_넘긴다():
    rows = _chain_rows()
    sel, why = _gate(rows, rights=[_rights()], review=_reviews(rows))
    assert [r["passage_id"] for r in sel] == [r["passage_id"] for r in rows] and why["n_rejected"] == 0


def test_필수_문맥이_못_넘어가면_사슬_끝까지_빠진다():
    rows = _chain_rows()
    sel, why = _gate(rows, rights=[_rights()], review=_reviews(rows[1:]))     # P1 의 검수 기록이 없다
    assert sel == []
    assert why["review:none"] == 1 and why["context:required_not_selectable"] == 2 and why["n_rejected"] == 3


def test_필수_문맥이_보류면_의존_구절이_빠진다():
    rows = _chain_rows()
    rows[0]["source_policy_holds"] = ["paid_standard_numeric"]
    sel, why = _gate(rows, rights=[_rights()], review=_reviews(rows))
    assert sel == [] and why["hold:source_policy:paid_standard_numeric"] == 1


@pytest.mark.parametrize(("over", "reason"), [
    ({"required_context_passage_ids": ["DOC-A:p999:9:00"]}, "context:required_missing"),
    ({"required_context_passage_ids": "DOC-A:p002:2:00"}, "context:malformed"),
    ({"lineage_group_id": "LG-OTHER"}, "context:cross_lineage"),
])
def test_필수_문맥의_꼴이_틀리면_빠진다(over, reason):
    rows = _chain_rows(**over)
    sel, why = _gate(rows, rights=[_rights()], review=_reviews(rows))
    assert [r["passage_id"] for r in sel] == [r["passage_id"] for r in rows[:2]] and why[reason] == 1


def _bank_files(tmp_path, rows, *, report_pointer=None):
    from corpus.parse.passage_bank import dump_jsonl

    full_sha = dump_jsonl(rows, tmp_path / "bank.jsonl.full")
    ptr_sha = dump_jsonl([pointer_row(r) for r in rows], tmp_path / "bank.jsonl")
    (tmp_path / "bank.build_report.json").write_text(json.dumps(
        {"outputs": {"full_bank_sha256": full_sha, "pointer_bank_sha256": report_pointer or ptr_sha}}), encoding="utf-8")
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump(_registry()), encoding="utf-8")
    dump_jsonl([_rights()], tmp_path / "rights.jsonl")
    dump_jsonl(_reviews(rows), tmp_path / "review.jsonl")
    return ptr_sha, full_sha


def test_단일_진입점은_지문을_묶고_사유별_수를_낸다(tmp_path):
    from corpus.parse.passage_bank import select_bank_for_generation

    rows = _chain_rows()
    ptr_sha, full_sha = _bank_files(tmp_path, rows)
    kw = {"registry": tmp_path / "reg.yaml", "rights_path": tmp_path / "rights.jsonl"}
    sel, summ = select_bank_for_generation(tmp_path / "bank.jsonl", tmp_path / "bank.jsonl.full", **kw)
    assert sel == [] and summ["selected"] == 0 and summ["excluded"] == 3 and summ["reasons"] == {"review:none": 3}
    assert summ["inputs_sha256"]["pointer_bank"] == ptr_sha and summ["inputs_sha256"]["full_bank"] == full_sha
    assert summ["inputs_sha256"]["review_records"] is None
    sel, summ = select_bank_for_generation(tmp_path / "bank.jsonl", tmp_path / "bank.jsonl.full",
                                           review_path=tmp_path / "review.jsonl", **kw)
    assert [r["passage_id"] for r in sel] == [r["passage_id"] for r in rows] and summ["reasons"] == {}
    assert all(isinstance(r["text"], str) for r in sel)


def test_단일_진입점은_본문_은행이_없으면_넘기지_않는다(tmp_path):
    from corpus.parse.passage_bank import select_bank_for_generation

    _bank_files(tmp_path, _chain_rows())
    sel, summ = select_bank_for_generation(tmp_path / "bank.jsonl", None, registry=tmp_path / "reg.yaml")
    assert sel == [] and summ["reasons"] == {"no_full_bank": 3} and summ["inputs_sha256"]["full_bank"] is None


def test_단일_진입점은_빌드_보고와_다른_은행을_거부한다(tmp_path):
    from corpus.parse.passage_bank import BankRefused, select_bank_for_generation

    _bank_files(tmp_path, _chain_rows(), report_pointer="0" * 64)
    with pytest.raises(BankRefused) as err:
        select_bank_for_generation(tmp_path / "bank.jsonl", tmp_path / "bank.jsonl.full", registry=tmp_path / "reg.yaml")
    assert err.value.code == "pointer_bank_sha_mismatch"


def test_단일_진입점은_계약이_아닌_은행을_조용히_0으로_만들지_않는다(tmp_path):
    from corpus.parse.passage_bank import BankRefused, select_bank_for_generation

    rows = _chain_rows()
    del rows[0]["scope"]                                          # 초안 꼴처럼 필수 칸이 빠진 은행
    _bank_files(tmp_path, rows)
    with pytest.raises(BankRefused) as err:
        select_bank_for_generation(tmp_path / "bank.jsonl", tmp_path / "bank.jsonl.full", registry=tmp_path / "reg.yaml")
    assert err.value.code == "bank_contract"
    (tmp_path / "bank.build_report.json").unlink()
    with pytest.raises(BankRefused) as err:
        select_bank_for_generation(tmp_path / "bank.jsonl", tmp_path / "bank.jsonl.full", registry=tmp_path / "reg.yaml")
    assert err.value.code == "build_report_missing"


def test_단일_진입점은_본문_행이_없는_포인터_행도_센다(tmp_path):
    from corpus.parse.passage_bank import dump_jsonl, select_bank_for_generation

    rows = _chain_rows()
    _bank_files(tmp_path, rows)
    full_sha = dump_jsonl(rows[:2], tmp_path / "bank.jsonl.full")      # 본문 은행에서 P3 이 빠졌다
    report = json.loads((tmp_path / "bank.build_report.json").read_text(encoding="utf-8"))
    report["outputs"]["full_bank_sha256"] = full_sha
    (tmp_path / "bank.build_report.json").write_text(json.dumps(report), encoding="utf-8")
    sel, summ = select_bank_for_generation(tmp_path / "bank.jsonl", tmp_path / "bank.jsonl.full", registry=tmp_path / "reg.yaml",
                                           rights_path=tmp_path / "rights.jsonl", review_path=tmp_path / "review.jsonl")
    assert len(sel) == 2 and summ["reasons"] == {"no_full_row": 1}
    assert summ["selected"] + summ["excluded"] == summ["bank_rows"] == 3


def test_필수_문맥이_돌아도_멈추고_한쪽이_빠지면_둘_다_빠진다():
    rows = _chain_rows()[:2]
    rows[0]["required_context_passage_ids"] = ["DOC-A:p002:2:00"]       # P1 ↔ P2
    sel, _ = _gate(rows, rights=[_rights()], review=_reviews(rows))
    assert len(sel) == 2
    sel, why = _gate(rows, rights=[_rights()], review=_reviews(rows[1:]))
    assert sel == [] and why["n_rejected"] == 2


def test_합친_행의_상대_경로는_기준점_파일이_지문으로_덮는다():
    anchor = json.loads((REPO / "corpus/parse/passage_bank_v3.provenance.json").read_text(encoding="utf-8"))
    assert anchor["bank_pointer_sha256"] == hashlib.sha256(POINTER.read_bytes()).hexdigest()
    merged = [r for r in load_jsonl(POINTER) if r.get("merged_from")]
    assert {r["passage_id"] for r in merged} == set(anchor["rows"])
    assert {r["merged_from"]["bank_sha256"] for r in merged} == {anchor["relative_paths_base"]["delivery_bank_sha256"]}
    for r in merged:
        got = anchor["rows"][r["passage_id"]]
        for field in anchor["path_fields"]:
            v = r
            for k in field.split("."):
                v = v.get(k) if isinstance(v, dict) else None
            if v is not None:
                assert got[field] == anchor["files"][v]["sha256"], (r["passage_id"], field)
        assert got["extraction.source_pdf"] == r["source_file_sha256"]


# --------------------------------------------------------------------------- run_cycle_corpus 연결


def test_사이클_실행기는_본문_은행이_없으면_구절이_0이다():
    from corpus.generate.run_cycle_corpus import load_passages

    out, gate = load_passages(10, full_bank=None)
    assert out == [] and gate["n_selected"] == 0


def test_사이클_실행기는_소비_경계를_지난_구절만_받는다(tmp_path):
    from corpus.generate.run_cycle_corpus import load_passages
    from corpus.parse.passage_bank import dump_jsonl

    full = [_row(), _row(passage_id="DOC-A:p002:2:00",
                         locator={"pdf_page_start": 2, "pdf_page_end": 2, "section_id": "2"})]
    full_sha = dump_jsonl(full, tmp_path / "full.jsonl")
    ptr_sha = dump_jsonl([pointer_row(r) for r in full], tmp_path / "ptr.jsonl")
    (tmp_path / "report.json").write_text(json.dumps(
        {"outputs": {"full_bank_sha256": full_sha, "pointer_bank_sha256": ptr_sha}}), encoding="utf-8")
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump(_registry()), encoding="utf-8")
    dump_jsonl([_rights()], tmp_path / "rights.jsonl")
    dump_jsonl([_review()], tmp_path / "review.jsonl")

    kw = {"full_bank": tmp_path / "full.jsonl", "pointer_bank": tmp_path / "ptr.jsonl",
          "registry": tmp_path / "reg.yaml", "build_report": tmp_path / "report.json"}
    out, gate = load_passages(10, **kw)
    assert out == [] and gate["n_selected"] == 0
    out, gate = load_passages(10, **kw, rights_path=tmp_path / "rights.jsonl",
                              review_path=tmp_path / "review.jsonl")
    assert [p["passage_id"] for p in out] == ["DOC-A:p001:1:00"]
    assert out[0]["source"]["text_sha256"] == text_sha256(TEXT)
    assert out[0]["source"]["scope"] == "test scope"
    assert gate["reasons"]["review:none"] == 1


def test_사이클_실행기는_빌드_보고와_지문이_다른_은행을_거부한다(tmp_path):
    from corpus.generate.run_cycle_corpus import load_passages
    from corpus.parse.passage_bank import dump_jsonl

    full = [_row()]
    full_sha = dump_jsonl(full, tmp_path / "full.jsonl")
    dump_jsonl([pointer_row(r) for r in full], tmp_path / "ptr.jsonl")
    (tmp_path / "report.json").write_text(json.dumps(
        {"outputs": {"full_bank_sha256": full_sha, "pointer_bank_sha256": "0" * 64}}), encoding="utf-8")
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump(_registry()), encoding="utf-8")
    dump_jsonl([_rights()], tmp_path / "rights.jsonl")
    dump_jsonl([_review()], tmp_path / "review.jsonl")
    out, gate = load_passages(10, full_bank=tmp_path / "full.jsonl", pointer_bank=tmp_path / "ptr.jsonl",
                              registry=tmp_path / "reg.yaml", build_report=tmp_path / "report.json",
                              rights_path=tmp_path / "rights.jsonl", review_path=tmp_path / "review.jsonl")
    assert out == [] and gate["reasons"] == {"pointer_bank_sha_mismatch": 1}


def test_소표본_실행기도_은행_경계만_거친다():
    from corpus.generate import run_pilot

    assert run_pilot.load_passages(10) == []
    assert not hasattr(run_pilot, "PASSAGE_DOCS")
