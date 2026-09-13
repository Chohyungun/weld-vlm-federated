"""색인→생성 및 export→채점의 실제 경로를 검증한다."""

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from evaluation.adapters import adapt_unified_generations
from rag.index import build_index, load_chunks, queries_from_gold_rows
from rag.judge import judge_image, load_prompt_template
from rag.retrieve import Query, RetrievalResult, retrieve


def meta(material="ST", scheme="none", text="최대 직경은 4 mm 이하이다."):
    return {"clause_id": "rule", "source_docs": ["doc"], "defect_codes": ["2011"],
            "materials": [material], "inspection_methods": ["RT"],
            "quality_schemes": [scheme], "quality_levels": ["ALL"],
            "thickness_min": "0", "thickness_max": "10", "text": text}


def index_file(tmp_path, **kw):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(meta(**kw)), encoding="utf-8")
    return path


def query(**kw):
    return replace(Query("RT", "2011", thickness_mm=Decimal(5), material="ST",
                         quality_scheme="none", quality_level="ALL"), **kw)


def test_embedded_body_reaches_actual_judge_prompt(tmp_path):
    idx = build_index(index_file(tmp_path))
    found = idx.search(query())
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return json.dumps({"verdict": "판정불가", "cited_clauses": ["rule"]})

    judge_image("img", [], found, idx.chunks, generate, load_prompt_template()[0])
    assert len(prompts) == 1
    assert meta()["text"] in prompts[0]
    assert "본문 미등재" not in prompts[0]


def test_explicit_body_override_and_material_are_preserved(tmp_path):
    path = index_file(tmp_path)
    chunks = load_chunks(path, {"rule": "다른 명시 본문"})
    assert chunks[0].text == "다른 명시 본문"
    assert chunks[0].materials == ("ST",)


@pytest.mark.parametrize("case", ["empty", "missing", "wrong"])
def test_bad_evidence_never_calls_generator(tmp_path, case):
    chunks = load_chunks(index_file(tmp_path, text="  " if case == "empty" else "기준"))
    supplied = [] if case == "missing" else chunks
    found = RetrievalResult(("wrong" if case == "wrong" else "rule",), False, 1)
    calls = []
    with pytest.raises(ValueError):
        judge_image("img", [], found, supplied, lambda p: calls.append(p), load_prompt_template()[0])
    assert not calls


@pytest.mark.parametrize("changes", [{"material": "AL"}, {"material": None},
                                     {"quality_scheme": "iso5817"}, {"quality_scheme": None}])
def test_real_meta_filters_reject_incompatible_context(tmp_path, changes):
    chunks = load_chunks(index_file(tmp_path))
    assert not retrieve(chunks, query(**changes)).found
    assert retrieve(chunks, query()).found


def test_all_material_clause_and_explicit_cross_scheme_map(tmp_path):
    chunks = load_chunks(index_file(tmp_path, material="ALL", scheme="iacs"))
    q = query(material="AL", quality_scheme="iso5817")
    assert not retrieve(chunks, q).found
    assert retrieve(chunks, q, {"iacs": {"ALL": ["ALL"]}}).found


def test_gold_query_preserves_material():
    row = {**meta(), "rule_id": "r", "material": "AL", "inspection_method": "RT",
           "defect_code": "2011", "quality_scheme": "none", "quality_level": "ALL"}
    q, _ = queries_from_gold_rows([row], n=1, seed=1)[0]
    assert q.material == "AL"


def test_material_changes_index_digest(tmp_path):
    idx = build_index(index_file(tmp_path))
    other = replace(idx, chunks=(replace(idx.chunks[0], materials=("AL",)),))
    assert idx.snapshot_digest() != other.snapshot_digest()


def adapt(row):
    return adapt_unified_generations([json.dumps(row)], cell="uni_central", seed=1,
                                     known_iso_codes=["2011"])


def test_actual_export_preserves_good_items_and_counts_bad(monkeypatch):
    from scripts import pilot_export_vlm as export
    from vlm.coords import CoordCfg, ImageGeom

    monkeypatch.setattr(export, "COORD_CFG", CoordCfg("ABS_ORIG"))
    good = {"iso_code": "2011", "bbox_2d": [1, 2, 10, 20]}
    bad = [None, {"bbox_2d": [1, 2]}, {"bbox_2d": [1, 2, "oops", 20]},
           {"bbox_2d": [float("nan"), 2, 10, 20]}, {"bbox_2d": [10, 2, 1, 20]}]
    obj = {"defects": [good, *bad], "verdict": "판정불가", "cited_clauses": [],
           "basis": "문자열의 { 괄호"}
    diagnostics = {}
    parsed, error = export.parse_and_backproject(json.dumps(obj), ImageGeom(100, 100),
                                                diagnostics=diagnostics)
    assert error is None
    rep = adapt({"image_id": "i", "bbox_px_parsed": parsed, "parse_error": error, **diagnostics})
    assert len(rep.records[0].defects) == 1
    assert rep.records[0].defects[0].bbox_px == (1, 2, 10, 20)
    assert rep.n_bad_items == len(bad)


def test_null_item_in_adapter_is_counted():
    rep = adapt({"image_id": "i", "bbox_px_parsed": {
        "defects": [None], "verdict": "판정불가", "cited_clauses": []}})
    assert rep.n_bad_items == 1 and rep.records[0].defects == []


@pytest.mark.parametrize("count", [-1, True, "2", 1.5, None])
def test_bad_export_counter_is_rejected(count):
    with pytest.raises(ValueError, match="n_bad_items_dropped"):
        adapt({"image_id": "i", "n_bad_items_dropped": count})
