"""생성 v2의 CPU 계약. 합성 주석만 사용하며 모델 판별력은 측정하지 않는다."""

from collections import Counter
from copy import deepcopy

import pytest

from corpus.production import generation_policy as G
from corpus.production.common import parse_json
from data.label_map import load_label_map


def make_task(groups=(), *, identifier="synthetic-v2"):
    labels = load_label_map().defect_types
    defects, positions = [], []
    for key, where, count in groups:
        label = labels[key]
        defects.extend({"type": label.iso_code, "bbox_px": [10, 10, 20, 20]} for _ in range(count))
        positions.extend({"name": label.name_ko, "where": where} for _ in range(count))
    task = {"id": identifier, "kind": "image_report", "locked_target": {
        "defects": defects, "defect_counts": dict(Counter(d["type"] for d in defects)), "clauses": []},
        "evidence": {"defect_counts_by_name": dict(Counter(p["name"] for p in positions)),
                     "defect_positions": positions, "material": "合成材質",
                     "has_applicable_clause": False, "size_mm_available": False,
                     "limitation": G.LIMITATION_FORMS[0]}}
    task["evidence"] = G.build_image_evidence_v2(task)
    task["prompt"] = G.build_image_prompt_v2(task["evidence"])
    return task


def backend(task, text, **overrides):
    return {"id": task["id"], "text": text, "finish_reason": "eos", "input_tokens": 41,
            "output_tokens": 17, **overrides}


def normal_text():
    return "표시된 결함이 없다. 결함이 표시되지 않았다는 것이 합격이나 안전을 뜻하지는 않는다."


def sample_text(task):
    name = next(iter(task["evidence"]["defect_counts_by_name"]))
    return f"{name} 1개가 영상 가운데에 표시되어 있다. 주석만으로는 결함의 모양과 크기를 알 수 없다."


def test_normal_has_no_prompt_model_or_tokens():
    task = make_task()
    G.validate_v2_task(task)
    assert task["prompt"] == ""
    assert G.render_generation_row(task) == {
        "id": task["id"], "generator": "template", "raw_text": None, "text": normal_text(),
        "finish_reason": "template", "input_tokens": 0, "output_tokens": 0,
        "format_normalized": False, "observation_problems": [], "language_ok": True}
    with pytest.raises(G.RegenerationRefused, match="template_model_called"):
        G.render_generation_row(task, backend(task, "모델을 부르면 안 된다."))


@pytest.mark.parametrize("prefix,suffix,join", [("", "", " "), (" \n", "\r\n ", "\n\n  "),
                                              ("\t", " ", "\r\n")])
def test_only_linebreak_boundaries_and_outer_whitespace_are_normalized(prefix, suffix, join):
    task = make_task([("porosity", "가운데-가운데", 1)])
    expected = sample_text(task)
    raw = prefix + expected.replace("다. 주석", "다." + join + "주석") + suffix
    before = deepcopy(task)
    result = G.render_generation_row(task, backend(task, raw))
    assert task == before
    assert result["raw_text"] == raw
    assert result["text"] == expected + " " + G.LIMITATION_FORMS[0]
    assert result["format_normalized"] is (raw != expected)
    assert result["observation_problems"] == []
    assert result["input_tokens"] == 41 and result["output_tokens"] == 17
    assert G.normalize_observation("가  나\t다.\n\n라.") == "가  나\t다. 라."


@pytest.mark.parametrize("foreign", ["缺陷", "，", "。", "；", "𠀀", "《"])
def test_han_and_chinese_punctuation_are_rejected_not_rewritten(foreign):
    task = make_task([("porosity", "가운데-가운데", 1)])
    raw = sample_text(task) + foreign
    row = G.render_generation_row(task, backend(task, raw))
    assert row["raw_text"] == raw and foreign in row["text"]
    assert row["format_normalized"] is False
    assert row["language_ok"] is False
    assert "obs_language" in row["observation_problems"]


def test_grouped_evidence_is_zone_then_name_order_and_drops_only_legacy_forms():
    task = make_task([("porosity", "아래-오른쪽", 2), ("crack", "위-왼쪽", 1),
                      ("porosity", "위-왼쪽", 2)])
    labels = load_label_map().defect_types
    expected = [
        *[{"zone": "위쪽 왼쪽", "name": name, "count": count} for name, count in sorted([
            (labels["crack"].name_ko, 1), (labels["porosity"].name_ko, 2)])],
        {"zone": "아래쪽 오른쪽", "name": labels["porosity"].name_ko, "count": 2},
    ]
    assert task["evidence"]["defects_by_zone"] == expected
    assert "limitation_forms" not in task["evidence"]
    assert len(task["evidence"]["defect_positions"]) == 5
    instruction, payload = task["prompt"].rsplit("\n주석: ", 1)
    assert set(parse_json(payload)) == {"defect_counts_by_name", "defects_by_zone", "observation_examples"}
    assert "合成材質" not in task["prompt"]
    assert G.LIMITATION_FORMS[0] not in task["prompt"]
    for example in G.observation_examples():
        assert example["글"] in payload and example["글"] not in instruction


@pytest.mark.parametrize("index,groups", [
    (0, [("porosity", "위-왼쪽", 2), ("porosity", "가운데-가운데", 1)]),
    (1, [("crack", "위-오른쪽", 1), ("slag_inclusion", "아래-가운데", 2)]),
    (2, [("lack_of_penetration", "가운데-오른쪽", 1)]),
])
def test_fewshot_passes_actual_leak_and_observation_contract(index, groups):
    from corpus.production.quality import leak_problems

    task = make_task(groups)
    raw = G.observation_examples()[index]["글"]
    result = G.render_generation_row(task, backend(task, raw))
    assert G.observation_problems(task, raw) == []
    assert leak_problems(result["text"], task) == []


def test_normal_template_passes_actual_leak_check():
    from corpus.production.quality import leak_problems

    task = make_task()
    assert leak_problems(G.render_generation_row(task)["text"], task) == []


@pytest.mark.parametrize("kind,expected", [("empty", "obs_empty"), ("one", "obs_sentence_count"),
                                         ("six", "obs_sentence_count"), ("long", "obs_length"),
                                         ("no_period", "obs_sentence_end"), ("polite", "obs_sentence_end"),
                                         ("alias", "obs_zone_word"), ("zone_missing", "obs_zone_missing"),
                                         ("limitation", "obs_limitation_written"), ("forbidden", "obs_forbidden"),
                                         ("split_name", "obs_name_form")])
def test_diagnostic_reasons_preserve_invalid_model_rows(kind, expected):
    task = make_task([("slag_inclusion", "가운데-가운데", 1)])
    raw = sample_text(task)
    if kind == "empty": raw = " \n "
    elif kind == "one": raw = raw.split(". ")[0] + "."
    elif kind == "six": raw += " 확인할 수 없다." * 4
    elif kind == "long": raw += " " + "가" * 301 + "다."
    elif kind == "no_period": raw = raw[:-1]
    elif kind == "polite": raw = raw.replace("있다.", "있습니다.")
    elif kind == "alias": raw = raw.replace("영상 가운데", "중앙")
    elif kind == "zone_missing": raw = raw.replace("영상 가운데", "위쪽 왼쪽")
    elif kind == "limitation": raw += " 모재 두께는 알 수 없다."
    elif kind == "forbidden": raw += " 안전하다고 본다."
    else:
        name = next(iter(task["evidence"]["defect_counts_by_name"]))
        raw = raw.replace(name, name[:2] + " " + name[2:])
    row = G.render_generation_row(task, backend(task, raw, finish_reason="length"))
    assert row["raw_text"] == raw
    assert row["finish_reason"] == "length"
    assert expected in row["observation_problems"]
    assert row["observation_problems"] == [code for code in G.OBSERVATION_REASON_ORDER
                                          if code in row["observation_problems"]]


@pytest.mark.parametrize("change", ["model_on_normal", "missing_model", "wrong_id", "zero_tokens", "bool_tokens"])
def test_renderer_rejects_metadata_fabrication(change):
    task = make_task([("porosity", "가운데-가운데", 1)])
    value = backend(task, sample_text(task))
    if change == "model_on_normal": task = make_task()
    elif change == "missing_model": value = None
    elif change == "wrong_id": value["id"] = "other"
    elif change == "zero_tokens": value["input_tokens"] = 0
    else: value["output_tokens"] = True
    with pytest.raises(G.RegenerationRefused):
        G.render_generation_row(task, value)


@pytest.mark.parametrize("change", ["prompt", "groups", "examples", "normal_prompt"])
def test_task_contract_checks_derivation_not_only_hash(change):
    task = make_task([] if change == "normal_prompt" else [("porosity", "가운데-가운데", 1)])
    if change == "prompt": task["prompt"] += " "
    elif change == "groups": task["evidence"]["defects_by_zone"][0]["count"] = 999
    elif change == "examples": task["evidence"]["observation_examples"][0]["글"] += " 바뀌었다."
    else: task["prompt"] = "모델을 불러라."
    with pytest.raises(G.RegenerationRefused, match="generation_evidence|generation_prompt"):
        G.validate_v2_task(task)


def make_run():
    policy = G.policy_identity()
    return {"config": {"recipe": G.POLICY_VERSION, "generation_policy": deepcopy(policy)},
            "identity_basis": {"generation_policy": deepcopy(policy), "prepare_code_sha256": policy["sha256"]},
            "regeneration": {"generation_policy": deepcopy(policy)}}


@pytest.mark.parametrize("change", ["config", "basis", "regeneration", "code", "downgrade", "unknown"])
def test_policy_links_fail_closed(change):
    run = make_run()
    if change == "code": run["identity_basis"]["prepare_code_sha256"] = "0" * 64
    elif change == "downgrade": run["config"]["recipe"] = G.LEGACY_POLICY_VERSION
    elif change == "unknown":
        run["config"]["recipe"] = "image_report_regeneration_v999"
        run["config"]["generation_policy"]["version"] = "image_report_regeneration_v999"
    else:
        key = {"config": "config", "basis": "identity_basis", "regeneration": "regeneration"}[change]
        run[key]["generation_policy"]["sha256"] = "0" * 64
    assert G.is_v2_run(run)
    with pytest.raises(G.RegenerationRefused, match="generation_policy"):
        G.validate_v2_policy(run)


def test_v4_current_identity_and_legacy_detection_are_separate():
    G.validate_v2_policy(make_run())
    assert not G.is_v2_run({"config": {"recipe": G.LEGACY_POLICY_VERSION,
                                      "generation_policy": {"version": G.LEGACY_POLICY_VERSION,
                                                            "sha256": "1" * 64}}})
    identity = G.policy_identity()
    assert identity["version"] == "image_report_regeneration_v4"
    assert identity["renderer"]["version"] == "image_report_render_v2"
    assert identity["renderer"]["normalization"] == "linebreak_flatten_strip_v1"
    assert identity["observation"]["min_sentences"] == 2


def test_label_cache_is_content_bound_bounded_and_returns_detached_names(tmp_path, monkeypatch):
    import data.label_map as labels

    path = tmp_path / "label_map.yaml"
    original = labels.DEFAULT_LABEL_MAP.read_bytes()
    path.write_bytes(original)
    monkeypatch.setattr(labels, "DEFAULT_LABEL_MAP", path)
    actual_load, calls = labels.load_label_map, []

    def counted(value=None):
        calls.append(value)
        return actual_load(value)

    monkeypatch.setattr(labels, "load_label_map", counted)
    G._label_snapshot.cache_clear()
    first = G._label_names()
    expected = dict(first)
    first.clear()
    assert G._label_names() == expected
    G.observation_examples()
    assert len(calls) == 1
    identity_before = G.policy_identity()
    path.write_bytes(original + b"\n# synthetic new bytes\n")
    assert G._label_names() == expected
    assert len(calls) == 2
    assert G.policy_identity()["label_map_sha256"] != identity_before["label_map_sha256"]
    assert G._label_snapshot.cache_info().maxsize == 4
    G._label_snapshot.cache_clear()

