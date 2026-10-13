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


# 명세의 추가 22개를 구현 목록에서 역산하지 않는다.
_FORBIDDEN_SUPPLEMENT = (
    "취약", "우려", "심각", "불량을 나타", "가능성", "때문", "탓", "판정", "충족",
    "가까이", "집중", "분포", "경향", "치우", "일직선", "패턴", "모여", "흩어", "나란히",
    "줄지어", "띠", "거리",
)


@pytest.mark.parametrize("word", _FORBIDDEN_SUPPLEMENT)
def test_forbidden_supplement_is_in_instruction_and_diagnostic(word):
    task = make_task([("porosity", "가운데-가운데", 1)])
    instruction = task["prompt"].rsplit("\n주석: ", 1)[0]
    forbidden_line = next(line for line in instruction.splitlines() if line.startswith("쓰지 않는 말: "))
    assert word in forbidden_line.removeprefix("쓰지 않는 말: ").removesuffix(".").split(", ")
    raw = sample_text(task) + f" {word}이라는 주장을 한다."
    assert G.observation_problems(task, raw) == ["obs_forbidden"]
    row = G.render_generation_row(task, backend(task, raw))
    assert row["raw_text"] == raw
    assert row["observation_problems"] == ["obs_forbidden"]
    assert row["text"] == raw + " " + G.LIMITATION_FORMS[0]
    assert row["format_normalized"] is False


@pytest.mark.parametrize("fragment", [
    "더 취약할 수 있다는 가능성을 제기할 수 있습니다",
    "용접품질에 대한 우려를 불러일으키지만",
    "용접 과정에서의 불량을 나타내지만",
    "이들 {crack}은 서로 가까이 위치해 있다",
    "중앙 부근에 집중되어 있다는 점을 시사한다",
    "이러한 분포는 {porosity}의 발생 경향을 보여주지만",
    "서로 가까이 위치해 있지만",
    "그 사이에 간격은 존재하지 않습니다",
    "그 사이에 충분한 간격이 있어",
    "그 사이의 거리는 정확히 일정하지 않습니다",
    "합격 기준을 충족하는지 판단하기 위해서는",
])
def test_observed_eleven_forbidden_fragments_are_detected(fragment):
    labels = {key: label.name_ko for key, label in load_label_map().defect_types.items()}
    task = make_task([("porosity", "가운데-가운데", 1)])
    raw = sample_text(task) + " " + fragment.format(**labels) + "."
    row = G.render_generation_row(task, backend(task, raw))
    assert "obs_forbidden" in row["observation_problems"]
    assert row["raw_text"] == raw


@pytest.mark.parametrize("label_key", sorted(load_label_map().defect_types))
@pytest.mark.parametrize("where,display", [
    ("위-왼쪽", "위쪽 왼쪽"), ("위-가운데", "위쪽 가운데"), ("위-오른쪽", "위쪽 오른쪽"),
    ("가운데-왼쪽", "가운데 왼쪽"), ("가운데-가운데", "영상 가운데"), ("가운데-오른쪽", "가운데 오른쪽"),
    ("아래-왼쪽", "아래쪽 왼쪽"), ("아래-가운데", "아래쪽 가운데"), ("아래-오른쪽", "아래쪽 오른쪽"),
])
def test_every_label_and_zone_is_safe_in_observation_frame(label_key, where, display):
    task = make_task([(label_key, where, 1)])
    name = load_label_map().defect_types[label_key].name_ko
    raw = f"{name} 1개가 {display}에 표시되어 있다. 주석만으로는 결함의 모양과 크기를 알 수 없다."
    assert G.observation_problems(task, raw) == []
    row = G.render_generation_row(task, backend(task, raw))
    # 제한 문장에 '판정'이 있어도 관찰 진단에는 넣지 않는다.
    assert "판정" in row["text"] and row["observation_problems"] == []


def test_forbidden_vocabulary_stays_separate_from_labels_and_normal_template():
    names = [label.name_ko for label in load_label_map().defect_types.values()]
    assert len(_FORBIDDEN_SUPPLEMENT) == len(set(_FORBIDDEN_SUPPLEMENT)) == 22
    assert len(G.OBSERVATION_FORBIDDEN_WORDS) == len(set(G.OBSERVATION_FORBIDDEN_WORDS)) == 45
    assert "불량" not in G.OBSERVATION_FORBIDDEN_WORDS
    assert all(not any(word in name for word in G.OBSERVATION_FORBIDDEN_WORDS) for name in names)
    assert any(name.endswith("불량") for name in names)  # 실제 사상표 회귀 대상이 빈집합이면 안 된다.
    normal = G.render_generation_row(make_task())
    assert normal["text"] == normal_text() and normal["observation_problems"] == []
    assert not any(word in normal["text"] for word in _FORBIDDEN_SUPPLEMENT)


def test_forbidden_stem_matching_includes_tti_and_casefolded_units():
    task = make_task([("porosity", "가운데-가운데", 1)])
    for suffix in (" 그런 양상을 띠다.", " 길이는 1MM이다.", " 길이는 1Cm이다."):
        assert "obs_forbidden" in G.observation_problems(task, sample_text(task) + suffix)


def test_old_vocabulary_prompt_is_not_reused_as_current_task(monkeypatch):
    original = G.OBSERVATION_FORBIDDEN_WORDS
    with monkeypatch.context() as context:
        context.setattr(G, "OBSERVATION_FORBIDDEN_WORDS", tuple(word for word in original if word not in _FORBIDDEN_SUPPLEMENT))
        task = make_task([("porosity", "가운데-가운데", 1)])
    with pytest.raises(G.RegenerationRefused, match="generation_prompt"):
        G.validate_v2_task(task)

