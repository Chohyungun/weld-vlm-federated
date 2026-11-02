"""생성 v2의 CPU 계약. 합성 주석만 사용하며 모델 판별력은 측정하지 않는다."""

from collections import Counter
from copy import deepcopy
import re

import pytest

from corpus.production import generation_policy as G
from corpus.production.common import canonical_bytes, parse_json
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


def split_v2_prompt(prompt):
    separator = "\n주석은 다음 줄과 같다.\n"
    assert prompt.count(separator) == 1
    instruction, payload = prompt.split(separator)
    assert payload == canonical_bytes(parse_json(payload)).decode("utf-8")
    return instruction, payload


def assert_symbol_free_instruction(prompt):
    instruction, _ = split_v2_prompt(prompt)
    fixed = instruction
    for words in (G.OBSERVATION_FORBIDDEN_WORDS, G.OBSERVATION_ZONE_ALIASES):
        fixed = fixed.replace(", ".join(words), " ")
    for identifier in ("defects_by_zone", "observation_examples", "name", "zone"):
        fixed = fixed.replace(identifier, " ")
    assert re.fullmatch(r"[가-힣 ,.\n]*", fixed), sorted(set(re.sub(r"[가-힣 ,.\n]", "", fixed)))


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


@pytest.mark.parametrize("foreign", ["缺陷", "，", "。", "；", "𠀀", "《", "〈", "〉"])
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
    instruction, payload = split_v2_prompt(task["prompt"])
    assert set(parse_json(payload)) == {"defect_counts_by_name", "defects_by_zone", "observation_examples"}
    assert "合成材質" not in task["prompt"]
    assert G.LIMITATION_FORMS[0] not in task["prompt"]
    for example in G.observation_examples():
        assert example["글"] in payload and example["글"] not in instruction


def test_model_visible_generation_prompt_contains_no_forbidden_characters():
    from corpus.production.infer import GENERATE_SYSTEM

    # 출력 규칙을 약하게 만들어 지시문 충돌을 숨기지 않는다.
    assert G._HAN_OR_CHINESE_PUNCT.findall("〈〉") == ["〈", "〉"]
    assert not G._HAN_OR_CHINESE_PUNCT.search(GENERATE_SYSTEM)
    for key in sorted(load_label_map().defect_types):
        for where in sorted(G.ZONE_DISPLAY):
            task = make_task([(key, where, 2)])
            # 본문뿐 아니라 모델에 함께 들어가는 JSON과 합성 예시까지 검사한다.
            forbidden = sorted(set(G._HAN_OR_CHINESE_PUNCT.findall(task["prompt"])))
            assert forbidden == [], (key, where, forbidden)
            assert_symbol_free_instruction(task["prompt"])
            payload = parse_json(split_v2_prompt(task["prompt"])[1])
            assert payload["observation_examples"] == G.observation_examples()


@pytest.mark.parametrize("symbol", ["〈", "〉", '"', "-", ":", "·", "/", "(", ")"])
def test_instruction_symbol_guard_rejects_reintroduced_symbol(symbol):
    task = make_task([("porosity", "가운데-가운데", 1)])
    with pytest.raises(AssertionError):
        assert_symbol_free_instruction(symbol + task["prompt"])


def test_new_prompt_boundary_preserves_canonical_payload_and_legacy_prompt():
    task = make_task([(key, where, 1) for key, where in
                      zip(sorted(load_label_map().defect_types), sorted(G.ZONE_DISPLAY))])
    _, payload = split_v2_prompt(task["prompt"])
    assert parse_json(payload) == {key: task["evidence"][key] for key in
                                  ("defect_counts_by_name", "defects_by_zone", "observation_examples")}
    assert "\n주석: " not in task["prompt"]
    assert_symbol_free_instruction(task["prompt"])
    legacy_evidence = G.build_image_evidence(task)
    legacy = G.build_image_prompt(legacy_evidence)
    assert legacy.rsplit("\n주석: ", 1)[1] == canonical_bytes(legacy_evidence).decode("utf-8")
    assert "\n주석은 다음 줄과 같다.\n" not in legacy


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


@pytest.mark.parametrize("kind,expected", [("empty", "obs_empty"),
                                         ("six", "obs_sentence_count"), ("long", "obs_length"),
                                         ("no_period", "obs_sentence_end"), ("polite", "obs_sentence_end"),
                                         ("alias", "obs_zone_word"), ("zone_missing", "obs_zone_missing"),
                                         ("limitation", "obs_limitation_written"), ("forbidden", "obs_forbidden"),
                                         ("split_name", "obs_name_form")])
def test_diagnostic_reasons_preserve_invalid_model_rows(kind, expected):
    task = make_task([("slag_inclusion", "가운데-가운데", 1)])
    raw = sample_text(task)
    if kind == "empty": raw = " \n "
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
    assert G.is_v2_run(make_run())
    G.validate_v2_policy(make_run())
    assert not G.is_v2_run({"config": {"recipe": G.LEGACY_POLICY_VERSION,
                                      "generation_policy": {"version": G.LEGACY_POLICY_VERSION,
                                                            "sha256": "1" * 64}}})
    identity = G.policy_identity()
    assert identity["version"] == "image_report_regeneration_v4"
    assert identity["renderer"]["version"] == "image_report_render_v2"
    assert identity["renderer"]["normalization"] == "linebreak_flatten_strip_v1"
    assert identity["spec_sha256"] == "43f94d60e0697efeda54655a24b2c6716c7141a996d7cff15f9f7f1e595e2867"
    assert identity["observation"]["version"] == "image_report_observation_v5"
    assert identity["observation"]["min_sentences"] == 1
    assert identity["observation"]["max_sentences"] == 5


@pytest.mark.parametrize("sentence_count,expected", [(0, True), (1, False), (5, False), (6, True)])
def test_observation_sentence_boundary_is_one_to_five(sentence_count, expected):
    task = make_task([("porosity", "가운데-가운데", 1)])
    name = load_label_map().defect_types["porosity"].name_ko
    raw = "" if sentence_count == 0 else (
        f"{name} 1개가 영상 가운데에 표시되어 있다." + " 확인할 수 없다." * (sentence_count - 1))
    row = G.render_generation_row(task, backend(task, raw))
    assert ("obs_sentence_count" in row["observation_problems"]) is expected
    if sentence_count in (1, 5):
        assert row["observation_problems"] == []
    assert row["raw_text"] == raw
    assert row["text"] == raw + " " + G.LIMITATION_FORMS[0]


def test_two_sentences_are_recommended_and_slash_frame_is_removed():
    task = make_task([("porosity", "가운데-가운데", 1)])
    instruction, _ = split_v2_prompt(task["prompt"])
    assert instruction.splitlines()[1] == (
        "주석의 결함 이름과 개수와 칸을 한국어 문장으로 옮겨 쓴다. 한 문단으로 쓰고 줄을 바꾸지 않는다. 문장은 다섯 개까지 쓴다.")
    assert instruction.splitlines()[2] == (
        "가능하면 두 문장 이상으로 쓴다. 첫 문장에는 이름별 총수를 쓴다.")
    assert "문장 꼴의 틀:" not in instruction
    assert "/" not in instruction
    assert "문장을 늘리지 않는다" not in instruction and "렌더러" not in instruction
    assert "주석에 있는 칸을 하나도 빠뜨리지 않는다" in instruction
    assert "한 칸의 개수와 이름 전체의 개수를 섞지 않는다" in instruction
    assert "칸이 여럿이면 칸마다 그 칸의 개수를 붙인다." not in instruction
    assert "개수로 문장을 끝내지 않는다. 칸과 개수를 쓴 뒤에도 있다 또는 표시되어 있다로 문장을 맺는다." in instruction
    assert "칸 이름은 두 낱말을 다 쓰고 줄이지 않는다. 영상 가운데를 가운데로 쓰지 않는다." in instruction
    assert task["evidence"]["observation_examples"] == G.observation_examples()


@pytest.mark.parametrize("multi_zone", [False, True])
def test_recommended_frame_output_passes_actual_leak_check(multi_zone):
    from corpus.production.quality import leak_problems

    groups = [("porosity", "위-왼쪽", 2)]
    if multi_zone:
        groups.append(("porosity", "가운데-가운데", 1))
    task = make_task(groups)
    name = load_label_map().defect_types["porosity"].name_ko
    raw = (f"{name} 3개가 표시되어 있다. 위쪽 왼쪽에 2개, 영상 가운데에 1개가 있다."
           if multi_zone else f"{name} 2개가 위쪽 왼쪽에 표시되어 있다.")
    row = G.render_generation_row(task, backend(task, raw))
    assert row["observation_problems"] == []
    assert leak_problems(row["text"], task) == []


def test_single_sentence_does_not_bypass_missing_zone_or_semantic_judgment():
    task = make_task([("porosity", "위-가운데", 1), ("porosity", "가운데-가운데", 2)])
    name = load_label_map().defect_types["porosity"].name_ko
    complete = f"{name} 3개가 위쪽 가운데에 1개, 영상 가운데에 2개 표시되어 있다."
    assert G.observation_problems(task, complete) == []
    missing = f"{name} 3개가 영상 가운데에 표시되어 있다."
    assert G.observation_problems(task, missing) == ["obs_zone_missing"]
    # 구조 진단이 통과해도 개수 사실의 승인은 아니다. 수 대조는 판정 단계에 남긴다.
    wrong_count = complete.replace("3개", "9개")
    assert G.observation_problems(task, wrong_count) == []


def test_old_two_sentence_contract_cannot_be_reused_under_new_identity(monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(G, "OBSERVATION_MIN_SENTENCES", 2)
        context.setattr(G, "OBSERVATION_VERSION", "image_report_observation_v2")
        previous = make_run()
    task = make_task([("porosity", "가운데-가운데", 1)])
    task["prompt"] = task["prompt"].replace(
        "한 문단으로 쓰고 줄을 바꾸지 않는다. 문장은 다섯 개까지 쓴다.",
        "문장은 2개에서 5개까지, 한 문단으로 쓰고 줄을 바꾸지 않는다.")
    with pytest.raises(G.RegenerationRefused, match="generation_policy_changed"):
        G.validate_v2_policy(previous)
    with pytest.raises(G.RegenerationRefused, match="generation_prompt"):
        G.validate_v2_task(task)


def test_multiple_zones_do_not_require_explicit_counts_per_zone():
    from corpus.production.judge_policy import deterministic_reasons_v7

    task = make_task([("porosity", "위-가운데", 1), ("porosity", "가운데-가운데", 2)])
    name = load_label_map().defect_types["porosity"].name_ko
    raw = f"{name} 3개가 위쪽 가운데와 영상 가운데에 표시되어 있다."
    row = G.render_generation_row(task, backend(task, raw))
    assert row["observation_problems"] == []
    assert deterministic_reasons_v7(task, row["text"]) == []


def test_observation_v3_identity_and_prompt_cannot_be_reused(monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(G, "OBSERVATION_VERSION", "image_report_observation_v3")
        context.setattr(G, "SPEC_SHA256", "a9422662d23334d37c58063f4279a4df8405ef056fdc9fd886907407bff8c465")
        previous = make_run()
    with pytest.raises(G.RegenerationRefused, match="generation_policy_changed"):
        G.validate_v2_policy(previous)
    task = make_task([("porosity", "가운데-가운데", 1)])
    task["prompt"] = task["prompt"].replace("\n주석은 다음 줄과 같다.\n", "\n주석: ")
    with pytest.raises(G.RegenerationRefused, match="generation_prompt"):
        G.validate_v2_task(task)


def test_observation_v4_identity_and_prompt_cannot_be_reused(monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(G, "OBSERVATION_VERSION", "image_report_observation_v4")
        context.setattr(G, "SPEC_SHA256", "dc50ddc05352746c8f18d05472bd010e69cb1c5909ea20cbb12819883476139c")
        previous = make_run()
    with pytest.raises(G.RegenerationRefused, match="generation_policy_changed"):
        G.validate_v2_policy(previous)
    task = make_task([("porosity", "가운데-가운데", 1)])
    task["prompt"] = task["prompt"].replace(
        "가능하면 두 문장 이상으로 쓴다. 첫 문장에는 이름별 총수를 쓴다.",
        "가능하면 두 문장 이상으로 쓴다. 첫 문장에는 이름별 총수를, 다음 문장에는 칸마다의 개수를 쓴다.")
    with pytest.raises(G.RegenerationRefused, match="generation_prompt"):
        G.validate_v2_task(task)


def test_b17b4_second_example_has_complete_predicates_and_three_zones():
    labels = load_label_map().defect_types
    crack, slag = labels["crack"].name_ko, labels["slag_inclusion"].name_ko
    example = G.observation_examples()[1]
    assert example == {"주석": [
        {"zone": "위쪽 오른쪽", "name": crack, "count": 1},
        {"zone": "아래쪽 가운데", "name": slag, "count": 1},
        {"zone": "아래쪽 오른쪽", "name": slag, "count": 1}],
        "글": f"{crack} 1개와 {slag} 2개가 표시되어 있다. {crack} 1개는 위쪽 오른쪽에 있다. "
              f"{slag} 2개는 아래쪽 가운데에 1개, 아래쪽 오른쪽에 1개가 있다. "
              "주석만으로는 결함의 모양과 크기를 알 수 없다."}
    task = make_task([("crack", "위-오른쪽", 1), ("slag_inclusion", "아래-가운데", 1),
                      ("slag_inclusion", "아래-오른쪽", 1)])
    assert G.observation_problems(task, example["글"]) == []
    assert "obs_sentence_end" in G.observation_problems(
        task, example["글"].replace("1개가 있다.", "1개."))
    center_task = make_task([("porosity", "가운데-가운데", 1)])
    assert "obs_zone_missing" in G.observation_problems(
        center_task, sample_text(center_task).replace("영상 가운데", "가운데"))


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
    instruction, _ = split_v2_prompt(task["prompt"])
    forbidden_line = next(line for line in instruction.splitlines() if line.endswith(" 같은 말은 쓰지 않는다."))
    assert word in forbidden_line.removesuffix(" 같은 말은 쓰지 않는다.").split(", ")
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

