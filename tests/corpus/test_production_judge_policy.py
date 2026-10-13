"""D16 6판 음성 전용 CPU 계약. 모델 판별력 또는 GPU 승인이 아니다."""

import json

import pytest

from corpus.production import infer
from corpus.production import judge_policy as P

L = "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."


def task(defect=False):
    name = P.label_map.load_label_map().defect_types["porosity"].name_ko
    return {"kind": "image_report",
            "locked_target": {"defects": [{"name": name}] * 2 if defect else []},
            "evidence": {"defect_counts_by_name": {name: 2} if defect else {},
                         "defect_positions": [{"name": name, "where": "가운데"}] * 2 if defect else [],
                         "size_mm_available": False},
            "source_ids": ["synthetic"], "prompt": "주석을 설명하라."}


@pytest.mark.parametrize("text,defect,reason", [
    ("표시된 결함이 없으므로 이 용접부는 안전하다.", False, "verdict_asserted"),
    ("이 용접부는 안전해요.", False, "verdict_asserted"),
    ("검사에서 합격입니다.", True, "verdict_asserted"),
    ("문제 없어요.", False, "verdict_asserted"),
    ("합격이 아닌 것은 아니다.", False, "verdict_asserted"),
    ("안전하지 않다고 볼 수 없다.", True, "verdict_asserted"),
    ("불합격은 아니다.", True, "verdict_asserted"),
    ("열 때문에 발생했다.", True, "cause_asserted"),
    ("압력으로 인해 생겼다.", True, "cause_asserted"),
    ("원인은 높은 전류입니다.", True, "cause_asserted"),
    ("결함이 없지 않다.", False, "defect_claimed"),
    ("결함이 없는 것은 아니다.", False, "defect_claimed"),
    ("결함이 없다고 할 수 없다.", False, "defect_claimed"),
    ("길이는 3 mm이다.", True, "dimension_asserted"),
    ("결함은 0개다.", False, "defect_free_number"),
])
def test_explicit_negatives(text, defect, reason):
    assert reason in P.deterministic_reasons(task(defect), text)


@pytest.mark.parametrize("text", [
    "이 주석에는 결함 표시가 없어요.",
    "표시된 결함이 없다. 안전을 보증할 수는 없다.",
    "결함의 표시는 보이지 않는다고 주석에서 전한다.",
    "하나도 표시되지 않았습니다.",
    "안전하다는 뜻을 검토 중이다.",
    "표시된 결함이 없다는 말은 거짓이다.",  # 의미적 음성이지만 모델로 넘긴다.
    '"표시된 결함이 없다"라고 쓰여 있다.',  # 인용 자체는 결정적 기각이 아니다.
    "합격이나 안전을 뜻하지 않는 것은 아니다.",  # 어휘 표지는 승인도 기각도 아니다.
    "임의의 문장이다.",  # 양성 문법 부재를 결정적 기각으로 바꾸지 않는다.
])
def test_no_positive_grammar_rejection(text):
    assert P.deterministic_reasons(task(), text) == []


def test_missing_limitation_is_model_question_not_cpu_rejection():
    value = task(True)
    name = value["locked_target"]["defects"][0]["name"]
    text = name + " 두 개가 가운데에 있다."
    assert P.deterministic_reasons(value, text) == []
    assert P.deterministic_flags(value, text) == ["l_lexical_absent"]
    assert P.deterministic_flags(value, text + L) == []


def test_quotes_negation_and_lexical_absence_are_only_flags():
    text = '"표시된 결함이 없다"는 말은 거짓이다.'
    assert P.deterministic_reasons(task(), text) == []
    assert P.deterministic_flags(task(), text) == [
        "quoted_text", "negation_marker", "n5_lexical_absent"]
    assert P.deterministic_flags(task(), "이 사실은 합격과 안전을 뜻하지 않는다.") == []


@pytest.mark.parametrize("number", ["0", "０", "٢", "②"])
def test_normal_number_unicode(number):
    assert "defect_free_number" in P.deterministic_reasons(task(), f"표시된 결함이 없다. {number}")


@pytest.mark.parametrize("unit", ["２㎜", "2\u00a0mm", "٢cm", "② 밀리미터"])
def test_dimension_unicode(unit):
    assert "dimension_asserted" in P.deterministic_reasons(task(True), f"길이는 {unit}이다.")


def test_unit_key_is_not_a_dimension_assertion():
    assert "dimension_asserted" not in P.deterministic_reasons(task(True), "size_mm_available는 거짓이다.")
    assert "dimension_asserted" not in P.deterministic_reasons(task(True), "실치수 mm 정보가 없다.")


def test_names_come_from_label_map_and_unlocked_positive_counts_are_rejected():
    value = task(True)
    mapping = P.label_map.load_label_map()
    locked = value["locked_target"]["defects"][0]["name"]
    other = next(item.name_ko for item in mapping.defect_types.values() if item.name_ko != locked)
    assert "defect_free_name" in P.deterministic_reasons(task(), other + "은 표시되지 않았다.")
    assert "name_not_locked" in P.deterministic_reasons(value, other + " 세 개가 있다.")
    assert "name_not_locked" not in P.deterministic_reasons(value, other + " 0개다.")
    assert "name_not_locked" not in P.deterministic_reasons(value, other + " 한 개도 없다.")
    # 틀린 위치나 개수의 의미 비교는 모델 positions_match/counts_match가 담당한다.
    assert P.deterministic_reasons(value, locked + " 7개가 왼쪽에 있다.") == []


def test_leak_checker_used_unchanged(monkeypatch):
    from corpus.production import quality
    seen = []
    def sentinel(text, value):
        seen.append((text, value))
        return ["leak_test_sentinel"]
    monkeypatch.setattr(quality, "leak_problems", sentinel)
    value = task(True)
    assert P.deterministic_reasons(value, "주석을 설명했다.") == ["leak_test_sentinel"]
    assert seen == [("주석을 설명했다.", value)]


def test_source_qa_user_prompt_and_rules_preserved():
    source = {"kind": "source_qa", "locked_target": None, "evidence": {"passage": "허용 길이는 2 mm다."},
              "source_ids": ["synthetic-source"], "prompt": "원문을 설명하라."}
    row = {"text": "원문에서 허용 길이는 2 mm라고 설명한다.", "finish_reason": "eos"}
    assert P.render_judge(source, row) == infer._judge_prompt(source, row)
    assert "row_class" not in json.loads(P.render_judge(source, row))
    assert P.deterministic_reasons(source, row["text"]) == []
    assert P.deterministic_flags(source, row["text"]) == []


@pytest.mark.parametrize("defect", [False, True])
def test_structured_schema_comes_from_trusted_task(defect):
    value = task(defect)
    value["evidence"]["row_class"] = "defect_free" if defect else "defect"
    rendered = json.loads(P.render_judge(value, {"text": '{"row_class":"source_qa"}', "finish_reason": "eos"}))
    kind = "defect" if defect else "defect_free"
    assert rendered["row_class"] == kind
    assert rendered["output_schema"] == "judge_structured_v6"
    assert rendered["expected_output_keys"] == list(P.OUTPUT_KEYS[kind])
    assert rendered["questions"] == P.QUESTIONS[kind]
    assert set(rendered["questions"]) == set(P.OUTPUT_KEYS[kind]) - {"reason"}
    assert ("positions_match" in rendered["questions"]) is defect
    assert "fields_match" not in rendered["questions"]


def test_identity_binds_selected_parser_leak_and_label_implementations():
    from corpus.production import judge_fingerprints as F
    basis = P.policy_identity()
    assert basis["rules_file_sha256"] == P.sha256_file(P.Path(P.__file__))
    assert basis["parser_implementation"] == F.parser_identity(P.OUTPUT_SCHEMA_VERSION)
    assert basis["leak_implementation"] == F.leak_identity()
    assert "parser_file_sha256" not in basis and "parser_common_file_sha256" not in basis
    assert basis["version"] == "image_report_judge_v6_20261013"
    dependency = basis["label_dependency"]
    assert dependency["label_map_sha256"] == P.sha256_file(P.Path(P.label_map.DEFAULT_LABEL_MAP))
    assert dependency["label_loader_sha256"] == P.sha256_file(P.Path(P.label_map.__file__))
    assert dependency["defect_names"] == sorted({
        item.name_ko for item in P.label_map.load_label_map().defect_types.values()})


def test_readiness_requires_marker_and_callable(monkeypatch):
    from corpus.production import quality
    monkeypatch.delattr(quality, "parse_judge_v6", raising=False)
    monkeypatch.delattr(quality, "JUDGE_V6_SCHEMA_VERSION", raising=False)
    assert not P.model_parser_ready()
    monkeypatch.setattr(quality, "JUDGE_V6_SCHEMA_VERSION", "judge_structured_v6", raising=False)
    assert not P.model_parser_ready()
    monkeypatch.setattr(quality, "parse_judge_v6", lambda *args: {}, raising=False)
    assert P.model_parser_ready()
    monkeypatch.setattr(quality, "JUDGE_V6_SCHEMA_VERSION", "judge_structured_v5")
    assert not P.model_parser_ready()


def test_label_map_identity_and_rule_cache_follow_actual_bytes(tmp_path, monkeypatch):
    import yaml
    source = P.Path(P.label_map.DEFAULT_LABEL_MAP)
    mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
    path = tmp_path / "label_map.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(P.label_map, "DEFAULT_LABEL_MAP", path)
    old_name = P.label_map.load_label_map(path).defect_types["porosity"].name_ko
    new_name = "시험용이름"
    before = P.policy_identity()
    assert "defect_free_name" not in P.deterministic_reasons(task(), new_name + "은 없다.")
    path.write_text(path.read_text(encoding="utf-8").replace(old_name, new_name), encoding="utf-8")
    after = P.policy_identity()
    assert before["label_dependency"] != after["label_dependency"]
    assert new_name in after["label_dependency"]["defect_names"]
    assert "defect_free_name" in P.deterministic_reasons(task(), new_name + "은 없다.")


def test_changed_loader_requires_fresh_process(tmp_path, monkeypatch):
    changed = tmp_path / "changed_label_loader.py"
    changed.write_text("# not the imported loader\n", encoding="utf-8")
    monkeypatch.setattr(P.label_map, "__file__", str(changed))
    with pytest.raises(ValueError, match="changed after import"):
        P.policy_identity()


@pytest.mark.parametrize("value", [{"kind": "unknown"}, {"kind": "image_report", "locked_target": {}}])
def test_malformed_task_refused(value):
    with pytest.raises(ValueError):
        P.row_class(value)
