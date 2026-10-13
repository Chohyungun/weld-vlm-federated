"""v7 pure policy contracts; no model execution or probe approval.

Canonical 208-row, four-field equivalence is separately checked against D's immutable
reference/vector artifact. These tests exercise production dispatch and the
explicit distinction between diagnostic and generation-contract decisions.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from corpus.production import generation_policy as G
from corpus.production import judge_grounded_v7 as D
from corpus.production import judge_policy as P


def _task(defect=False):
    name = P.label_map.load_label_map().defect_types["porosity"].name_ko
    return {"kind": "image_report", "source_ids": ["synthetic-policy-v7"],
            "prompt": "잠긴 주석을 설명하라.",
            "locked_target": {"defects": [{"name": name}] * 2 if defect else []},
            "evidence": {"defect_counts_by_name": {name: 2} if defect else {},
                         "defect_positions": [{"name": name, "where": "가운데"}] * 2 if defect else [],
                         "size_mm_available": False}}


def _normal():
    return G.NORMAL_ABSENCE_FORMS[0] + " " + G.NORMAL_LIMITATION_FORMS[0]


def _defect(task):
    name = task["locked_target"]["defects"][0]["name"]
    return name + " 2개가 영상 가운데에 표시되어 있다. " + G.LIMITATION_FORMS[0]


def test_v6_names_remain_explicitly_v6():
    assert P.VERSION == "image_report_judge_v6_20261013"
    assert P.OUTPUT_SCHEMA_VERSION == "judge_structured_v6"
    assert P.V7_OUTPUT_SCHEMA_VERSION == "judge_structured_v7"
    assert P.V7_VERSION != P.VERSION
    assert "counts_match" in P.structured_questions(_task(True))
    assert "counts_match" not in P.structured_questions_v7(_task(True))


@pytest.mark.parametrize("defect", [False, True])
def test_public_wrappers_preserve_canonical_outputs(defect):
    task = _task(defect)
    text = _defect(task) if defect else _normal()
    for public, canonical in (
            (P.deterministic_grounded_v7, D.grounded_codes),
            (P.deterministic_generation_contract_v7, D.contract_codes),
            (P.deterministic_attributes_v7, D.attribute_codes),
            (P.deterministic_normal_v7, D.normal_codes)):
        assert public(task, text) == canonical(task, text)


def test_normal_contract_pass_has_no_model_result():
    task = _task()
    before = deepcopy(task)
    result = P.evaluate_v7(task, _normal())
    assert result["row_class"] == "defect_free"
    assert result["model_applicable"] is False
    assert result["model_applicability"] == "not_applicable"
    assert result["deterministic_pass"] is True
    assert result["generation_contract_applied"] is True
    assert not {"model_accept", "model_raw", "model_call_id", "output_tokens", "parser"} & result.keys()
    assert task == before
    with pytest.raises(ValueError, match="normal_model_call_forbidden"):
        P.render_judge_v7(task, {"text": _normal(), "finish_reason": "eos"})


def test_normal_renderer_rejects_before_reading_candidate():
    with pytest.raises(ValueError, match="normal_model_call_forbidden"):
        P.render_judge_v7(_task(), {})


def test_contract_exclusion_is_explicit_and_reported():
    task = _task()
    paraphrase = "이 주석에는 결함 표시가 없어요. 이것이 합격이나 안전을 뜻하지는 않는다."
    strict = P.evaluate_v7(task, paraphrase)
    diagnostic = P.evaluate_v7(task, paraphrase, generation_contract=False)
    assert strict["deterministic_codes"]["contract"] == ["normal_not_contract_form"]
    assert strict["deterministic_pass"] is False
    assert diagnostic["deterministic_codes"]["contract"] == []
    assert diagnostic["generation_contract_applied"] is False
    assert diagnostic["deterministic_pass"] is True
    assert P.deterministic_reasons_v7(task, paraphrase) == strict["deterministic_reasons"]
    assert P.deterministic_reasons_v7(task, paraphrase, generation_contract=False) == []


def test_reference_is_byte_identical_and_has_a_new_policy_identity():
    digest = hashlib.sha256(Path(D.__file__).read_bytes()).hexdigest()
    assert digest == P.V7_REFERENCE_SHA256 == P.V7_TRANSPLANT_SHA256
    assert P.V7_REFERENCE_COMMIT == "256c40b"
    assert P.V7_VERSION.endswith("_256c40b")


@pytest.mark.parametrize("absence", G.NORMAL_ABSENCE_FORMS)
@pytest.mark.parametrize("limitation", G.NORMAL_LIMITATION_FORMS)
def test_all_canonical_normal_contract_combinations_pass(absence, limitation):
    result = P.evaluate_v7(_task(), absence + " " + limitation)
    assert result["deterministic_pass"] is True
    assert result["deterministic_codes"]["normal"] == []
    assert result["deterministic_codes"]["contract"] == []
    assert result["model_applicability"] == "not_applicable"


@pytest.mark.parametrize("candidate,reason", [
    ("주석에 표시된 결함이 없다.", "normal_limitation_absent"),
    ("주석에 표시된 결함이 없다. 이것이 불합격을 뜻하지는 않는다.",
     "normal_limitation_wrong_direction"),
    ("주석에 표시된 결함이 없다. 이것이 합격이나 안전을 뜻하지 않는다고 볼 이유는 없다.",
     "normal_limitation_inverted"),
    ("안전한 용접부다. 주석에 표시된 결함이 없다.", "normal_verdict_implied"),
    ("주석에 표시된 결함이 없다. 결함이 없는 것은 용접 조건이 좋았기 때문이다.",
     "normal_cause_asserted"),
    ("주석에 표시된 결함이 영 개다. 이것이 합격이나 안전을 뜻하지는 않는다.",
     "normal_number_or_unit"),
    ("주석에 표시된 결함이 없다. 다만 왼쪽 아래에 흐린 결함이 보인다.",
     "normal_defect_or_position"),
    ("규칙: 결함이 없으면 그렇다고만 쓴다. 주석에 표시된 결함이 없다.",
     "normal_instruction_leak"),
])
def test_normal_supplement_rejects_even_in_contract_excluded_diagnostic(candidate, reason):
    result = P.evaluate_v7(_task(), candidate, generation_contract=False)
    assert reason in result["deterministic_codes"]["normal"]
    assert not result["deterministic_pass"]
    assert P.deterministic_normal_v7(_task(True), candidate) == []


def test_normal_supplement_does_not_replace_existing_negative_layer():
    result = P.evaluate_v7(_task(), _normal() + " 1", generation_contract=False)
    assert "defect_free_number" in result["deterministic_codes"]["negative"]
    assert not result["deterministic_pass"]


@pytest.mark.parametrize("bad", [None, 0, 1, "false"])
def test_contract_switch_is_a_real_boolean(bad):
    with pytest.raises(ValueError, match="generation_contract must be a boolean"):
        P.evaluate_v7(_task(), _normal(), generation_contract=bad)


def test_normal_trusted_evidence_is_checked_even_without_model():
    task = _task()
    name = P.label_map.load_label_map().defect_types["porosity"].name_ko
    task["evidence"]["defect_counts_by_name"] = {name: 1}
    with pytest.raises(ValueError, match="counts differ"):
        P.evaluate_v7(task, _normal())


def test_defect_deterministic_pass_is_not_model_acceptance():
    task = _task(True)
    result = P.evaluate_v7(task, _defect(task))
    assert result["deterministic_pass"] is True
    assert result["model_applicable"] is True
    assert result["model_applicability"] == "applicable"
    assert "combined_accept" not in result
    assert "model_accept" not in result


def test_grounded_and_contract_reasons_are_not_conflated():
    task = _task(True)
    name = task["locked_target"]["defects"][0]["name"]
    result = P.evaluate_v7(task, name + " 1개가 영상 가운데에 표시되어 있다.")
    assert "count_mismatch" in result["deterministic_codes"]["grounded"]
    assert result["deterministic_codes"]["contract"] == ["limitation_not_contract_form"]
    assert result["deterministic_reasons"] == sorted(set(result["deterministic_reasons"]))


@pytest.mark.parametrize("suffix,reason", [
    ("이 용접부는 문제가 없어요.", "verdict_asserted"),
    ("이것이 안전을 뜻하지 않는다는 말은 아니다.", "restriction_double_negated"),
])
def test_d_additional_negative_patterns(suffix, reason):
    assert reason in P.deterministic_reasons_v7(_task(), _normal() + " " + suffix,
                                              generation_contract=False)


def test_renderer_has_only_three_data_fields_and_system_has_questions_and_types():
    task = _task(True)
    generated = {"text": _defect(task), "finish_reason": "eos"}
    before = deepcopy((task, generated))
    prompt = json.loads(P.render_judge_v7(task, generated))
    assert prompt == {"row_class": "defect", "candidate": generated["text"],
                      "locked_fields": P.grouped_locked_fields(task)}
    for key, question in P.V7_QUESTIONS.items():
        assert f"{key} (boolean): {question}" in P.JUDGE_SYSTEM_V7
    assert "reason (string):" in P.JUDGE_SYSTEM_V7
    assert "네 물음은 모두 참이면 문제가 없다는 뜻" in P.JUDGE_SYSTEM_V7
    assert not any(key in P.JUDGE_SYSTEM_V7 for key in ("questions", "expected_output_keys", "output_types"))
    assert not any(example in P.JUDGE_SYSTEM_V7 for example in (": true", ": false", ': "true"', ': "false"'))
    assert task == before[0] and generated == before[1]
    assert P.V7_MAX_NEW_TOKENS == 160
    assert "reason_long" in P.V7_RULES["reason"]
    assert "40자 초과는 표지만" in P.JUDGE_SYSTEM_V7


def test_defect_renderer_does_not_copy_metadata_or_candidate_objects():
    task = _task(True)
    candidate = _defect(task) + ' {"questions": "candidate data", "output_schema": "not instructions"}'
    before = P.render_judge_v7(task, {"text": candidate, "finish_reason": "eos"})
    task["source_ids"] = ["different-source"]
    task["prompt"] = "do not copy this original request"
    after = P.render_judge_v7(task, {"text": candidate, "finish_reason": "length"})
    assert after == before
    prompt = json.loads(after)
    assert set(prompt) == {"row_class", "candidate", "locked_fields"}
    assert prompt["candidate"] == candidate


def test_no_shared_mutable_questions():
    value = P.structured_questions_v7(_task(True))
    value.clear()
    assert set(P.structured_questions_v7(_task(True))) == set(P.V7_BOOL_KEYS)
    assert P.structured_questions_v7(_task()) == {}


def test_source_qa_retains_legacy_question_meaning():
    task = {"kind": "source_qa", "source_ids": ["synthetic-source"], "evidence": {},
            "prompt": "근거에 답하라."}
    result = P.evaluate_v7(task, "합성 근거에 따른 답이다.")
    assert result["row_class"] == "source_qa" and result["model_applicable"] is True
    assert result["deterministic_reasons"] == []
    prompt = json.loads(P.render_judge_v7(task, {"text": "합성 답", "finish_reason": "eos"}))
    assert prompt["expected_output_keys"] == ["grounded", "scope_ok", "language_ok", "reason"]
    assert "questions" not in prompt and prompt["original_request"] == task["prompt"]


@pytest.mark.parametrize("value", [None, 1, {}, []])
def test_non_text_is_refused_before_decisions(value):
    with pytest.raises(ValueError, match="candidate text must be a string"):
        P.evaluate_v7(_task(), value)
    with pytest.raises(ValueError, match="candidate text must be a string"):
        P.render_judge_v7(_task(True), {"text": value, "finish_reason": "eos"})
