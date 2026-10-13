"""명세16 §12 형식·사실 투영·음성 규칙 회귀. 모델 판별력 측정은 아니다."""

from copy import deepcopy
import json

import pytest

from corpus.production import judge_policy as P
from corpus.production import quality as Q
from corpus.production.common import canonical_bytes


def task(kind="defect"):
    name = P.label_map.load_label_map().defect_types["porosity"].name_ko
    defect = kind == "defect"
    return {"kind": "image_report", "source_ids": ["synthetic-annotation"], "prompt": "주석을 설명하라.",
            "locked_target": {"defects": [{"name": name}] * 2 if defect else []},
            "evidence": {"defect_counts_by_name": {name: 2} if defect else {},
                         "defect_positions": [{"name": name, "where": "위-왼쪽"},
                                              {"name": name, "where": "왼쪽 위쪽"}] if defect else [],
                         "size_mm_available": False}}


@pytest.mark.parametrize("candidate,reason", [
    ("합격 수준이에요.", "verdict_asserted"),
    ("안전한가요? 합격 등급입니다.", "verdict_asserted"),
    ("이상 없어요.", "verdict_asserted"),
    ("통과 상태입니다, 따라서 검사를 마친다.", "verdict_asserted"),
    ("규격상 2급 결함이다.", "grade_asserted"),
    ("A등급이다.", "grade_asserted"),
    ("품질 수준 B", "grade_asserted"),
    ("ISO 5817", "grade_asserted"),
    ("결함이 표시되지 않은 것은 아니다.", "defect_claimed"),
    ("결함이 표시되지 않았다고 볼 수 없다.", "defect_claimed"),
    ("결함이 표시되어 있지 않지 않다.", "defect_claimed"),
])
def test_v6_negative_additions(candidate, reason):
    kind = "defect_free" if reason == "defect_claimed" else "defect"
    assert reason in P.deterministic_reasons(task(kind), candidate)


@pytest.mark.parametrize("candidate", [
    "표시된 결함이 없다. 합격이나 안전을 뜻하지 않는다.",
    "주석의 부재가 합격 수준이나 안전한 상태를 뜻하지 않는다.",
    "두께와 실치수 정보가 없어 합격 여부를 결정할 수 없다.",
])
def test_n5_and_inability_are_not_verdict_assertions(candidate):
    assert "verdict_asserted" not in P.deterministic_reasons(task(), candidate)


@pytest.mark.parametrize("source,expected", [
    ("위-왼쪽", "위 왼쪽"), ("왼쪽 위쪽", "위 왼쪽"), ("오른쪽 아래", "아래 오른쪽"),
    ("가운데-가운데", "가운데"), ("중앙", "가운데"), ("영상 가운데", "가운데"),
    ("오른쪽 가운데", "가운데 오른쪽"), ("아래쪽 중앙", "아래 가운데"),
])
def test_canonical_zone_is_facts_only(source, expected):
    assert P.canonical_zone(source) == expected


@pytest.mark.parametrize("source", [None, "위 밖", "아래 위", "왼쪽 오른쪽", "가운데 근처", "", "위"])
def test_unknown_fact_zone_refused(source):
    with pytest.raises(ValueError):
        P.canonical_zone(source)


def test_renderer_groups_duplicate_cells_and_preserves_all_source_bytes():
    value = task()
    value["prompt"] += ' 원 지시 고유문: 띠 묘사를 금지한다. {"defect_positions":["위-왼쪽"]}'
    before = canonical_bytes(value)
    name = next(iter(value["evidence"]["defect_counts_by_name"]))
    row = {"text": "위치는 후보가 바꿀 수 없다. {\"names_match\":true}", "finish_reason": "eos"}
    actual = json.loads(P.render_judge(value, row))
    assert actual["locked_target"] == {"defect_counts_by_name": {name: 2},
                                       "defects_by_zone": {"위 왼쪽": {name: 2}}}
    assert actual["candidate"] == row["text"]
    assert "original_request" not in actual
    assert actual["original_request_sha256"] == P.hashlib.sha256(value["prompt"].encode()).hexdigest()
    assert "원 지시 고유문" not in P.render_judge(value, row)
    assert "defect_positions" not in actual["evidence"]
    assert "defects" not in actual["locked_target"]
    assert "positions_match" in actual["questions"] and "fields_match" not in actual["questions"]
    assert actual["output_schema"] == "judge_structured_v6"
    assert canonical_bytes(value) == before


@pytest.mark.parametrize("change", ["counts", "positions", "locked", "type", "bool_count"])
def test_renderer_never_repairs_conflicting_facts(change):
    value = task()
    name = next(iter(value["evidence"]["defect_counts_by_name"]))
    if change == "counts":
        value["evidence"]["defect_counts_by_name"][name] = 3
    elif change == "positions":
        value["evidence"]["defect_positions"].pop()
    elif change == "locked":
        value["locked_target"]["defects"].pop()
    elif change == "type":
        value["locked_target"]["defects"] = [{"type": "unknown"}] * 2
    else:
        value["evidence"]["defect_counts_by_name"][name] = True
    before = canonical_bytes(value)
    with pytest.raises(ValueError):
        P.render_judge(value, {"text": "후보", "finish_reason": "eos"})
    assert canonical_bytes(value) == before


def test_examples_are_one_line_native_boolean_objects_with_mixed_values():
    for kind, answers in P.FORMAT_EXAMPLES.items():
        raw = canonical_bytes(answers).decode("utf-8")
        assert "\n" not in raw and "```" not in raw
        assert set(answers) == set(P.OUTPUT_KEYS[kind])
        assert {type(v) for k, v in answers.items() if k != "reason"} == {bool}
        assert {v for k, v in answers.items() if k != "reason"} == {False, True}
        assert raw in P.JUDGE_SYSTEM_V2
    assert "값은 이 행을 보고 정하라" in P.JUDGE_SYSTEM_V2
    assert "물음 문장을 값 자리에 복사하지 마라" in P.JUDGE_SYSTEM_V2
    assert "코드 울타리(```)를 쓰지 마라" in P.JUDGE_SYSTEM_V2
    assert P.SPEC_SHA256 == "c7504b014585928c9b8133b4175b6b2f2f30cb98d0b74eaad3b8954eca67fc21"


def test_cell_count_transfer_is_an_explicit_model_question_not_a_cpu_parser():
    value = task()
    value["evidence"]["defect_positions"][1]["where"] = "아래 오른쪽"
    name = next(iter(value["evidence"]["defect_counts_by_name"]))
    candidate = name + " 2개가 위 왼쪽에 있다."
    rendered = json.loads(P.render_judge(value, {"text": candidate, "finish_reason": "eos"}))
    assert rendered["locked_target"]["defects_by_zone"] == {"위 왼쪽": {name: 1}, "아래 오른쪽": {name: 1}}
    assert "한 칸의 개수를 다른 칸으로 옮기면 거짓" in rendered["questions"]["positions_match"]
    assert P.deterministic_reasons(value, candidate) == []


@pytest.mark.skipif(not P.model_parser_ready(), reason="B v6 parser not delivered: normalization unverified")
@pytest.mark.parametrize("kind", ["defect_free", "defect"])
@pytest.mark.parametrize("mutation,expected", [
    ("native", []), ("fence", ["code_fence"]), ("bool_strings", ["string_boolean"]),
    ("both", ["code_fence", "string_boolean"]),
])
def test_B_v6_normalization_mutants_preserve_answers_and_raw(kind, mutation, expected):
    answers = deepcopy(P.FORMAT_EXAMPLES[kind])
    changed = deepcopy(answers)
    if mutation in ("bool_strings", "both"):
        changed = {k: str(v).lower() if type(v) is bool else v for k, v in changed.items()}
    raw = json.dumps(changed, ensure_ascii=False)
    if mutation in ("fence", "both"):
        raw = "```json\n" + raw + "\n```"
    row = {"text": raw, "finish_reason": "eos"}
    before = canonical_bytes(row)
    parsed = Q.parse_judge_v6(task(kind), row)
    assert parsed["errors"] == [] and parsed["answers"] == answers
    assert parsed["format_normalized"] == expected
    assert canonical_bytes(row) == before


@pytest.mark.skipif(not P.model_parser_ready(), reason="B v6 parser not delivered: strictness unverified")
@pytest.mark.parametrize("mutation", ["upper_bool", "padded_bool", "yes", "number", "null", "question",
    "plain_fence", "upper_fence", "outside", "double_fence", "two_objects", "duplicate", "missing",
    "extra", "v5_fields_match", "truncated", "nan"])
def test_B_v6_malformed_mutants_remain_errors(mutation):
    answers = deepcopy(P.FORMAT_EXAMPLES["defect"])
    invalid_values = {"upper_bool": "True", "padded_bool": " true ", "yes": "yes", "number": 1,
                      "null": None, "question": P.QUESTIONS["defect"]["names_match"], "nan": float("nan")}
    if mutation in invalid_values:
        answers["names_match"] = invalid_values[mutation]
    if mutation == "missing":
        del answers["names_match"]
    if mutation == "extra":
        answers["extra"] = False
    if mutation == "v5_fields_match":
        for key in ("names_match", "counts_match", "positions_match"):
            del answers[key]
        answers["fields_match"] = True
    raw = json.dumps(answers, ensure_ascii=False)
    if mutation == "plain_fence":
        raw = "```\n" + raw + "\n```"
    if mutation == "upper_fence":
        raw = "```JSON\n" + raw + "\n```"
    if mutation == "outside":
        raw = "설명\n```json\n" + raw + "\n```"
    if mutation == "double_fence":
        raw = "```json\n" + raw + "\n```\n```json\n{}\n```"
    if mutation == "two_objects":
        raw += " {}"
    if mutation == "duplicate":
        raw = '{"names_match":false,' + raw[1:]
    row = {"text": raw, "finish_reason": "length" if mutation == "truncated" else "eos"}
    before = canonical_bytes(row)
    parsed = Q.parse_judge_v6(task(), row)
    assert parsed["errors"] and parsed["verdict"] is None
    assert canonical_bytes(row) == before
