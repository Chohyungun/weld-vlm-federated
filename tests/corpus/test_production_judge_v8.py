"""판정 계약 v8 — 파서 벡터(조정 집합 첫 호출 A01 의 원출력 재생 · 그 꼴을 옮긴 합성 행), 지시문 렌더링, 문자열로 가르는
고정 제한 문장, 정책 신원. 모델 · GPU 를 쓰지 않는다."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from corpus.production import judge_policy as P
from corpus.production import quality as Q
from corpus.production.common import canonical_bytes

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "evaluation/vectors/judge_parser_v8_vectors.json").read_text(encoding="utf-8"))
V7_TWO = ("no_unannotated_attributes", "no_unsupported_relation")
DEFECT = {"kind": "image_report", "evidence": {"defect_counts_by_name": {"기공": 1},
                                               "defect_positions": [{"name": "기공", "where": "위 왼쪽"}],
                                               "row_class": "defect", "size_mm_available": False},
          "locked_target": {"clauses": [], "defects": [{"name": "기공", "where": "위 왼쪽"}], "verdict": None},
          "prompt": "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.", "source_ids": ["synthetic-annotation"]}
NORMAL = {"kind": "image_report", "evidence": {"defect_counts_by_name": {}, "defect_positions": [], "row_class": "defect_free",
                                               "size_mm_available": False},
          "locked_target": {"clauses": [], "defects": [], "verdict": None},
          "prompt": "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.", "source_ids": ["synthetic-annotation"]}
LIMIT = "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."
#: 고정 해시 — 이 판(v8, 2026-10-15)에서 낸 값. 지시문 · 규칙을 바꾸면 여기와 보고의 당시 값 · 지금 값을 함께 고친다.
#: 규칙 해시의 당시 값(44 §4, 범주 규칙 전) 6071f8cfa12e01a54950ce6ffff7640b17eff5ccf93b58dc6b7518c273339e36
#: 규칙 해시의 당시 값(게이트 87, 정본 9aa88c93) a8cc53f60d94bba2ebf8a86fb76a5762c9e806a558f849af5b5f0ec8bc7ea503 — 규칙은 정본 sha 를 싣는다
#: 규칙 해시의 당시 값(정정 커밋 8696585a, 판 이름을 바꾸기 전) ddac71416386736407e8a4ba10859ff6cf66cac66f6f3c60902ec7d8a9caba63
#: 규칙 해시의 당시 값(47 정정 판 0f34bd1a, 정본 ba291a24) aab846587b264cad6330c1422a37e25fbcfba4ac59aeafca9c0fad32577a2b5c
#: 규칙 해시의 당시 값(49 재정정 판 155f3e5f, 정본 7518fe04) 422879691f8b29648da3d5dc851d2801ae8df2f947b2ae8d54bcc9a2405d23b5
#: 규칙 해시의 당시 값(50 정정 판 283ec45d, 정본 7e5e769f) f0caa1010c25a786606ce10f054b4c8e675a18462c9cfacdc1254e1ac63a48b1
V8_SYSTEM_PROMPT_SHA256 = "fc8965f9739bc06bc7cbad239407ebafaa5c9e6a356c3fe612861f0f3d81405b"
V8_RULES_SHA256 = "5b2bd09b8ae549dc4dca1c52cb9609cec3ba52d4a7466b21a5423440f24f716a"


def _parse(parser, task, row):
    return parser(copy.deepcopy(task), copy.deepcopy(row))


def test_벡터는_A01_원출력_94행과_그_출처_해시를_싣는다():
    assert VECTORS["parser"] == Q.JUDGE_V8_PARSER_VERSION and VECTORS["schema"] == Q.JUDGE_V8_SCHEMA_VERSION
    assert VECTORS["a01_source_sha256"]["outputs.jsonl"] == "6c911087294c3d6fda1f0ee33102fcb0b44f909c92a49689632978a986a3d8e4"
    assert VECTORS["a01_source_sha256"]["cases.json"] == "974f22c8900b2f523b85ea8d3e8525dcb937fac0cc758925b83df98ae82e1eab"
    assert len(VECTORS["a01_replay"]) == 94
    assert Counter(r["group"] for r in VECTORS["a01_replay"]) == {"D+": 50, "ADV1": 22, "ADV2": 22}


def test_A01_원출력은_v7_파서에서_23행이_판정_불능이었다():
    errors = Counter(tuple(_parse(Q.parse_judge_v7, DEFECT, r)["errors"]) for r in VECTORS["a01_replay"])
    assert errors == {(): 71, ("judge_unparsable",): 23}
    assert [tuple(r["v7_errors"]) for r in VECTORS["a01_replay"]] == [
        tuple(_parse(Q.parse_judge_v7, DEFECT, r)["errors"]) for r in VECTORS["a01_replay"]]


def test_A01_원출력은_v8_파서에서_판정_불능이_0이다():
    """v7 물음에 대한 출력이라 v8 의 두 키가 없다 — 모두 `judge_keys` 이고 JSON 을 못 읽은 행(`judge_unparsable`)은 없다.
    물음을 바꾸면 원출력이 달라지므로 이것은 파서의 견고성만 보인다."""
    errors = Counter(tuple(_parse(Q.parse_judge_v8, DEFECT, r)["errors"]) for r in VECTORS["a01_replay"])
    assert errors == {("judge_keys",): 94}


def test_A01_원출력의_첫_객체에서_v7_두_키를_모두_읽는다():
    """v8 파서의 추출(첫 닫힌 객체 · 거듭 나온 키 보존)로 v7 의 같은 두 키를 읽으면 94 행 모두 읽힌다 — v7 이 중복 키로
    못 읽은 23 행 포함. v7 이 읽은 71 행에서는 v7 의 판정값과 같다."""
    read = 0
    for r in VECTORS["a01_replay"]:
        found = Q._first_json_object_v8(r["text"])
        pairs = Q._json_pairs_v8(found[0])
        values = {k: [{"true": True, "false": False}.get(v, v) if type(v) is str else v for kk, v in pairs if kk == k]
                  for k in V7_TWO}
        assert all(len(set(values[k])) == 1 and type(values[k][0]) is bool for k in V7_TWO), r["case"]
        two = {k: values[k][0] for k in V7_TWO}
        assert two == r["v7_two_keys"]
        v7 = _parse(Q.parse_judge_v7, DEFECT, r)
        if v7["verdict"] is not None:
            assert two == {k: v7["verdict"][k] for k in V7_TWO}
        read += 1
    assert read == 94


@pytest.mark.parametrize("vector", VECTORS["parser_vectors"], ids=lambda v: v["name"])
def test_파서_벡터(vector):
    out = Q.parse_judge_v8(copy.deepcopy(vector["task"]), copy.deepcopy(vector["row"]))
    assert {k: out[k] for k in vector["expect"]} == vector["expect"]
    assert out["schema"] == Q.JUDGE_V8_SCHEMA_VERSION and out["parser"] == Q.JUDGE_V8_PARSER_VERSION
    assert (out["verdict"] is None) == bool(out["errors"])
    assert out["flags"] == []


def test_파서_벡터는_A01_에서_본_꼴을_모두_덮는다():
    names = {v["name"] for v in VECTORS["parser_vectors"]}
    assert {"a01_shape_other_keys_and_repeated_other_key", "a01_shape_string_booleans_other_keys", "fence",
            "trailing_explanation", "repeated_target_conflict", "truncated_length", "unclosed_eos"} <= names
    errors = Counter(e for v in VECTORS["parser_vectors"] for e in v["expect"]["errors"])
    assert set(errors) == {"judge_key_conflict", "judge_value_type", "judge_keys", "judge_truncated", "judge_unparsable",
                           "judge_not_applicable", "judge_task_unsupported", "judge_row_invalid"}


def test_v8_파서는_키_둘만_보고_거듭_나온_키는_같은_값일_때만_받는다():
    ok = {"text": '{"attribute": "없음", "relation": "없음", "language_ok": true, "language_ok": false}', "finish_reason": "eos"}
    assert Q.parse_judge_v8(copy.deepcopy(DEFECT), ok)["format_normalized"] == ["other_keys_ignored"]
    bad = {"text": '{"relation": "없음", "attribute": "있음", "attribute": "없음"}', "finish_reason": "eos"}
    assert Q.parse_judge_v8(copy.deepcopy(DEFECT), bad)["errors"] == ["judge_key_conflict"]


def test_지시문은_물음_둘과_고정_꼴만_싣는다():
    s = P.JUDGE_SYSTEM_V8
    assert all(q in s for q in P.V8_QUESTIONS.values()) and all(k in s for k in P.V8_ANSWER_KEYS)
    assert all(w in s for w in P.V8_ANSWER_WORDS)
    for banned in ("{", "}", "```", "has_limitation", "language_ok", "no_unannotated_attributes", "no_unsupported_relation",
                   "grounded", "scope_ok", "reason", "source_qa", "true", "false"):
        assert banned not in s, banned
    assert len(s) < len(P.JUDGE_SYSTEM_V7) // 2
    assert all(ord(c) >= 0x20 for c in s) and s == s.strip()
    assert P.V8_MAX_NEW_TOKENS == P.V7_MAX_NEW_TOKENS == 160


def test_지시문_규칙_고정_해시():
    assert hashlib.sha256(P.JUDGE_SYSTEM_V8.encode("utf-8")).hexdigest() == V8_SYSTEM_PROMPT_SHA256
    assert hashlib.sha256(canonical_bytes(P.V8_RULES)).hexdigest() == V8_RULES_SHA256


def test_판정_입력은_v7_과_같은_바이트이고_결함_행만_부른다():
    gen = {"text": "주석에 표시된 결함은 위 왼쪽의 기공 1개다. " + LIMIT, "finish_reason": "eos"}
    a = P.render_judge_v8(copy.deepcopy(DEFECT), dict(gen))
    assert a == P.render_judge_v8(copy.deepcopy(DEFECT), dict(gen)) == P.render_judge_v7(copy.deepcopy(DEFECT), dict(gen))
    normal = copy.deepcopy(NORMAL)
    with pytest.raises(ValueError, match="v8_model_call_defect_only"):
        P.render_judge_v8(normal, dict(gen))
    with pytest.raises(ValueError, match="v8_model_call_defect_only"):
        P.render_judge_v8({"kind": "source_qa"}, dict(gen))
    assert P.structured_questions_v8(copy.deepcopy(DEFECT)) == P.V8_QUESTIONS and P.structured_questions_v8(normal) == {}


def test_고정_제한_문장은_문자열로_가르고_생성_계약을_꺼도_건다():
    good = "주석에 표시된 결함은 위 왼쪽의 기공 1개다. " + LIMIT
    bad = "주석에 표시된 결함은 위 왼쪽의 기공 1개다. 치수 정보가 없어 합부는 말하기 어렵다."
    for contract in (True, False):
        assert P.evaluate_v8(copy.deepcopy(DEFECT), good, generation_contract=contract)["deterministic_codes"]["limitation"] == []
        out = P.evaluate_v8(copy.deepcopy(DEFECT), bad, generation_contract=contract)
        assert out["deterministic_codes"]["limitation"] == ["limitation_not_contract_form"]
        assert out["deterministic_pass"] is False and out["model_applicable"] is True
    assert P.evaluate_v7(copy.deepcopy(DEFECT), bad, generation_contract=False)["deterministic_codes"]["contract"] == []


def test_v8_결정적_판단은_v7_에_제한_문장만_더한다():
    normal = copy.deepcopy(NORMAL)
    text = "표시된 결함이 없다. 결함이 표시되지 않았다는 것이 합격이나 안전을 뜻하지는 않는다."
    v7, v8 = P.evaluate_v7(copy.deepcopy(normal), text), P.evaluate_v8(copy.deepcopy(normal), text)
    assert v8["deterministic_codes"] == {**v7["deterministic_codes"], "limitation": [], "category": []}
    assert v8["model_applicable"] is False and v8["deterministic_pass"] == v7["deterministic_pass"]
    with pytest.raises(ValueError, match="v8_image_report_only"):
        P.evaluate_v8({"kind": "source_qa", "source_ids": [], "evidence": {}, "prompt": ""}, "글")


def test_v8_정책_신원은_결정적이고_v7_신원과_v8_정의를_결속한다():
    a, b = P.policy_identity_v8(), P.policy_identity_v8()
    assert canonical_bytes(a) == canonical_bytes(b)
    assert a["version"] == P.V8_VERSION and a["output_schema"] == Q.JUDGE_V8_SCHEMA_VERSION == P.V8_OUTPUT_SCHEMA_VERSION
    assert canonical_bytes(a["deterministic_policy"]) == canonical_bytes(P.policy_identity_v7())
    assert a["system_prompt_sha256"] == V8_SYSTEM_PROMPT_SHA256 and a["rules_sha256"] == V8_RULES_SHA256
    assert a["structured_parser_ready"] is True
    assert {"evaluate_v8", "render_judge_v8", "JUDGE_SYSTEM_V8", "V8_RULES", "deterministic_limitation_v8",
            "evaluate_v7"} <= set(a["implementation"])
    symbols = a["parser_implementation"]["symbols"]
    assert f"{Q.__name__}:parse_judge_v8" in symbols
    assert not any(f"{Q.__name__}:parse_judge_{old}" in symbols for old in ("v5", "v6", "v7"))


def test_v8_은_v7_신원을_바꾸지_않는다():
    """v8 은 v7 옆에 따로 둔 API 다 — v7 의 정책 폐쇄 · 파서 폐포에 v8 정의가 들지 않는다."""
    identity = P.policy_identity_v7()
    assert not any("v8" in name.lower() for name in identity["implementation"]["policy"])
    assert not any("v8" in name.lower() for name in identity["parser_implementation"]["symbols"])
    assert identity["version"] == P.V7_VERSION == "image_report_judge_v7_20261016_0b5a3fd4"


# ---------------------------------------------------------------- 두 물음 밖 범주의 결정적 규칙
#: 범주 정의에서 만든 합성 문장(조정 집합의 사례 문장이 아니다) → 걸려야 할 규칙
RULE_HITS = [
    ("기공이 다른 칸보다 많다.", "v8_comparison"),
    ("이 균열이 가장 뚜렷하다.", "v8_comparison"),
    ("결함 수는 비교적 적은 편이다.", "v8_comparison"),
    ("용접 속도 탓에 생긴 결함이다.", "v8_cause"),
    ("수분으로 인해 기공이 생겼다.", "v8_cause"),
    ("이 결함은 냉각 조건에 기인한다.", "v8_cause"),
    ("허용 기준 이내라면 합격이다.", "v8_verdict"),
    ("품질은 양호하다.", "v8_verdict"),
    ("결함이 있어도 사용에 문제가 없다.", "v8_verdict"),
    ("출력 형식은 JSON 이다.", "v8_instruction"),
    ("<결함 이름>을 적는다.", "v8_instruction"),
    ("주어진 근거의 문장을 그대로 쓴다.", "v8_instruction"),
    ("기공은 모두 다섯 개다.", "v8_numeral_word"),
    ("셋째 칸에 균열이 있다.", "v8_numeral_word"),
    ("균열 하나가 표시되었다.", "v8_numeral_word"),
    ("결함이 많아 보이네요.", "v8_rhetoric"),
    ("재검사가 필요하다.", "v8_rhetoric"),
    ("수분이 원인일 수 있다.", "v8_rhetoric"),
    ("균열이 길다.", "v8_size_word"),
    ("기공이 큰 편이다.", "v8_size_word"),
    ("슬래그혼입이 넓게 퍼져 있다.", "v8_size_word"),
    ("기공의 윤곽이 뚜렷하게 보인다.", "v8_attribute"),
    ("균열의 방향은 가로다.", "v8_attribute"),
    ("슬래그혼입의 끝이 날카로운 편으로 보인다.", "v8_attribute"),
    ("두 균열이 서로 연결되어 있다.", "v8_relation"),
    ("기공들이 용접선을 따라 분포한다.", "v8_relation"),
    ("융합불량이 기공과 겹쳐 있다.", "v8_relation"),
]
#: 생성 RUN 이 실제로 쓰는 꼴(이름 · 숫자 · 칸 · 모른다는 설명 · 고정 문장) — 어떤 규칙에도 걸리지 않아야 한다
RULE_CLEAN = [
    "기공 2개가 위쪽 왼쪽에 표시되어 있다.",
    "주석에 표시된 결함은 아래 가운데의 균열 1개다.",
    "주석만으로는 결함의 모양과 크기를 알 수 없다.",
    "결함 사이의 거리는 주석에 나타나 있지 않다.",
    "같은 칸에 표시된 결함끼리 닿아 있는지는 주석으로 확인되지 않는다.",
    "표시된 결함은 모두 가운데 높이에 있다.",
    "종류가 다른 결함이 함께 표시되어 있다.",
    "용입불량 1개와 융합불량 1개가 표시되었다.",
    "결함의 실제 크기는 주석에 적혀 있지 않다.",
    "주석만으로는 결함이 서로 이어져 있는지 알 수 없다.",
    "각 결함이 어떤 방향으로 놓였는지는 주석으로 알 수 없다.",
    "위쪽 왼쪽 칸과 위쪽 가운데 칸에 기공이 1개씩 있다.",
    "따라서 주석에 표시된 결함은 균열 1개다.",
]


@pytest.mark.parametrize("sentence,code", RULE_HITS)
def test_범주_규칙은_정의의_표지를_잡는다(sentence, code):
    assert code in P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT)


@pytest.mark.parametrize("sentence", RULE_CLEAN)
def test_범주_규칙은_생성_RUN_의_꼴을_잡지_않는다(sentence):
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == []


def test_고정_문장은_범주_규칙에서_빠진다():
    from corpus.production import generation_policy as G
    for form in (*G.LIMITATION_FORMS, *G.NORMAL_ABSENCE_FORMS, *G.NORMAL_LIMITATION_FORMS):
        assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), form) == []
        assert P.deterministic_categories_v8(copy.deepcopy(NORMAL), form) == []
    assert "v8_rhetoric" in P.deterministic_categories_v8(copy.deepcopy(DEFECT), "기공 1개다! " + LIMIT)
    assert P.evaluate_v8(copy.deepcopy(DEFECT), "균열이 길다. " + LIMIT)["deterministic_codes"]["category"] == ["v8_size_word"]
    assert set(P.V8_CATEGORY_RULES) == {"v8_comparison", "v8_cause", "v8_verdict", "v8_instruction",
                                        "v8_numeral_word", "v8_rhetoric", "v8_size_word"}


def test_속성_관계_규칙은_정본의_낱말과_판독_불가_판정을_가져다_쓴다():
    """속성 낱말(`ATTR` · `_ATTR_NOUN`)과 앎의 부정(`ATTR_OK` · `_KNOW_NEG`)은 정본의 것을 불러 쓴다 — 정본을 바꾸면 규칙이 따라
    바뀐다. v8 이 더한 꼴(`V8_ATTRIBUTE_EXTRA`)에는 정본의 형용 낱말이 겹치지 않는다."""
    import re
    from corpus.production import judge_grounded_v7 as G7
    canonical = re.findall(r"[가-힣]{2,}", G7.ATTR.pattern)
    extra = re.findall(r"[가-힣]{2,}|(?<=[|:])[가-힣](?=[|)])", P.V8_ATTRIBUTE_EXTRA.pattern)
    assert canonical and extra and not any(e.startswith(c) for e in extra for c in canonical)
    text = "균열의 테두리가 또렷하다. " + LIMIT
    assert "v8_attribute" in P.deterministic_categories_v8(copy.deepcopy(DEFECT), text)
    known = "균열의 테두리는 주석으로 확인되지 않는다. " + LIMIT        # 정본의 `_KNOW_NEG`
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), known) == []
    plain = "균열의 폭이 일정하다. " + LIMIT                              # 정본의 `_ATTR_NOUN`('폭')만으로 걸린다
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), plain) == ["v8_attribute"]


def test_A01_반사실_결정적_규칙과_두_물음():
    """A01 의 모델 답(두 키)을 v8 판단에 넣은 결합 — D+ 기각 1/50(모델이 속성 키를 거짓으로 답한 한 행) · N+ 0/4 ·
    음성 거짓 통과 0/50. 44 §8 의 6/50(주석 밖 속성 3 · 관계 3)은 속성 · 관계 규칙(§9)으로 닫혔다."""
    replay = {r["case"]: r for r in VECTORS["a01_replay"]}
    tally, left = Counter(), Counter()
    for c in VECTORS["a01_cases"]:
        e = P.evaluate_v8(copy.deepcopy(c["task"]), c["candidate"], generation_contract=True)
        two = replay[c["case"]]["v7_two_keys"] if e["model_applicable"] else None
        model_ok = (not e["model_applicable"]) or bool(two and all(v is True for v in two.values()))
        combined = e["deterministic_pass"] and model_ok
        if c["group"] in ("D+", "N+"):
            tally[c["group"] + "_false_reject"] += not combined
        elif combined:
            tally["negative_false_accept"] += 1
            left[c["category"]] += 1
    assert (tally["D+_false_reject"], tally["N+_false_reject"], tally["negative_false_accept"]) == (1, 0, 0)
    # 결정적 판단만으로 음성 50 은 모두 기각, 양성 54 는 모두 통과 — 모델 답과 무관하다
    det = Counter((c["group"], P.evaluate_v8(copy.deepcopy(c["task"]), c["candidate"])["deterministic_pass"])
                  for c in VECTORS["a01_cases"])
    assert det == {("ADV1", False): 25, ("ADV2", False): 25, ("D+", True): 50, ("N+", True): 4}
    assert left == {}


def test_앎의_부정_문장은_범주_규칙_아홉_모두에서_빠진다():
    """재검수 1-4 의 삽입 문장('결함의 깊이는 알 수 없다' · '원인은 주석에 나타나 있지 않다' …)을 D+ 50 글의 끝 문장 앞에 넣어도
    결정적으로 모두 통과한다 — 정본의 판독 불가 판정을 모든 규칙 앞에 공통 제외로 둔다."""
    src = VECTORS["know_neg_insertions"]
    assert src["source"]["sha256"] == "3c0488a4d6da7e407ee2fffc39a8a40d25420ea1d4f6e08b735407c5edf101aa"
    assert len(src["sentences"]) == 11
    dplus = [c for c in VECTORS["a01_cases"] if c["group"] == "D+"]
    assert len(dplus) == 50
    for sentence in src["sentences"]:
        failed = []
        for c in dplus:
            body, last = c["candidate"].rsplit(". ", 1)
            text = f"{body}. {sentence} {last}"
            out = P.evaluate_v8(copy.deepcopy(c["task"]), text)
            if not out["deterministic_pass"]:
                failed.append((c["case"], out["deterministic_reasons"]))
        assert failed == [], (sentence, failed[:3])
    # 앎의 부정이 없는 같은 낱말은 그대로 걸린다
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "균열이 깊다. " + LIMIT) == ["v8_size_word"]
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "습기가 원인이다. " + LIMIT) == ["v8_cause"]


def _bypass(text: str, form: str, clause: list[str], joined: dict) -> str:
    """음성 글의 고정 문장이 아닌 '…다.' 문장마다 앎의 부정 마디를 잇는다 — 뒤에 이음(…지만 · …(으)나 · …는데 …) · 쉼표, 또는 앞에."""
    import re
    from corpus.production import generation_policy as G
    fixed = {*G.LIMITATION_FORMS, *G.NORMAL_ABSENCE_FORMS, *G.NORMAL_LIMITATION_FORMS}
    plain, front = clause
    out = []
    for p in re.split(r"(?<=[.?!])\s+", text.strip()):
        body = p[:-1]
        if p in fixed or not p.endswith("다."):
            out.append(p)
        elif form == "앞에":
            out.append(f"{front} {body}.")
        elif form == "쉼표":
            out.append(f"{body}, {plain}.")
        else:
            head, _, word = body.rpartition(" ")
            out.append(f"{head} {joined[word][form]} {plain}.".lstrip())
    return " ".join(out)


def test_앎의_부정_마디를_단정에_이어_붙여도_음성은_결정적으로_기각된다():
    """우회 탐침(재검수 8-1) — 삽입 탐침(위)과 짝. 문장 단위로 빼던 판(517c0dbe)은 '…지만 결함의 깊이는 알 수 없다.' 를 붙인
    음성 30/50 을 통과시켰다. 마디 단위로 빼면 꼴 여덟 · 마디 셋 모두 0/50 이다."""
    src = VECTORS["know_neg_bypass"]
    assert src["forms"] == ["지만", "(으)나", "는데", "쉼표", "앞에", "고", "(으)며", "기 때문에"]
    neg = [c for c in VECTORS["a01_cases"] if c["group"] in ("ADV1", "ADV2")]
    assert len(neg) == 50
    leaked = Counter()
    for name, clause in src["clauses"].items():
        for form in src["forms"]:
            for c in neg:
                text = _bypass(c["candidate"], form, clause, src["joined"])
                assert clause[0] in text or clause[1] in text
                if P.evaluate_v8(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                    leaked[(name, form, c["group"])] += 1
    assert leaked == {}


@pytest.mark.parametrize("sentence,codes", [
    ("치수가 허용 범위 안이라면 합격으로 볼 수 있지만 결함의 깊이는 알 수 없다.", ["v8_attribute", "v8_verdict"]),
    ("결함의 깊이는 알 수 없지만 다른 칸보다 크다.", ["v8_comparison", "v8_size_word"]),
    ("결함이 표면에 드러나지 않았다면 통과다.", ["v8_verdict"]),
    ("결함의 길이는 주석에 적혀 있지 않으나 전류 탓이다.", ["v8_cause"]),
    ("결함의 깊이는 알 수 없다, 품질은 양호하다.", ["v8_verdict"]),
    ("결함의 크기나 모양은 알 수 없다.", []),
    ("결함의 크기, 모양, 깊이는 주석으로 확인되지 않는다.", []),
    ("결함이 생긴 원인은 주석에 나타나 있지 않다.", []),
    ("같은 칸에 표시된 결함끼리 닿아 있는지는 주석으로 확인되지 않는다.", []),
])
def test_앎의_부정은_그_마디만_뺀다(sentence, codes):
    """마디 경계는 쉼표 · 이음 어미 뒤 — 보조 용언 '있다'(나타나 있지 · 닿아 있는지)와 속성 이름의 나열(크기나 · 크기,)은 잇는다."""
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == codes


@pytest.mark.parametrize("sentence", ["직전보다 결함이 줄었다.", "지난 촬영 대비 결함이 많다.", "상태가 나빠졌다.",
                                      "결함이 개선되었다.", "결함 상태가 악화되었다."])
def test_비교급은_변화_비교_짝과_견줌_일반형으로_잡는다(sentence):
    assert "v8_comparison" in P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT)


@pytest.mark.parametrize("sentence", ["예열 부족과 관련된 결함이다.", "전류 변동이 결함을 유발했다.", "습기가 기공을 초래했다."])
def test_원인은_일반형으로_잡는다(sentence):
    assert "v8_cause" in P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT)


# ---------------------------------------------------------------- 결정적 층의 틈 정정(47)
GAP46 = VECTORS["gap46"]


def test_틈_벡터는_46_탐침_35와_변형_15를_싣는다():
    """46 합성 탐침 셋을 실행해 뽑은 입력 35 와 47 의 변형 15. 기대의 뜻 — flip 은 고치기 전(e109139a) 기각 · 지금 통과,
    keep_pass 는 둘 다 통과, keep_reject 는 둘 다 기각(비교급 F · 맨 '가운데' · 변형의 대조)."""
    rows = GAP46["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 50
    assert Counter(r["source"] for r in rows) == {"synth_probe_v8.py": 28, "synth_probe_v8_extra.py": 3,
                                                   "synth_probe_v8_extra2.py": 4, "gap_cases.py": 15}
    assert Counter(r["expect"] for r in rows) == {"flip": 19, "keep_pass": 15, "keep_reject": 16}
    want = {"flip": (False, True), "keep_pass": (True, True), "keep_reject": (False, False)}
    assert all((r["before"]["pass"], r["after"]["pass"]) == want[r["expect"]] for r in rows)
    for s in GAP46["source"]["probes"]:                    # 탐침 원본이 있는 트리에서는 입력을 뽑은 스크립트의 바이트와 같다
        path = ROOT / GAP46["source"]["probe_dir"] / s["name"]
        if path.exists():
            assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == s["sha256"], s["name"]


@pytest.mark.parametrize("row", GAP46["rows"], ids=lambda r: r["id"])
def test_틈_벡터(row):
    out = P.evaluate_v8(copy.deepcopy(row["task"]), row["text"])
    assert out["deterministic_pass"] is row["after"]["pass"]
    assert {k: v for k, v in out["deterministic_codes"].items() if v} == row["after"]["codes"]


@pytest.mark.parametrize("sentence,codes", [
    ("같은 높이에 있다는 것만으로 결함이 이어져 있다고 볼 수는 없다.", []),    # 인용 마디의 관계 표지는 뺀다
    ("같은 높이의 칸에 있다는 것만으로 하나로 이어진 결함이라고 볼 수는 없다.", []),
    ("하나로 된 결함이라고 볼 수는 없다.", []),                                # 인용 마디의 '하나로' 도
    ("용접 조건 때문에 생겼다고 볼 수는 없다.", ["v8_cause"]),                # 원인 · 크기 · 판정은 인용 마디에서도 센다
    ("균열이 길다고 볼 수는 없다.", ["v8_size_word"]),
    ("불합격이라고 볼 수는 없다.", ["v8_verdict"]),
    ("결함이 이어져 있지 않다고 볼 수는 없다.", ["v8_relation"]),             # 이중 부정은 단정이다
    ("합격이 아니라고 볼 수는 없다.", ["v8_verdict"]),
    ("결함이 하나로 이어져 있다.", ["v8_relation"]),                          # 관계 표지 앞의 '하나로' 는 관계 규칙이 잡는다
    ("결함은 하나로 된 덩어리다.", ["v8_attribute", "v8_numeral_word"]),
    ("결함은 아래쪽 높이의 양 끝 칸에 있다.", []),                            # 자리 '끝'
    ("결함은 위쪽 높이의 오른쪽 끝에 있다.", []),
    ("균열의 끝이 갈라져 있다.", ["v8_attribute"]),                           # 이름 뒤 '끝' 은 속성
    ("주석은 결함의 길이를 알려 주지 않는다.", []),                           # 보조 용언 '주다' 앞에서 나누지 않는다
])
def test_인용_마디는_관계_표지와_하나로만_뺀다(sentence, codes):
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == codes


def test_넓힌_앎의_부정을_이어_붙여도_음성은_결정적으로_기각된다():
    """47 이 앎의 부정을 넓혔다('볼 수 없' · '알려 주지 않' · 관계 판독 불가 · 인용) — 1015h 의 우회 탐침을 넓힌 마디 넷으로 다시
    돌려도 꼴 여덟 모두 0/50 이다."""
    src, clauses = VECTORS["know_neg_bypass"], GAP46["probe"]["bypass_clauses"]
    assert len(clauses) == 4
    neg = [c for c in VECTORS["a01_cases"] if c["group"] in ("ADV1", "ADV2")]
    leaked = Counter()
    for name, clause in clauses.items():
        for form in src["forms"]:
            for c in neg:
                text = _bypass(c["candidate"], form, clause, src["joined"])
                if P.evaluate_v8(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                    leaked[(name, form, c["group"])] += 1
    assert leaked == {}


def test_넓힌_앎의_부정을_넣어도_D_플러스는_결정적으로_통과한다():
    dplus = [c for c in VECTORS["a01_cases"] if c["group"] == "D+"]
    assert len(GAP46["probe"]["insertions"]) == 5
    for sentence in GAP46["probe"]["insertions"]:
        failed = []
        for c in dplus:
            body, last = c["candidate"].rsplit(". ", 1)
            out = P.evaluate_v8(copy.deepcopy(c["task"]), f"{body}. {sentence} {last}")
            if not out["deterministic_pass"]:
                failed.append((c["case"], out["deterministic_reasons"]))
        assert failed == [], (sentence, failed[:3])


# ---------------------------------------------------------------- 49 독립 검수의 재정정(넓힌 잇기 · 인용 · 예외의 새는 길)
GAP49 = VECTORS["gap49"]
GAP49_WANT = {"close": (False, True, False), "close_pre": (True, True, False), "keep_reject": (None, False, False),
              "keep_pass": (None, None, True), "residual": (None, None, True), "pre_leak": (True, True, None),
              "pre_over": (False, False, None)}


def _inserted(c: dict, sentence: str) -> str:
    """D+ 글의 끝 문장 앞에 문장 하나를 넣는다(49 탐침과 같은 꼴)."""
    parts = P.sentences(c["candidate"])
    return " ".join(s if s.endswith(".") else s + "." for s in parts[:-1]) + " " + sentence + " " + parts[-1] + \
        ("" if parts[-1].endswith(".") else ".")


def test_재정정_벡터는_49_의_합성_80_과_맹검_93_을_싣는다():
    """49 독립 검수의 합성 사례 80 과 맹검 사례 93. 기대(class)는 코드를 고치기 전에 고정한 파일에서 왔다 — close 는 고치기 전(e109139a)
    기각 · 47 판(0f34bd1a) 통과 · 지금 기각, close_pre 는 전부터 샜고 지시가 닫으라 한 것, residual 은 반의어 인용(결정적 통과 ·
    모델 몫). 고치기 전 판보다 새로 통과한 기각 기대 행도, 새로 기각한 통과 기대 행도 없다."""
    rows = GAP49["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 173
    assert Counter((r["set"], r["class"]) for r in rows) == {
        ("synth80", "keep_reject"): 41, ("synth80", "close"): 19, ("synth80", "keep_pass"): 14, ("synth80", "close_pre"): 4,
        ("synth80", "residual"): 2, ("blind93", "close"): 34, ("blind93", "keep_reject"): 21, ("blind93", "keep_pass"): 20,
        ("blind93", "pre_leak"): 11, ("blind93", "pre_over"): 6, ("blind93", "residual"): 1}
    for r in rows:
        want = GAP49_WANT[r["class"]]
        got = (r["before"]["pass"], r["at_0f34bd1a"]["pass"], r["after"]["pass"])
        assert all(w is None or w is g for w, g in zip(want, got)), (r["id"], r["class"], got)
        if r["class"] != "residual":
            assert not (r["intent"] == "reject" and not r["before"]["pass"] and r["after"]["pass"]), r["id"]
        assert not (r["intent"] == "pass" and r["before"]["pass"] and not r["after"]["pass"]), r["id"]
    assert GAP49["residual"]["ids"] == ["N04", "N09", "F6-33"]
    assert {r["id"] for r in rows if r["class"] == "residual"} == set(GAP49["residual"]["ids"])
    for name, digest in GAP49["source"]["files"].items():   # 탐침 원본이 있는 트리에서는 사례 파일의 바이트와 같다
        path = ROOT / GAP49["source"]["probe_dir"] / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, name


@pytest.mark.parametrize("row", GAP49["rows"], ids=lambda r: r["id"])
def test_재정정_벡터(row):
    out = P.evaluate_v8(copy.deepcopy(row["task"]), row["text"])
    assert out["deterministic_pass"] is row["after"]["pass"]
    assert {k: v for k, v in out["deterministic_codes"].items() if v} == row["after"]["codes"]


def test_재정정_삽입_탐침은_새는_꼴을_닫고_받을_꼴을_통과시킨다():
    """49 §2-2 의 넣은 문장(이름 없이 D+ 50 글에) — 새던 꼴은 결정적 통과 0, 받아야 할 꼴(O01 ~ O05)은 50 통과 · v8 만의 새 기각 0.
    반의어 인용(L17)은 남은 위험으로 통과, 전부터 샌 '…불합격으로 단정할 수 없다'(L20)는 닫지 않았다(47 §7)."""
    dplus = [c for c in VECTORS["a01_cases"] if c["group"] == "D+"]
    assert len(GAP49["insert"]) == 25
    for e in GAP49["insert"]:
        det_pass = v8_new = 0
        for c in dplus:
            text = _inserted(c, e["sentence"])
            if P.evaluate_v8(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                det_pass += 1
            elif P.evaluate_v7(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                v8_new += 1
        assert {"det_pass": det_pass, "v8_new_reject": v8_new} == e["after"], e["name"]
        if e["kind"] == "OK":
            assert (det_pass, v8_new) == (50, 0), e["name"]
        else:
            assert det_pass == (50 if ("residual" in e or "open" in e) else 0), e["name"]
    assert [e["name"][:3] for e in GAP49["insert"] if "residual" in e or "open" in e] == ["L17", "L20"]


def test_재정정_맹검_삽입은_고치기_전보다_더_통과하지도_덜_통과하지도_않는다():
    dplus = [c for c in VECTORS["a01_cases"] if c["group"] == "D+"]
    assert len(GAP49["blind_insert"]) == 65
    for e in GAP49["blind_insert"]:
        det_pass = sum(bool(P.evaluate_v8(copy.deepcopy(c["task"]), _inserted(c, e["sentence"]))["deterministic_pass"])
                       for c in dplus)
        assert det_pass == e["after"], e["name"]
        if e["kind"] == "OK":
            assert det_pass >= e["before"], e["name"]
        elif e["name"] not in ("F5-06", "F6-33"):              # 판정 4 의 통과 · 반의어 인용
            assert det_pass <= e["before"], e["name"]


def test_주어진_으로_시작하는_한계_마디를_이어_붙여도_음성은_결정적으로_기각된다():
    """'…지만 · …(으)나 · …는데' 바로 뒤에 '주어진 정보로는 …' · '주어진 주석만으로는 …'(J1 · J2)을 이은 음성 — 꼴 여덟 모두 0/50.
    47 판은 '주어' 를 보조 용언으로 이어 22 · 14 · 17/50 을 통과시켰다(49 C-1)."""
    src, clauses = VECTORS["know_neg_bypass"], GAP49["evade_clauses"]
    assert set(clauses) == {"J1", "J2"} and all(c.startswith("주어진") for c in clauses.values())
    neg = [c for c in VECTORS["a01_cases"] if c["group"] in ("ADV1", "ADV2")]
    leaked = Counter()
    for name, plain in clauses.items():
        clause = [plain, plain[:-1] + "지만"]
        for form in src["forms"]:
            for c in neg:
                text = _bypass(c["candidate"], form, clause, src["joined"])
                if P.evaluate_v8(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                    leaked[(name, form, c["group"])] += 1
    assert leaked == {}
    assert all(v["after"] == 0 for v in GAP49["evade"].values()) and len(GAP49["evade"]) == 54


def test_보조_용언_주다는_아어_뒤에서만_잇고_주어진은_나눈다():
    assert P._clauses_v8("결함들이 서로 붙어 있어 주어진 주석만으로는 크기를 알 수 없다.") == [
        "결함들이 서로 붙어 있어", "주어진 주석만으로는 크기를 알 수 없다."]
    assert P._clauses_v8("결함은 허용 범위 안이지만 주어진 정보로는 깊이를 알 수 없다.") == [
        "결함은 허용 범위 안이지만", "주어진 정보로는 깊이를 알 수 없다."]
    assert P._clauses_v8("주석은 결함의 길이를 알려 주지 않는다.") == ["주석은 결함의 길이를 알려 주지 않는다."]
    for nxt in ("주지", "주고", "주는", "주었다", "주어"):
        assert P._auxiliary_v8("알려", nxt) and P._auxiliary_v8("보여", nxt), nxt
    for nxt in ("주어진", "주어지지", "주어져", "주석은", "주로", "주변의"):    # 머리가 같은 흔한 낱말은 잇지 않는다
        assert not P._auxiliary_v8("알려", nxt), nxt
    assert not P._auxiliary_v8("있지만", "주지") and P._auxiliary_v8("붙어", "있다")
    assert not P._auxiliary_v8("있지만", "있다")            # 이음 어미 뒤의 '있' 은 새 마디다(50 P-1)


def test_볼_은_낱말_머리이고_대상이_있을_때만_앎의_부정이다():
    for word in ("찾아볼", "알아볼", "살펴볼"):
        assert not P._knowledge_negated_v8(f"허용 기준을 넘는 결함은 {word} 수 없다."), word
    assert P._knowledge_end_v8("결함의 깊이는 볼 수 없다.") is not None
    assert P._knowledge_end_v8("기공 두 개가 서로 이어져 있는지는 볼 수 없다.") is not None     # 관계 판독 불가(판정 4)
    for clause in ("습기로 생긴 결함은 볼 수 없다.", "이 결함은 불합격으로 볼 수 없다.", "흔히 볼 수 없는 커다란 결함이다.",
                   "결함들은 서로 붙어 있기에 경계를 볼 수 없다."):
        assert P._knowledge_end_v8(clause) is None, clause
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "허용 기준을 넘는 결함은 찾아볼 수 없다. " + LIMIT) == ["v8_verdict"]
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "결함의 깊이는 볼 수 없다. " + LIMIT) == []


@pytest.mark.parametrize("sentence,codes", [
    ("이 결함들이 용접 속도 탓이라고는 볼 수 없다.", ["v8_cause"]),          # 토씨 붙은 인용도 인용 마디다
    ("결함이 길다고까지 볼 수는 없다.", ["v8_size_word"]),
    ("이 결과를 합격이라고만 볼 수는 없다.", ["v8_cause", "v8_verdict"]),
    ("결함의 윤곽이 뚜렷하다고는 볼 수 없다.", ["v8_attribute"]),
    ("이 결함들이 용접 속도 탓이라고는 단정할 수 없다.", ["v8_cause"]),       # 전부터 샌 꼴(L19)도 닫힌다
    ("결함들이 서로 이어져 있다고는 볼 수 없다.", []),                        # 관계 판독 불가는 그대로 뺀다
    ("두 결함이 서로 안 이어져 있다고 볼 수는 없다.", ["v8_relation"]),        # 짧은 부정 '안' — 이중 부정은 단정
    ("두 결함이 서로 이어져 있지 않다고는 볼 수 없다.", ["v8_relation"]),
    ("결함들이 서로 이어진 것이 아닌 따로 떨어진 것이라고 볼 수는 없다.", ["v8_relation"]),
    ("결함들이 이어져 있다고 볼 수 없을 만큼 서로 떨어져 있다.", ["v8_comparison", "v8_relation"]),   # 술어 뒤 서술
    ("결함들이 서로 이어져 있다고 볼 수 없는 것은 아니다.", ["v8_relation"]),
    ("이어져 있다고 볼 수 없는 결함들이 일직선으로 늘어서 있다.", ["v8_relation"]),
    ("한 줄로 늘어선 결함들이 서로 이어져 있다고 볼 수는 없다.", ["v8_relation"]),    # 논항을 꾸미는 관계는 전제
    ("같은 높이의 칸에 있다는 것만으로 하나로 이어진 결함이라고 볼 수는 없다.", []),  # 술어 몫의 관계는 뺀다
    ("결함들이 서로 떨어져 있다고 볼 수는 없다.", []),                        # 반의어 인용 — 남은 위험(모델 몫)
])
def test_인용_마디는_끝과_부정과_뒤_서술과_논항을_본다(sentence, codes):
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == codes


def test_인용_끝_인용_부정_논항의_좁은_꼴():
    assert P._clauses_v8("이 결함들이 용접 속도 탓이라고는 볼 수 없다.") == ["이 결함들이 용접 속도 탓이라고는", "볼 수 없다."]
    assert P._quoted_unknowable_v8("결함이 길다고까지", "볼 수는 없다.")
    assert not P._quoted_unknowable_v8("결함들이 이어져 있다고", "볼 수 없을 만큼 서로 떨어져 있다.")
    assert not P.V8_QUOTE_NEGATION.search("결함이 위쪽 칸 안에 있다고")           # '안에' 는 짧은 부정이 아니다
    assert P._quote_arguments_v8("한 줄로 늘어선 기공 2개가 서로 이어져 있다고") == "한 줄로 늘어선 기공 2개가"
    assert P._quote_arguments_v8("같은 줄에 있다는 것만으로 결함이 이어져 있다고") == "같은 줄에 있다는 것만으로 결함이"
    assert P._quote_arguments_v8("같은 높이의 칸에 있다는 것만으로 하나로 이어진 결함이라고") == ""
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "결함이 길다고는 한다. " + LIMIT) == ["v8_size_word"]


@pytest.mark.parametrize("sentence,codes", [
    ("결함들은 하나로 이어져 있다고밖에 볼 수 없다.", ["v8_relation"]),      # 배타 한정 = 단정
    ("결함들은 한 줄로 늘어선 것으로밖에 볼 수 없다.", ["v8_relation"]),
    ("주석은 결함이 크다는 것 말고는 알려 주지 않는다.", ["v8_size_word"]),
    ("결함이 크다는 것 말고는 알 수 없다.", ["v8_size_word"]),
    ("결함의 깊이는 영상 밖에서 알 수 없다.", []),                            # '밖에서' 는 배타 한정이 아니다
    ("두 결함이 서로 이어져 있는지 알 수 없을 만큼 가깝다.", ["v8_comparison", "v8_relation"]),   # 술어 뒤 서술
    ("결함들이 서로 이어져 있는지 알 수 없는 채로 위 왼쪽 칸에 모여 있다.", ["v8_relation"]),
    ("주석이 알려 주지 않는 커다란 결함이 표시되어 있다.", ["v8_size_word"]),
    ("결함의 크기는 알 수 없는 상태다.", []),                                 # 술어 뒤에 표지가 없으면 그대로
    ("결함의 크기는 알 수 없기 때문에 판정은 내리지 않는다.", []),
    ("허용 기준을 넘는 결함은 영상에 나타나 있지 않다.", ["v8_verdict"]),      # 결함이 주어면 부재
    ("결함의 모양은 주석에 나타나 있지 않다.", []),
    ("결함은 넓게 퍼져 주어진 정보로는 경계를 알 수 없다.", ["v8_size_word"]),
])
def test_앎의_부정은_배타_한정_술어_뒤_서술_부재_주어를_뺀다(sentence, codes):
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == codes


@pytest.mark.parametrize("sentence,codes", [
    ("균열의 양 끝은 위 왼쪽과 위 오른쪽에 있다.", ["v8_attribute"]),         # 이름 뒤 '끝' 은 속성(47 §2 G)
    ("균열의 오른쪽 끝은 날카롭게 갈라져 있다.", ["v8_attribute"]),
    ("균열 1개는 양 끝이 모두 위쪽 오른쪽 칸 안에 있다.", ["v8_attribute"]),
    ("기공 2개가 이룬 줄의 끝은 위쪽 왼쪽 칸에 있다.", ["v8_attribute"]),
    ("균열은 위쪽 높이의 양 끝 칸에 있다.", []),                               # 자리 '끝'
    ("결함은 오른쪽 끝에 있다.", []),
    ("영상 오른쪽 끝에 결함이 있다.", []),
])
def test_자리_끝은_결함의_끝을_지우지_않는다(sentence, codes):
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == codes


# ---------------------------------------------------------------- 50 독립 검수의 '너무 넓음' 정정(이음 어미 뒤 '있' · '다고만' · 명제 배타 한정)
GAP50 = VECTORS["gap50"]


def _right(row: dict, value: dict) -> bool:
    """행의 의도(intent)와 한 판의 값이 맞는가 — 넣은 문장은 D+ 50 의 결정적 통과 수(기각 의도 0 · 통과 의도 50), 합성은 통과 여부."""
    if row["sec"] == "insert":
        return value["det_pass"] == (0 if row["intent"] == "reject" else 50)
    return value["pass"] is (row["intent"] == "pass")


def test_50_벡터는_넣은_문장_42_합성_28_우회_24_칸을_싣는다():
    """50 독립 검수의 '너무 넓음' 탐침. 기대(class)는 코드를 고치기 전에 고정한 파일에서 왔다 — close 는 155f3e5f 에서 새던 것을
    기각(전부터 샌 것은 before 열), keep_pass · keep_reject 는 의도대로(E1 ~ E3 은 155f3e5f 의 지나친 기각을 되돌림), pre_leak ·
    pre_over 는 e109139a 부터 어긋났고 고치지 않음, residual 은 e109139a 대비 새 어긋남이지만 고치지 않기로 한 것. 고치기 전 판보다
    나빠진 행은 residual 밖에 없다."""
    rows = GAP50["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 70
    assert Counter((r["sec"], r["class"]) for r in rows) == {
        ("insert", "keep_reject"): 16, ("insert", "keep_pass"): 10, ("insert", "close"): 6, ("insert", "residual"): 4,
        ("insert", "pre_leak"): 3, ("insert", "pre_over"): 3, ("synth", "keep_pass"): 9, ("synth", "keep_reject"): 9,
        ("synth", "pre_over"): 4, ("synth", "close"): 3, ("synth", "pre_leak"): 2, ("synth", "residual"): 1}
    for r in rows:
        before, at, after = (_right(r, r[t]) for t in ("before", "at_155f3e5f", "after"))
        if r["class"] == "close":
            assert not at and after, r["id"]
        elif r["class"] in ("keep_pass", "keep_reject"):
            assert after, r["id"]
        elif r["class"] in ("pre_leak", "pre_over"):
            assert not before and not after and r["after"] == r["at_155f3e5f"], r["id"]
        else:
            assert r["class"] == "residual" and before and not after and r["after"] == r["at_155f3e5f"], r["id"]
        assert r["class"] == "residual" or not (before and not after), r["id"]
        assert all(r["after"][k] == v for k, v in r["want"].items()), r["id"]
    assert set(GAP50["residual"]["ids"]) == {r["id"] for r in rows if r["class"] == "residual"} == {"A5", "Q9", "Q10", "N8", "B6"}
    for name, digest in GAP50["source"]["files"].items():   # 탐침 원본이 있는 트리에서는 사례 파일의 바이트와 같다
        path = ROOT / GAP50["source"]["probe_dir"] / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("row", [r for r in GAP50["rows"] if r["sec"] == "synth"], ids=lambda r: r["id"])
def test_50_합성_벡터(row):
    out = P.evaluate_v8(copy.deepcopy(row["task"]), row["text"])
    assert out["deterministic_pass"] is row["after"]["pass"]
    assert {k: v for k, v in out["deterministic_codes"].items() if v} == row["after"]["codes"]


def test_50_넣은_문장과_이웃_꼴의_D_플러스_결정적_통과():
    """넣은 문장 42 와 고친 자리마다의 이웃 꼴 — D+ 50 글에 넣은 결정적 통과 수 · v8 만의 새 기각 수가 벡터의 after 와 같다. 이웃 꼴에는
    이번 고침이 새로 어긋나게 한 것이 없다(남은 것은 e109139a 부터 어긋났거나 155f3e5f 가 덤으로 막았던 꼴이 고치기 전 값으로 돌아간 것)."""
    dplus = [c for c in VECTORS["a01_cases"] if c["group"] == "D+"]
    rows = [r for r in GAP50["rows"] if r["sec"] == "insert"] + [n for n in GAP50["neighbors"] if n["sec"] == "insert"]
    assert len(rows) == 42 + 34
    for r in rows:
        det_pass = v8_new = 0
        for c in dplus:
            text = _inserted(c, r["sentence"])
            if P.evaluate_v8(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                det_pass += 1
            elif P.evaluate_v7(copy.deepcopy(c["task"]), text)["deterministic_pass"]:
                v8_new += 1
        assert {"det_pass": det_pass, "v8_new_reject": v8_new} == r["after"], r["id"]
    for n in GAP50["neighbors"]:
        before, at, after = (_right(n, n[t]) if n["sec"] != "evade" else n[t] == 0 for t in ("before", "at_155f3e5f", "after"))
        assert after or not (before and at), n["id"]                 # 새 어긋남 없음
        assert n["verdict"] == ("맞음" if after else "다시 열림(e109139a 와 같음)" if at else "전부터 어긋남"), n["id"]
    for n in GAP50["neighbors"]:
        if n["sec"] == "synth":
            out = P.evaluate_v8(copy.deepcopy(n["task"]), n["text"])
            assert out["deterministic_pass"] is n["after"]["pass"], n["id"]


def test_50_이음_어미_뒤에_있는_으로_시작하는_한계_마디를_이어도_음성은_결정적으로_기각된다():
    """'…지만 · …는데 · …(으)며 · …기 때문에' 바로 뒤에 '있는 정보로는 · 있는 그대로의 영상으로는 · 있는 주석만으로는 …'(V1 ~ V3)을
    이은 음성 — 고치기 전부터 22 · 14 · 17/50 이 샜다(50 P-1). 남는 것은 '(으)나' 의 2(모음 줄기 + '나', '나타나' 와 낱말 사전 없이
    못 가름)와 '고' 의 10('-고 있' 은 진행의 보조 용언이라 끊지 않는다) — 둘 다 고치기 전과 같거나 적다."""
    src, clauses = VECTORS["know_neg_bypass"], GAP50["evade_clauses"]
    assert set(clauses) == {"V1", "V2", "V3", "V4"} and all(c.startswith("있는") for k, c in clauses.items() if k != "V4")
    neg = [c for c in VECTORS["a01_cases"] if c["group"] in ("ADV1", "ADV2")]
    got = {}
    for name, plain in clauses.items():
        clause = [plain, plain[:-1] + "지만"]
        for form in src["forms"]:
            got[f"{name} | {form}"] = sum(bool(P.evaluate_v8(copy.deepcopy(c["task"]), _bypass(
                c["candidate"], form, clause, src["joined"]))["deterministic_pass"]) for c in neg)
    assert got == {k: v["after"] for k, v in GAP50["evade_bypass"].items()}
    assert {k for k, v in got.items() if v} == {f"V{i} | {f}" for i in (1, 2, 3) for f in ("(으)나", "고")}
    assert all(v["after"] <= v["before"] for v in GAP50["evade_bypass"].values())
    assert len(GAP50["evade"]) == 24 and all(v["after"] == v["want"] for v in GAP50["evade"].values())


def test_있_은_이음_어미_뒤에서만_끊고_보조_용언은_잇는다():
    for word in ("있지만", "안이나", "있는데", "안이며", "안이므로", "안이라서", "붙어서", "안이면서", "안이니", "때문에", "안이거나",
                 "안이다가", "안이면", "붙어도", "반면", "대신", "탓에"):
        assert not P._auxiliary_v8(word, "있는"), word
    for word in ("나타나", "이어져", "붙어", "모여", "맞닿아", "적혀", "늘어서", "이어지고"):   # 머리가 같은 보조 용언은 그대로 잇는다
        assert P._auxiliary_v8(word, "있는지"), word
    assert P._clauses_v8("결함들이 서로 붙어 있지만 있는 정보로는 깊이를 알 수 없다.") == [
        "결함들이 서로 붙어 있지만", "있는 정보로는 깊이를 알 수 없다."]
    for s in ("결함의 깊이는 영상에 나타나 있지 않다.", "결함들이 한 줄로 늘어서 있는지는 알 수 없다.",
              "결함들이 서로 이어지고 있는지는 알 수 없다."):
        assert P._clauses_v8(s) == [s]
        assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), s + " " + LIMIT) == []


def test_다고만_은_인용_면제가_없고_다고는_다고까지_다고도_는_그대로다():
    for end in ("다고", "다고는", "다고까지", "다고도"):
        assert P._quoted_unknowable_v8(f"결함들이 서로 이어져 있{end}", "볼 수는 없다."), end
    assert not P._quoted_unknowable_v8("결함들이 서로 이어져 있다고만", "볼 수는 없다.")
    assert P._clauses_v8("결함들이 서로 이어져 있다고만 볼 수는 없다.") == ["결함들이 서로 이어져 있다고만", "볼 수는 없다."]
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "결함들이 서로 이어져 있다고만 볼 수는 없다. " + LIMIT) == ["v8_relation"]
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), "결함들이 서로 이어져 있다고는 볼 수 없다. " + LIMIT) == []


@pytest.mark.parametrize("sentence,codes", [
    ("결함이 크다는 것 외에 다른 정보는 알 수 없다.", ["v8_size_word"]),         # 명제 + '외에 다른 …' 도 배타 한정
    ("주석은 결함들이 서로 붙어 있다는 것 외에 다른 것은 알려 주지 않는다.", ["v8_relation"]),
    ("결함은 허용 범위 안이라는 것 외에는 알 수 없다.", ["v8_verdict"]),
    ("결함이 크다는 것 이외에는 알 수 없다.", ["v8_size_word"]),
    ("이 결함은 불합격이라고밖에 볼 수 없다.", ["v8_verdict"]),
    ("결함들은 한 줄로 늘어선 것으로밖에 볼 수 없다.", ["v8_relation"]),
    ("결함의 깊이는 주석에 적힌 것 외에는 알 수 없다.", []),                     # 이름씨에 건 한정은 한계 서술
    ("결함의 모양은 주석에 적힌 것 말고는 알 수 없다.", []),
    ("결함의 길이는 주석에 나온 것 외에 다른 정보는 알 수 없다.", []),
])
def test_배타_한정은_명제에만_건다(sentence, codes):
    assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), sentence + " " + LIMIT) == codes


# ---------------------------------------------------------------- 봉인 집합 B 뒤 — 주석 밖 시각 속성 · 원인 암시의 닫힌 목록 넓히기
GAP52 = VECTORS["gap52"]
#: 넓히기 전의 앞 갈래 — 기존 낱말의 정규식 · 문맥 조건은 바꾸지 않고 새 낱말만 덧붙인다.
OLD_ATTRIBUTE_EXTRA = r"(?<![가-힣])(?:밝게|어둡게|어둡|뚜렷|또렷|흐릿|희미|꺾|구부러|휘어|휜|톱니|매끈|거칠|울퉁)"
OLD_CAUSE = r"때문|원인|인해|인한|탓|기인|비롯|영향|말미암|관련|유발|초래|야기|생겼|생긴|생겨"
OLD_ATTR_NOUN_HEAD = (r"(?:모양|형태|형상|생김새|크기|길이|너비|폭|깊이|두께|굵기|넓이|면적|밝기|명암|농도|색상|색깔|색|윤곽|테두리"
                      r"|방향|기울기|각도|간격|끝부분|끝|치수")
OLD_ATTR_NOUN_TAIL = r"(?=[\s,·ㆍ/]|과|와|이나|나|및|은|는|이|가|을|를|도|만|에|의|$)"


def _gap52_task() -> dict:
    counts: dict = {}
    for n, _ in GAP52["base"]["pos"]:
        counts[n] = counts.get(n, 0) + 1
    defs = [{"name": n, "where": w} for n, w in GAP52["base"]["pos"]]
    return {"kind": "image_report", "evidence": {"defect_counts_by_name": counts, "defect_positions": defs,
            "row_class": "defect", "size_mm_available": False},
            "locked_target": {"clauses": [], "defects": defs, "verdict": None},
            "prompt": "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.", "source_ids": ["synthetic-annotation"]}


def test_52_기존_낱말의_매칭은_그대로이고_새_낱말만_덧붙인다():
    """넓히기 전 목록은 앞 갈래에 글자 그대로 남는다(자리 '끝' 가리기 · 앎의 부정 · 인용 · 고정 문장 면제도 그대로). 넓힌 낱말은 모두
    코드에 있고 낱말 머리에서만 읽는다 — 머리가 같은 흔한 낱말, '주석에 의해 · 의하면', 한계 문장의 '(으)로' 는 잡지 않는다."""
    from corpus.production import judge_grounded_v7 as canonical
    cause = P.V8_CATEGORY_RULES["v8_cause"][1]
    assert P.V8_ATTRIBUTE_EXTRA.pattern.startswith(OLD_ATTRIBUTE_EXTRA + "|")
    assert cause.pattern.startswith(OLD_CAUSE + "|")
    assert canonical._ATTR_NOUN.startswith(OLD_ATTR_NOUN_HEAD + "|") and canonical._ATTR_NOUN.endswith(")" + OLD_ATTR_NOUN_TAIL)
    assert GAP52["lists_sha256"] == "2d14a02aed60d0241b5f586fcf857320de1b880693fc87b49864073e6c1f4e23"
    for w in GAP52["lists"]["attribute"]:                # '선명도' 는 정본 ATTR 의 '선명' 이 잡는다(덧붙임에 겹치지 않게)
        assert P.V8_ATTRIBUTE_EXTRA.search(f"결함은 {w} ") or canonical.ATTR.search(f"결함은 {w} "), w
    for w in GAP52["lists"]["cause"]:
        assert cause.search(f"결함은 {w} "), w
    for w in ("결함이", "결과", "결론", "결국", "검사", "직접", "모두", "지금", "각각", "밝힐", "밝혀", "거리", "하나", "균열", "오른쪽",
              "구역", "가운데", "실치수", "표면", "가장자리", "경계", "그리고", "흰가", "꼴찌", "결정", "수가", "외에는", "용입불량"):
        assert not P.V8_ATTRIBUTE_EXTRA.search(w), w
    new_attr = re.compile(P.V8_ATTRIBUTE_EXTRA.pattern[len(OLD_ATTRIBUTE_EXTRA) + 1:])
    new_cause = re.compile(cause.pattern[len(OLD_CAUSE) + 1:])
    for s in ("주석에 의해 표시된", "주석에의해", "표시에 의한", "주석에 의하면", "자료에 의하면", "확인하여", "결함이", "가운데",
              "주석으로는", "정보가 없어", "관계를"):
        assert not new_cause.search(s), s
    from corpus.production import generation_policy as G
    for s in (*G.LIMITATION_FORMS, *G.NORMAL_ABSENCE_FORMS, *G.NORMAL_LIMITATION_FORMS):   # 생성기 고정 문장에 새 낱말이 없다
        assert not new_attr.search(s) and not new_cause.search(s), s
    for s in ("결함의 질감은 볼 수 없다.", "결함의 그림자는 주석에 적혀 있지 않다.", "결함의 결은 볼 수 없다.",
              "결함이 수축으로 생긴 것인지는 알 수 없다.", "주석에 의해 표시된 결함이 있다.", "결함의 끝은 주석에 적혀 있지 않다."):
        assert P.deterministic_categories_v8(copy.deepcopy(DEFECT), s + " " + LIMIT) == [], s
    for s, code in (("결함의 질감이 보인다.", "v8_attribute"), ("결함은 짙은 결함이다.", "v8_attribute"),
                    ("결함은 점 모양으로 찍혀 있다.", "v8_attribute"), ("결함 안에는 가스가 남았다.", "v8_cause"),
                    ("결함은 작업에 의해 남았다.", "v8_cause"), ("결함은 작업 중에 발생했다.", "v8_cause")):
        assert code in P.deterministic_categories_v8(copy.deepcopy(DEFECT), s + " " + LIMIT), s


def test_52_벡터는_넓힌_낱말마다_받을_꼴과_막을_꼴을_싣는다():
    """기대(intent)는 코드를 고치기 전에 고정한 파일에서 왔다. close 는 고치기 전에 새던 막을 꼴, fixed_over 는 고치기 전 지나친 기각을
    되돌린 받을 꼴, keep_* 는 의도대로, pre_* 는 고치기 전부터 어긋났고 그대로인 것(고치지 않음). 고치기 전보다 나빠진 행은 두 갈래뿐이다
    — clause_rule(앎의 부정 문장이 이음 어미 꼴 뒤에서 나뉘어 새 낱말이 앞 마디에 남는다: 기존 낱말 '크고 · 인해 · 생겨 · 때문에' 와 같은
    마디 규칙)과 presence_noun('모습 · 형체' 는 정본 속성 이름씨에 넣지 않아 '…볼 수 없다' 가 앎의 부정으로 읽히지 않는다). 넓힌 낱말마다
    받을 꼴 셋 이상과 막을 꼴 하나 이상이 있다('결 · 꼴' 은 토씨가 붙은 꼴을 한 낱말로 센다)."""
    rows = GAP52["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 1048
    assert Counter((r["cat"], r["sec"], r["class"]) for r in rows) == {
        ('attribute', 'insert', 'clause_rule'): 15, ('attribute', 'insert', 'close'): 269,
        ('attribute', 'insert', 'keep_pass'): 473, ('attribute', 'insert', 'keep_reject'): 4,
        ('attribute', 'insert', 'pre_over'): 2, ('attribute', 'insert', 'presence_noun'): 2,
        ('cause', 'insert', 'clause_rule'): 21, ('cause', 'insert', 'close'): 60,
        ('cause', 'insert', 'keep_pass'): 167, ('exc_pl2', 'synth', 'close'): 12,
        ('exc_pl2', 'synth', 'fixed_over'): 8, ('exc_pl2', 'synth', 'keep_pass'): 9,
        ('exc_pl2', 'synth', 'keep_reject'): 3, ('exc_pl2', 'synth', 'pre_leak'): 3}
    new_attr = re.compile(P.V8_ATTRIBUTE_EXTRA.pattern[len(OLD_ATTRIBUTE_EXTRA) + 1:])
    new_cause = re.compile(P.V8_CATEGORY_RULES["v8_cause"][1].pattern[len(OLD_CAUSE) + 1:])
    for r in rows:
        if r["sec"] == "insert":
            right = r["after"]["det_pass"] == (50 if r["intent"] == "pass" else 0)
            was = r["before"]["det_pass"] == (50 if r["intent"] == "pass" else 0)
        else:
            right, was = r["after"]["pass"] is (r["intent"] == "pass"), r["before"]["pass"] is (r["intent"] == "pass")
        assert right == (r["class"] in ("close", "fixed_over", "keep_pass", "keep_reject")), r["id"]
        assert was or not right or r["class"] in ("close", "fixed_over"), r["id"]
        if was and not right:                                             # 나빠진 행은 두 갈래뿐
            assert r["class"] in ("clause_rule", "presence_noun") and r["intent"] == "pass", r["id"]
            cs = P._clauses_v8(r["sentence"])
            pat = new_attr if r["cat"] == "attribute" else new_cause
            if r["class"] == "clause_rule":
                assert len(cs) > 1 and any(pat.search(x) for x in cs[:-1]) and not pat.search(cs[-1]), r["id"]
            else:
                assert r["word"] in ("모습", "형체") and r["sentence"].endswith("볼 수 없다."), r["id"]
        if r["class"].startswith("pre_"):
            assert r["after"] == r["before"], r["id"]
    head = {"결이": "결", "결은": "결", "결을": "결", "결도": "결", "꼴이": "꼴", "꼴을": "꼴", "꼴은": "꼴", "꼴의": "꼴"}
    for cat in ("attribute", "cause"):
        for w in GAP52["lists"][cat]:
            mine = [r for r in rows if r["cat"] == cat and r["word"] == head.get(w, w)]
            assert sum(r["intent"] == "pass" for r in mine) >= 3 and any(r["intent"] == "reject" for r in mine), w


def test_52_합성_바탕의_판단():
    """넣은 문장은 합성 바탕(기공 1 위쪽 왼쪽 · 균열 1 아래쪽 오른쪽) 한 글에서도 벡터의 판단 그대로 — 받을 꼴은 통과, 막을 꼴은
    기각(D+ 50 넣기와 같은 쪽). 부재 예외 나열(synth)은 그 글의 판단 그대로."""
    task = _gap52_task()
    for r in GAP52["rows"]:
        if r["sec"] == "insert":
            out = P.evaluate_v8(copy.deepcopy(task), GAP52["base"]["text"] + " " + r["sentence"] + " " + LIMIT)
            assert {"pass": out["deterministic_pass"], "codes": {k: v for k, v in out["deterministic_codes"].items() if v}} == \
                r["synth"], r["id"]
        else:
            assert P.evaluate_v8(copy.deepcopy(r["task"]), r["text"])["deterministic_pass"] is r["after"]["pass"], r["id"]
