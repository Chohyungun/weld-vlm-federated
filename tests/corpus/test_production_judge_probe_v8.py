"""v8 판정 계약의 배선 — 등록 탐침 · 판정 신원 · 정지 계약 · 집계 파서 표 · CLI. CPU 만, 대역 모델, GPU 증거가 아니다."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from corpus.production import infer as I
from corpus.production import judge_cases as C
from corpus.production import judge_identity as J
from corpus.production import judge_policy as P
from corpus.production import judge_probe as K
from corpus.production import judge_stopping as S
from corpus.production import quality as Q
from tests.corpus.test_production_judge_probe_v7 import _registered, _spec, v7  # noqa: F401 -- 같은 합성 고정물


class FakeV8:
    """합성 v8 출력 — attribute · relation 에 있음 · 없음. 정상 행이 오면 실패한다."""
    requests: list = []
    reject_dplus = 0
    accept_negative = False
    finish_reason = "json_closed"

    def __init__(self, cfg, stage):
        self.identity = {"backend": "fake", "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"],
                         "quantization": cfg["quantization"], "tokenizer_sha256": "c" * 64}
        self.last_metrics = {"peak_reserved_bytes": 0, "peak_allocated_bytes": 0}
        self.enabled = False

    def enable_json_object_stopping(self):
        self.enabled = True

    def generate_batch(self, prompts, system, deadline):
        assert self.enabled and system == "synthetic-v8-test-only"
        outputs = []
        for prompt in prompts:
            request = json.loads(prompt)
            assert request["row_class"] == "defect", "normal row reached backend"
            FakeV8.requests.append(deepcopy(request))
            group, index = request["candidate"].split(":")
            accept = (group == "D+" and int(index) >= FakeV8.reject_dplus) or (group == "ADV2" and FakeV8.accept_negative)
            value = {"attribute": "없음" if accept else "있음", "relation": "없음"}
            # A01 에서 본 꼴 — 원문 QA 키를 덧붙인 출력도 v8 파서는 두 키만 읽는다
            text = json.dumps(value, ensure_ascii=False)[:-1] + ', "grounded": true, "language_ok": true, "language_ok": true}'
            outputs.append({"text": text, "finish_reason": FakeV8.finish_reason, "input_tokens": 20, "output_tokens": 18})
        return outputs

    def close(self):
        pass


@pytest.fixture
def v8(v7, monkeypatch):
    """v7 의 합성 고정물 위에 v8 의 정책 · 렌더러 · 파서 자리를 잇는다(실제 v8 파서, 합성 정책 신원)."""
    monkeypatch.setattr(P, "evaluate_v8", lambda task, text, **kw: P.evaluate_v7(task, text, **kw))
    monkeypatch.setattr(P, "render_judge_v8", lambda task, generated: json.dumps(
        {"row_class": P.row_class(task), "candidate": generated["text"]}, ensure_ascii=False))
    monkeypatch.setattr(J, "require_v8_parser", lambda: Q.parse_judge_v8)
    monkeypatch.setattr(J, "_policy_v8", lambda: (P, "synthetic-v8-test-only", {"synthetic_v8_test_only": True}))
    FakeV8.requests = []
    FakeV8.reject_dplus = 0
    FakeV8.accept_negative = False
    FakeV8.finish_reason = "json_closed"
    return v7


def _rows(root, name):
    return [json.loads(line) for line in (root / name / "outputs.jsonl").read_bytes().splitlines()]


def test_v8_프로토콜은_v7_계열로_등록되고_v7_과_섞이지_않는다():
    assert C.V8_A_PROTOCOL == "judge_v8_g150_a_v2" and C.V8_B_PROTOCOL == "judge_v8_g150_b_v2"
    assert C.V7_REGISTERED_SCOPES[C.V8_A_PROTOCOL] == "adjustment_operational_only"
    assert C.V7_REGISTERED_SCOPES[C.V8_B_PROTOCOL] == "final_blinded"
    assert C.v7_registered_evaluation_scope(C.V8_B_PROTOCOL)["final_approval_eligible"] is True
    assert C.v7_registered_evaluation_scope(C.V8_A_PROTOCOL)["final_approval_eligible"] is False
    for protocol in C.V8_PROTOCOLS:
        assert K._is_v7(protocol) and K._is_v8(protocol) and K._parser_schema(protocol) == "judge_structured_v8"
    for protocol in (K.V7_PROTOCOL, C.V7_A_PROTOCOL, C.V7_B_PROTOCOL):
        assert not K._is_v8(protocol) and K._parser_schema(protocol) == "judge_structured_v7"


@pytest.mark.parametrize("protocol", [C.V8_A_PROTOCOL, C.V8_B_PROTOCOL])
def test_등록_v8_탐침은_v8_파서와_두_키로_가르고_정상_행은_부르지_않는다(v8, monkeypatch, protocol):
    monkeypatch.setattr(P, "evaluate_v7", P.evaluate_v7)
    path, expected = _registered(v8, monkeypatch, protocol)
    monkeypatch.setattr(P, "evaluate_v8", lambda task, text, **kw: P.evaluate_v7(task, text, **kw))
    cpu = K.check_rules(path, **expected)
    assert cpu["exit_code"] == cpu["positive_deterministic_false_reject"] == 0
    result = K.run_probe(path, v8[1], v8[2] / "first", backend_factory=FakeV8, **expected)
    summary = result["summary"]
    assert summary["protocol"] == protocol and summary["thresholds_passed"]
    assert summary["normal_model_calls"] == 0 and summary["groups"]["D+"]["model"]["applicable"] == 50
    rows = _rows(v8[2], "first")
    model_rows = [r for r in rows if r["model_applicability"] == "applicable"]
    assert model_rows and all(r["parser"]["parser"] == "parse_judge_v8" for r in model_rows)
    assert all(set(r["derived_verdict"]) == set(K._V8_BOOL_KEYS) for r in model_rows)
    assert all(r["format_normalized"] == ["other_keys_ignored"] for r in model_rows)
    receipt, _, _ = K._verify_probe(v8[2] / "first")
    assert receipt["identity_basis"]["input_basis"]["parser_schema_version"] == "judge_structured_v8"
    assert receipt["identity_basis"]["input_basis"]["policy_identity"] == {"synthetic_v8_test_only": True}


def test_등록_v8_탐침의_음성_거짓_통과와_D_기각을_센다(v8, monkeypatch):
    path, expected = _registered(v8, monkeypatch, C.V8_A_PROTOCOL)
    monkeypatch.setattr(P, "evaluate_v8", lambda task, text, **kw: P.evaluate_v7(task, text, **kw))
    FakeV8.reject_dplus, FakeV8.accept_negative = 3, True
    summary = K.run_probe(path, v8[1], v8[2] / "first", backend_factory=FakeV8, **expected)["summary"]
    assert summary["groups"]["D+"]["model"]["false_reject"] == 3
    assert summary["groups"]["ADV2"]["combined"]["false_accept"] > 0
    assert not summary["thresholds_passed"]


def test_v8_잘린_출력은_통과하지_못한다(v8, monkeypatch):
    path, expected = _registered(v8, monkeypatch, C.V8_A_PROTOCOL)
    monkeypatch.setattr(P, "evaluate_v8", lambda task, text, **kw: P.evaluate_v7(task, text, **kw))
    FakeV8.finish_reason = "length"
    summary = K.run_probe(path, v8[1], v8[2] / "first", backend_factory=FakeV8, **expected)["summary"]
    assert summary["parser_error_rows"] == summary["truncated_rows"] > 0
    assert not summary["thresholds_passed"]
    rows = [r for r in _rows(v8[2], "first") if r["model_applicability"] == "applicable"]
    assert rows and all(r["model_errors"] == ["judge_truncated"] for r in rows)


def test_v7_매니페스트로_v8_프로토콜을_세울_수_없다(v8, monkeypatch):
    path, expected = _registered(v8, monkeypatch, C.V7_A_PROTOCOL)
    relabeled = dict(expected, expected_protocol=C.V8_A_PROTOCOL)
    with pytest.raises((I.StageRefused, ValueError)):
        K.check_rules(path, **relabeled)


def test_판정_신원의_v8_실행_정체와_정지_계약():
    spec = _spec("phi")

    class Fake:
        identity = {"backend": "fake", "model_id": spec["model_id"], "revision": spec["revision"],
                    "family": "phi", "model_type": "phi", "quantization": "bf16", "tokenizer_sha256": "c" * 64}

    v8 = J.judge_runtime_identity(Fake(), spec, policy_version="v8")
    v7 = J.judge_runtime_identity(Fake(), spec, policy_version="v7")
    assert v8["judge_runtime"]["policy_version"] == "v8" and v7["judge_runtime"]["policy_version"] == "v7"
    assert v8["judge_runtime"]["json_stop_protocol"] == S.JSON_STOP_PROTOCOL_V8 == "judge_json_object_v8"
    assert v7["judge_runtime"]["json_stop_protocol"] == S.JSON_STOP_PROTOCOL == "judge_json_object_v7"
    assert v8["judge_runtime"]["decoding"]["json_stop_protocol"] == "judge_json_object_v8"
    J._check_runtime_v7(v8, spec, "v8")
    with pytest.raises(I.StageRefused, match="v8 cannot reuse"):
        J._check_runtime_v7(v7, spec, "v8")
    with pytest.raises(I.StageRefused, match="v7 cannot reuse"):
        J._check_runtime_v7(v8, spec, "v7")
    with pytest.raises(I.StageRefused, match="requires exactly 160"):
        J.judge_runtime_identity(Fake(), {**spec, "max_new_tokens": 200}, policy_version="v8")
    with pytest.raises(I.StageRefused, match="unknown judge policy version"):
        J.judge_runtime_identity(Fake(), spec, policy_version="v9")


def test_판정_신원의_v8_정책_파서_승인_결속():
    policy, system, identity = J._policy_v8()
    assert system == P.JUDGE_SYSTEM_V8 and identity["output_schema"] == "judge_structured_v8"
    assert J.require_v8_parser() is Q.parse_judge_v8 and J.require_v7_parser() is Q.parse_judge_v7
    assert J._CONTRACTS["v8"]["record"] != J._CONTRACTS["v7"]["record"]
    approval = {"receipt_sha256": "a" * 64, "identity_sha256": "b" * 64,
                "identity_basis": {"protocol": C.V7_B_PROTOCOL, "approval_scope": "final_blinded",
                                   "final_blinded_approval": True, "judge_config": {}, "probe_identity_sha256": "c" * 64,
                                   "probe_call_ids": [], "common_implementation_sha256": {},
                                   "input_basis": {"protocol": C.V7_B_PROTOCOL, "policy_identity": identity,
                                                   "cases_file_sha256": "d" * 64,
                                                   "registration": {"expected_protocol": C.V7_B_PROTOCOL,
                                                                    "expected_approval_scope": "final_blinded"}}}}
    with pytest.raises(I.StageRefused, match="v8 production requires"):
        J._approval_binding_v7(approval, {"identity_sha256": "e" * 64}, {}, identity, {}, "v8")
    for key in ("protocol",):
        approval["identity_basis"][key] = C.V8_B_PROTOCOL
    approval["identity_basis"]["input_basis"]["protocol"] = C.V8_B_PROTOCOL
    approval["identity_basis"]["input_basis"]["registration"]["expected_protocol"] = C.V8_B_PROTOCOL
    bound = J._approval_binding_v7(approval, {"identity_sha256": "e" * 64}, {}, identity, {}, "v8")
    assert bound["protocol"] == C.V8_B_PROTOCOL
    other = J._policy_v7()[2]
    with pytest.raises(I.StageRefused, match="approved policy"):
        J._approval_binding_v7(approval, {"identity_sha256": "e" * 64}, {}, other, {}, "v8")


def test_집계는_v8_출처를_v8_파서와_v8_정책으로_읽는다():
    assert Q.JUDGE_PARSER_VERSIONS[Q.JUDGE_V8_SCHEMA_VERSION] == "parse_judge_v8"
    assert Q.V7_FAMILY_CONTRACTS == {"judge_structured_v7": "v7", "judge_structured_v8": "v8"}
    assert Q._declared_contract({"output_schema": "judge_structured_v8"}) == "v8"
    assert Q._declared_contract({"identity_basis": {"output_schema": "judge_structured_v7"}}) == "v7"
    assert Q._declared_contract({"output_schema": "judge_structured_v6"}) is None
    _, identity = Q._resolve_policy_v7(None, "v8")
    assert identity["output_schema"] == "judge_structured_v8"
    _, identity7 = Q._resolve_policy_v7(None, "v7")
    assert identity7["output_schema"] == "judge_structured_v7"
    task = {"kind": "image_report", "locked_target": {"defects": [{"name": "기공"}], "clauses": [], "verdict": None}}
    errors, rejected, verdict, *_ = Q._read_judge(task, {"text": '{"attribute": "있음", "relation": "없음"}',
                                                        "finish_reason": "eos"}, Q.JUDGE_V8_SCHEMA_VERSION)
    assert errors == [] and rejected == ["judge_no_unannotated_attributes_false"]
    assert verdict == {"no_unannotated_attributes": False, "no_unsupported_relation": True}


def test_CLI_는_v8_을_받고_v7_은_그대로다(monkeypatch, tmp_path, capsys):
    from corpus.production import __main__ as M
    from corpus.production import infer
    seen = []
    monkeypatch.setattr(infer, "run_judge", lambda *a, **kw: seen.append(kw) or {"exit_code": 0})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"judge": _spec("phi")}), encoding="utf-8")
    for version in ("v8", "v7", "v6"):
        code = M.main(["judge-separate", "--run", str(tmp_path), "--config", str(config),
                       "--judge-dir", str(tmp_path / "j"), "--policy-version", version])
        assert code == 0
    assert [kw.get("policy_version") for kw in seen] == ["v8", "v7", None]
    with pytest.raises(SystemExit):
        M.main(["judge-separate", "--run", str(tmp_path), "--config", str(config),
                "--judge-dir", str(tmp_path / "j"), "--policy-version", "v9"])
    capsys.readouterr()


@pytest.mark.parametrize("protocol", [C.V7_A_PROTOCOL, C.V7_B_PROTOCOL, C.V8_A_PROTOCOL, C.V8_B_PROTOCOL])
def test_최종_맹검_승인은_등록_스코프를_따른다(v8, monkeypatch, protocol):
    """최종 맹검 승인은 등록 스코프(`V7_REGISTERED_SCOPES` 의 final_blinded — v7 B · v8 B)를 따른다. 한 호출의 요약은 승인이 아니고, B 의
    요약만 '노출된 조정 묶음은 최종 승인을 낼 수 없다' 가 거짓이다. 대역 두 호출의 비교는 verified 가 아니라 승인이 없고, 실제 두 호출로
    본 비교(불 논리 단위 입력)는 B 만 비교 보고 · 승인 근거의 final_blinded_approval 이 참이다. A 는 v7 · v8 모두 거짓이다."""
    from tests.corpus.test_production_judge_probe_v7 import FakeV7, _comparison_only_real_metadata
    path, expected = _registered(v8, monkeypatch, protocol)
    if protocol in C.V8_PROTOCOLS:
        monkeypatch.setattr(P, "evaluate_v8", lambda task, text, **kw: P.evaluate_v7(task, text, **kw))
    final = C.V7_REGISTERED_SCOPES[protocol] == "final_blinded"
    assert final is (protocol in (C.V7_B_PROTOCOL, C.V8_B_PROTOCOL))
    for name in ("first", "second"):
        summary = K.run_probe(path, v8[1], v8[2] / name, backend_factory=FakeV8 if protocol in C.V8_PROTOCOLS else FakeV7,
                              **expected)["summary"]
        assert summary["thresholds_passed"] and summary["final_blinded_approval"] is False
        assert summary["exposed_adjustment_set_cannot_mint_final_approval"] is (not final)
    first, left, sha1 = K._verify_probe(v8[2] / "first")
    second, right, sha2 = K._verify_probe(v8[2] / "second")
    fake = K._compare_verified(first, left, sha1, second, right, sha2)
    assert fake["gpu_model_verified"] is False and fake["final_blinded_approval"] is False
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K._approval_basis(fake, first, second, v8[1]["judge"])
    first, second = map(_comparison_only_real_metadata, (first, second))
    report = K._compare_verified(first, left, sha1, second, right, sha2)
    assert report["gpu_model_verified"] is True and report["approval_scope"] == C.V7_REGISTERED_SCOPES[protocol]
    assert report["final_blinded_approval"] is final
    basis = K._approval_basis(report, first, second, v8[1]["judge"])
    assert basis["final_blinded_approval"] is final and basis["approval_scope"] == expected["expected_approval_scope"]
