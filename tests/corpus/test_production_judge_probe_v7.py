"""CPU plumbing only: invented cases and injected backend, never GPU evidence."""

import hashlib
import json
from copy import deepcopy

import pytest

from corpus.production import infer as I
from corpus.production import judge_cases as C
from corpus.production import judge_identity as J
from corpus.production import judge_policy as P
from corpus.production import judge_probe as K
from corpus.production import quality as Q
from data import label_map


def _spec(family):
    return {"model_id": "local/" + family, "revision": "a" * 40, "family": family,
            "local_files_only": True, "quantization": "bf16", "batch_size": 16,
            "max_new_tokens": 160, "max_input_tokens": 4096, "seed": 1,
            "wall_seconds": 1800, "enable_thinking": False}


class FakeV7:
    """All generated outputs are synthetic, including the JSON-stop label."""
    requests = []
    reject_dplus = 0
    accept_negative = False
    finish_reason = "json_closed"
    fail_at = None
    calls = 0
    reason = "합성 시험 문장이다."

    def __init__(self, cfg, stage):
        self.identity = {"backend": "fake", "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"],
                         "quantization": cfg["quantization"], "tokenizer_sha256": "c" * 64}
        self.last_metrics = {"peak_reserved_bytes": 0, "peak_allocated_bytes": 0}
        self.enabled = False

    def enable_json_object_stopping(self):
        self.enabled = True

    def generate_batch(self, prompts, system, deadline):
        assert self.enabled
        FakeV7.calls += 1
        if FakeV7.calls == FakeV7.fail_at:
            raise I.StageTimeLimit()
        outputs = []
        for prompt in prompts:
            request = json.loads(prompt)
            assert request["row_class"] == "defect", "normal row reached backend"
            FakeV7.requests.append(deepcopy(request))
            group, index = request["candidate"].split(":")
            accept = group == "D+" and int(index) >= FakeV7.reject_dplus
            if group == "ADV2" and FakeV7.accept_negative:
                accept = True
            value = dict.fromkeys(K._V7_BOOL_KEYS, True)
            value.update(has_limitation=accept, reason=FakeV7.reason)
            outputs.append({"text": json.dumps(value, ensure_ascii=False),
                            "finish_reason": FakeV7.finish_reason, "input_tokens": 20, "output_tokens": 30})
        return outputs

    def close(self):
        pass


@pytest.fixture
def v7(tmp_path, monkeypatch):
    isolated = tmp_path / "isolated"
    (isolated / ".git").mkdir(parents=True)
    monkeypatch.setattr(K, "_repository_root", lambda: isolated)
    names = label_map.load_label_map().defect_types
    label = names["porosity"].name_ko
    sets = {"N+": "judge_nplus_g1", "D+": "judge_dplus_g1",
            "ADV1": "judge_adv_negative_g1", "ADV2": "judge_adv_negative_g2"}
    paths, pins = {}, {}
    for group, count in K.V7_GROUP_COUNTS.items():
        normal = group in ("N+", "ADV1")
        source = {"set": sets[group], "row_kind": "defect_free" if normal else "defect",
                  "cases": [{"id": f"invented-{group}-{i}", "text": f"{group}:{i}",
                             "expected": "accept" if group in ("N+", "D+") else "reject",
                             "fields": {"defect_counts_by_name": {} if normal else {label: 1},
                                        "defect_positions": {} if normal else {label: ["가운데"]}}}
                            for i in range(count)]}
        raw = json.dumps(source, ensure_ascii=False).encode("utf-8")
        path = tmp_path / (group.replace("+", "plus") + ".json")
        path.write_bytes(raw)
        paths[group], pins[group] = path, hashlib.sha256(raw).hexdigest()
    value = C.build_v7_cases(paths, expected_source_sha256=pins)
    monkeypatch.setattr(K, "V7_SOURCE_SHA256", pins)

    def evaluate(task, text, *, generation_contract=True):
        # Explicitly invented policy plumbing; actual grounded policy has separate
        # D-vector equivalence tests. Normals satisfy the strict contract by
        # default; keep defect full-contract diagnostic failures observable.
        negative = ["invented_negative"] if text.startswith("ADV1:") else []
        codes = {"negative": negative, "grounded": [], "attribute": [],
                 "contract": ["old_positive_contract"] if generation_contract and text.startswith("D+:") else []}
        reasons = sorted({code for values in codes.values() for code in values})
        normal = P.row_class(task) == "defect_free"
        return {"row_class": P.row_class(task), "model_applicable": not normal,
                "model_applicability": "not_applicable" if normal else "applicable",
                "deterministic_codes": codes, "deterministic_reasons": reasons,
                "deterministic_pass": not reasons, "flags": [],
                "generation_contract_applied": generation_contract}

    monkeypatch.setattr(P, "evaluate_v7", evaluate)
    monkeypatch.setattr(P, "render_judge_v7", lambda task, generated: json.dumps(
        {"row_class": P.row_class(task), "candidate": generated["text"]}, ensure_ascii=False))
    # Real B parser, intentionally synthetic policy identity. Never model approval.
    monkeypatch.setattr(J, "require_v7_parser", lambda: Q.parse_judge_v7)
    monkeypatch.setattr(J, "_policy_v7", lambda: (P, "synthetic-test-only", {"synthetic_test_only": True}))
    FakeV7.requests = []
    FakeV7.reject_dplus = FakeV7.calls = 0
    FakeV7.accept_negative = False
    FakeV7.finish_reason = "json_closed"
    FakeV7.fail_at = None
    FakeV7.reason = "합성 시험 문장이다."
    cases = tmp_path / "v7-cases.json"
    cases.write_bytes(I._canonical(value))
    return cases, {"generation": _spec("qwen"), "judge": _spec("phi")}, tmp_path


def _run(v7, name="one"):
    path, config, root = v7
    return K.run_probe(path, config, root / name, backend_factory=FakeV7)


def _rows(v7, name="one"):
    return [json.loads(line) for line in (v7[2] / name / "outputs.jsonl").read_bytes().splitlines()]


def test_v7_adapter_preserves_sources_and_classifies_exposed_set(v7):
    value = json.loads(v7[0].read_bytes())
    assert value["schema_version"] == 4 and value["protocol"] == K.V7_PROTOCOL
    assert all(c["candidate"] == c["source_case"]["text"] for c in value["cases"])
    assert value["provenance"]["evaluation_scope"]["prior_model_output_exposed"] is True
    assert value["provenance"]["evaluation_scope"]["final_approval_eligible"] is False
    assert len(K._inputs(v7[0].read_bytes())) == 150


def test_v7_normals_have_no_raw_parser_or_model_and_do_not_enter_batch(v7, monkeypatch):
    original = Q.parse_judge_v7
    seen = []

    def parser(task, raw):
        assert P.row_class(task) == "defect"
        seen.append(raw["id"])
        return original(task, raw)

    monkeypatch.setattr(J, "require_v7_parser", lambda: parser)
    before = v7[0].read_bytes()
    result = _run(v7)
    rows = _rows(v7)
    normal = [r for r in rows if r["row_class"] == "defect_free"]
    assert len(normal) == 75 and len(FakeV7.requests) == 75
    assert all(r["raw"] is r["raw_sha256"] is r["model_accept"] is r["parser"] is None for r in normal)
    assert all(r["model_provenance"] == "deterministic_only_no_model" for r in normal)
    assert all("N+" not in cid and "ADV1" not in cid for cid in seen)
    assert result["summary"]["nplus_model_metric"] == "not_applicable"
    assert result["summary"]["groups"]["N+"]["model"]["false_reject"] is None
    assert result["summary"]["thresholds_passed"]
    assert v7[0].read_bytes() == before
    events = [json.loads(x) for x in (v7[2] / "one/backend_batches.jsonl").read_bytes().splitlines()]
    assert sum(len(event["case_ids"]) for event in events) == 75


@pytest.mark.parametrize("rejects,passed", [(0, True), (2, True), (3, False)])
def test_v7_dplus_model_denominator_never_includes_nplus(v7, rejects, passed):
    FakeV7.reject_dplus = rejects
    result = _run(v7)["summary"]
    assert result["dplus_model_denominator"] == 50
    assert result["dplus_model_false_reject"] == rejects
    assert result["thresholds_passed"] is passed


def test_v7_defect_contract_diagnostic_is_not_hidden(v7):
    result = _run(v7)["summary"]
    assert result["positive_full_generation_contract_rejected"] == 50
    assert not result["full_generation_contract_passed"]
    assert result["thresholds_passed"]
    assert result["normal_generation_contract_applied_to_probe_gate"] is True
    assert result["normal_positive_contract_rejected"] == 0
    rule = K.check_rules(v7[0])
    assert rule["positive_full_generation_contract_rejected"] == 50
    assert rule["positive_deterministic_false_reject"] == 0
    assert not rule["model_discrimination_measured"]


def test_v7_normal_contract_failures_block_probe_but_preserve_exposed_regression(v7, monkeypatch):
    evaluate = P.evaluate_v7

    def old_normal(task, text, *, generation_contract=True):
        item = evaluate(task, text, generation_contract=generation_contract)
        if generation_contract and text.startswith("N+:"):
            item["deterministic_codes"]["contract"] = ["normal_not_contract_form"]
            item["deterministic_reasons"].append("normal_not_contract_form")
            item["deterministic_pass"] = False
        return item

    monkeypatch.setattr(P, "evaluate_v7", old_normal)
    result = _run(v7)["summary"]
    assert result["normal_positive_contract_rejected"] == 50
    assert result["positive_deterministic_false_reject"] == 50
    assert not result["thresholds_passed"]
    normals = [r for r in _rows(v7) if r["group"] == "N+"]
    assert all(r["generation_contract_applied_to_probe_gate"] for r in normals)
    assert all(not r["combined_accept"] and r["model_accept"] is None for r in normals)
    rule = K.check_rules(v7[0])
    assert rule["positive_deterministic_false_reject"] == 0
    assert rule["normal_production_positive_rejected"] == 50
    assert rule["normal_production_negative_false_accept"] == 0
    assert not rule["approval_eligible"]


def test_v7_normal_negative_false_accept_is_deterministic_and_fatal(v7, monkeypatch):
    evaluate = P.evaluate_v7

    def allow(task, text, **kw):
        item = evaluate(task, text, **kw)
        if text == "ADV1:0":
            item.update(deterministic_codes=dict.fromkeys(("negative", "grounded", "attribute", "contract"), []),
                        deterministic_reasons=[], deterministic_pass=True)
        return item

    monkeypatch.setattr(P, "evaluate_v7", allow)
    summary = _run(v7)["summary"]
    assert summary["negative_combined_denominator"] == 50
    assert summary["negative_combined_false_accept"] == 1
    assert not summary["thresholds_passed"]


@pytest.mark.parametrize("group", ["N+", "D+"])
def test_v7_positive_deterministic_reject_cannot_shrink_dplus_model_calls(v7, monkeypatch, group):
    evaluate = P.evaluate_v7

    def reject(task, text, **kw):
        item = evaluate(task, text, **kw)
        if text == group + ":0":
            item["deterministic_reasons"].append("injected_positive_rejection")
            item["deterministic_pass"] = False
        return item

    monkeypatch.setattr(P, "evaluate_v7", reject)
    summary = _run(v7)["summary"]
    assert summary["positive_deterministic_false_reject"] == 1
    assert summary["groups"]["D+"]["model"]["applicable"] == 50
    assert not summary["thresholds_passed"]


def test_v7_two_fake_calls_cannot_be_real_determinism_or_approval(v7):
    _run(v7, "one")
    _run(v7, "two")
    report = K.compare_probes(v7[2] / "one", v7[2] / "two")
    assert report["raw_text_deterministic"] and report["raw_row_hashes_equal"]
    assert not report["gpu_determinism_measured"] and not report["gpu_model_verified"]
    assert report["approval_scope"] == "adjustment_operational_only"
    assert not report["final_blinded_approval"]
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K.seal_probe_approval(v7[2] / "one", v7[2] / "two", v7[2] / "approval")
    assert not (v7[2] / "approval").exists()


def test_v7_long_reason_is_flag_not_rejection(v7):
    FakeV7.reason = "합성" * 25
    result = _run(v7)["summary"]
    assert result["thresholds_passed"] and result["model_flags"]["reason_long"] == 75


def test_v7_defect_negative_model_accept_is_counted_in_fixed_negative_denominator(v7):
    FakeV7.accept_negative = True
    result = _run(v7)["summary"]
    assert result["negative_combined_denominator"] == 50
    assert result["negative_combined_false_accept"] == 25
    assert not result["thresholds_passed"]


def test_v7_parser_missing_refuses_before_output_or_backend(v7, monkeypatch):
    def missing():
        raise I.StageRefused("parser_missing", "deliberately disconnected")

    monkeypatch.setattr(J, "require_v7_parser", missing)
    with pytest.raises(I.StageRefused, match="parser_missing"):
        _run(v7)
    assert not FakeV7.requests and not (v7[2] / "one").exists()


def test_v7_truncated_response_cannot_pass(v7):
    FakeV7.finish_reason = "length"
    result = _run(v7)["summary"]
    assert result["parser_error_rows"] == result["truncated_rows"] == 75
    assert not result["thresholds_passed"]


def test_v7_token_limit_refuses_before_backend_or_output(v7):
    v7[1]["judge"]["max_new_tokens"] = 159
    with pytest.raises(I.StageRefused, match="max_new_tokens=160"):
        _run(v7)
    assert not FakeV7.requests and not (v7[2] / "one").exists()


def test_v7_partial_failure_preserves_all_normal_rows_without_approval(v7):
    FakeV7.fail_at = 2
    result = _run(v7)
    rows = _rows(v7)
    assert result["status"] == "time_limit" and result["exit_code"] == 3
    assert len(rows) == 75 + 16
    assert sum(r["raw"] is None for r in rows) == 75
    assert not result["summary"]["thresholds_passed"]
    K._verify_probe(v7[2] / "one")


@pytest.mark.parametrize("change", ["candidate", "expectation", "version", "scope"])
def test_v7_adapter_rejects_source_or_scope_tamper(v7, change):
    data = json.loads(v7[0].read_bytes())
    if change == "candidate":
        data["cases"][0]["candidate"] = "changed"
    elif change == "expectation":
        data["cases"][0]["expected_model_accept"] = False
    elif change == "version":
        data["cases"][0]["protocol"] = K.V6_PROTOCOL
    else:
        data["provenance"]["evaluation_scope"]["prior_model_output_exposed"] = False
    with pytest.raises(I.StageRefused):
        K._inputs(I._canonical(data))


def test_v7_normal_cannot_be_given_a_fabricated_model_response(v7):
    case = K._inputs(v7[0].read_bytes())[0]
    with pytest.raises(I.StageRefused, match="normal_model_call_forbidden"):
        K._analysis(case, {"text": "{}", "finish_reason": "eos"})
