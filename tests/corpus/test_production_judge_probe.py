"""CPU-only probe contracts; no actual model, images, production rows or GPU."""

import base64
import hashlib
import json
import shutil
from collections import Counter
from types import SimpleNamespace

import pytest

from corpus.production import infer as I
from corpus.production import judge_policy as P
from corpus.production import judge_probe as K
from corpus.production import judge_cases as C
from corpus.production import judge_fingerprints as F


def spec(family):
    return {"model_id": f"local/{family}", "revision": "a" * 40, "family": family,
            "local_files_only": True, "quantization": "bf16", "batch_size": 50,
            "max_new_tokens": 64, "max_input_tokens": 4096, "seed": 1,
            "wall_seconds": 1800, "enable_thinking": False}


def cases():
    rows = []
    for group in K.GROUPS:
        for index in range(50):
            positive = group in ("N+", "D+")
            expected_det = positive or group == "D-ungrounded"
            task = {"kind": "image_report", "source_ids": ["synthetic"],
                    "prompt": "주석에 따라 설명하라.", "evidence": {},
                    "locked_target": {"defects": [] if group.startswith("N") else [{"name": "기공"}]},
                    "synthetic_expected_deterministic": expected_det}
            rows.append({"id": f"synthetic-{group}-{index}", "group": group, "task": task,
                         "candidate": f"{group} sample {index}",
                         "expected_model_accept": positive,
                         "expected_deterministic_accept": expected_det})
    return {"schema_version": 1, "cases": rows}


class Fake:
    made = 0
    closed = 0
    calls = 0
    failure_at = None
    failure = "time"
    tokenizer = "c" * 64
    reason = "합성 근거에 따른다."
    malformed = False
    length = False
    reject_nplus = 0
    reject_dplus = 0
    structured = False
    extra_false_accept = False
    memory = 0

    def __init__(self, cfg, stage):
        Fake.made += 1
        self.identity = {"backend": "fake", "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"],
                         "quantization": cfg["quantization"], "tokenizer_sha256": Fake.tokenizer}
        self.last_metrics = {"peak_reserved_bytes": Fake.memory, "peak_allocated_bytes": 0}

    def generate_batch(self, prompts, system, deadline):
        Fake.calls += 1
        if Fake.calls == Fake.failure_at:
            if Fake.failure == "time":
                raise I.StageTimeLimit()
            raise RuntimeError("CUDA out of memory")
        result = []
        for prompt in prompts:
            candidate = json.loads(prompt)["candidate"]
            group, _, index = candidate.split()
            accept = group in ("N+", "D+")
            if group == "N+" and int(index) < Fake.reject_nplus:
                accept = False
            if group == "D+" and int(index) < Fake.reject_dplus:
                accept = False
            if group in {"D-missing", "ADV1", "ADV2"} and int(index) == 0 and Fake.extra_false_accept:
                accept = True
            obj = {"grounded": accept, "scope_ok": accept, "language_ok": True, "reason": Fake.reason}
            if Fake.structured:
                if json.loads(prompt)["row_class"] == "defect_free":
                    obj = {"claims_defect": not accept, "asserts_verdict": False,
                           "states_no_marked_defect": accept, "has_limitation": False,
                           "language_ok": True, "reason": Fake.reason}
                else:
                    obj = {"fields_match": accept, "has_limitation": True,
                           "asserts_verdict_dimension_or_cause": False,
                           "language_ok": True, "reason": Fake.reason}
                    if Fake.structured == "v6":
                        del obj["fields_match"]
                        obj.update(names_match=accept, counts_match=accept, positions_match=accept)
            if Fake.malformed:
                obj["extra"] = "strict parser must reject"
            result.append({"text": I._canonical(obj).decode(), "finish_reason": "length" if Fake.length else "eos",
                           "input_tokens": 20, "output_tokens": 15})
        return result

    def close(self):
        Fake.closed += 1


class ProductionLookingFake(Fake):
    """Only Python objects: never evidence of real hardware or actual models."""
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        self.identity["backend"] = "transformers"
        defaults = {"eos_token_id": [2], "pad_token_id": 0,
                    "do_sample": False, "num_beams": 1}
        self.model = SimpleNamespace(generation_config=SimpleNamespace(
            eos_token_id=[2], to_dict=lambda: dict(defaults)))
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                                         chat_template="synthetic template only")


@pytest.fixture(autouse=True)
def reset(tmp_path, monkeypatch):
    # Exercise the real gate checker against an isolated repository. Unit tests
    # must not depend on a concurrent document/merge gate in the working copy;
    # source/runtime hash checks still read the actual implementation files.
    repo = tmp_path / "probe-unit-repository"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(K, "_repository_root", lambda: repo)
    Fake.made = Fake.closed = Fake.calls = 0
    Fake.failure_at, Fake.failure = None, "time"
    Fake.tokenizer, Fake.reason = "c" * 64, "합성 근거에 따른다."
    Fake.malformed = Fake.length = Fake.extra_false_accept = Fake.structured = False
    Fake.reject_nplus = Fake.reject_dplus = Fake.memory = 0


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # This isolates runner/statistical plumbing from policy grammar. Policy gets
    # its own independent fixtures and tests; this is not discrimination evidence.
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        [] if task["synthetic_expected_deterministic"] else ["synthetic_rejected"])
    monkeypatch.setattr(P, "deterministic_flags", lambda task, text: [])
    # Renderer/semantic grammar has separate tests. These contract cases use no
    # real annotation fields and must not be mistaken for policy integration.
    monkeypatch.setattr(P, "render_judge", lambda task, generated: I._canonical(
        {"row_class": P.row_class(task), "candidate": generated["text"]}).decode())
    path = tmp_path / "cases.json"
    path.write_bytes(I._canonical(cases()))
    return path, {"generation": spec("qwen"), "judge": spec("phi")}, tmp_path


def synthetic_v5_parser(task, raw):
    """Test-only structured adapter, not the actual B parser or its validation."""
    normal = not task["locked_target"]["defects"]
    keys = ({"claims_defect", "asserts_verdict", "states_no_marked_defect", "has_limitation",
             "language_ok", "reason"} if normal else
            {"fields_match", "has_limitation", "asserts_verdict_dimension_or_cause", "language_ok", "reason"})
    try:
        obj = json.loads(raw["text"])
        valid = (raw["finish_reason"] == "eos" and isinstance(obj, dict) and set(obj) == keys and
                 all(type(obj[k]) is bool for k in keys - {"reason"}) and isinstance(obj["reason"], str))
    except (KeyError, ValueError, TypeError):
        valid = False
    if not valid:
        return {"errors": ["synthetic_parse_error"], "rejected": [], "verdict": None, "flags": []}
    verdict = {"grounded": obj["states_no_marked_defect"] and not obj["claims_defect"] if normal else obj["fields_match"],
               "scope_ok": not obj["asserts_verdict"] if normal else
               obj["has_limitation"] and not obj["asserts_verdict_dimension_or_cause"],
               "language_ok": obj["language_ok"], "reason": obj["reason"]}
    return {"errors": [], "rejected": [k for k in ("grounded", "scope_ok", "language_ok") if not verdict[k]],
            "verdict": verdict, "flags": ["n5_absent"] if normal and not obj["has_limitation"] else []}


@pytest.fixture
def v5_fixture(tmp_path, monkeypatch):
    # Synthetic original files only: this fixture never reads any G source. Pins
    # are independently computed here and replace the production constants only
    # within this unit test. It proves contracts, not policy discrimination.
    loaded = C.label_map.load_label_map(C.label_map.DEFAULT_LABEL_MAP)
    labels = sorted(value.name_ko for value in loaded.defect_types.values())[:2][::-1]
    assert len(labels) == 2  # Deliberately opposite to canonical object-key order.
    sets = {"N+": "judge_nplus_g1", "D+": "judge_dplus_g1",
            "ADV1": "judge_adv_negative_g1", "ADV2": "judge_adv_negative_g2"}
    paths, pins = {}, {}
    for group, count in K.V5_GROUP_COUNTS.items():
        normal = group in {"N+", "ADV1"}
        kind = "defect_free" if normal else "defect"
        fields = {"defect_counts_by_name": {} if normal else {label: 1 for label in labels},
                  "defect_positions": {} if normal else {label: ["가운데"] for label in labels}}
        original = {"set": sets[group], "row_kind": kind,
                    "cases": [{"id": f"unit-{group}-{i}", "text": f"{group} sample {i}",
                               "expected": "accept" if group in {"N+", "D+"} else "reject",
                               "fields": fields} for i in range(count)]}
        raw = json.dumps(original, ensure_ascii=False).encode("utf-8")
        path = tmp_path / (group.replace("+", "plus") + "-synthetic.json")
        path.write_bytes(raw)
        paths[group], pins[group] = path, hashlib.sha256(raw).hexdigest()
    value = C.build_v5_cases(paths, expected_source_sha256=pins)
    monkeypatch.setattr(K, "V5_SOURCE_SHA256", pins)
    base = P.policy_identity()  # Before installing the intentionally synthetic parser.
    parser_id = {"capabilities": {"legacy_four_keys": True, "judge_structured_v5": True},
                 "closure_sha256": "7" * 64, "test_only": True}
    monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "judge_structured_v5", raising=False)
    monkeypatch.setattr(K.Q, "JUDGE_V5_SCHEMA_VERSION", "judge_structured_v5", raising=False)
    monkeypatch.setattr(K.Q, "parse_judge_v5", synthetic_v5_parser, raising=False)
    monkeypatch.setattr(F, "require_v5_parser", lambda: parser_id, raising=False)
    monkeypatch.setattr(P, "policy_identity", lambda: {**base, "parser_implementation": parser_id})
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text: [])
    monkeypatch.setattr(P, "deterministic_flags", lambda task, text: [])
    monkeypatch.setattr(P, "render_judge", lambda task, generated: I._canonical(
        {"row_class": P.row_class(task), "candidate": generated["text"]}).decode())
    Fake.structured = True
    path = tmp_path / "v5-cases.json"
    path.write_bytes(I._canonical(value))
    return path, {"generation": spec("qwen"), "judge": spec("phi")}, tmp_path


def synthetic_v6_parser(task, raw):
    """Test-only plumbing, not B's parser, normalization or GPU validation."""
    if not task["locked_target"]["defects"]:
        result = synthetic_v5_parser(task, raw)
    else:
        expected = {"names_match", "counts_match", "positions_match", "has_limitation",
                    "asserts_verdict_dimension_or_cause", "language_ok", "reason"}
        try:
            value = json.loads(raw["text"])
            assert isinstance(value, dict) and set(value) == expected
            assert all(type(value[k]) is bool for k in expected - {"reason"})
            fields_match = value.pop("names_match") & value.pop("counts_match") & value.pop("positions_match")
            value["fields_match"] = fields_match
            result = synthetic_v5_parser(task, {**raw, "text": I._canonical(value).decode()})
        except (AssertionError, KeyError, TypeError, ValueError):
            result = {"errors": ["synthetic_parse_error"], "rejected": [], "verdict": None, "flags": []}
    return {**result, "format_normalized": []}


@pytest.fixture
def v6_fixture(v5_fixture, monkeypatch):
    # The four sources are synthetic fixture files. No actual G text is read or
    # used to tune expectations; v6 parser injection is explicitly test-only.
    _, config, temp = v5_fixture
    paths = {group: temp / (group.replace("+", "plus") + "-synthetic.json") for group in K.V6_GROUP_COUNTS}
    pins = dict(K.V5_SOURCE_SHA256)
    value = C.build_v6_cases(paths, expected_source_sha256=pins)
    monkeypatch.setattr(K, "V6_SOURCE_SHA256", pins)
    parser_id = {"selected_schema": "judge_structured_v6", "capabilities": {"judge_structured_v6": True},
                 "closure_sha256": "8" * 64, "test_only": True}
    base = P.policy_identity()
    monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "judge_structured_v6", raising=False)
    monkeypatch.setattr(K.Q, "JUDGE_V6_SCHEMA_VERSION", "judge_structured_v6", raising=False)
    monkeypatch.setattr(K.Q, "parse_judge_v6", synthetic_v6_parser, raising=False)
    monkeypatch.setattr(F, "require_v6_parser", lambda: parser_id, raising=False)
    monkeypatch.setattr(P, "policy_identity", lambda: {**base, "output_schema": "judge_structured_v6",
                                                     "structured_parser_ready": True,
                                                     "parser_implementation": parser_id})
    Fake.structured = "v6"
    path = temp / "v6-cases.json"
    path.write_bytes(I._canonical(value))
    return path, config, temp


def run(fixture, name="first"):
    source, config, temp = fixture
    return K.run_probe(source, config, temp / name, backend_factory=Fake)


def test_v5_canonical_bundle_preserves_original_reverse_label_list_order(v5_fixture):
    data = json.loads(v5_fixture[0].read_bytes())
    source = json.loads(base64.b64decode(data["provenance"]["source_files_base64"]["D+"]))
    original = source["cases"][0]
    adapted = next(case for case in data["cases"] if case["group"] == "D+")
    original_order = list(original["fields"]["defect_counts_by_name"])
    canonical_order = list(adapted["source_case"]["fields"]["defect_counts_by_name"])
    assert len(original_order) == 2 and original_order == list(reversed(canonical_order))
    assert I._digest(original) == I._digest(adapted["source_case"])
    assert [fact["name"] for fact in adapted["task"]["evidence"]["defect_positions"]] == original_order
    assert [fact["name"] for fact in adapted["task"]["locked_target"]["defects"]] == original_order
    assert len(K._inputs(v5_fixture[0].read_bytes())) == 150
    assert K.check_rules(v5_fixture[0])["exit_code"] == 0 and not Fake.made
    # The dictionary serialization order may change, but actual ordered model
    # input lists cannot be silently rebuilt in canonical-key order.
    adapted["task"]["evidence"]["defect_positions"].reverse()
    adapted["task"]["locked_target"]["defects"].reverse()
    with pytest.raises(I.StageRefused, match="probe_task"):
        K._inputs(I._canonical(data))


def test_v5_two_fake_calls_and_portable_source_contract(v5_fixture):
    source = json.loads(v5_fixture[0].read_bytes())
    for name in ("first", "second"):
        summary = run(v5_fixture, name)["summary"]
        assert summary["protocol"] == K.V5_PROTOCOL and summary["planned_cases"] == 150
        assert summary["thresholds"] == K.V5_THRESHOLDS and summary["thresholds_passed"]
        assert summary["model_flags"]["n5_absent"] == 75  # Flags are not normal-row rejection.
        for group in ("ADV1", "ADV2"):
            assert summary["groups"][group]["deterministic"]["false_accept"] is None
            assert summary["groups"][group]["deterministic"]["model_pending"] == 25
            assert summary["groups"][group]["combined"]["false_accept"] == 0
    report = K.compare_probes(v5_fixture[2] / "first", v5_fixture[2] / "second")
    assert report["case_count"] == 150 and not report["approval_eligible"]
    assert report["raw_text_deterministic"] and report["thresholds_passed"]
    assert not report["gpu_model_verified"] and report["exit_code"] == 2
    saved = json.loads((v5_fixture[2] / "first/PROBE.json").read_bytes())["identity_basis"]
    assert saved["input_basis"]["protocol"] == K.V5_PROTOCOL
    assert saved["input_basis"]["case_provenance_sha256"] == I._digest(source["provenance"])
    assert saved["input_basis"]["parser_schema_version"] == "judge_structured_v5"
    assert saved["metrics"]["generation_batches"] == 3


@pytest.mark.parametrize("group", ["N+", "D+"])
@pytest.mark.parametrize("rejects,passed", [(2, True), (3, False)])
def test_v5_positive_model_floor_five_percent(v5_fixture, group, rejects, passed):
    if group == "N+":
        Fake.reject_nplus = rejects
    else:
        Fake.reject_dplus = rejects
    summary = run(v5_fixture)["summary"]
    assert summary["groups"][group]["model"]["false_reject"] == rejects
    assert summary["groups"][group]["deterministic"]["false_reject"] == 0
    assert summary["thresholds_passed"] is passed


@pytest.mark.parametrize("group", ["N+", "D+"])
def test_v5_positive_deterministic_fr_one_vetoes_combined_tolerance(v5_fixture, monkeypatch, group):
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        ["synthetic_negative_error"] if text == f"{group} sample 0" else [])
    summary = run(v5_fixture)["summary"]
    assert summary["groups"][group]["deterministic"]["false_reject"] == 1
    assert summary["groups"][group]["combined"]["false_reject"] == 1 <= 2
    assert not summary["thresholds_passed"]
    report = K.check_rules(v5_fixture[0])
    assert report["exit_code"] == 2 and not report["positive_deterministic_gate_passed"]


@pytest.mark.parametrize("rules_reject,passed", [(True, True), (False, False)])
def test_v5_negative_combined_not_model_only_gate(v5_fixture, monkeypatch, rules_reject, passed):
    Fake.extra_false_accept = True
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        ["synthetic_negative"] if rules_reject and text.startswith("ADV") else [])
    summary = run(v5_fixture)["summary"]
    for group in ("ADV1", "ADV2"):
        assert summary["groups"][group]["model"]["false_accept"] == 1
        assert summary["groups"][group]["combined"]["false_accept"] == (0 if rules_reject else 1)
        assert summary["groups"][group]["deterministic"]["false_accept"] is None
    assert summary["thresholds_passed"] is passed


def test_v5_cpu_negatives_are_unapproved_model_pending(v5_fixture):
    raw = v5_fixture[0].read_bytes()
    cases_value = json.loads(raw)["cases"]
    result = K.check_rules(v5_fixture[0])
    assert result["rows_recorded"] == len(result["rows"]) == 150 and result["exit_code"] == 0
    for case, row in zip(cases_value, result["rows"]):
        assert row["case_sha256"] == I._digest(case) and row["id"] == case["id"]
        assert row["expected_deterministic_accept"] is (True if case["group"] in {"N+", "D+"} else None)
        assert row["deterministic_status"] == ("not_rejected" if case["group"] in {"N+", "D+"} else "model_pending")
    assert result["groups"]["ADV1"]["model_pending"] == result["groups"]["ADV2"]["model_pending"] == 25
    assert not result["approval_eligible"] and not result["gpu_model_verified"]
    assert not result["model_discrimination_measured"] and not Fake.made


@pytest.mark.parametrize("kind", ["count", "duplicate", "group", "expectation", "source_sha", "source_bytes",
                                  "source_member", "source_index", "metadata", "task", "label_names",
                                  "adapter", "thresholds", "base64", "task_bool", "task_count_bool",
                                  "source_count_bool", "threshold_bool"])
def test_v5_input_binding_refuses_before_model_or_output(v5_fixture, kind):
    path, _, temp = v5_fixture
    data = json.loads(path.read_bytes())
    case, provenance = data["cases"][0], data["provenance"]
    if kind == "count":
        data["cases"].pop()
    elif kind == "duplicate":
        data["cases"][1] = case
    elif kind == "group":
        case["group"] = "ADV1"
    elif kind == "expectation":
        data["cases"][-1]["expected_deterministic_accept"] = True
    elif kind == "source_sha":
        provenance["source_sha256"]["N+"] = "f" * 64
    elif kind == "source_bytes":
        raw = base64.b64decode(provenance["source_files_base64"]["N+"])
        provenance["source_files_base64"]["N+"] = base64.b64encode(raw + b"\n").decode()
    elif kind == "source_member":
        case["source_case"]["text"] = case["candidate"] = "changed but internally rehashed"
        case["source_case_sha256"] = I._digest(case["source_case"])
    elif kind == "source_index":
        case["source_index"] = 1
    elif kind == "metadata":
        provenance["source_metadata"]["N+"]["changed"] = True
    elif kind == "task":
        case["task"]["prompt"] = "leaked expected answer"
    elif kind == "task_bool":
        case["task"]["evidence"]["size_mm_available"] = 0
    elif kind in {"task_count_bool", "source_count_bool"}:
        selected = data["cases"][50]
        obj = (selected["task"]["evidence"] if kind == "task_count_bool" else selected["source_case"]["fields"])
        key = next(iter(obj["defect_counts_by_name"]))
        obj["defect_counts_by_name"][key] = True
        if kind == "source_count_bool":
            selected["source_case_sha256"] = I._digest(selected["source_case"])
    elif kind == "label_names":
        provenance["label_names"]["invented"] = "wrong"
    elif kind == "adapter":
        provenance["adapter_sha256"] = "f" * 64
    elif kind == "thresholds":
        data["thresholds"] = K._THRESHOLDS
    elif kind == "threshold_bool":
        data["thresholds"] = {**K.V5_THRESHOLDS, "positive_deterministic_false_reject": False}
    else:
        provenance["source_files_base64"]["N+"] = "not base64"
    path.write_bytes(I._canonical(data))
    with pytest.raises(I.StageRefused):
        run(v5_fixture)
    assert not Fake.made and not (temp / "first").exists()


@pytest.mark.parametrize("missing", ["callable", "marker", "policy_schema", "closure", "unbound"])
def test_v5_missing_parser_refuses_before_model_output_but_cpu_check_works(v5_fixture, monkeypatch, missing):
    if missing == "callable":
        monkeypatch.delattr(K.Q, "parse_judge_v5")
    elif missing == "marker":
        monkeypatch.setattr(K.Q, "JUDGE_V5_SCHEMA_VERSION", "old")
    elif missing == "policy_schema":
        monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "old")
    elif missing == "closure":
        monkeypatch.setattr(F, "require_v5_parser", lambda: {"capabilities": {"judge_structured_v5": False}})
    else:
        monkeypatch.setattr(P, "policy_identity", lambda: {"parser_implementation": {"wrong": True}})
    assert K.check_rules(v5_fixture[0])["exit_code"] == 0
    with pytest.raises(I.StageRefused, match="parser_missing"):
        run(v5_fixture)
    assert not Fake.made and not (v5_fixture[2] / "first").exists()


@pytest.mark.parametrize("malformed,length", [(True, False), (False, True)])
def test_v5_raw_parser_error_and_length_do_not_pass(v5_fixture, malformed, length):
    Fake.malformed, Fake.length = malformed, length
    summary = run(v5_fixture)["summary"]
    assert summary["parser_error_rows"] == 150
    assert summary["truncated_rows"] == (150 if length else 0)
    assert not summary["thresholds_passed"]


def test_v5_parser_contract_failure_keeps_raw_batch_without_receipt(v5_fixture, monkeypatch):
    monkeypatch.setattr(K.Q, "parse_judge_v5", lambda task, row: {"verdict": True})
    with pytest.raises(I.StageRefused, match="parser_contract"):
        run(v5_fixture)
    folder = v5_fixture[2] / "first"
    events = [json.loads(line) for line in (folder / "backend_batches.jsonl").read_bytes().splitlines()]
    assert len(events[0]["output"]) == 50 and not (folder / "PROBE.json").exists()
    assert Fake.closed == 1


def test_v5_does_not_fall_back_to_legacy_parser(v5_fixture, monkeypatch):
    def forbidden(row):
        pytest.fail("v5 must not use legacy parser")
    monkeypatch.setattr(K.Q, "_judge", forbidden)
    assert run(v5_fixture)["summary"]["thresholds_passed"]


def test_legacy_default_backend_shape_never_grants_new_approval(fixture, monkeypatch):
    monkeypatch.setattr(I, "TransformersBackend", ProductionLookingFake)
    source, config, temp = fixture
    for name in ("first", "second"):
        K.run_probe(source, config, temp / name)
    report = K.compare_probes(temp / "first", temp / "second")
    assert report["real_backend_pair"] and report["thresholds_passed"]
    assert report["protocol"] == K.LEGACY_PROTOCOL and not report["approval_eligible"]
    assert not report["gpu_model_verified"] and report["exit_code"] == 2
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K.seal_probe_approval(temp / "first", temp / "second", temp / "approval")
    assert not (temp / "approval").exists()


def test_two_fake_calls_are_not_gpu_verification(fixture):
    first = run(fixture)
    second = run(fixture, "second")
    assert Fake.made == Fake.closed == 2
    assert first["call_id"] != second["call_id"]
    assert first["summary"]["thresholds_passed"] and second["summary"]["thresholds_passed"]
    report = K.compare_probes(fixture[2] / "first", fixture[2] / "second")
    assert report["raw_text_deterministic"] and report["raw_row_hashes_equal"]
    assert not report["real_backend_pair"] and not report["gpu_model_verified"]
    assert report["exit_code"] == 2
    assert not report["ready_for_training"] and report["population_error_bound"] is None
    assert report["independent_sample_count"] is None
    assert report["first_summary"]["groups"]["D-ungrounded"]["deterministic"]["false_accept"] == 0


def test_transformers_looking_injected_backend_is_not_gpu_verification(fixture):
    # A synthetic backend can expose production-shaped metadata without ever
    # loading transformers or a model. The injected-call flag must still veto it.
    source, config, temp = fixture
    for name in ("first", "second"):
        result = K.run_probe(source, config, temp / name, backend_factory=ProductionLookingFake)
        assert result["summary"]["thresholds_passed"]
        basis = json.loads((temp / name / "PROBE.json").read_text(encoding="utf-8"))["identity_basis"]
        assert basis["runtime_model"]["backend"] == "transformers"
        assert basis["runtime_model"]["judge_runtime"]["test_only"] is False
        assert basis["call"]["backend_injected"] is True
    report = K.compare_probes(temp / "first", temp / "second")
    assert Fake.made == Fake.closed == 2
    assert report["complete"] and report["thresholds_passed"]
    assert report["raw_text_deterministic"] and report["raw_row_hashes_equal"]
    assert not report["real_backend_pair"] and not report["gpu_model_verified"]
    assert report["exit_code"] == 2 and not report["ready_for_training"]


def test_existing_output_always_refused(fixture):
    run(fixture)
    made = Fake.made
    with pytest.raises(I.StageRefused, match="probe_output_exists"):
        run(fixture)
    assert Fake.made == made
    (fixture[2] / "empty").mkdir()
    with pytest.raises(I.StageRefused, match="probe_output_exists"):
        run(fixture, "empty")


def test_copied_or_same_results_not_second_invocation(fixture):
    run(fixture)
    first, copied = fixture[2] / "first", fixture[2] / "copied"
    shutil.copytree(first, copied)
    with pytest.raises(I.StageRefused, match="probe_same_call"):
        K.compare_probes(first, copied)
    with pytest.raises(I.StageRefused, match="probe_same_call"):
        K.compare_probes(first, first)


def test_runtime_mismatch_refused(fixture):
    run(fixture)
    Fake.tokenizer = "d" * 64
    run(fixture, "second")
    with pytest.raises(I.StageRefused, match="probe_pair_identity"):
        K.compare_probes(fixture[2] / "first", fixture[2] / "second")


def test_raw_text_nondeterminism_is_reported(fixture):
    run(fixture)
    Fake.reason = "다른 이유 문장이다."
    run(fixture, "second")
    report = K.compare_probes(fixture[2] / "first", fixture[2] / "second")
    assert report["thresholds_passed"]
    assert not report["raw_text_deterministic"] and not report["raw_row_hashes_equal"]
    assert not report["gpu_model_verified"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_group", "wrong_expected", "wrong_class"])
def test_invalid_case_contract_before_load_or_output(fixture, kind):
    source, _, temp = fixture
    value = cases()
    if kind == "missing":
        value["cases"].pop()
    elif kind == "duplicate":
        value["cases"][1]["id"] = value["cases"][0]["id"]
    elif kind == "wrong_group":
        value["cases"][0]["group"] = "N-new"
    elif kind == "wrong_class":
        value["cases"][0]["task"]["locked_target"]["defects"] = [{"name": "기공"}]
    else:
        value["cases"][0]["expected_model_accept"] = False
    source.write_bytes(I._canonical(value))
    with pytest.raises(I.StageRefused):
        run(fixture)
    assert not Fake.made and not (temp / "first").exists()


@pytest.mark.parametrize("malformed,length", [(True, False), (False, True)])
def test_strict_parser_errors_and_truncation_never_pass(fixture, malformed, length):
    Fake.malformed, Fake.length = malformed, length
    result = run(fixture)
    assert result["summary"]["complete"]
    assert result["summary"]["parser_error_rows"] == 500
    assert result["summary"]["truncated_rows"] == (500 if length else 0)
    assert not result["summary"]["thresholds_passed"]


@pytest.mark.parametrize("rejected,passed", [(2, True), (3, False)])
def test_positive_combined_false_reject_integer_threshold(fixture, rejected, passed):
    Fake.reject_nplus = rejected
    summary = run(fixture)["summary"]
    assert summary["groups"]["N+"]["combined"]["false_reject"] == rejected
    assert summary["thresholds_passed"] is passed


def test_nplus_deterministic_false_reject_vetoes_combined_tolerance(fixture, monkeypatch):
    previous = P.deterministic_reasons
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        ["synthetic_false_reject"] if text == "N+ sample 0" else previous(task, text))
    for name in ("first", "second"):
        summary = run(fixture, name)["summary"]
        group = summary["groups"]["N+"]
        assert summary["complete"] and summary["parser_error_rows"] == 0
        assert summary["truncated_rows"] == 0
        assert group["model"]["false_reject"] == 0
        assert group["deterministic"]["false_reject"] == 1
        assert group["combined"]["false_reject"] == 1 <= 2
        assert not summary["thresholds_passed"]
    report = K.compare_probes(fixture[2] / "first", fixture[2] / "second")
    assert report["complete"] and report["raw_text_deterministic"]
    assert not report["thresholds_passed"] and not report["gpu_model_verified"]
    assert report["exit_code"] == 2
    rules = K.check_rules(fixture[0])
    assert not rules["nplus_deterministic_gate_passed"] and rules["exit_code"] == 2


def test_negative_model_false_accept_fails_even_when_rules_reject(fixture):
    Fake.extra_false_accept = True
    summary = run(fixture)["summary"]
    assert summary["groups"]["D-missing"]["model"]["false_accept"] == 1
    assert summary["groups"]["D-missing"]["combined"]["false_accept"] == 0
    assert not summary["thresholds_passed"]


def test_deterministic_false_accept_fails_even_if_model_rejects(fixture, monkeypatch):
    previous = P.deterministic_reasons
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        [] if text == "N-number sample 0" else previous(task, text))
    summary = run(fixture)["summary"]
    group = summary["groups"]["N-number"]
    assert group["deterministic"]["false_accept"] == 1
    assert group["model"]["false_accept"] == group["combined"]["false_accept"] == 0
    assert not summary["thresholds_passed"]
    assert K.check_rules(fixture[0])["exit_code"] == 2


@pytest.mark.parametrize("failure,exit_code", [("time", 3), ("oom", 5)])
def test_partial_probe_has_receipt_but_no_pass_or_resume(fixture, failure, exit_code):
    Fake.failure_at, Fake.failure = 2, failure
    result = run(fixture)
    assert result["exit_code"] == exit_code and result["summary"]["completed_cases"] == 50
    assert not result["summary"]["complete"] and not result["summary"]["thresholds_passed"]
    assert Fake.closed == 1
    assert (fixture[2] / "first/PROBE.json").is_file()


def test_memory_budget_uses_judge_config(fixture):
    fixture[1]["judge"]["max_memory_reserved_bytes"] = 100
    Fake.memory = 101
    result = run(fixture)
    assert result["status"] == "memory_limit" and result["exit_code"] == 5
    assert not result["summary"]["complete"]


def test_artifact_hash_tamper_is_refused(fixture):
    run(fixture)
    path = fixture[2] / "first/outputs.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(I.StageRefused, match="probe_artifacts"):
        K.compare_probes(path.parent, path.parent)


@pytest.mark.parametrize("missing", ["all", "judge_probe.py", "judge_identity.py", "infer.py"])
def test_missing_implementation_fingerprint_refused_even_with_rehashed_receipt(fixture, missing):
    run(fixture)
    run(fixture, "second")
    path = fixture[2] / "first/PROBE.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    input_basis = receipt["identity_basis"]["input_basis"]
    if missing == "all":
        del input_basis["implementation_sha256"]
    else:
        del input_basis["implementation_sha256"][missing]
    # Rehashing an internally consistent receipt cannot waive the external
    # current-implementation comparison. No production artifact is touched.
    receipt["identity_sha256"] = I._digest(receipt["identity_basis"])
    path.write_bytes(I._canonical(receipt) + b"\n")
    with pytest.raises(I.StageRefused, match="probe_identity_changed"):
        K.compare_probes(path.parent, fixture[2] / "second")


@pytest.mark.parametrize("gate", ["CTO_GATE_OPEN", "CT0_GATE_OPEN", ".git/CTO_GATE_OPEN",
                                  ".git/CT0_GATE_OPEN", ".git/MERGE_HEAD"])
def test_local_gate_refuses_before_backend(fixture, monkeypatch, gate):
    repo = fixture[2] / "local-repo"
    (repo / ".git").mkdir(parents=True)
    lock = repo / gate
    lock.write_text("not-finished", encoding="utf-8")
    monkeypatch.setattr(K, "_repository_root", lambda: repo)
    with pytest.raises(I.StageRefused, match="repository_gate"):
        run(fixture)
    assert not Fake.made
    assert lock.read_text(encoding="utf-8") == "not-finished"


def test_no_git_artifact_runtime_is_explicit(fixture, monkeypatch):
    monkeypatch.setattr(K, "_repository_root", lambda: fixture[2] / "artifact-root")
    run(fixture)
    call = json.loads((fixture[2] / "first/CALL.json").read_text(encoding="utf-8"))
    assert call["environment"] == "artifact-runtime" and call["backend_injected"] is True


@pytest.mark.parametrize("mismatches", [False, True])
def test_rules_only_check_never_loads_model_or_claims_model_validation(fixture, monkeypatch, mismatches):
    # Deliberately synthetic labels exercise receipt plumbing, not policy grammar.
    source_bytes = fixture[0].read_bytes()
    inputs = json.loads(source_bytes)["cases"]
    original_reasons = P.deterministic_reasons
    overrides = ({"N-number sample 0": [], "N+ sample 0": ["synthetic_false_reject"]}
                 if mismatches else {})
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        overrides[text] if text in overrides else original_reasons(task, text))
    monkeypatch.setattr(P, "deterministic_flags", lambda task, text:
                        ["synthetic_flag", "synthetic_normal"] if not task["locked_target"]["defects"]
                        else ["synthetic_flag"])
    report = K.check_rules(fixture[0])
    assert report["nplus_deterministic_gate_passed"] is (not mismatches)
    assert report["deterministic_expectations_match"] is (not mismatches)
    assert report["exit_code"] == (2 if mismatches else 0)
    assert report["cases"] == report["rows_recorded"] == len(report["rows"]) == 500
    expected_rows = []
    for case in inputs:
        reasons = overrides.get(case["candidate"],
                                [] if case["expected_deterministic_accept"] else ["synthetic_rejected"])
        flags = (["synthetic_flag", "synthetic_normal"] if case["group"].startswith("N")
                 else ["synthetic_flag"])
        expected_rows.append({"id": case["id"], "group": case["group"],
                              "expected_deterministic_accept": case["expected_deterministic_accept"],
                              "deterministic_accept": not reasons, "reasons": reasons, "flags": flags,
                              "case_sha256": hashlib.sha256(I._canonical(case)).hexdigest()})
    assert report["rows"] == expected_rows  # All 500 IDs, order, decisions and hashes.
    for group in K.GROUPS:
        group_rows = [row for row in expected_rows if row["group"] == group]
        assert report["groups"][group] == {
            "cases": len(group_rows),
            "false_accept": sum(row["deterministic_accept"] and not row["expected_deterministic_accept"]
                                for row in group_rows),
            "false_reject": sum(row["expected_deterministic_accept"] and not row["deterministic_accept"]
                                for row in group_rows),
            "expected_deterministic_accept": sum(row["expected_deterministic_accept"] for row in group_rows),
        }
    assert report["deterministic_flags"] == dict(Counter(flag for row in expected_rows for flag in row["flags"]))
    assert report["deterministic_flags"] == {"synthetic_flag": 500, "synthetic_normal": 250}
    assert fixture[0].read_bytes() == source_bytes
    assert report["groups"]["D-ungrounded"]["expected_deterministic_accept"] == 50
    assert not report["model_discrimination_measured"]
    assert not report["combined_discrimination_measured"] and not report["determinism_measured"]
    assert not report["gpu_model_verified"] and not Fake.made


def test_changed_policy_refuses_old_receipt(fixture, monkeypatch):
    run(fixture)
    original = P.policy_identity
    monkeypatch.setattr(P, "policy_identity", lambda: {**original(), "rules_sha256": "e" * 64})
    with pytest.raises(I.StageRefused, match="probe_identity_changed"):
        K.compare_probes(fixture[2] / "first", fixture[2] / "first")


def test_unexpected_analysis_failure_keeps_raw_backend_output(fixture, monkeypatch):
    def failed_parser(case, row):
        raise I.StageRefused("synthetic_parser_error", "injected test failure")
    monkeypatch.setattr(K, "_analysis", failed_parser)
    with pytest.raises(I.StageRefused, match="synthetic_parser_error"):
        run(fixture)
    output = fixture[2] / "first"
    assert not (output / "PROBE.json").exists()
    events = [json.loads(line) for line in (output / "backend_batches.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["event"] == "batch_returned" and len(events[0]["output"]) == 50
    assert events[-1]["event"] == "call_failed"
    assert Fake.closed == 1


@pytest.mark.parametrize("gate", ["CTO_GATE_OPEN", "CT0_GATE_OPEN", "MERGE_HEAD"])
def test_worktree_common_git_gate_refuses_before_load(fixture, monkeypatch, gate):
    root = fixture[2] / "worktree"
    git = fixture[2] / "shared-git/worktrees/unit"
    root.mkdir()
    git.mkdir(parents=True)
    (root / ".git").write_text(f"gitdir: {git.as_posix()}", encoding="utf-8")
    (git / "commondir").write_text("../..", encoding="utf-8")
    lock = git.parent.parent / gate
    lock.write_text("owner", encoding="utf-8")
    monkeypatch.setattr(K, "_repository_root", lambda: root)
    with pytest.raises(I.StageRefused, match="repository_gate"):
        run(fixture)
    assert not Fake.made and lock.read_text(encoding="utf-8") == "owner"


@pytest.mark.parametrize("kind", ["policy", "code"])
def test_probe_read_rechecks_policy_and_code_after_analysis(fixture, monkeypatch, kind):
    run(fixture)
    original_analysis = K._analysis
    calls = []
    def changed_after_analysis(case, raw):
        result = original_analysis(case, raw)
        calls.append(case["id"])
        if len(calls) == 500:
            if kind == "policy":
                identity = P.policy_identity()
                monkeypatch.setattr(P, "policy_identity", lambda: {**identity, "changed": True})
            else:
                pins = K._code_pins()
                monkeypatch.setattr(K, "_code_pins", lambda: {**pins, "infer.py": "e" * 64})
        return result
    monkeypatch.setattr(K, "_analysis", changed_after_analysis)
    with pytest.raises(I.StageRefused, match="probe_identity_changed"):
        K.compare_probes(fixture[2] / "first", fixture[2] / "first")


@pytest.mark.parametrize("factory", [Fake, ProductionLookingFake])
def test_fake_or_injected_pair_cannot_emit_approval(fixture, factory):
    source, config, temp = fixture
    for name in ("first", "second"):
        K.run_probe(source, config, temp / name, backend_factory=factory)
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K.seal_probe_approval(temp / "first", temp / "second", temp / "approval")
    assert not (temp / "approval").exists()


@pytest.mark.parametrize("kind", ["partial", "threshold"])
def test_incomplete_or_failed_emulated_pair_never_emits_approval(fixture, monkeypatch, kind):
    monkeypatch.setattr(I, "TransformersBackend", ProductionLookingFake)
    source, config, temp = fixture
    if kind == "partial":
        Fake.failure_at = 2
    else:
        Fake.reject_nplus = 3
    for name in ("first", "second"):
        Fake.calls = 0
        K.run_probe(source, config, temp / name)
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K.seal_probe_approval(temp / "first", temp / "second", temp / "approval")
    assert not (temp / "approval").exists()


@pytest.fixture
def emulated_approval(v6_fixture, monkeypatch):
    # Emulate the default-backend branch solely to exercise the portable receipt
    # protocol. These monkeypatched artifacts are not real GPU approval evidence.
    monkeypatch.setattr(I, "TransformersBackend", ProductionLookingFake)
    source, config, temp = v6_fixture
    for name in ("first", "second"):
        K.run_probe(source, config, temp / name)
    result = K.seal_probe_approval(temp / "first", temp / "second", temp / "approval")
    return temp, result["approval_sha256"]


def test_approval_requires_external_pin_and_is_portable(emulated_approval):
    temp, sha = emulated_approval
    for bad in (None, "", "true", "d" * 64):
        with pytest.raises(I.StageRefused, match="probe_approval_pin"):
            K.read_probe_approval(temp / "approval", bad)
    expected = K.read_probe_approval(temp / "approval", sha)
    assert len(set(expected["identity_basis"]["probe_call_ids"])) == 2
    assert set(expected["identity_basis"]["common_implementation_sha256"]) == {"infer.py", "judge_identity.py"}
    copied = K.copy_probe_approval(temp / "approval", temp / "moved", sha)
    assert copied == expected
    # Original probe locations are not part of the portable identity.
    (temp / "first").rename(temp / "old-first")
    (temp / "second").rename(temp / "old-second")
    assert K.read_probe_approval(temp / "moved", sha) == expected
    with pytest.raises(I.StageRefused, match="probe_approval_exists"):
        K.copy_probe_approval(temp / "approval", temp / "moved", sha)


def test_approval_output_cannot_change_original_probe_tree(fixture):
    run(fixture)
    run(fixture, "second")
    first = fixture[2] / "first"
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(I.StageRefused, match="probe_approval_overlap"):
        K.seal_probe_approval(first, fixture[2] / "second", first / "approval")
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before


@pytest.mark.parametrize("kind", ["flag", "source_row", "same_call"])
def test_portable_approval_revalidates_evidence_not_true_flags(emulated_approval, kind):
    temp, sha = emulated_approval
    folder = temp / "approval"
    if kind == "flag":
        path = folder / "APPROVAL.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["identity_basis"]["comparison"]["raw_text_deterministic"] = False
        receipt["identity_sha256"] = I._digest(receipt["identity_basis"])
        path.write_bytes(I._canonical(receipt) + b"\n")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    elif kind == "source_row":
        path = folder / "first/outputs.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        for path in (folder / "first").iterdir():
            (folder / "second" / path.name).write_bytes(path.read_bytes())
    with pytest.raises(I.StageRefused, match="probe_approval_changed|probe_artifacts|probe_same_call"):
        K.read_probe_approval(folder, sha)


def test_v6_adapter_keeps_four_source_bytes_and_exposed_scope(v6_fixture):
    source, _, temp = v6_fixture
    old = json.loads((temp / "v5-cases.json").read_bytes())
    new = json.loads(source.read_bytes())
    assert new["schema_version"] == 3 and new["protocol"] == K.V6_PROTOCOL
    assert new["provenance"]["evaluation_scope"] == K.V6_EVALUATION_SCOPE
    for key in old["provenance"]:
        assert new["provenance"][key] == old["provenance"][key]
    for a, b in zip(old["cases"], new["cases"]):
        assert b == {**a, "protocol": K.V6_PROTOCOL}
    result = K.check_rules(source)
    assert result["cases"] == result["rows_recorded"] == 150
    assert result["protocol"] == K.V6_PROTOCOL and result["exit_code"] == 0
    assert result["evaluation_scope"]["prior_model_output_exposed"] is True
    assert result["evaluation_scope"]["new_independent_discrimination_claimed"] is False
    assert not result["gpu_model_verified"] and not result["approval_eligible"] and not Fake.made


def test_v6_two_fake_calls_keep_raw_proof_but_cannot_approve(v6_fixture):
    for name in ("first", "second"):
        result = run(v6_fixture, name)
        assert result["summary"]["thresholds_passed"]
        assert result["summary"]["format_normalized_rows"] == 0
        assert result["summary"]["format_normalizations"] == {}
        receipt = json.loads((v6_fixture[2] / name / "PROBE.json").read_bytes())
        assert receipt["schema_version"] == 2
        assert receipt["identity_basis"]["input_basis"]["parser_schema_version"] == "judge_structured_v6"
    report = K.compare_probes(v6_fixture[2] / "first", v6_fixture[2] / "second")
    assert report["protocol"] == K.V6_PROTOCOL and report["approval_eligible"]
    assert report["raw_text_deterministic"] and not report["gpu_model_verified"]
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K.seal_probe_approval(v6_fixture[2] / "first", v6_fixture[2] / "second", v6_fixture[2] / "approval")
    assert not (v6_fixture[2] / "approval").exists()


@pytest.mark.parametrize("group", ["N+", "D+"])
@pytest.mark.parametrize("rejects,passed", [(2, True), (3, False)])
def test_v6_positive_model_fr_boundary_stays_two_per_fifty(v6_fixture, group, rejects, passed):
    if group == "N+":
        Fake.reject_nplus = rejects
    else:
        Fake.reject_dplus = rejects
    result = run(v6_fixture)["summary"]
    assert result["thresholds"] == K.V6_THRESHOLDS
    assert result["groups"][group]["model"]["false_reject"] == rejects
    assert result["thresholds_passed"] is passed


@pytest.mark.parametrize("group", ["N+", "D+"])
def test_v6_one_deterministic_positive_rejection_is_fatal(v6_fixture, monkeypatch, group):
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        ["synthetic_rule_failure"] if text == f"{group} sample 0" else [])
    assert not run(v6_fixture)["summary"]["thresholds_passed"]
    assert K.check_rules(v6_fixture[0])["exit_code"] == 2


@pytest.mark.parametrize("rules_reject,passed", [(False, False), (True, True)])
def test_v6_negative_combined_false_accept_zero(v6_fixture, monkeypatch, rules_reject, passed):
    Fake.extra_false_accept = True
    monkeypatch.setattr(P, "deterministic_reasons", lambda task, text:
                        ["synthetic_negative"] if rules_reject and text.startswith("ADV") else [])
    result = run(v6_fixture)["summary"]
    for group in ("ADV1", "ADV2"):
        assert result["groups"][group]["model"]["false_accept"] == 1
        assert result["groups"][group]["combined"]["false_accept"] == (0 if rules_reject else 1)
    assert result["thresholds_passed"] is passed


@pytest.mark.parametrize("missing", ["callable", "marker", "policy_schema", "closure", "unbound"])
def test_v6_missing_b_connection_refuses_before_model_or_output(v6_fixture, monkeypatch, missing):
    if missing == "callable":
        monkeypatch.delattr(K.Q, "parse_judge_v6")
    elif missing == "marker":
        monkeypatch.setattr(K.Q, "JUDGE_V6_SCHEMA_VERSION", "judge_structured_v5")
    elif missing == "policy_schema":
        monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "judge_structured_v5")
    elif missing == "closure":
        monkeypatch.setattr(F, "require_v6_parser", lambda: {"capabilities": {"judge_structured_v6": False}})
    else:
        monkeypatch.setattr(P, "policy_identity", lambda: {"parser_implementation": {"wrong": True}})
    # CPU rules are explicitly not B parser or GPU validation.
    assert K.check_rules(v6_fixture[0])["exit_code"] == 0
    with pytest.raises(I.StageRefused, match="parser_missing"):
        run(v6_fixture)
    assert not Fake.made and not (v6_fixture[2] / "first").exists()


@pytest.mark.parametrize("kind", ["case_v5", "outer_v5", "schema_v5", "independence", "source_text",
                                  "source_expected", "source_raw", "task", "threshold_bool"])
def test_v6_protocol_and_original_source_binding(v6_fixture, kind):
    path = v6_fixture[0]
    value = json.loads(path.read_bytes())
    row = value["cases"][0]
    if kind == "case_v5":
        row["protocol"] = K.V5_PROTOCOL
    elif kind == "outer_v5":
        value["protocol"] = K.V5_PROTOCOL
    elif kind == "schema_v5":
        value["schema_version"] = 2
    elif kind == "independence":
        value["provenance"]["evaluation_scope"]["new_independent_discrimination_claimed"] = True
    elif kind == "source_text":
        row["candidate"] = row["source_case"]["text"] = "changed but rehashed"
        row["source_case_sha256"] = I._digest(row["source_case"])
    elif kind == "source_expected":
        row["source_case"]["expected"] = "reject"
        row["source_case_sha256"] = I._digest(row["source_case"])
    elif kind == "source_raw":
        data = base64.b64decode(value["provenance"]["source_files_base64"]["N+"])
        value["provenance"]["source_files_base64"]["N+"] = base64.b64encode(data + b"\n").decode()
    elif kind == "task":
        row["task"]["evidence"]["size_mm_available"] = 0
    else:
        value["thresholds"] = {**K.V6_THRESHOLDS, "negative_combined_false_accept": False}
    path.write_bytes(I._canonical(value))
    with pytest.raises(I.StageRefused):
        run(v6_fixture)
    assert not Fake.made and not (v6_fixture[2] / "first").exists()


@pytest.mark.parametrize("normalizations", [[], ["code_fence"], ["string_boolean"],
                                             ["code_fence", "string_boolean"]])
def test_v6_normalized_metadata_rows_counts_and_raw_bytes(v6_fixture, monkeypatch, normalizations):
    # Only tests metadata transport. No claim that this stub validates B's exact
    # fence/string rules: the actual parser owns those separate tests.
    def parser(task, raw):
        return {**synthetic_v6_parser(task, raw), "format_normalized": list(normalizations)}
    monkeypatch.setattr(K.Q, "parse_judge_v6", parser)
    result = run(v6_fixture)
    summary = result["summary"]
    assert summary["format_normalized_rows"] == (150 if normalizations else 0)
    assert summary["format_normalizations"] == {name: 150 for name in normalizations}
    for group, count in K.V6_GROUP_COUNTS.items():
        assert summary["groups"][group]["format_normalized_rows"] == (count if normalizations else 0)
    folder = v6_fixture[2] / "first"
    rows = [json.loads(line) for line in (folder / "outputs.jsonl").read_bytes().splitlines()]
    events = [json.loads(line) for line in (folder / "backend_batches.jsonl").read_bytes().splitlines()]
    raw = [row for event in events for row in event["output"]]
    for row, original in zip(rows, raw):
        assert row["raw"] == {**original, "id": row["id"]}
        assert row["raw_sha256"] == I._digest(row["raw"])
        assert row["format_normalized"] == normalizations
        assert row["format_normalized_any"] is bool(normalizations)


@pytest.mark.parametrize("normalizations", [None, True, "code_fence", ["guess"], ["code_fence", "code_fence"],
                                             ["string_boolean", "code_fence"], [1]])
def test_v6_invalid_normalization_contract_preserves_raw_before_failure(v6_fixture, monkeypatch, normalizations):
    monkeypatch.setattr(K.Q, "parse_judge_v6", lambda task, raw:
                        {**synthetic_v6_parser(task, raw), "format_normalized": normalizations})
    with pytest.raises(I.StageRefused, match="parser_contract"):
        run(v6_fixture)
    folder = v6_fixture[2] / "first"
    events = [json.loads(line) for line in (folder / "backend_batches.jsonl").read_bytes().splitlines()]
    assert len(events[0]["output"]) == 50 and not (folder / "PROBE.json").exists()
    assert Fake.closed == 1


def test_v6_never_falls_back_to_v5_parser(v6_fixture, monkeypatch):
    def forbidden(*args):
        pytest.fail("v6 must not use the archived v5 parser")
    monkeypatch.setattr(K.Q, "parse_judge_v5", forbidden)
    monkeypatch.setattr(K.J, "require_v5_parser", forbidden)
    assert run(v6_fixture)["summary"]["thresholds_passed"]


@pytest.mark.parametrize("kind", ["protocol", "receipt_schema", "normalization"])
def test_v6_rehashed_receipt_or_row_cannot_hide_version_or_metadata_tamper(v6_fixture, kind):
    run(v6_fixture)
    folder = v6_fixture[2] / "first"
    receipt = json.loads((folder / "PROBE.json").read_bytes())
    if kind == "protocol":
        receipt["identity_basis"]["input_basis"]["protocol"] = K.V5_PROTOCOL
    elif kind == "receipt_schema":
        receipt["schema_version"] = 1
    else:
        path = folder / "outputs.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["format_normalized"] = ["code_fence"]
        rows[0]["format_normalized_any"] = True
        path.write_bytes(b"".join(I._canonical(row) + b"\n" for row in rows))
        receipt["identity_basis"]["artifacts_sha256"]["outputs.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt["identity_sha256"] = I._digest(receipt["identity_basis"])
    (folder / "PROBE.json").write_bytes(I._canonical(receipt) + b"\n")
    with pytest.raises(I.StageRefused):
        K._verify_probe(folder)


def test_v5_pair_even_real_shaped_cannot_seal_v6_approval(v5_fixture, monkeypatch):
    monkeypatch.setattr(I, "TransformersBackend", ProductionLookingFake)
    source, config, temp = v5_fixture
    for name in ("first", "second"):
        K.run_probe(source, config, temp / name)
    report = K.compare_probes(temp / "first", temp / "second")
    assert report["real_backend_pair"] and report["thresholds_passed"]
    assert not report["approval_eligible"] and not report["gpu_model_verified"]
    with pytest.raises(I.StageRefused, match="probe_not_approved"):
        K.seal_probe_approval(temp / "first", temp / "second", temp / "approval")
    assert not (temp / "approval").exists()


def test_v6_approval_receipt_cannot_be_downgraded_to_v5_schema(emulated_approval):
    temp, _ = emulated_approval
    path = temp / "approval/APPROVAL.json"
    value = json.loads(path.read_bytes())
    assert value["schema_version"] == 2 and value["identity_basis"]["protocol"] == K.V6_PROTOCOL
    value["schema_version"] = 1
    path.write_bytes(I._canonical(value))
    with pytest.raises(I.StageRefused, match="probe_approval_receipt"):
        K.read_probe_approval(path.parent, hashlib.sha256(path.read_bytes()).hexdigest())
