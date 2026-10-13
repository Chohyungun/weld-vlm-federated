"""합성 입력 작성 계약만 검사한다. 독립 N+ 원본의 기대를 판정하거나 변경하지 않는다."""

import base64
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from corpus.production import judge_cases as C
from corpus.production.common import read_json, sha256_file, write_json_new


def _nplus(tmp_path):
    source = {"schema_version": 1, "group": "N+",
              "metadata": {"independent_expected_cases": True, "policy_source_not_read": True},
              "cases": [{"id": f"synthetic-fixture-{i:03d}", "task": C._task(i, False),
                         "candidate": "표시된 결함이 없다.", "expected_model_accept": True,
                         "expected_deterministic_accept": True} for i in range(50)]}
    path = tmp_path / "independent.json"
    write_json_new(path, source)
    return path, source


def test_cases_keep_independent_expectations_and_scope(tmp_path):
    path, source = _nplus(tmp_path)
    before = path.read_bytes()
    result = C.build_cases(path)
    assert path.read_bytes() == before
    assert result["cases"][:50] == [{**v, "group": "N+"} for v in source["cases"]]
    assert Counter(c["group"] for c in result["cases"]) == {group: 50 for group in (
        "N+", "N-invented", "N-number", "N-assertion", "N-leak",
        "D+", "D-missing", "D-half", "D-trap", "D-ungrounded")}
    assert len({c["id"] for c in result["cases"]}) == 500
    assert result["provenance"]["independent_nplus_sha256"] == sha256_file(path)
    assert result["model_discrimination_measured"] is False
    assert result["provenance"]["statistical_generalization_claimed"] is False
    assert result["provenance"]["case_reuse"]["N+"]["unique_candidate_texts"] == 1
    assert all("image_path" not in c["task"] for c in result["cases"])
    assert all(not c["expected_model_accept"] for c in result["cases"] if c["group"] == "D-ungrounded")
    assert any(c["expected_deterministic_accept"] for c in result["cases"] if c["group"] == "D-ungrounded")


@pytest.mark.parametrize("change", ["count", "expectation", "authorship", "duplicate", "not_normal"])
def test_invalid_independent_source_is_not_relabelled(tmp_path, change):
    _, original = _nplus(tmp_path)
    value = deepcopy(original)
    if change == "count":
        value["cases"].pop()
    elif change == "expectation":
        value["cases"][0]["expected_deterministic_accept"] = False
    elif change == "authorship":
        value["metadata"]["policy_source_not_read"] = False
    elif change == "duplicate":
        value["cases"][1]["id"] = value["cases"][0]["id"]
    else:
        value["cases"][0]["task"]["locked_target"]["defects"] = [{"type": "101"}]
    path = tmp_path / "invalid.json"
    write_json_new(path, value)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        C.build_cases(path)
    assert path.read_bytes() == before


def test_write_is_new_only_and_gate_guarded(tmp_path, monkeypatch):
    path, _ = _nplus(tmp_path)
    output = tmp_path / "cases.json"
    calls = []
    monkeypatch.setattr(C, "assert_gate_closed", lambda root: calls.append(root))
    receipt = C.write_cases(path, output, repo_root=tmp_path)
    assert calls == [tmp_path]
    assert receipt["sha256"] == sha256_file(output)
    with pytest.raises((OSError, ValueError)):
        C.write_cases(path, output, repo_root=tmp_path)
    assert receipt["sha256"] == sha256_file(output)


def test_open_gate_prevents_any_output(tmp_path, monkeypatch):
    path, _ = _nplus(tmp_path)

    def refuse(root):
        raise ValueError("gate_open")

    monkeypatch.setattr(C, "assert_gate_closed", refuse)
    output = tmp_path / "cases.json"
    with pytest.raises(ValueError, match="gate_open"):
        C.write_cases(path, output)
    assert not output.exists()


def _label_map_copy(tmp_path):
    path = tmp_path / "label_map.yaml"
    path.write_bytes(Path(C.label_map.DEFAULT_LABEL_MAP).read_bytes())
    return path


def test_case_labels_and_provenance_follow_validated_yaml(tmp_path, monkeypatch):
    source, _ = _nplus(tmp_path)
    path = _label_map_copy(tmp_path)
    monkeypatch.setattr(C.label_map, "DEFAULT_LABEL_MAP", path)
    loader = C.label_map.load_label_map
    calls = []

    def tracked_loader(value):
        calls.append(value)
        return loader(value)

    monkeypatch.setattr(C.label_map, "load_label_map", tracked_loader)
    original = C.build_cases(source)
    before_names = original["provenance"]["label_names"]
    after_names = {key: f"대조_{key}" for key in before_names}
    text = path.read_text(encoding="utf-8")
    for key, name in before_names.items():
        original_field = f'name_ko: "{name}"'
        assert text.count(original_field) == 1
        text = text.replace(original_field, f'name_ko: "{after_names[key]}"')
    path.write_text(text, encoding="utf-8")
    changed = C.build_cases(source)

    assert calls == [path.resolve(), path.resolve()]
    expected = json.dumps(original["cases"], ensure_ascii=False)
    for key, name in before_names.items():
        expected = expected.replace(name, after_names[key])
    assert changed["cases"] == json.loads(expected)
    assert changed["cases"][:50] == original["cases"][:50]
    assert changed["thresholds"] == original["thresholds"]
    assert changed["provenance"]["label_names"] == after_names
    assert changed["provenance"]["label_map_sha256"] == sha256_file(path)
    assert changed["provenance"]["label_map_sha256"] != original["provenance"]["label_map_sha256"]
    assert changed["provenance"]["label_map_loader_sha256"] == sha256_file(C.label_map.__file__)
    assert changed["provenance"]["independent_nplus_sha256"] == sha256_file(source)


@pytest.mark.parametrize("input_key", ["label_map", "loader", "independent_nplus", "builder"])
def test_inputs_changed_during_case_construction_are_rejected(tmp_path, monkeypatch, input_key):
    source, _ = _nplus(tmp_path)
    paths = {"label_map": _label_map_copy(tmp_path), "independent_nplus": source}
    # 원본 코드와 봉인 자산에는 쓰지 않는다. 지문 검사는 임시 사본으로 검사한다.
    for key, original in (("loader", C.label_map.__file__), ("builder", C.__file__)):
        paths[key] = tmp_path / f"{key}.py"
        paths[key].write_bytes(Path(original).read_bytes())
    monkeypatch.setattr(C.label_map, "DEFAULT_LABEL_MAP", paths["label_map"])
    monkeypatch.setattr(C.label_map, "__file__", str(paths["loader"]))
    monkeypatch.setattr(C, "__file__", str(paths["builder"]))
    make_case = C._case
    mutated = False

    def mutate_after_first_case(*args, **kwargs):
        nonlocal mutated
        case = make_case(*args, **kwargs)
        if not mutated:
            paths[input_key].write_bytes(paths[input_key].read_bytes() + b"\n")
            mutated = True
        return case

    monkeypatch.setattr(C, "_case", mutate_after_first_case)
    monkeypatch.setattr(C, "assert_gate_closed", lambda root: None)
    output = tmp_path / "not_created.json"
    with pytest.raises(ValueError, match="input changed during construction"):
        C.write_cases(source, output)
    assert mutated
    assert not output.exists()


def test_label_map_change_inside_loader_is_rejected(tmp_path, monkeypatch):
    source, _ = _nplus(tmp_path)
    path = _label_map_copy(tmp_path)
    monkeypatch.setattr(C.label_map, "DEFAULT_LABEL_MAP", path)
    loader = C.label_map.load_label_map

    def changed_after_read(value):
        loaded = loader(value)
        path.write_bytes(path.read_bytes() + b"\n")
        return loaded

    monkeypatch.setattr(C.label_map, "load_label_map", changed_after_read)
    with pytest.raises(ValueError, match="input changed during construction: label_map_sha256"):
        C.build_cases(source)


def test_case_builder_has_no_embedded_label_names():
    # 기존 AST 검사의 완전 일치뿐 아니라 문장 중간에 박힌 이름도 금지한다.
    text = Path(C.__file__).read_text(encoding="utf-8")
    names = C.label_map.load_label_map().defect_types.values()
    assert all(defect.name_ko not in text for defect in names)


def test_current_yaml_rebuilds_all_frozen_initial_cases_without_changes():
    root = Path(__file__).resolve().parents[2]
    frozen_path = root / "_workspace" / "2026-10-12-policy" / "cases_v3_initial.json"
    if not frozen_path.is_file():
        pytest.skip("로컬 봉인 사례 대조본이 없다. 이 시험은 사례 내용의 불변을 확인하지 못했다.")
    source = Path(__file__).parent / "fixtures" / "judge_nplus_v3.json"
    frozen_bytes, source_bytes = frozen_path.read_bytes(), source.read_bytes()
    frozen = read_json(frozen_path)
    assert frozen["provenance"]["independent_nplus_sha256"] == sha256_file(source)
    rebuilt = C.build_cases(source)
    assert len(rebuilt["cases"]) == 500
    assert rebuilt["cases"] == frozen["cases"]
    for key in ("schema_version", "thresholds", "model_discrimination_measured"):
        assert rebuilt[key] == frozen[key]
    for key, value in frozen["provenance"].items():
        if key != "builder_sha256":
            assert rebuilt["provenance"][key] == value
    assert rebuilt["provenance"]["builder_sha256"] == sha256_file(C.__file__)
    assert frozen_path.read_bytes() == frozen_bytes
    assert source.read_bytes() == source_bytes


def _v5_sources(tmp_path):
    """G 원문이나 D 참조 구현을 쓰지 않는 구조 대조용 입력이다."""
    labels = C.label_map.load_label_map().defect_types
    first, second = labels["porosity"].name_ko, labels["crack"].name_ko
    sources, originals = {}, {}
    specifications = (("N+", "judge_nplus_g1", 50), ("D+", "judge_dplus_g1", 50),
                      ("ADV1", "judge_adv_negative_g1", 25), ("ADV2", "judge_adv_negative_g2", 25))
    for group, source_set, count in specifications:
        value = {"set": source_set, "written_by": "합성 구조 시험",
                 "not_read": ["D 참조 구현"], "cases": []}
        if group in ("N+", "ADV1"):
            value["row_kind"] = "defect_free"
        for i in range(count):
            item = {"id": f"fixture-{group}-{i:03d}", "bundle": f"원본 묶음 {group}",
                    "text": f"  원문 ‘{i}’\n둘째 줄은 그대로 남는다.  ",
                    "expected": "accept" if group in ("N+", "D+") else "reject",
                    "reason": "모델 입력에는 없어야 할 원본 작성 사유", "extra": {"original": i}}
            if group not in ("N+", "ADV1"):
                defect = group == "D+" or i % 2 == 1
                item["row_kind"] = "defect" if defect else "defect_free"
                item["fields"] = {
                    "defect_counts_by_name": {first: 2, second: 1} if defect else {},
                    "defect_positions": {first: ["위 왼쪽", "아래 오른쪽"], second: ["가운데"]} if defect else {}}
            value["cases"].append(item)
        path = tmp_path / f"{source_set}.json"
        write_json_new(path, value)
        sources[group], originals[group] = path, value
    return sources, {group: sha256_file(path) for group, path in sources.items()}, originals


def _object_sha(value):
    # 변환기의 헬퍼를 재사용하지 않고 원객체 지문 산식을 대조한다.
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _registration_manifest(tmp_path, protocol, pins, originals, sources=None):
    if sources is not None:
        originals["N+"]["cases"] = originals["N+"]["cases"][:4]
        sources["N+"].write_bytes(json.dumps(originals["N+"], ensure_ascii=False).encode("utf-8"))
        pins["N+"] = sha256_file(sources["N+"])
    value = {"schema_version": 1, "protocol": protocol,
             "approval_scope": C.V7_REGISTERED_SCOPES[protocol],
             "source_sha256": pins, "source_sets": {g: row["set"] for g, row in originals.items()},
             "group_counts": dict(C.V7_REGISTERED_GROUP_COUNTS)}
    path = tmp_path / "invented-registration-manifest.json"
    write_json_new(path, value)
    kwargs = {"expected_manifest_sha256": sha256_file(path), "expected_protocol": protocol,
              "expected_approval_scope": value["approval_scope"], "expected_source_sha256": pins}
    return path, kwargs


@pytest.mark.parametrize("protocol", [C.V7_A_PROTOCOL, C.V7_B_PROTOCOL])
def test_registered_v7_preserves_all_source_bytes_and_expectations(tmp_path, protocol):
    sources, pins, originals = _v5_sources(tmp_path)
    old = C.build_v5_cases(sources, expected_source_sha256=pins)
    manifest, expected = _registration_manifest(tmp_path, protocol, pins, originals, sources)
    before = {g: path.read_bytes() for g, path in sources.items()}
    new = C.build_v7_registered_cases(sources, manifest_path=manifest, **expected)
    assert new["schema_version"] == 5 and new["protocol"] == protocol
    assert new["provenance"]["evaluation_scope"]["generation_contract_applied_to_probe_gate"] is True
    assert new["provenance"]["evaluation_scope"]["final_approval_eligible"] is (protocol == C.V7_B_PROTOCOL)
    assert not new["model_discrimination_measured"]
    assert len(new["cases"]) == 104
    old_cases = [c for c in old["cases"] if c["group"] != "N+" or c["source_index"] < 4]
    assert all(b == {**a, "protocol": protocol, "source_file_sha256": pins[a["group"]]}
               for a, b in zip(old_cases, new["cases"]))
    assert {g: path.read_bytes() for g, path in sources.items()} == before
    for name, value in expected.items():
        assert new["provenance"]["registration"][name] == value


def test_v7_manifest_validation_never_reads_source_files(tmp_path, monkeypatch):
    # Metadata-only path; the source paths deliberately do not exist.
    pins = {g: str(i + 1) * 64 for i, g in enumerate(C.V5_GROUP_COUNTS)}
    originals = {g: {"set": "invented-unopened-" + g} for g in pins}
    manifest, expected = _registration_manifest(tmp_path, C.V7_B_PROTOCOL, pins, originals)
    raw = manifest.read_bytes()

    def no_source_read(*args, **kwargs):
        raise AssertionError("metadata validation opened a file")

    monkeypatch.setattr(Path, "read_bytes", no_source_read)
    value = C.validate_v7_manifest(raw, **expected)
    assert value["source_sha256"] == pins


@pytest.mark.parametrize("change", ["outer_sha", "source_sha", "scope", "protocol", "extra_key",
                                    "counts", "boolean_count", "sets", "missing_pin"])
def test_registered_v7_manifest_rejects_mutable_or_incompatible_claims(tmp_path, change):
    sources, pins, originals = _v5_sources(tmp_path)
    path, expected = _registration_manifest(tmp_path, C.V7_A_PROTOCOL, pins, originals, sources)
    value = read_json(path)
    if change == "outer_sha":
        expected["expected_manifest_sha256"] = "0" * 64
    elif change == "missing_pin":
        expected["expected_source_sha256"] = {"N+": pins["N+"]}
    else:
        if change == "source_sha":
            value["source_sha256"]["N+"] = "0" * 64
        elif change == "scope":
            value["approval_scope"] = "final_blinded"
        elif change == "protocol":
            value["protocol"] = C.V7_B_PROTOCOL
        elif change == "extra_key":
            value["approval_eligible"] = True
        elif change == "counts":
            value["group_counts"]["N+"] = 49
        elif change == "boolean_count":
            value["group_counts"]["N+"] = True
        else:
            value["source_sets"]["D+"] = value["source_sets"]["N+"]
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        path.write_bytes(raw)
        expected["expected_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="v7"):
        C.build_v7_registered_cases(sources, manifest_path=path, **expected)


def test_registered_v7_writer_rechecks_gate_and_never_overwrites(tmp_path, monkeypatch):
    sources, pins, originals = _v5_sources(tmp_path)
    manifest, expected = _registration_manifest(tmp_path, C.V7_A_PROTOCOL, pins, originals, sources)
    seen = []
    monkeypatch.setattr(C, "assert_gate_closed", lambda root: seen.append(root))
    output = tmp_path / "registered-cases.json"
    result = C.write_v7_registered_cases(sources, output, manifest_path=manifest,
                                         repo_root=tmp_path, **expected)
    assert seen == [tmp_path, tmp_path] and result["sha256"] == sha256_file(output)
    assert not result["final_blinded_approval"] and not result["model_discrimination_measured"]
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new path"):
        C.write_v7_registered_cases(sources, output, manifest_path=manifest, **expected)
    assert output.read_bytes() == before


def test_v6_new_protocol_preserves_v5_source_task_candidate_and_expectation(tmp_path):
    sources, pins, originals = _v5_sources(tmp_path)
    before = {group: path.read_bytes() for group, path in sources.items()}
    old = C.build_v5_cases(sources, expected_source_sha256=pins)
    new = C.build_v6_cases(sources, expected_source_sha256=pins)
    assert new["schema_version"] == 3 and new["protocol"] == "judge_v6_g150"
    assert len(new["cases"]) == 150 and new["model_discrimination_measured"] is False
    assert new["provenance"]["evaluation_scope"] == C.V6_EVALUATION_SCOPE
    assert new["provenance"]["evaluation_scope"]["prior_model_output_exposed"] is True
    assert new["provenance"]["evaluation_scope"]["new_independent_discrimination_claimed"] is False
    for key in old["provenance"]:
        assert new["provenance"][key] == old["provenance"][key]
    for a, b in zip(old["cases"], new["cases"]):
        assert b == {**a, "protocol": C.V6_PROTOCOL}
        original = originals[b["source_group"]]["cases"][b["source_index"]]
        assert b["candidate"] == original["text"]
        assert b["source_case_sha256"] == _object_sha(original)
        assert b["source_file_sha256"] == pins[b["source_group"]]
    assert {group: path.read_bytes() for group, path in sources.items()} == before


def test_v6_writer_never_overwrites_v5_and_rechecks_gate(tmp_path, monkeypatch):
    sources, pins, _ = _v5_sources(tmp_path)
    seen = []
    monkeypatch.setattr(C, "assert_gate_closed", lambda root: seen.append(root))
    old_path, new_path = tmp_path / "v5.json", tmp_path / "v6.json"
    C.write_v5_cases(sources, old_path, expected_source_sha256=pins)
    frozen = old_path.read_bytes()
    seen.clear()
    receipt = C.write_v6_cases(sources, new_path, expected_source_sha256=pins, repo_root=tmp_path)
    assert seen == [tmp_path, tmp_path]
    assert receipt["protocol"] == C.V6_PROTOCOL and receipt["sha256"] == sha256_file(new_path)
    assert read_json(new_path)["schema_version"] == 3
    with pytest.raises(ValueError, match="new path"):
        C.write_v6_cases(sources, old_path, expected_source_sha256=pins)
    assert old_path.read_bytes() == frozen


@pytest.mark.parametrize("reject_on_call", [1, 2])
def test_v6_gate_refusal_never_publishes(tmp_path, monkeypatch, reject_on_call):
    sources, pins, _ = _v5_sources(tmp_path)
    count = 0
    def gate(root):
        nonlocal count
        count += 1
        if count == reject_on_call:
            raise RuntimeError("synthetic open gate")
    monkeypatch.setattr(C, "assert_gate_closed", gate)
    output = tmp_path / "not_created.json"
    with pytest.raises(RuntimeError, match="synthetic open gate"):
        C.write_v6_cases(sources, output, expected_source_sha256=pins)
    assert not output.exists()


def test_v5_preserves_all_originals_and_converts_mixed_fields_without_answer_leak(tmp_path):
    sources, pins, originals = _v5_sources(tmp_path)
    before = {group: path.read_bytes() for group, path in sources.items()}
    result = C.build_v5_cases(sources, expected_source_sha256=pins)
    assert result["schema_version"] == 2 and result["protocol"] == "judge_v5_g150"
    assert result["model_discrimination_measured"] is False
    assert Counter(case["group"] for case in result["cases"]) == {"N+": 50, "D+": 50, "ADV1": 25, "ADV2": 25}
    assert len({case["id"] for case in result["cases"]}) == 150
    for case in result["cases"]:
        group, index = case["source_group"], case["source_index"]
        original = originals[group]["cases"][index]
        assert case["group"] == group and case["source_file_sha256"] == pins[group]
        assert case["source_case"] == original and case["source_case_sha256"] == _object_sha(original)
        assert case["candidate"] == original["text"] and case["id"] == original["id"]
        positive = original["expected"] == "accept"
        assert case["expected_model_accept"] is positive
        assert case["expected_deterministic_accept"] is (True if positive else None)
        fields = original.get("fields", {"defect_counts_by_name": {}, "defect_positions": {}})
        evidence, locked = case["task"]["evidence"], case["task"]["locked_target"]
        assert evidence["defect_counts_by_name"] == fields["defect_counts_by_name"]
        assert evidence["row_class"] == original.get("row_kind", originals[group].get("row_kind"))
        recovered = {}
        for fact in evidence["defect_positions"]:
            recovered.setdefault(fact["name"], []).append(fact["where"])
        assert recovered == fields["defect_positions"]
        assert Counter(item["name"] for item in locked["defects"]) == fields["defect_counts_by_name"]
        assert [{"name": item["name"], "where": item["where"]} for item in locked["defects"]] == evidence["defect_positions"]
        assert all(set(item) == {"name", "where"} for item in locked["defects"])
        assert locked["clauses"] == [] and locked["verdict"] is None
        assert set(case["task"]) == {"kind", "source_ids", "prompt", "evidence", "locked_target"}
        assert case["task"]["source_ids"] == ["synthetic-annotation"]
        assert set(evidence) == {"row_class", "defect_counts_by_name", "defect_positions", "size_mm_available"}
        task_text = json.dumps(case["task"], ensure_ascii=False)
        assert original["reason"] not in task_text and original["text"] not in task_text
        assert all(example not in task_text for example in C.LIMITATIONS)
    provenance = result["provenance"]
    assert provenance["source_sha256"] == pins
    for group, encoded in provenance["source_files_base64"].items():
        raw = base64.b64decode(encoded, validate=True)
        assert raw == before[group] and hashlib.sha256(raw).hexdigest() == pins[group]
        embedded_source = json.loads(raw)
        for case in (item for item in result["cases"] if item["group"] == group):
            assert embedded_source["cases"][case["source_index"]] == case["source_case"]
            assert C.task_from_v5_source(embedded_source, case["source_case"],
                                         label_names=provenance["label_names"]) == case["task"]
    assert provenance["source_metadata"] == {
        group: {key: value for key, value in source.items() if key != "cases"} for group, source in originals.items()}
    assert provenance["adapter_sha256"] == sha256_file(C.__file__)
    assert provenance["label_map_sha256"] == sha256_file(C.label_map.DEFAULT_LABEL_MAP)
    assert provenance["label_map_loader_sha256"] == sha256_file(C.label_map.__file__)
    assert provenance["deterministic_results_are_not_model_approval"] is True
    assert {group: path.read_bytes() for group, path in sources.items()} == before


@pytest.mark.parametrize("bad_pin", [None, {}, {"N+": "a" * 64}, "self-hash-is-not-a-pin"])
def test_v5_requires_four_external_pins(tmp_path, bad_pin):
    sources, _, _ = _v5_sources(tmp_path)
    with pytest.raises(ValueError, match="all four"):
        C.build_v5_cases(sources, expected_source_sha256=bad_pin)


@pytest.mark.parametrize("bad_pin", [True, "a" * 63, "A" * 64, "b" * 64])
def test_v5_malformed_or_wrong_pin_is_not_replaced_with_observed_hash(tmp_path, bad_pin):
    sources, pins, _ = _v5_sources(tmp_path)
    pins["D+"] = bad_pin
    with pytest.raises(ValueError, match="pin"):
        C.build_v5_cases(sources, expected_source_sha256=pins)


@pytest.mark.parametrize("change", ["expectation", "count", "duplicate", "missing_kind", "conflicting_kind",
                                   "bool_count", "zero_count", "wrong_positions", "unknown_position", "unknown_label",
                                   "normal_fields", "missing_fields", "blank_text"])
def test_v5_invalid_source_is_refused_not_relabelled_or_repaired(tmp_path, change):
    sources, pins, originals = _v5_sources(tmp_path)
    source = originals["D+"]
    first = source["cases"][0]
    counts, positions = first["fields"]["defect_counts_by_name"], first["fields"]["defect_positions"]
    name = next(iter(counts))
    if change == "expectation":
        first["expected"] = "reject"
    elif change == "count":
        source["cases"].pop()
    elif change == "duplicate":
        source["cases"][1]["id"] = first["id"]
    elif change == "missing_kind":
        del first["row_kind"]
    elif change == "conflicting_kind":
        source["row_kind"] = "defect_free"
    elif change == "bool_count":
        counts[name] = True
    elif change == "zero_count":
        counts[name] = 0
    elif change == "wrong_positions":
        positions[name].pop()
    elif change == "unknown_position":
        positions[name][0] = "위 밖"
    elif change == "unknown_label":
        counts["not_in_map"] = counts.pop(name)
        positions["not_in_map"] = positions.pop(name)
    elif change == "normal_fields":
        first["row_kind"] = "defect_free"
    elif change == "missing_fields":
        del first["fields"]
    else:
        first["text"] = "  "
    path = tmp_path / "changed.json"
    write_json_new(path, source)
    sources["D+"], pins["D+"] = path, sha256_file(path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        C.build_v5_cases(sources, expected_source_sha256=pins)
    assert path.read_bytes() == before


@pytest.mark.parametrize("changed_input", ["N+", "D+", "ADV1", "ADV2", "label_map", "loader", "adapter"])
def test_v5_input_drift_during_conversion_never_publishes(tmp_path, monkeypatch, changed_input):
    sources, pins, _ = _v5_sources(tmp_path)
    paths = dict(sources)
    for key, original in (("label_map", C.label_map.DEFAULT_LABEL_MAP),
                          ("loader", C.label_map.__file__), ("adapter", C.__file__)):
        paths[key] = tmp_path / f"{key}.copy"
        paths[key].write_bytes(Path(original).read_bytes())
    monkeypatch.setattr(C.label_map, "DEFAULT_LABEL_MAP", paths["label_map"])
    monkeypatch.setattr(C.label_map, "__file__", str(paths["loader"]))
    monkeypatch.setattr(C, "__file__", str(paths["adapter"]))
    convert, changed = C._v5_task, False

    def mutate(*args):
        nonlocal changed
        task = convert(*args)
        if not changed:
            paths[changed_input].write_bytes(paths[changed_input].read_bytes() + b"\n")
            changed = True
        return task

    monkeypatch.setattr(C, "_v5_task", mutate)
    monkeypatch.setattr(C, "assert_gate_closed", lambda root: None)
    output = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="changed during construction|differs from caller pin"):
        C.write_v5_cases(sources, output, expected_source_sha256=pins)
    assert changed and not output.exists()


def test_v5_write_is_new_only_and_rechecks_gate(tmp_path, monkeypatch):
    sources, pins, _ = _v5_sources(tmp_path)
    before = {group: path.read_bytes() for group, path in sources.items()}
    calls = []
    monkeypatch.setattr(C, "assert_gate_closed", lambda root: calls.append(root))
    output = tmp_path / "v5.json"
    receipt = C.write_v5_cases(sources, output, expected_source_sha256=pins, repo_root=tmp_path)
    assert calls == [tmp_path, tmp_path] and receipt["sha256"] == sha256_file(output)
    assert receipt["cases"] == 150
    output_bytes = output.read_bytes()
    with pytest.raises(ValueError, match="new path"):
        C.write_v5_cases(sources, output, expected_source_sha256=pins)
    assert output.read_bytes() == output_bytes
    with pytest.raises(ValueError, match="new path"):
        C.write_v5_cases(sources, sources["N+"], expected_source_sha256=pins)
    assert {group: path.read_bytes() for group, path in sources.items()} == before


@pytest.mark.parametrize("reject_on_call", [1, 2])
def test_v5_gate_refusal_before_build_or_publish_leaves_no_output(tmp_path, monkeypatch, reject_on_call):
    sources, pins, _ = _v5_sources(tmp_path)
    calls = 0

    def gate(root):
        nonlocal calls
        calls += 1
        if calls == reject_on_call:
            raise ValueError("gate_open")

    monkeypatch.setattr(C, "assert_gate_closed", gate)
    output = tmp_path / "blocked.json"
    with pytest.raises(ValueError, match="gate_open"):
        C.write_v5_cases(sources, output, expected_source_sha256=pins)
    assert not output.exists()
