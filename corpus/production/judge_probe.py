"""Single-use synthetic judge probes and a read-only two-call comparison.

No production rows or images are read. A probe is never resumed: every invocation
needs a fresh output directory. Its final receipt is written only after all saved
artifacts have been bound. Fake/injected backend runs cannot verify a GPU model.
GPU-wide exclusivity is the external operator/supervisor's responsibility: this
standalone probe has no production run directory from which to acquire its lock.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import infer as I
from . import judge_identity as J
from . import judge_policy as P
from . import quality as Q
from .judge_cases import (V7_EVALUATION_SCOPE, V7_A_PROTOCOL, V7_B_PROTOCOL,
                          V7_REGISTERED_SCOPES, V7_REGISTERED_GROUP_COUNTS, validate_v7_manifest,
                          v7_registered_evaluation_scope)

_POSITIVE = {"N+", "D+"}
GROUPS = ("N+", "N-invented", "N-number", "N-assertion", "N-leak", "D+",
          "D-missing", "D-half", "D-trap", "D-ungrounded")
_THRESHOLDS = {"cases_per_group": 50, "negative_false_accept": 0,
               "nplus_deterministic_false_reject": 0,
               "positive_combined_false_reject_maximum": 2, "actual_independent_calls": 2}
LEGACY_PROTOCOL = "legacy_v3_500_regression_only"
V5_PROTOCOL = "judge_v5_g150"
V5_GROUP_COUNTS = {"N+": 50, "D+": 50, "ADV1": 25, "ADV2": 25}
V5_THRESHOLDS = {"positive_model_false_reject_maximum": 2,
                 "positive_deterministic_false_reject": 0,
                 "negative_combined_false_accept": 0, "actual_independent_calls": 2,
                 "parser_error_rows": 0, "truncated_rows": 0}
V5_SOURCE_SHA256 = {
    "N+": "a776e5063bbee58d8e44ab3757c041451e3c137079fc98e1f3f1fe546d4b0227",
    "D+": "2dbf12cbba37b81a9157ad1f1da8f1d50e53dd49d533bb1e9a3ac3d6e0da9672",
    "ADV1": "28a94e03d58b247d79d9d540f5bf1e41969d4007cad392fe8a93ec12eccdd58b",
    "ADV2": "389af5560ac9270dff7671937f7e62b6ca16c03e681f6f15f17ac5ef88d82a40",
}
V6_PROTOCOL = "judge_v6_g150"
V6_GROUP_COUNTS = dict(V5_GROUP_COUNTS)
V6_THRESHOLDS = dict(V5_THRESHOLDS)
V6_SOURCE_SHA256 = dict(V5_SOURCE_SHA256)
V6_EVALUATION_SCOPE = {
    "previous_model_protocol": V5_PROTOCOL,
    "prior_model_output_exposed": True,
    "new_independent_discrimination_claimed": False,
    "role": "fixed-source regression and preregistered operational gate",
}
V7_PROTOCOL = "judge_v7_g150"
V7_GROUP_COUNTS = dict(V5_GROUP_COUNTS)
V7_SOURCE_SHA256 = dict(V5_SOURCE_SHA256)
V7_THRESHOLDS = {"dplus_model_false_reject_maximum": 2, "dplus_model_denominator": 50,
                 "nplus_model_metric": "not_applicable", "nplus_deterministic_false_reject": 0,
                 "positive_deterministic_false_reject": 0,
                 "negative_combined_false_accept": 0, "negative_combined_denominator": 50,
                 "actual_independent_calls": 2, "parser_error_rows": 0, "truncated_rows": 0}
V7_REGISTERED_THRESHOLDS = {**V7_THRESHOLDS, "nplus_deterministic_denominator": 4, "planned_cases": 104}
_V7_BOOL_KEYS = ("has_limitation", "language_ok", "no_unannotated_attributes", "no_unsupported_relation")
_V7_PROTOCOLS = (V7_PROTOCOL, V7_A_PROTOCOL, V7_B_PROTOCOL)
_STRUCTURED_PROTOCOLS = (V5_PROTOCOL, V6_PROTOCOL, *_V7_PROTOCOLS)


def _is_v7(protocol):
    return protocol in _V7_PROTOCOLS
_FORMAT_NORMALIZATIONS = ("code_fence", "string_boolean")
_ARTIFACTS = {"CALL.json", "cases.json", "config.json", "backend_batches.jsonl", "outputs.jsonl", "PROBE.json"}


def _repository_root():
    return Path(__file__).absolute().parents[2]


def _check_gate():
    root = _repository_root()
    for name in ("CTO_GATE_OPEN", "CT0_GATE_OPEN"):
        if (root / name).exists() or (root / name).is_symlink():
            raise I.StageRefused("repository_gate", "local document/merge gate is open")
    git = root / ".git"
    if not git.exists():
        return "artifact-runtime"
    if git.is_file():
        value = git.read_text(encoding="utf-8").strip()
        if not value.startswith("gitdir: "):
            raise I.StageRefused("repository_gate", "unrecognised git directory locator")
        git = (root / value[8:]).absolute()
    if not git.is_dir():
        raise I.StageRefused("repository_gate", "git directory is unavailable")
    directories = [git]
    common = git / "commondir"
    if common.exists() or common.is_symlink():
        J._safe_path(common)
        I._plain_file(common)
        value = common.read_text(encoding="utf-8").strip()
        if not value:
            raise I.StageRefused("repository_gate", "empty common git directory locator")
        shared = J._safe_path((git / value).absolute())
        if not shared.is_dir():
            raise I.StageRefused("repository_gate", "common git directory is unavailable")
        directories.append(shared)
    for directory in directories:
        for name in ("CTO_GATE_OPEN", "CT0_GATE_OPEN", "MERGE_HEAD"):
            if (directory / name).exists() or (directory / name).is_symlink():
                raise I.StageRefused("repository_gate", "local document/merge gate is open")
    return "repository-runtime"


def _inputs(raw, *, expected_manifest_sha256=None, expected_protocol=None,
            expected_approval_scope=None, expected_source_sha256=None):
    data = I._json(raw)
    if isinstance(data, dict) and (data.get("schema_version") in (2, 3, 4) or "protocol" in data):
        return _structured_inputs(data, expected_registration={
            "expected_manifest_sha256": expected_manifest_sha256,
            "expected_protocol": expected_protocol,
            "expected_approval_scope": expected_approval_scope,
            "expected_source_sha256": expected_source_sha256})
    allowed = {"schema_version", "cases", "provenance", "thresholds", "model_discrimination_measured"}
    if (not isinstance(data, dict) or set(data) - allowed or
            type(data.get("schema_version")) is not int or data["schema_version"] != 1 or "cases" not in data):
        raise I.StageRefused("probe_cases", "expected schema_version=1 and cases")
    if "thresholds" in data and data["thresholds"] != _THRESHOLDS:
        raise I.StageRefused("probe_thresholds", "preregistered thresholds cannot be relaxed")
    if "model_discrimination_measured" in data and data["model_discrimination_measured"] is not False:
        raise I.StageRefused("probe_cases", "a case fixture cannot claim measured discrimination")
    if "provenance" in data:
        provenance = data["provenance"]
        if (not isinstance(provenance, dict) or provenance.get("real_images_or_ids_used") is not False or
                provenance.get("expectations_fixed_before_model_execution") is not True or
                any(not I._HEX64.fullmatch(str(provenance.get(key, "")))
                    for key in ("independent_nplus_sha256", "builder_sha256"))):
            raise I.StageRefused("probe_provenance", "invalid synthetic fixture provenance")
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != 500:
        raise I.StageRefused("probe_cases", "exactly 500 preregistered cases are required")
    ids, groups = set(), Counter()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
                "id", "group", "task", "candidate", "expected_model_accept", "expected_deterministic_accept"}:
            raise I.StageRefused("probe_case", "unexpected case fields")
        cid, group, task = case["id"], case["group"], case["task"]
        if not isinstance(cid, str) or not cid or cid in ids:
            raise I.StageRefused("probe_case_id", "empty or duplicate case id")
        ids.add(cid)
        if group not in GROUPS:
            raise I.StageRefused("probe_group", "unrecognised preregistered group")
        if not isinstance(task, dict) or task.get("kind") != "image_report":
            raise I.StageRefused("probe_task", "the fixed ten groups are synthetic image reports")
        if not isinstance(task.get("evidence"), dict) or not isinstance(task.get("prompt"), str):
            raise I.StageRefused("probe_task", "missing evidence or request")
        if not isinstance(task.get("source_ids"), list) or not all(isinstance(s, str) for s in task["source_ids"]):
            raise I.StageRefused("probe_task", "invalid synthetic source IDs")
        try:
            row_class = P.row_class(task)
        except (ValueError, TypeError) as exc:
            raise I.StageRefused("probe_task", str(exc)) from exc
        expected_class = "defect_free" if group.startswith("N") else "defect"
        if row_class != expected_class:
            raise I.StageRefused("probe_group", "group and locked row class differ")
        if not isinstance(case["candidate"], str) or not case["candidate"].strip():
            raise I.StageRefused("probe_candidate", "candidate must be an unchanged nonempty string")
        for name in ("expected_model_accept", "expected_deterministic_accept"):
            if type(case[name]) is not bool:
                raise I.StageRefused("probe_expectation", "expected decisions must be booleans")
        if case["expected_model_accept"] != (group in _POSITIVE):
            raise I.StageRefused("probe_expectation", "positive/negative model expectation differs from group")
        if group in _POSITIVE and case["expected_deterministic_accept"] is not True:
            raise I.StageRefused("probe_expectation", "positive groups must expect deterministic acceptance")
        groups[group] += 1
    if set(groups) != set(GROUPS) or set(groups.values()) != {50}:
        raise I.StageRefused("probe_groups", "ten groups of exactly 50 including N+ and D+ are required")
    return cases


def _registration(provenance, protocol, expected_registration):
    """Validate externally supplied registration before decoding source content."""
    if protocol not in V7_REGISTERED_SCOPES:
        if expected_registration and any(value is not None for value in expected_registration.values()):
            raise I.StageRefused("probe_registration", "external registration cannot relabel an old protocol")
        return None
    record = provenance.get("registration") if isinstance(provenance, dict) else None
    keys = {"expected_manifest_sha256", "expected_protocol", "expected_approval_scope", "expected_source_sha256"}
    if (not isinstance(record, dict) or set(record) != keys | {"manifest_base64"} or
            not isinstance(expected_registration, dict) or set(expected_registration) != keys or
            any(expected_registration[key] is None or record[key] != expected_registration[key] for key in keys) or
            expected_registration["expected_protocol"] != protocol):
        raise I.StageRefused("probe_registration", "external manifest, scope, protocol and four source pins are required")
    try:
        if not isinstance(record["manifest_base64"], str):
            raise ValueError("manifest encoding must be text")
        raw = base64.b64decode(record["manifest_base64"], validate=True)
        return validate_v7_manifest(raw, **expected_registration)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise I.StageRefused("probe_registration", "registered manifest does not match external pins") from exc


def _registration_pins(data):
    """Replay pins already admitted by run_probe; not a public input trust anchor."""
    if data.get("protocol") not in V7_REGISTERED_SCOPES:
        return {}
    record = data.get("provenance", {}).get("registration", {})
    return {key: record.get(key) for key in (
        "expected_manifest_sha256", "expected_protocol", "expected_approval_scope", "expected_source_sha256")}


def _structured_inputs(data, *, expected_registration=None):
    protocol = data.get("protocol")
    versioned = protocol in (V6_PROTOCOL, *_V7_PROTOCOLS)
    registered = protocol in V7_REGISTERED_SCOPES
    group_counts = _group_counts(protocol)
    manifest = _registration(data.get("provenance"), protocol, expected_registration)
    pins = manifest["source_sha256"] if manifest else _source_pins(protocol)
    required = {"schema_version", "protocol", "cases", "provenance", "model_discrimination_measured"}
    if (set(data) not in (required, required | {"thresholds"}) or
            type(data.get("schema_version")) is not int or
            data["schema_version"] != (5 if registered else 4 if _is_v7(protocol) else 3 if versioned else 2) or
            protocol not in _STRUCTURED_PROTOCOLS or data.get("model_discrimination_measured") is not False):
        raise I.StageRefused("probe_protocol_contract", "explicit versioned 150-case protocol required")
    if "thresholds" in data and I._digest(data["thresholds"]) != I._digest(_thresholds(protocol)):
        raise I.StageRefused("probe_thresholds", "structured thresholds cannot be replaced or relaxed")
    provenance = data["provenance"]
    keys = {"source_sha256", "adapter_sha256", "label_map_sha256", "label_map_loader_sha256",
            "label_names", "source_metadata", "real_images_or_ids_used",
            "expectations_fixed_before_model_execution", "statistical_generalization_claimed",
            "deterministic_results_are_not_model_approval", "source_files_base64"}
    if versioned:
        keys.add("evaluation_scope")
    if registered:
        keys.add("registration")
    if (not isinstance(provenance, dict) or set(provenance) != keys or
            provenance["source_sha256"] != pins or
            provenance["real_images_or_ids_used"] is not False or
            provenance["expectations_fixed_before_model_execution"] is not True or
            provenance["statistical_generalization_claimed"] is not False or
            provenance["deterministic_results_are_not_model_approval"] is not True or
            any(not isinstance(provenance[k], str) or not I._HEX64.fullmatch(provenance[k])
                for k in ("adapter_sha256", "label_map_sha256", "label_map_loader_sha256")) or
            not isinstance(provenance["label_names"], dict) or not provenance["label_names"] or
            not all(isinstance(k, str) and k and isinstance(v, str) and v
                    for k, v in provenance["label_names"].items()) or
            not isinstance(provenance["source_metadata"], dict) or
            set(provenance["source_metadata"]) != set(V5_GROUP_COUNTS) or
            not all(isinstance(v, dict) for v in provenance["source_metadata"].values())):
        raise I.StageRefused("probe_provenance", "structured probes require pinned source and adapter provenance")
    if versioned and I._digest(provenance["evaluation_scope"]) != I._digest(_evaluation_scope(protocol)):
        raise I.StageRefused("probe_evaluation_scope", "exposed G sources are not new independent samples")
    sources = _structured_sources(provenance, protocol, manifest=manifest)
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != sum(group_counts.values()):
        raise I.StageRefused("probe_cases", "structured probes require the exact registered source-preserving count")
    fields = {"id", "group", "task", "candidate", "expected_model_accept", "expected_deterministic_accept",
              "source_case", "source_case_sha256", "source_file_sha256", "source_group", "source_index"}
    if versioned:
        fields.add("protocol")
    ids, positions, counts = set(), {g: [] for g in group_counts}, Counter()
    for case in cases:
        if not isinstance(case, dict) or set(case) != fields:
            raise I.StageRefused("probe_case", "unexpected versioned case fields")
        if versioned and case["protocol"] != protocol:
            raise I.StageRefused("probe_protocol_contract", "mixed-version case cannot enter a versioned probe")
        cid, group, task, original = case["id"], case["group"], case["task"], case["source_case"]
        if not isinstance(cid, str) or not cid or cid in ids or group not in V5_GROUP_COUNTS:
            raise I.StageRefused("probe_case_id", "invalid structured group or duplicate case ID")
        ids.add(cid)
        if (not isinstance(original, dict) or original.get("id") != cid or
                case["candidate"] != original.get("text") or not isinstance(case["candidate"], str) or
                not case["candidate"].strip() or case["source_case_sha256"] != I._digest(original) or
                case["source_file_sha256"] != pins[group] or case["source_group"] != group or
                type(case["source_index"]) is not int or case["source_index"] < 0):
            raise I.StageRefused("probe_source_case", "source text, ID, position or hash was not preserved")
        index = case["source_index"]
        if (index >= len(sources[group]["cases"]) or I._digest(original) != I._digest(sources[group]["cases"][index])):
            raise I.StageRefused("probe_source_case", "case is not the pinned source member at its declared index")
        positive = group in _POSITIVE
        if (case["expected_model_accept"] is not positive or
                (case["expected_deterministic_accept"] is not True if positive
                 else case["expected_deterministic_accept"] is not None)):
            raise I.StageRefused("probe_expectation", "structured negatives are model-pending, not deterministic acceptance labels")
        if (not isinstance(task, dict) or task.get("kind") != "image_report" or
                not isinstance(task.get("prompt"), str) or not isinstance(task.get("evidence"), dict) or
                not isinstance(task.get("source_ids"), list) or
                not all(isinstance(s, str) for s in task["source_ids"])):
            raise I.StageRefused("probe_task", "invalid synthetic structured task")
        try:
            row_class = P.row_class(task)
        except (ValueError, TypeError) as exc:
            raise I.StageRefused("probe_task", str(exc)) from exc
        if (row_class not in {"defect_free", "defect"} or
                (group == "N+" and row_class != "defect_free") or
                (group == "D+" and row_class != "defect")):
            raise I.StageRefused("probe_group", "structured group and trusted task class differ")
        from .judge_cases import task_from_v5_source
        try:
            # Object-key order is lost when the portable case bundle is written
            # canonically. Ordered positions/defects were built from the pinned
            # original bytes, so reconstruct from that authenticated member.
            expected_task = task_from_v5_source(sources[group], sources[group]["cases"][index],
                                               label_names=provenance["label_names"])
        except (ValueError, TypeError, KeyError) as exc:
            raise I.StageRefused("probe_task", "pinned source cannot construct a trusted task") from exc
        if I._digest(task) != I._digest(expected_task):
            raise I.StageRefused("probe_task", "adapted task differs from the pinned source fields")
        positions[group].append(case["source_index"])
        counts[group] += 1
    if dict(counts) != group_counts or any(positions[g] != list(range(n)) for g, n in group_counts.items()):
        raise I.StageRefused("probe_groups", "structured probes require ordered registered group source indices")
    return cases


def _structured_sources(provenance, protocol, *, manifest=None):
    """Authenticate original bytes, membership and adapter dependencies portably."""
    from . import judge_cases as adapter
    from data import label_map
    paths = {"adapter_sha256": Path(adapter.__file__),
             "label_map_sha256": Path(label_map.DEFAULT_LABEL_MAP),
             "label_map_loader_sha256": Path(label_map.__file__)}
    for key, path in paths.items():
        J._safe_path(path)
        I._plain_file(path)
        if I._file_sha(path) != provenance[key]:
            raise I.StageRefused("probe_adapter_changed", "source adapter or label dependency changed")
    loaded = label_map.load_label_map(paths["label_map_sha256"])
    names = {value.name_ko: key for key, value in loaded.defect_types.items()}
    if len(names) != len(loaded.defect_types) or provenance["label_names"] != names:
        raise I.StageRefused("probe_label_names", "adapter label names differ from the pinned label map")
    encoded = provenance["source_files_base64"]
    if not isinstance(encoded, dict) or set(encoded) != set(V5_GROUP_COUNTS):
        raise I.StageRefused("probe_source_files", "all four original byte strings are required")
    sources = {}
    sets = {"N+": "judge_nplus_g1", "D+": "judge_dplus_g1",
            "ADV1": "judge_adv_negative_g1", "ADV2": "judge_adv_negative_g2"}
    if protocol in V7_REGISTERED_SCOPES:
        if manifest is None:
            raise I.StageRefused("probe_registration", "source verification requires authenticated registration")
        sets = manifest["source_sets"]
    for group, count in _group_counts(protocol).items():
        value = encoded[group]
        try:
            if not isinstance(value, str):
                raise ValueError("base64 must be text")
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise I.StageRefused("probe_source_files", "invalid source byte encoding") from exc
        pins = manifest["source_sha256"] if manifest else _source_pins(protocol)
        if hashlib.sha256(raw).hexdigest() != pins[group]:
            raise I.StageRefused("probe_source_pin", "source bytes differ from the pinned four-file G source")
        source = I._json(raw)
        if (not isinstance(source, dict) or source.get("set") != sets[group] or
                not isinstance(source.get("cases"), list) or len(source["cases"]) != count or
                I._digest(provenance["source_metadata"][group]) !=
                I._digest({k: v for k, v in source.items() if k != "cases"})):
            raise I.StageRefused("probe_source_files", "source metadata/set/count differs")
        positive = group in _POSITIVE
        if any(not isinstance(row, dict) or row.get("expected") != ("accept" if positive else "reject")
               for row in source["cases"]):
            raise I.StageRefused("probe_expectation", "source expectation cannot be relabelled")
        sources[group] = source
    return sources


def _source_dependencies_unchanged(raw):
    data = I._json(raw)
    if data.get("protocol") in _STRUCTURED_PROTOCOLS:
        # Includes a second raw-source SHA and loaded label-map check; no images,
        # production rows, models, mutable original paths or generated fixtures.
        manifest = _registration(data["provenance"], data["protocol"], _registration_pins(data))
        _structured_sources(data["provenance"], data["protocol"], manifest=manifest)


def _protocol(cases):
    # Called only after complete input validation, never inferred from model output.
    if cases and _is_v7(cases[0].get("protocol")) and all(
            c.get("protocol") == cases[0]["protocol"] for c in cases):
        return cases[0]["protocol"]
    if cases and all(c.get("protocol") == V6_PROTOCOL for c in cases):
        return V6_PROTOCOL
    return V5_PROTOCOL if len(cases) == 150 else LEGACY_PROTOCOL


def _thresholds(protocol):
    if protocol in V7_REGISTERED_SCOPES:
        return V7_REGISTERED_THRESHOLDS
    if _is_v7(protocol):
        return V7_THRESHOLDS
    return V6_THRESHOLDS if protocol == V6_PROTOCOL else V5_THRESHOLDS if protocol == V5_PROTOCOL else _THRESHOLDS


def _group_counts(protocol):
    return V7_REGISTERED_GROUP_COUNTS if protocol in V7_REGISTERED_SCOPES else V5_GROUP_COUNTS


def _parser_schema(protocol):
    if _is_v7(protocol):
        return "judge_structured_v7"
    return "judge_structured_v6" if protocol == V6_PROTOCOL else (
        "judge_structured_v5" if protocol == V5_PROTOCOL else "legacy_four_keys")


def _require_parser(protocol):
    if _is_v7(protocol):
        return J.require_v7_parser()
    if protocol == V6_PROTOCOL:
        return J.require_v6_parser()
    if protocol == V5_PROTOCOL:
        return J.require_v5_parser()
    return None


def _source_pins(protocol):
    if protocol in V7_REGISTERED_SCOPES:
        raise I.StageRefused("probe_registration", "registered source pins require authenticated manifest")
    return V7_SOURCE_SHA256 if protocol == V7_PROTOCOL else (
        V6_SOURCE_SHA256 if protocol == V6_PROTOCOL else V5_SOURCE_SHA256)


def _evaluation_scope(protocol):
    if protocol in V7_REGISTERED_SCOPES:
        return v7_registered_evaluation_scope(protocol)
    return V7_EVALUATION_SCOPE if protocol == V7_PROTOCOL else V6_EVALUATION_SCOPE


def _approval_scope(protocol):
    return V7_REGISTERED_SCOPES.get(protocol, "regression_diagnostic_only")


def _policy_for(protocol):
    if _is_v7(protocol):
        return J._policy_v7()
    return J._policy()


def _runtime_for(model, spec, protocol):
    if _is_v7(protocol):
        return J.judge_runtime_identity(model, spec, policy_version="v7")
    return J.judge_runtime_identity(model, spec)


def _model_cases(cases):
    if _is_v7(_protocol(cases)):
        return [case for case in cases if P.row_class(case["task"]) == "defect"]
    return cases


def _raw_check(rows, ids, spec, protocol):
    if _is_v7(protocol):
        return I._check_rows(rows, ids, spec, allow_json_closed=True)
    return I._check_rows(rows, ids, spec)


def _receipt_schema(protocol):
    return 3 if _is_v7(protocol) else 2 if protocol == V6_PROTOCOL else 1


def _settings(config):
    if not isinstance(config, dict):
        raise I.StageRefused("probe_config", "config must be an object")
    spec = J._clone(I._model_config(config, "judge"))
    generation = I._model_config(config, "generate")
    if spec["model_id"] == generation["model_id"] or I._family(spec["family"]) == I._family(generation["family"]):
        raise I.StageRefused("self_judging", "probe judge must be cross-family")
    memory_limit = spec.get("max_memory_reserved_bytes")
    if memory_limit is not None:
        I._positive_int(memory_limit, "judge.max_memory_reserved_bytes")
    return spec, generation, memory_limit


def _code_pins():
    return {"judge_probe.py": I._file_sha(Path(__file__)),
            "judge_identity.py": I._file_sha(Path(J.__file__)),
            "infer.py": I._file_sha(Path(I.__file__))}


def _render(cases):
    if _is_v7(_protocol(cases)):
        return [P.render_judge_v7(c["task"], {"text": c["candidate"], "finish_reason": "eos"})
                for c in _model_cases(cases)]
    return [P.render_judge(c["task"], {"text": c["candidate"], "finish_reason": "eos"}) for c in cases]


def check_rules(cases_path, *, expected_manifest_sha256=None, expected_protocol=None,
                expected_approval_scope=None, expected_source_sha256=None) -> dict:
    """Read-only CPU discrimination check; no backend, model or output file."""
    _check_gate()
    path = J._safe_path(Path(cases_path))
    I._plain_file(path)
    raw = path.read_bytes()
    cases = _inputs(raw, expected_manifest_sha256=expected_manifest_sha256,
                    expected_protocol=expected_protocol, expected_approval_scope=expected_approval_scope,
                    expected_source_sha256=expected_source_sha256)
    protocol = _protocol(cases)
    if _is_v7(protocol):
        return _check_rules_v7(path, raw, cases)
    structured = protocol in _STRUCTURED_PROTOCOLS
    _, system, policy = J._policy()
    code = _code_pins()
    groups = {g: {"cases": 0, "false_accept": None if structured and g not in _POSITIVE else 0,
                  "false_reject": 0, "expected_deterministic_accept": 0,
                  **({"negative_detected": 0, "model_pending": 0} if structured else {})}
              for g in (V5_GROUP_COUNTS if structured else GROUPS)}
    flags = Counter()
    rows = []
    for case in cases:
        reasons = P.deterministic_reasons(case["task"], case["candidate"])
        actual, expected = not reasons, case["expected_deterministic_accept"]
        group = groups[case["group"]]
        group["cases"] += 1
        group["expected_deterministic_accept"] += expected is True
        if expected is None:
            group["negative_detected"] += not actual
            group["model_pending"] += actual
        else:
            group["false_accept"] += actual and not expected
            group["false_reject"] += expected and not actual
        row_flags = P.deterministic_flags(case["task"], case["candidate"])
        flags.update(row_flags)
        rows.append({"id": case["id"], "group": case["group"],
                     "expected_deterministic_accept": expected,
                     "deterministic_accept": actual, "reasons": reasons, "flags": row_flags,
                     "case_sha256": hashlib.sha256(I._canonical(case)).hexdigest(),
                     **({"deterministic_status": ("model_pending" if actual else "negative_detected")
                         if expected is None else ("not_rejected" if actual else "unexpected_rejection")}
                        if structured else {})})
    if (I._file_sha(path) != hashlib.sha256(raw).hexdigest() or _code_pins() != code or
            J._policy()[1:] != (system, policy)):
        raise I.StageRefused("probe_inputs_changed", "case/policy/source changed during rule check")
    _source_dependencies_unchanged(raw)
    _check_gate()
    matches = (all(groups[g]["false_reject"] == 0 for g in _POSITIVE) if structured else
               all(g["false_accept"] == g["false_reject"] == 0 for g in groups.values()))
    return {"schema_version": 1, "cases": len(cases), "groups": groups, "protocol": protocol,
            "rows_recorded": len(rows), "rows": rows,
            "exit_code": 0 if matches else 2,
            "cases_file_sha256": hashlib.sha256(raw).hexdigest(), "policy_identity": policy,
            "implementation_sha256": code, "deterministic_flags": dict(sorted(flags.items())),
            "nplus_deterministic_gate_passed": groups["N+"]["false_reject"] == 0,
            "positive_deterministic_gate_passed": all(groups[g]["false_reject"] == 0 for g in _POSITIVE),
            "deterministic_expectations_match": matches,
            "negative_non_rejection_is_model_pending": structured,
            **({"evaluation_scope": V6_EVALUATION_SCOPE} if protocol == V6_PROTOCOL else {}),
            "approval_eligible": False,
            "semantic_negative_expected_deterministic_accept_is_not_a_rule_error": True,
            "model_discrimination_measured": False, "combined_discrimination_measured": False,
            "determinism_measured": False, "gpu_model_verified": False,
            "ready_for_training": False, "population_error_bound": None}


def _write_new(path, raw):
    temp = path.with_name(path.name + ".tmp")
    if path.exists():
        raise I.StageRefused("output_exists", str(path))
    try:
        with temp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise I.StageRefused("partial_or_unknown_output", str(temp)) from exc
    if path.exists():
        raise I.StageRefused("output_exists", str(path))
    os.replace(temp, path)


def _analysis(case, raw):
    if _is_v7(case.get("protocol")):
        return _analysis_v7(case, raw)
    parser_flags = []
    normalized = []
    v6 = case.get("protocol") == V6_PROTOCOL
    if "source_case" in case:
        parsed = _require_parser(V6_PROTOCOL if v6 else V5_PROTOCOL)(case["task"], raw)
        if (not isinstance(parsed, dict) or not {"errors", "rejected", "verdict", "flags"} <= set(parsed) or
                any(not isinstance(parsed[k], list) or not all(isinstance(v, str) for v in parsed[k])
                    for k in ("errors", "rejected", "flags"))):
            raise I.StageRefused("parser_contract", "structured parser must return errors/rejected/verdict/flags")
        errors, rejected, verdict, parser_flags = (parsed[k] for k in ("errors", "rejected", "verdict", "flags"))
        if v6:
            normalized = parsed.get("format_normalized")
            if (not isinstance(normalized, list) or
                    any(not isinstance(v, str) or v not in _FORMAT_NORMALIZATIONS for v in normalized) or
                    normalized != [v for v in _FORMAT_NORMALIZATIONS if v in normalized]):
                raise I.StageRefused("parser_contract", "v6 requires an ordered unique format_normalized list")
        if verdict is None and not errors:
            raise I.StageRefused("parser_contract", "missing derived verdict must carry a parser error")
        if verdict is not None and (not isinstance(verdict, dict) or
                set(verdict) != {"grounded", "scope_ok", "language_ok", "reason"} or
                any(type(verdict[k]) is not bool for k in ("grounded", "scope_ok", "language_ok")) or
                not isinstance(verdict["reason"], str)):
            raise I.StageRefused("parser_contract", "structured parser must expose strict derived verdict values")
        # A parser's empty rejection list alone cannot approve false derived values.
        if verdict is not None:
            rejected = sorted(set(rejected) | {k for k in ("grounded", "scope_ok", "language_ok") if not verdict[k]})
    else:
        errors, rejected, verdict = Q._judge(raw)
    reasons = P.deterministic_reasons(case["task"], case["candidate"])
    flags = P.deterministic_flags(case["task"], case["candidate"])
    model_accept = not errors and not rejected and verdict is not None
    deterministic_accept = not reasons
    return {"id": case["id"], "group": case["group"], "raw": raw,
            "raw_sha256": I._digest(raw), "case_sha256": I._digest(case),
            "model_errors": errors, "model_rejected": rejected,
            "deterministic_reasons": reasons, "deterministic_flags": flags,
            "model_accept": model_accept, "deterministic_accept": deterministic_accept,
            "combined_accept": model_accept and deterministic_accept,
            **({"model_flags": parser_flags, "derived_verdict": verdict,
                "deterministic_status": ("model_pending" if deterministic_accept else "negative_detected")
                if case["expected_deterministic_accept"] is None else
                ("not_rejected" if deterministic_accept else "unexpected_rejection")}
               if "source_case" in case else {}),
            **({"format_normalized": list(normalized), "format_normalized_any": bool(normalized)} if v6 else {})}


def _summary(cases, rows, status):
    protocol = _protocol(cases)
    if _is_v7(protocol):
        return _summary_v7(cases, rows, status)
    v5 = protocol in _STRUCTURED_PROTOCOLS
    v6 = protocol == V6_PROTOCOL
    planned = Counter(c["group"] for c in cases)
    groups = {}
    for group in sorted({c["group"] for c in cases}):
        groups[group] = {"planned": planned[group], "completed": 0,
                         **{name: {"false_accept": None if v5 and name == "deterministic" and group not in _POSITIVE else 0,
                                   "false_reject": 0}
                            for name in ("model", "deterministic", "combined")}}
        if v5:
            groups[group]["deterministic"].update(negative_detected=0, model_pending=0)
        if v6:
            groups[group]["format_normalized_rows"] = 0
    parser_errors = truncated = 0
    flags, model_flags = Counter(), Counter()
    normalized_rows, normalization_counts = 0, Counter()
    for case, row in zip(cases, rows):
        group = groups[case["group"]]
        group["completed"] += 1
        parser_errors += bool(row["model_errors"])
        truncated += row["raw"]["finish_reason"] != "eos"
        flags.update(row["deterministic_flags"])
        model_flags.update(row.get("model_flags", []))
        if v6:
            normalized = row["format_normalized"]
            normalization_counts.update(normalized)
            normalized_rows += bool(normalized)
            group["format_normalized_rows"] += bool(normalized)
        for name in ("model", "deterministic", "combined"):
            expected = case["expected_deterministic_accept"] if name == "deterministic" else case["expected_model_accept"]
            actual = row[name + "_accept"]
            if expected is None:
                group[name]["negative_detected"] += not actual
                group[name]["model_pending"] += actual
            else:
                group[name]["false_accept"] += actual and not expected
                group[name]["false_reject"] += expected and not actual
    complete = status == "complete" and len(rows) == len(cases)
    thresholds = (complete and parser_errors == 0 and truncated == 0 and
                  all(groups[g]["deterministic"]["false_reject"] == 0 and
                      groups[g]["model"]["false_reject"] <= 2 for g in _POSITIVE) and
                  all(groups[g]["combined"]["false_accept"] == 0 for g in ("ADV1", "ADV2"))) if v5 else (
                  complete and parser_errors == 0 and truncated == 0 and
                  groups["N+"]["deterministic"]["false_reject"] == 0 and
                  all(g["deterministic"]["false_accept"] == 0 for g in groups.values()) and
                  all(groups[g]["combined"]["false_reject"] <= 2 for g in _POSITIVE) and
                  all(v["model"]["false_accept"] == 0 and v["combined"]["false_accept"] == 0
                      for g, v in groups.items() if g not in _POSITIVE))
    return {"planned_cases": len(cases), "completed_cases": len(rows), "complete": complete,
            "protocol": protocol, "thresholds": _thresholds(protocol),
            "approval_eligible": v6,
            "parser_error_rows": parser_errors, "truncated_rows": truncated,
            "groups": groups, "deterministic_flags": dict(sorted(flags.items())),
            "model_flags": dict(sorted(model_flags.items())),
            **({"format_normalized_rows": normalized_rows,
                "format_normalizations": dict(sorted(normalization_counts.items())),
                "evaluation_scope": V6_EVALUATION_SCOPE} if v6 else {}),
            "thresholds_passed": bool(thresholds),
            "negative_non_rejection_is_model_pending": v5,
            "deterministic_false_accept_zero_not_required_for_ungrounded_groups": True,
            "single_call_is_not_determinism_verification": True}


def _v7_evaluations(case):
    # The exposed G150 predates the generation contract. Keep both modes; the
    # regression gate must not silently relabel old positives or discard failures.
    task, text = case["task"], case["candidate"]
    regression = P.evaluate_v7(task, text, generation_contract=False)
    full = P.evaluate_v7(task, text, generation_contract=True)
    return regression, full


def _analysis_v7(case, raw):
    regression, full = _v7_evaluations(case)
    normal = P.row_class(case["task"]) == "defect_free"
    # D 900657c: normal rows have no model fallback. Their operational gate
    # must enforce the strict production contract, not only the exposed-set
    # diagnostic below. Do not rewrite the pinned old N+ prose to make it pass.
    registered = case.get("protocol") in V7_REGISTERED_SCOPES
    gate = full if normal or registered else regression
    det = not gate["deterministic_reasons"]
    base = {"id": case["id"], "group": case["group"], "case_sha256": I._digest(case),
            "row_class": P.row_class(case["task"]),
            "deterministic_codes": gate["deterministic_codes"],
            "deterministic_reasons": gate["deterministic_reasons"],
            "deterministic_flags": gate["flags"], "deterministic_accept": det,
            "generation_contract_applied_to_probe_gate": normal or registered,
            "full_generation_contract": full,
            "full_generation_contract_accept": not full["deterministic_reasons"]}
    if normal:
        if raw is not None:
            raise I.StageRefused("normal_model_call_forbidden", "normal v7 rows cannot carry fabricated model output")
        return {**base, "raw": None, "raw_sha256": None, "parser": None, "model_call_id": None,
                "model_applicability": "not_applicable", "model_accept": None,
                "model_errors": [], "model_rejected": [], "model_flags": [],
                "derived_verdict": None, "format_normalized": [], "format_normalized_any": False,
                "model_provenance": "deterministic_only_no_model", "combined_accept": det}
    if not isinstance(raw, dict):
        raise I.StageRefused("probe_rows", "defect v7 rows require preserved model output")
    # No parser or policy helper receives a mutable alias to the saved response.
    parsed = _require_parser(V7_PROTOCOL)(J._clone(case["task"]), J._clone(raw))
    if (not isinstance(parsed, dict) or
            not {"errors", "rejected", "verdict", "flags", "format_normalized"} <= set(parsed) or
            any(not isinstance(parsed[k], list) or not all(isinstance(v, str) for v in parsed[k])
                for k in ("errors", "rejected", "flags", "format_normalized"))):
        raise I.StageRefused("parser_contract", "v7 requires B errors/rejected/verdict/flags/normalization")
    normalized = parsed["format_normalized"]
    if normalized != [key for key in _FORMAT_NORMALIZATIONS if key in normalized]:
        raise I.StageRefused("parser_contract", "invalid v7 format normalization metadata")
    verdict = parsed["verdict"]
    if verdict is None and not parsed["errors"]:
        raise I.StageRefused("parser_contract", "missing v7 verdict must carry a parser error")
    if verdict is not None and (not isinstance(verdict, dict) or set(verdict) != set(_V7_BOOL_KEYS) | {"reason"}
                               or any(type(verdict[key]) is not bool for key in _V7_BOOL_KEYS)
                               or not isinstance(verdict["reason"], str) or not verdict["reason"].strip()):
        raise I.StageRefused("parser_contract", "v7 verdict must contain four strict booleans and reason")
    rejected = sorted(set(parsed["rejected"]) | (
        {"judge_" + key + "_false" for key in _V7_BOOL_KEYS if verdict[key] is not True}
        if verdict is not None else set()))
    accept = verdict is not None and not parsed["errors"] and not rejected
    return {**base, "raw": raw, "raw_sha256": I._digest(raw), "parser": parsed,
            "model_applicability": "applicable", "model_accept": accept,
            "model_errors": parsed["errors"], "model_rejected": rejected,
            "model_flags": parsed["flags"], "derived_verdict": verdict,
            "format_normalized": list(normalized), "format_normalized_any": bool(normalized),
            "model_provenance": "preserved_backend_output", "combined_accept": det and accept}


def _summary_v7(cases, rows, status):
    protocol = _protocol(cases)
    registered = protocol in V7_REGISTERED_SCOPES
    planned = Counter(c["group"] for c in cases)
    if dict(planned) != _group_counts(protocol):
        raise I.StageRefused("probe_groups", "v7 denominators must equal the fixed protocol registry")
    case_by_id = {c["id"]: c for c in cases}
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or any(cid not in case_by_id for cid in ids):
        raise I.StageRefused("probe_rows", "v7 summary requires distinct authenticated input IDs")
    groups = {g: {"planned": count, "completed": 0,
                  "model": {"applicable": 0, "not_applicable": 0, "false_accept": 0, "false_reject": 0},
                  "deterministic": {"false_accept": None if g not in _POSITIVE else 0,
                                    "false_reject": 0, "negative_detected": 0, "model_pending": 0},
                  "combined": {"false_accept": 0, "false_reject": 0},
                  "full_generation_contract_rejected": 0}
              for g, count in planned.items()}
    parser_errors = truncated = normalized_rows = 0
    flags, model_flags, norms = Counter(), Counter(), Counter()
    for row in rows:
        case = case_by_id[row["id"]]
        if row["group"] != case["group"]:
            raise I.StageRefused("probe_rows", "summary group differs from source")
        group = groups[case["group"]]
        group["completed"] += 1
        normal = P.row_class(case["task"]) == "defect_free"
        if normal:
            if row["raw"] is not None or row["model_accept"] is not None or row["model_applicability"] != "not_applicable":
                raise I.StageRefused("normal_model_call_forbidden", "normal summary cannot claim a model result")
            group["model"]["not_applicable"] += 1
        else:
            group["model"]["applicable"] += 1
            parser_errors += bool(row["model_errors"])
            truncated += row["raw"]["finish_reason"] not in ("eos", "json_closed")
            group["model"]["false_reject"] += case["expected_model_accept"] and not row["model_accept"]
            group["model"]["false_accept"] += not case["expected_model_accept"] and row["model_accept"]
        expected = case["expected_deterministic_accept"]
        if expected is None:
            group["deterministic"]["negative_detected"] += not row["deterministic_accept"]
            group["deterministic"]["model_pending"] += row["deterministic_accept"] and not normal
        else:
            group["deterministic"]["false_reject"] += expected and not row["deterministic_accept"]
        group["combined"]["false_reject"] += case["expected_model_accept"] and not row["combined_accept"]
        group["combined"]["false_accept"] += not case["expected_model_accept"] and row["combined_accept"]
        group["full_generation_contract_rejected"] += not row["full_generation_contract_accept"]
        flags.update(row["deterministic_flags"])
        model_flags.update(row["model_flags"])
        norms.update(row["format_normalized"])
        normalized_rows += row["format_normalized_any"]
    # A normal-only group's model rate has no denominator, not a zero numerator.
    for name, group in groups.items():
        planned_model = sum(c["group"] == name and P.row_class(c["task"]) == "defect" for c in cases)
        group["model"]["planned"] = planned_model
        if planned_model == 0:
            group["model"].update(false_accept=None, false_reject=None, status="not_applicable")
    complete = status == "complete" and len(rows) == len(cases)
    dplus = groups["D+"]["model"]["false_reject"]
    negative = sum(groups[g]["combined"]["false_accept"] for g in ("ADV1", "ADV2"))
    positive_det = sum(groups[g]["deterministic"]["false_reject"] for g in _POSITIVE)
    thresholds = (complete and parser_errors == 0 and truncated == 0 and positive_det == 0 and
                  dplus <= 2 and negative == 0 and groups["D+"]["model"]["applicable"] == 50)
    return {"planned_cases": len(cases), "completed_cases": len(rows), "complete": complete,
            "protocol": protocol, "thresholds": _thresholds(protocol), "approval_eligible": registered,
            "approval_scope": _approval_scope(protocol), "final_blinded_approval": False,
            "evaluation_scope": _evaluation_scope(protocol), "groups": groups,
            "nplus_model_metric": "not_applicable", "dplus_model_denominator": 50,
            "nplus_deterministic_denominator": planned["N+"],
            "dplus_model_false_reject": dplus, "negative_combined_denominator": 50,
            "negative_combined_false_accept": negative,
            "positive_deterministic_false_reject": positive_det,
            "model_target_rows": len(_model_cases(cases)),
            "completed_model_rows": sum(r["raw"] is not None for r in rows),
            "normal_model_calls": 0,
            "parser_error_rows": parser_errors, "truncated_rows": truncated,
            "deterministic_flags": dict(sorted(flags.items())), "model_flags": dict(sorted(model_flags.items())),
            "format_normalized_rows": normalized_rows, "format_normalizations": dict(sorted(norms.items())),
            "generation_contract_applied_to_probe_gate": registered,
            "normal_generation_contract_applied_to_probe_gate": True,
            "normal_positive_contract_rejected": sum(
                bool(r["full_generation_contract"]["deterministic_codes"]["contract"])
                for r in rows if r["group"] == "N+"),
            "full_generation_contract_rejected": sum(g["full_generation_contract_rejected"] for g in groups.values()),
            "positive_full_generation_contract_rejected": sum(groups[g]["full_generation_contract_rejected"] for g in _POSITIVE),
            "full_generation_contract_passed": complete and not any(g["full_generation_contract_rejected"] for g in groups.values()),
            "thresholds_passed": bool(thresholds), "single_call_is_not_determinism_verification": True,
            "exposed_adjustment_set_cannot_mint_final_approval": protocol != V7_B_PROTOCOL}


def _check_rules_v7(path, raw, cases):
    protocol = _protocol(cases)
    registered = protocol in V7_REGISTERED_SCOPES
    _, system, policy = _policy_for(protocol)
    code = _code_pins()
    rows, positive_rejects, normal_false_accepts = [], 0, 0
    for case in cases:
        regression, full = _v7_evaluations(case)
        normal = P.row_class(case["task"]) == "defect_free"
        gate = full if registered else regression
        accept = not gate["deterministic_reasons"]
        positive_rejects += case["group"] in _POSITIVE and not accept
        normal_false_accepts += case["group"] not in _POSITIVE and normal and accept
        rows.append({"id": case["id"], "group": case["group"], "case_sha256": I._digest(case),
                     "regression": regression, "full_generation_contract": full,
                     "model_applicability": "not_applicable" if normal else "applicable",
                     "deterministic_only_negative_false_accept": case["group"] not in _POSITIVE and normal and accept})
    if (I._file_sha(path) != hashlib.sha256(raw).hexdigest() or _code_pins() != code or
            _policy_for(protocol)[1:] != (system, policy)):
        raise I.StageRefused("probe_inputs_changed", "v7 case/policy/source changed during rule check")
    _source_dependencies_unchanged(raw)
    _check_gate()
    return {"schema_version": 2, "protocol": protocol, "cases": len(cases), "rows": rows,
            "rows_recorded": len(rows), "exit_code": 0 if positive_rejects == normal_false_accepts == 0 else 2,
            "policy_identity": policy, "implementation_sha256": code,
            "cases_file_sha256": hashlib.sha256(raw).hexdigest(), "evaluation_scope": _evaluation_scope(protocol),
            "positive_deterministic_false_reject": positive_rejects,
            "normal_negative_deterministic_false_accept": normal_false_accepts,
            "normal_production_positive_rejected": sum(
                not r["full_generation_contract"]["deterministic_pass"]
                for r in rows if r["group"] == "N+"),
            "normal_production_negative_false_accept": sum(
                r["full_generation_contract"]["deterministic_pass"]
                for r in rows if r["group"] not in _POSITIVE
                and r["model_applicability"] == "not_applicable"),
            "scope": ("registered strict generation contract for every row" if registered else
                      "exposed-set regression without contract; strict normal production counted separately"),
            "registration": _registration_pins(I._json(raw)),
            "positive_full_generation_contract_rejected": sum(not r["full_generation_contract"]["deterministic_pass"]
                                                               for r in rows if r["group"] in _POSITIVE),
            "model_discrimination_measured": False, "combined_discrimination_measured": False,
            "determinism_measured": False, "gpu_model_verified": False, "approval_eligible": False,
            "ready_for_training": False}


def run_probe(cases_path, config: dict, output_dir, *, backend_factory=None,
              expected_manifest_sha256=None, expected_protocol=None,
              expected_approval_scope=None, expected_source_sha256=None) -> dict:
    """One fresh call. Only separately registered A/B inputs are approval-eligible."""
    environment = _check_gate()
    source, target = J._safe_path(Path(cases_path)), J._safe_path(Path(output_dir))
    I._plain_file(source)
    if target.exists() or target.is_symlink():
        raise I.StageRefused("probe_output_exists", "each probe requires a new exclusive directory")
    config = J._clone(config)
    spec, generation, memory_limit = _settings(config)
    raw_cases = source.read_bytes()
    registration = {"expected_manifest_sha256": expected_manifest_sha256,
                    "expected_protocol": expected_protocol,
                    "expected_approval_scope": expected_approval_scope,
                    "expected_source_sha256": expected_source_sha256}
    cases = _inputs(raw_cases, **registration)
    protocol = _protocol(cases)
    if protocol == V7_PROTOCOL and backend_factory is None:
        raise I.StageRefused("diagnostic_only_protocol", "old G150 is CPU/injected-backend regression only")
    _require_parser(protocol)  # Before output creation or backend/model loading.
    if protocol in V7_REGISTERED_SCOPES:
        for case in cases:
            _, full = _v7_evaluations(case)
            accepted = not full["deterministic_reasons"]
            if ((case["group"] in _POSITIVE and not accepted) or
                    (case["group"] not in _POSITIVE and P.row_class(case["task"]) == "defect_free" and accepted)):
                raise I.StageRefused("probe_cpu_admission", "registered strict-contract CPU gate failed")
    if _is_v7(protocol) and spec["max_new_tokens"] != 160:
        raise I.StageRefused("probe_config", "v7 requires the preregistered max_new_tokens=160")
    source_sha = hashlib.sha256(raw_cases).hexdigest()
    policy, system, identity = _policy_for(protocol)
    model_cases = _model_cases(cases)
    prompts = _render(cases)
    render_sha = I._digest([{"system": system, "user": p} for p in prompts])
    code = _code_pins()
    input_basis = {"cases_file_sha256": source_sha, "config_sha256": I._digest(config),
                   "rendered_input_sha256": render_sha, "policy_identity": identity,
                   "implementation_sha256": code, "protocol": protocol,
                   "thresholds": _thresholds(protocol),
                   "case_provenance_sha256": I._digest(I._json(raw_cases).get("provenance")),
                   "parser_schema_version": _parser_schema(protocol),
                   **({"registration": J._clone(registration)} if protocol in V7_REGISTERED_SCOPES else {}),
                   **({"evaluation_scope": _evaluation_scope(protocol)}
                      if protocol in (V6_PROTOCOL, *_V7_PROTOCOLS) else {})}

    def unchanged(model=None, runtime=None):
        _check_gate()
        _require_parser(protocol)
        J._safe_path(source)
        J._safe_path(target)
        if I._file_sha(source) != source_sha or _code_pins() != code:
            raise I.StageRefused("probe_inputs_changed", "input/source bytes changed")
        _source_dependencies_unchanged(raw_cases)
        _, current_system, current_policy = _policy_for(protocol)
        if current_system != system or current_policy != identity:
            raise I.StageRefused("probe_policy_changed", "policy changed during invocation")
        if model is not None and _runtime_for(model, spec, protocol) != runtime:
            raise I.StageRefused("runtime_changed", "loaded defaults/tokenizer/runtime changed")

    unchanged()
    target.mkdir(parents=True, exist_ok=False)
    call = {"call_id": uuid.uuid4().hex, "started_at_utc": datetime.now(UTC).isoformat(),
            "environment": environment, "backend_injected": backend_factory is not None,
            "gpu_exclusivity": "external-supervisor-required"}
    _write_new(target / "CALL.json", I._canonical(call) + b"\n")
    _write_new(target / "cases.json", raw_cases)
    _write_new(target / "config.json", I._canonical(config) + b"\n")
    log_path = target / "backend_batches.jsonl"
    _write_new(log_path, b"")
    log_hash = hashlib.sha256()

    def log_event(event):
        J._safe_path(log_path)
        I._plain_file(log_path)
        if I._file_sha(log_path) != log_hash.hexdigest():
            raise I.StageRefused("probe_log_changed", "previous raw backend events changed")
        line = I._canonical({"call_id": call["call_id"], **event}) + b"\n"
        with log_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        log_hash.update(line)
        if I._file_sha(log_path) != log_hash.hexdigest():
            raise I.StageRefused("probe_log_changed", "raw event append could not be verified")
    started = time.monotonic()
    deadline = started + spec["wall_seconds"]
    model, runtime, rows, batches = None, None, [], 0
    normal_rows = ([_analysis(c, None) for c in cases if P.row_class(c["task"]) == "defect_free"]
                   if _is_v7(protocol) else [])
    status, exit_code = "complete", 0
    memory = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    try:
        unchanged()
        model = (backend_factory or I.TransformersBackend)(dict(spec), "judge")
        if _is_v7(protocol):
            enable = getattr(model, "enable_json_object_stopping", None)
            if not callable(enable):
                raise I.StageRefused("probe_json_stop", "v7 backend must explicitly enable JSON object stopping")
            enable()
        runtime = _runtime_for(model, spec, protocol)
        if I._family(runtime["model_type"]) == I._family(generation["family"]):
            raise I.StageRefused("self_judging", "actual judge architecture matches configured generator")
        unchanged(model, runtime)
        for offset in range(0, len(model_cases), spec["batch_size"]):
            unchanged(model, runtime)
            if time.monotonic() >= deadline:
                raise I.StageTimeLimit()
            batch_cases = model_cases[offset:offset + spec["batch_size"]]
            output = model.generate_batch(prompts[offset:offset + spec["batch_size"]], system, deadline)
            batches += 1
            # Preserve returned bytes/values before shape checks, strict parsing,
            # policy analysis, deadline checks, or another backend invocation.
            try:
                log_event({"event": "batch_returned", "batch_index": batches - 1,
                           "offset": offset, "case_ids": [c["id"] for c in batch_cases],
                           "output": output})
            except (TypeError, ValueError):
                log_event({"event": "batch_unserializable", "batch_index": batches - 1,
                           "offset": offset, "output_repr": repr(output)})
                raise I.StageRefused("probe_raw_output", "non-JSON backend output preserved as representation")
            if not isinstance(output, list) or len(output) != len(batch_cases):
                raise I.StageRefused("backend_row_count", "wrong synthetic batch row count")
            raw_rows = []
            for case, row in zip(batch_cases, output):
                if not isinstance(row, dict):
                    raise I.StageRefused("row_content", case["id"])
                if _is_v7(protocol) and "id" in row and row["id"] != case["id"]:
                    raise I.StageRefused("probe_raw_id", "returned model ID cannot be overwritten")
                raw_rows.append({**row, "id": case["id"]})
            _raw_check(raw_rows, [c["id"] for c in batch_cases], spec, protocol)
            unchanged(model, runtime)
            if time.monotonic() >= deadline:
                raise I.StageTimeLimit()
            for key in memory:
                measured = getattr(model, "last_metrics", {}).get(key, 0)
                if type(measured) is not int or measured < 0:
                    raise I.StageRefused("probe_metrics", "invalid memory measurement")
                memory[key] = max(memory[key], measured)
            rows.extend(_analysis(case, raw) for case, raw in zip(batch_cases, raw_rows))
            if memory_limit is not None and memory["peak_reserved_bytes"] > memory_limit:
                status, exit_code = "memory_limit", 5
                log_event({"event": "call_error", "status": status})
                break
    except I.StageTimeLimit:
        status, exit_code = "time_limit", 3
        log_event({"event": "call_error", "status": status})
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            log_event({"event": "call_failed", "error_type": type(exc).__name__, "message": str(exc)})
            raise
        status, exit_code = "out_of_memory", 5
        log_event({"event": "call_error", "status": status})
    except BaseException as exc:
        log_event({"event": "call_failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if model is not None:
            # Check the loaded runtime before close; close can discard it.
            try:
                if runtime is not None:
                    unchanged(model, runtime)
            finally:
                model.close()
    unchanged()
    if I._file_sha(log_path) != log_hash.hexdigest():
        raise I.StageRefused("probe_log_changed", "raw backend log changed before receipt")
    if _is_v7(protocol):
        observed = {row["id"]: row for row in normal_rows + rows}
        rows = [observed[c["id"]] for c in cases if c["id"] in observed]
    outputs = b"".join(I._canonical(row) + b"\n" for row in rows)
    _write_new(target / "outputs.jsonl", outputs)
    summary = _summary(cases, rows, status)
    basis = {"call": call, "input_basis": input_basis, "runtime_model": runtime,
             "status": status, "exit_code": exit_code, "summary": summary,
             "finished_at_utc": datetime.now(UTC).isoformat(),
             "metrics": {"wall_seconds": round(time.monotonic() - started, 6),
                         "generation_batches": batches, **memory},
             "artifacts_sha256": {name: I._file_sha(target / name)
                                  for name in sorted(_ARTIFACTS - {"PROBE.json"})},
             "raw_row_sha256": [r["raw_sha256"] for r in rows],
             "gpu_model_verified": False, "ready_for_training": False}
    receipt = {"schema_version": _receipt_schema(protocol),
               "identity_basis": basis, "identity_sha256": I._digest(basis)}
    _write_new(target / "PROBE.json", I._canonical(receipt) + b"\n")
    # Self-readback does not repeat inference and cannot establish determinism.
    _verify_probe(target)
    return {"status": status, "exit_code": exit_code, "call_id": call["call_id"],
            "probe_identity_sha256": receipt["identity_sha256"],
            "receipt_sha256": I._file_sha(target / "PROBE.json"),
            "output_dir": str(target), "summary": summary,
            "gpu_model_verified": False, "ready_for_training": False}


def _verify_probe(folder):
    _check_gate()
    folder = J._safe_path(Path(folder))
    if not folder.is_dir() or {p.name for p in folder.iterdir()} != _ARTIFACTS:
        raise I.StageRefused("probe_artifacts", "missing, partial or unknown probe files")
    for path in folder.iterdir():
        J._safe_path(path)
        I._plain_file(path)
    before = {p.name: I._file_sha(p) for p in folder.iterdir()}
    receipt = I._json((folder / "PROBE.json").read_bytes())
    if (not isinstance(receipt, dict) or set(receipt) != {"schema_version", "identity_basis", "identity_sha256"} or
            type(receipt.get("schema_version")) is not int or
            receipt["schema_version"] not in (1, 2, 3) or
            not isinstance(receipt.get("identity_basis"), dict) or
            receipt.get("identity_sha256") != I._digest(receipt["identity_basis"])):
        raise I.StageRefused("probe_receipt", "invalid receipt identity")
    basis = receipt["identity_basis"]
    if basis.get("artifacts_sha256") != {k: v for k, v in before.items() if k != "PROBE.json"}:
        raise I.StageRefused("probe_artifacts", "artifact hash mismatch")
    cases_raw = (folder / "cases.json").read_bytes()
    saved_input = basis.get("input_basis")
    if not isinstance(saved_input, dict):
        raise I.StageRefused("probe_identity_changed", "missing input basis")
    registered = I._json(cases_raw).get("protocol") in V7_REGISTERED_SCOPES
    registration = saved_input.get("registration") if registered else {}
    if not isinstance(registration, dict) or (registered and set(registration) != {
            "expected_manifest_sha256", "expected_protocol", "expected_approval_scope", "expected_source_sha256"}):
        raise I.StageRefused("probe_registration", "receipt must preserve admitted external registration")
    cases = _inputs(cases_raw, **registration)
    protocol = _protocol(cases)
    if receipt["schema_version"] != _receipt_schema(protocol):
        raise I.StageRefused("probe_receipt", "receipt schema differs from the authenticated input protocol")
    _require_parser(protocol)
    config = I._json((folder / "config.json").read_bytes())
    spec, generation, memory_limit = _settings(config)
    if _is_v7(protocol) and spec["max_new_tokens"] != 160:
        raise I.StageRefused("probe_config", "v7 requires max_new_tokens=160")
    model_cases = _model_cases(cases)
    _, system, policy = _policy_for(protocol)
    input_basis = {"cases_file_sha256": hashlib.sha256(cases_raw).hexdigest(),
                   "config_sha256": I._digest(config),
                   "rendered_input_sha256": I._digest([{"system": system, "user": p} for p in _render(cases)]),
                   "policy_identity": policy, "implementation_sha256": _code_pins(),
                   "protocol": protocol,
                   "thresholds": _thresholds(protocol),
                   "case_provenance_sha256": I._digest(I._json(cases_raw).get("provenance")),
                   "parser_schema_version": _parser_schema(protocol),
                   **({"registration": J._clone(registration)} if registered else {}),
                   **({"evaluation_scope": _evaluation_scope(protocol)}
                      if protocol in (V6_PROTOCOL, *_V7_PROTOCOLS) else {})}
    if basis.get("input_basis") != input_basis:
        raise I.StageRefused("probe_identity_changed", "cases, config, renderer, parser or implementation changed")
    call = I._json((folder / "CALL.json").read_bytes())
    if (basis.get("call") != call or not isinstance(call, dict) or
            not isinstance(call.get("call_id"), str) or not re_full_uuid(call["call_id"]) or
            type(call.get("backend_injected")) is not bool):
        raise I.StageRefused("probe_call", "invalid independent call identity")
    raw_outputs = (folder / "outputs.jsonl").read_bytes()
    rows = []
    for line in raw_outputs.splitlines():
        if not line.strip():
            raise I.StageRefused("probe_rows", "blank output row")
        rows.append(I._json(line))
    if len(rows) > len(cases):
        raise I.StageRefused("probe_rows", "too many output rows")
    analysed_cases = cases
    if _is_v7(protocol):
        ids = [row.get("id") for row in rows if isinstance(row, dict)]
        analysed_cases = [case for case in cases if case["id"] in ids]
        if ids != [case["id"] for case in analysed_cases]:
            raise I.StageRefused("probe_rows", "v7 source ordering or unique IDs changed")
        normals = [c["id"] for c in cases if P.row_class(c["task"]) == "defect_free"]
        if any(cid not in ids for cid in normals):
            raise I.StageRefused("probe_rows", "v7 must retain all deterministic-only normal results")
    for case, row in zip(analysed_cases, rows):
        normal = _is_v7(protocol) and P.row_class(case["task"]) == "defect_free"
        if not isinstance(row, dict) or (not normal and not isinstance(row.get("raw"), dict)):
            raise I.StageRefused("probe_rows", "invalid raw row")
        if normal:
            if row.get("raw") is not None:
                raise I.StageRefused("normal_model_call_forbidden", "normal receipt carries model output")
        else:
            _raw_check([row["raw"]], [case["id"]], spec, protocol)
        if I._digest(row) != I._digest(_analysis(case, row["raw"])):
            raise I.StageRefused("probe_rows", "row order, hashes or analysis changed")
    returned_rows, returned_batches, terminal_events = [], 0, []
    for line in (folder / "backend_batches.jsonl").read_bytes().splitlines():
        event = I._json(line)
        if not isinstance(event, dict) or event.get("call_id") != call["call_id"]:
            raise I.StageRefused("probe_raw_log", "raw event has wrong call identity")
        if event.get("event") == "batch_returned":
            offset = returned_batches * spec["batch_size"]
            batch_cases = model_cases[offset:offset + spec["batch_size"]]
            ids = [c["id"] for c in batch_cases]
            if (terminal_events or event.get("batch_index") != returned_batches or
                    event.get("offset") != offset or event.get("case_ids") != ids or
                    not isinstance(event.get("output"), list) or len(event["output"]) != len(ids)):
                raise I.StageRefused("probe_raw_log", "raw backend batch order/count differs")
            enriched = []
            for cid, raw_row in zip(ids, event["output"]):
                if not isinstance(raw_row, dict):
                    raise I.StageRefused("probe_raw_log", "invalid raw backend row")
                if _is_v7(protocol) and "id" in raw_row and raw_row["id"] != cid:
                    raise I.StageRefused("probe_raw_id", "saved returned model ID differs from requested ID")
                enriched.append({**raw_row, "id": cid})
            _raw_check(enriched, ids, spec, protocol)
            returned_rows.extend(enriched)
            returned_batches += 1
        elif event.get("event") == "call_error":
            terminal_events.append(event.get("status"))
        else:
            raise I.StageRefused("probe_raw_log", "failed or unknown event cannot have a terminal receipt")
    analysed_raw = [r["raw"] for r in rows if r["raw"] is not None]
    if analysed_raw != returned_rows[:len(analysed_raw)] or len(analysed_raw) > len(returned_rows):
        raise I.StageRefused("probe_raw_log", "analysed output differs from preserved backend output")
    expected_terminal = [] if basis.get("status") == "complete" else [basis.get("status")]
    if terminal_events != expected_terminal:
        raise I.StageRefused("probe_raw_log", "terminal status differs from raw call events")
    runtime = basis.get("runtime_model")
    if runtime is not None:
        J._check_runtime(runtime, spec)
        if I._family(runtime["model_type"]) == I._family(generation["family"]):
            raise I.StageRefused("self_judging", "stored probe is self-judging")
    elif any(row.get("raw") is not None for row in rows) or basis.get("status") == "complete":
        raise I.StageRefused("probe_runtime", "completed output requires a runtime identity")
    allowed = {"complete": 0, "time_limit": 3, "out_of_memory": 5, "memory_limit": 5}
    if basis.get("status") not in allowed or basis.get("exit_code") != allowed[basis["status"]]:
        raise I.StageRefused("probe_status", "invalid terminal status")
    if I._digest(basis.get("summary")) != I._digest(_summary(cases, rows, basis["status"])):
        raise I.StageRefused("probe_summary", "summary does not match raw outcomes")
    if basis.get("raw_row_sha256") != [r["raw_sha256"] for r in rows]:
        raise I.StageRefused("probe_rows", "raw row digest list differs")
    if basis.get("gpu_model_verified") is not False or basis.get("ready_for_training") is not False:
        raise I.StageRefused("probe_approval", "a single call is not production approval")
    metrics = basis.get("metrics")
    if not isinstance(metrics, dict):
        raise I.StageRefused("probe_metrics", "missing bounded invocation metrics")
    for name in ("generation_batches", "peak_allocated_bytes", "peak_reserved_bytes"):
        if type(metrics.get(name)) is not int or metrics[name] < 0:
            raise I.StageRefused("probe_metrics", "invalid invocation metrics")
    if metrics["generation_batches"] != returned_batches:
        raise I.StageRefused("probe_metrics", "actual preserved batch count differs")
    wall = metrics.get("wall_seconds")
    if type(wall) not in (int, float) or wall < 0:
        raise I.StageRefused("probe_metrics", "invalid wall clock duration")
    if basis["status"] == "complete":
        expected_batches = (len(model_cases) + spec["batch_size"] - 1) // spec["batch_size"]
        if len(rows) != len(cases) or metrics["generation_batches"] != expected_batches:
            raise I.StageRefused("probe_incomplete", "complete status requires all cases and actual batches")
        if memory_limit is not None and metrics["peak_reserved_bytes"] > memory_limit:
            raise I.StageRefused("probe_metrics", "completed probe exceeded memory budget")
    if {p.name: I._file_sha(p) for p in folder.iterdir()} != before:
        raise I.StageRefused("probe_snapshot_changed", "probe artifacts changed during read")
    if _code_pins() != input_basis["implementation_sha256"] or _policy_for(protocol)[1:] != (system, policy):
        raise I.StageRefused("probe_identity_changed", "policy or implementation changed during read")
    _require_parser(protocol)
    _source_dependencies_unchanged(cases_raw)
    _check_gate()
    return receipt, rows, before["PROBE.json"]


def re_full_uuid(value):
    try:
        return len(value) == 32 and uuid.UUID(hex=value).hex == value
    except (ValueError, AttributeError):
        return False


def _compare_verified(first, left, first_sha, second, right, second_sha):
    a, b = first["identity_basis"], second["identity_basis"]
    if a["call"]["call_id"] == b["call"]["call_id"]:
        raise I.StageRefused("probe_same_call", "copied/reused results are not a second invocation")
    if a["input_basis"] != b["input_basis"] or a["runtime_model"] != b["runtime_model"]:
        raise I.StageRefused("probe_pair_identity", "both calls must use identical cases, policy and loaded runtime")
    complete = a["summary"]["complete"] and b["summary"]["complete"]
    row_hashes_equal = complete and a["raw_row_sha256"] == b["raw_row_sha256"]
    raw_text_equal = complete and [r["raw"]["text"] for r in left if r["raw"] is not None] == [
        r["raw"]["text"] for r in right if r["raw"] is not None]
    real = (a["runtime_model"] is not None and a["runtime_model"].get("backend") == "transformers" and
            not a["call"]["backend_injected"] and not b["call"]["backend_injected"] and
            a["runtime_model"]["judge_runtime"].get("test_only") is False)
    thresholds = a["summary"]["thresholds_passed"] and b["summary"]["thresholds_passed"]
    protocol = a["input_basis"].get("protocol")
    eligible = (protocol in (V6_PROTOCOL, *V7_REGISTERED_SCOPES) and
                a["input_basis"].get("parser_schema_version") == _parser_schema(protocol) and
                a["summary"].get("approval_eligible") is True and b["summary"].get("approval_eligible") is True)
    verified = bool(eligible and real and complete and thresholds and raw_text_equal and row_hashes_equal)
    return {"schema_version": 1, "exit_code": 0 if verified else 2,
            "first_call_id": a["call"]["call_id"], "second_call_id": b["call"]["call_id"],
            "first_receipt_sha256": first_sha, "second_receipt_sha256": second_sha,
            "complete": bool(complete), "raw_text_deterministic": bool(raw_text_equal),
            "raw_row_hashes_equal": bool(row_hashes_equal), "thresholds_passed": bool(thresholds),
            "real_backend_pair": bool(real), "gpu_model_verified": verified,
            "protocol": a["input_basis"]["protocol"], "approval_eligible": bool(eligible),
            **({"approval_scope": _approval_scope(protocol),
                "final_blinded_approval": protocol == V7_B_PROTOCOL and verified,
                "nplus_model_metric": "not_applicable", "model_determinism_rows": len([r for r in left if r["raw"] is not None]),
                "gpu_determinism_measured": bool(real and complete)} if _is_v7(protocol) else {}),
            "first_summary": a["summary"], "second_summary": b["summary"],
            "ready_for_training": False, "population_error_bound": None,
            "case_count": a["summary"]["planned_cases"], "independent_sample_count": None, "model_invocations": 2}


def _pair(first_dir, second_dir):
    first, left, first_sha = _verify_probe(first_dir)
    second, right, second_sha = _verify_probe(second_dir)
    report = _compare_verified(first, left, first_sha, second, right, second_sha)
    for folder, receipt, sha in ((first_dir, first, first_sha), (second_dir, second, second_sha)):
        expected = {**receipt["identity_basis"]["artifacts_sha256"], "PROBE.json": sha}
        if _probe_files(folder) != expected:
            raise I.StageRefused("probe_snapshot_changed", "pair changed during comparison")
    if (_code_pins() != first["identity_basis"]["input_basis"]["implementation_sha256"] or
            _policy_for(first["identity_basis"]["input_basis"]["protocol"])[2] !=
            first["identity_basis"]["input_basis"]["policy_identity"]):
        raise I.StageRefused("probe_identity_changed", "pair policy or implementation changed")
    return report, first, second


def compare_probes(first_dir, second_dir) -> dict:
    """Read and compare two distinct fresh calls. Never infer or change files."""
    return _pair(first_dir, second_dir)[0]


def _probe_files(folder):
    folder = J._safe_path(Path(folder))
    if not folder.is_dir() or {p.name for p in folder.iterdir()} != _ARTIFACTS:
        raise I.StageRefused("probe_artifacts", "missing or unknown portable probe files")
    result = {}
    for path in folder.iterdir():
        J._safe_path(path)
        I._plain_file(path)
        result[path.name] = I._file_sha(path)
    return result


def _approval_basis(report, first, second, judge_config):
    if (report.get("gpu_model_verified") is not True or report.get("exit_code") != 0 or
            report.get("protocol") not in (V6_PROTOCOL, *V7_REGISTERED_SCOPES) or report.get("approval_eligible") is not True):
        raise I.StageRefused("probe_not_approved", "two real, un-injected passing versioned probes are required")
    a, b = first["identity_basis"], second["identity_basis"]
    protocol = report["protocol"]
    common = {key: a["input_basis"]["implementation_sha256"][key]
              for key in ("infer.py", "judge_identity.py")}
    return {"kind": "two-probe-judge-approval", "protocol": protocol,
            "thresholds": _thresholds(protocol), "comparison": report,
            "evaluation_scope": _evaluation_scope(protocol),
            **({"approval_scope": _approval_scope(protocol), "final_blinded_approval": protocol == V7_B_PROTOCOL}
               if _is_v7(protocol) else {}),
            "input_basis": a["input_basis"], "judge_config": judge_config,
            "runtime_model": a["runtime_model"], "common_implementation_sha256": common,
            "probe_identity_sha256": [first["identity_sha256"], second["identity_sha256"]],
            "probe_call_ids": [a["call"]["call_id"], b["call"]["call_id"]],
            "external_call_authentication": "operator must independently pin call logs and receipt SHA",
            "ready_for_training": False}


def approval_files(folder):
    """Hash the complete portable evidence without following aliases."""
    folder = J._safe_path(Path(folder))
    if not folder.is_dir() or {p.name for p in folder.iterdir()} != {"first", "second", "APPROVAL.json"}:
        raise I.StageRefused("probe_approval_artifacts", "incomplete or unknown approval evidence")
    receipt = folder / "APPROVAL.json"
    J._safe_path(receipt)
    I._plain_file(receipt)
    return {"APPROVAL.json": I._file_sha(receipt),
            **{f"{name}/{key}": sha for name in ("first", "second")
               for key, sha in _probe_files(folder / name).items()}}


def read_probe_approval(folder, expected_sha256) -> dict:
    """Recompute approval from both portable probes, not a JSON true flag.

    The caller's separately recorded receipt SHA is the trust anchor. Self-hashes
    and UUIDs cannot authenticate real hardware calls without operator evidence.
    """
    _check_gate()
    if not isinstance(expected_sha256, str) or not I._HEX64.fullmatch(expected_sha256):
        raise I.StageRefused("probe_approval_pin", "a separately recorded approval SHA256 is required")
    folder = J._safe_path(Path(folder))
    before = approval_files(folder)
    if before["APPROVAL.json"] != expected_sha256:
        raise I.StageRefused("probe_approval_pin", "approval receipt differs from the explicit SHA256")
    receipt = I._json((folder / "APPROVAL.json").read_bytes())
    if (not isinstance(receipt, dict) or set(receipt) != {"schema_version", "identity_basis", "identity_sha256"}
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 2
            or not isinstance(receipt["identity_basis"], dict)
            or receipt["identity_sha256"] != I._digest(receipt["identity_basis"])):
        raise I.StageRefused("probe_approval_receipt", "invalid approval identity")
    report, first, second = _pair(folder / "first", folder / "second")
    config = I._json((folder / "first/config.json").read_bytes())
    expected = _approval_basis(report, first, second, _settings(config)[0])
    if receipt["identity_basis"] != expected or approval_files(folder) != before:
        raise I.StageRefused("probe_approval_changed", "approval does not match its verified evidence")
    if (_code_pins() != expected["input_basis"]["implementation_sha256"] or
            _policy_for(expected["input_basis"]["protocol"])[2] != expected["input_basis"]["policy_identity"]):
        raise I.StageRefused("probe_identity_changed", "approval policy or implementation changed during read")
    _check_gate()
    return {"receipt_sha256": expected_sha256, "identity_sha256": receipt["identity_sha256"],
            "identity_basis": expected, "files_sha256": before}


def _copy_pinned_files(source, target, pins):
    source, target = J._safe_path(Path(source)), J._safe_path(Path(target))
    if source == target or source in target.parents or target in source.parents:
        raise I.StageRefused("probe_approval_overlap", "portable copy cannot overlap its source")
    if target.exists():
        raise I.StageRefused("probe_approval_exists", "portable evidence requires a new directory")
    target.mkdir(parents=True, exist_ok=False)
    for name, sha in sorted(pins.items(), key=lambda item: (item[0] == "APPROVAL.json", item[0])):
        # Callers construct names from the fixed artifact allowlist, never JSON paths.
        original = J._safe_path(source / name)
        I._plain_file(original)
        raw = original.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise I.StageRefused("probe_snapshot_changed", "source changed while sealing evidence")
        final = target / name
        final.parent.mkdir(parents=True, exist_ok=True)
        J._safe_path(final.parent)
        _write_new(final, raw)


def copy_probe_approval(source, target, expected_sha256):
    """Copy only a fully revalidated immutable approval to a new writer folder."""
    approval = read_probe_approval(source, expected_sha256)
    _copy_pinned_files(source, target, approval["files_sha256"])
    copied = read_probe_approval(target, expected_sha256)
    if copied != approval or approval_files(source) != approval["files_sha256"]:
        raise I.StageRefused("probe_approval_changed", "portable copy changed")
    return copied


def seal_probe_approval(first_dir, second_dir, output_dir) -> dict:
    """Seal a reverified passing pair; failed/fake probes never produce approval."""
    _check_gate()
    target = J._safe_path(Path(output_dir))
    for folder in (first_dir, second_dir):
        source = J._safe_path(Path(folder))
        if source == target or source in target.parents or target in source.parents:
            raise I.StageRefused("probe_approval_overlap", "approval cannot overlap an original probe")
    if target.exists() or target.is_symlink():
        raise I.StageRefused("probe_approval_exists", "approval output must be new")
    report, first, second = _pair(first_dir, second_dir)
    config = I._json((Path(first_dir) / "config.json").read_bytes())
    basis = _approval_basis(report, first, second, _settings(config)[0])
    target.mkdir(parents=True, exist_ok=False)
    for name, source, receipt, sha in (
            ("first", first_dir, first, report["first_receipt_sha256"]),
            ("second", second_dir, second, report["second_receipt_sha256"])):
        _copy_pinned_files(source, target / name,
                           {**receipt["identity_basis"]["artifacts_sha256"], "PROBE.json": sha})
    copied_report, copied_first, copied_second = _pair(target / "first", target / "second")
    if _approval_basis(copied_report, copied_first, copied_second, _settings(config)[0]) != basis:
        raise I.StageRefused("probe_approval_changed", "copied pair differs")
    for source, receipt, sha in ((first_dir, first, report["first_receipt_sha256"]),
                                 (second_dir, second, report["second_receipt_sha256"])):
        if _probe_files(source) != {**receipt["identity_basis"]["artifacts_sha256"], "PROBE.json": sha}:
            raise I.StageRefused("probe_snapshot_changed", "original pair changed while sealing")
    receipt = {"schema_version": 2, "identity_basis": basis, "identity_sha256": I._digest(basis)}
    _write_new(target / "APPROVAL.json", I._canonical(receipt) + b"\n")
    sha = I._file_sha(target / "APPROVAL.json")
    read_probe_approval(target, sha)
    return {"exit_code": 0, "approval_dir": str(target), "approval_sha256": sha,
            "approval_identity_sha256": receipt["identity_sha256"], "gpu_model_verified": True,
            "ready_for_training": False}
