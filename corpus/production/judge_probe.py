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
_STRUCTURED_PROTOCOLS = (V5_PROTOCOL, V6_PROTOCOL)
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


def _inputs(raw):
    data = I._json(raw)
    if isinstance(data, dict) and (data.get("schema_version") in (2, 3) or "protocol" in data):
        return _structured_inputs(data)
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


def _structured_inputs(data):
    protocol = data.get("protocol")
    v6 = protocol == V6_PROTOCOL
    pins = V6_SOURCE_SHA256 if v6 else V5_SOURCE_SHA256
    required = {"schema_version", "protocol", "cases", "provenance", "model_discrimination_measured"}
    if (set(data) not in (required, required | {"thresholds"}) or
            type(data.get("schema_version")) is not int or data["schema_version"] != (3 if v6 else 2) or
            protocol not in _STRUCTURED_PROTOCOLS or data.get("model_discrimination_measured") is not False):
        raise I.StageRefused("probe_protocol_contract", "explicit versioned 150-case protocol required")
    if "thresholds" in data and I._digest(data["thresholds"]) != I._digest(_thresholds(protocol)):
        raise I.StageRefused("probe_thresholds", "structured thresholds cannot be replaced or relaxed")
    provenance = data["provenance"]
    keys = {"source_sha256", "adapter_sha256", "label_map_sha256", "label_map_loader_sha256",
            "label_names", "source_metadata", "real_images_or_ids_used",
            "expectations_fixed_before_model_execution", "statistical_generalization_claimed",
            "deterministic_results_are_not_model_approval", "source_files_base64"}
    if v6:
        keys.add("evaluation_scope")
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
    if v6 and I._digest(provenance["evaluation_scope"]) != I._digest(V6_EVALUATION_SCOPE):
        raise I.StageRefused("probe_evaluation_scope", "exposed G sources are not new independent samples")
    sources = _structured_sources(provenance, protocol)
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != 150:
        raise I.StageRefused("probe_cases", "structured probes require exactly 150 source-preserving cases")
    fields = {"id", "group", "task", "candidate", "expected_model_accept", "expected_deterministic_accept",
              "source_case", "source_case_sha256", "source_file_sha256", "source_group", "source_index"}
    if v6:
        fields.add("protocol")
    ids, positions, counts = set(), {g: [] for g in V5_GROUP_COUNTS}, Counter()
    for case in cases:
        if not isinstance(case, dict) or set(case) != fields:
            raise I.StageRefused("probe_case", "unexpected versioned case fields")
        if v6 and case["protocol"] != V6_PROTOCOL:
            raise I.StageRefused("probe_protocol_contract", "mixed-version case cannot enter a v6 probe")
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
    if dict(counts) != V5_GROUP_COUNTS or any(positions[g] != list(range(n)) for g, n in V5_GROUP_COUNTS.items()):
        raise I.StageRefused("probe_groups", "structured probes require ordered 50/50/25/25 source indices")
    return cases


def _structured_sources(provenance, protocol):
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
    for group, count in V5_GROUP_COUNTS.items():
        value = encoded[group]
        try:
            if not isinstance(value, str):
                raise ValueError("base64 must be text")
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise I.StageRefused("probe_source_files", "invalid source byte encoding") from exc
        pins = V6_SOURCE_SHA256 if protocol == V6_PROTOCOL else V5_SOURCE_SHA256
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
        _structured_sources(data["provenance"], data["protocol"])


def _protocol(cases):
    # Called only after complete input validation, never inferred from model output.
    if cases and all(c.get("protocol") == V6_PROTOCOL for c in cases):
        return V6_PROTOCOL
    return V5_PROTOCOL if len(cases) == 150 else LEGACY_PROTOCOL


def _thresholds(protocol):
    return V6_THRESHOLDS if protocol == V6_PROTOCOL else V5_THRESHOLDS if protocol == V5_PROTOCOL else _THRESHOLDS


def _parser_schema(protocol):
    return "judge_structured_v6" if protocol == V6_PROTOCOL else (
        "judge_structured_v5" if protocol == V5_PROTOCOL else "legacy_four_keys")


def _require_parser(protocol):
    if protocol == V6_PROTOCOL:
        return J.require_v6_parser()
    if protocol == V5_PROTOCOL:
        return J.require_v5_parser()
    return None


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
    return [P.render_judge(c["task"], {"text": c["candidate"], "finish_reason": "eos"}) for c in cases]


def check_rules(cases_path) -> dict:
    """Read-only CPU discrimination check; no backend, model or output file."""
    _check_gate()
    path = J._safe_path(Path(cases_path))
    I._plain_file(path)
    raw = path.read_bytes()
    cases = _inputs(raw)
    protocol = _protocol(cases)
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


def run_probe(cases_path, config: dict, output_dir, *, backend_factory=None) -> dict:
    """Perform one fresh call. Only v6-150 can qualify for a new approval."""
    environment = _check_gate()
    source, target = J._safe_path(Path(cases_path)), J._safe_path(Path(output_dir))
    I._plain_file(source)
    if target.exists() or target.is_symlink():
        raise I.StageRefused("probe_output_exists", "each probe requires a new exclusive directory")
    config = J._clone(config)
    spec, generation, memory_limit = _settings(config)
    raw_cases = source.read_bytes()
    cases = _inputs(raw_cases)
    protocol = _protocol(cases)
    _require_parser(protocol)  # Before output creation or backend/model loading.
    source_sha = hashlib.sha256(raw_cases).hexdigest()
    policy, system, identity = J._policy()
    prompts = _render(cases)
    render_sha = I._digest([{"system": system, "user": p} for p in prompts])
    code = _code_pins()
    input_basis = {"cases_file_sha256": source_sha, "config_sha256": I._digest(config),
                   "rendered_input_sha256": render_sha, "policy_identity": identity,
                   "implementation_sha256": code, "protocol": protocol,
                   "thresholds": _thresholds(protocol),
                   "case_provenance_sha256": I._digest(I._json(raw_cases).get("provenance")),
                   "parser_schema_version": _parser_schema(protocol),
                   **({"evaluation_scope": V6_EVALUATION_SCOPE} if protocol == V6_PROTOCOL else {})}

    def unchanged(model=None, runtime=None):
        _check_gate()
        _require_parser(protocol)
        J._safe_path(source)
        J._safe_path(target)
        if I._file_sha(source) != source_sha or _code_pins() != code:
            raise I.StageRefused("probe_inputs_changed", "input/source bytes changed")
        _source_dependencies_unchanged(raw_cases)
        _, current_system, current_policy = J._policy()
        if current_system != system or current_policy != identity:
            raise I.StageRefused("probe_policy_changed", "policy changed during invocation")
        if model is not None and J.judge_runtime_identity(model, spec) != runtime:
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
    status, exit_code = "complete", 0
    memory = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    try:
        unchanged()
        model = (backend_factory or I.TransformersBackend)(dict(spec), "judge")
        runtime = J.judge_runtime_identity(model, spec)
        if I._family(runtime["model_type"]) == I._family(generation["family"]):
            raise I.StageRefused("self_judging", "actual judge architecture matches configured generator")
        unchanged(model, runtime)
        for offset in range(0, len(cases), spec["batch_size"]):
            unchanged(model, runtime)
            if time.monotonic() >= deadline:
                raise I.StageTimeLimit()
            batch_cases = cases[offset:offset + spec["batch_size"]]
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
                raw_rows.append({**row, "id": case["id"]})
            I._check_rows(raw_rows, [c["id"] for c in batch_cases], spec)
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
    receipt = {"schema_version": 2 if protocol == V6_PROTOCOL else 1,
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
            receipt["schema_version"] not in (1, 2) or
            not isinstance(receipt.get("identity_basis"), dict) or
            receipt.get("identity_sha256") != I._digest(receipt["identity_basis"])):
        raise I.StageRefused("probe_receipt", "invalid receipt identity")
    basis = receipt["identity_basis"]
    if basis.get("artifacts_sha256") != {k: v for k, v in before.items() if k != "PROBE.json"}:
        raise I.StageRefused("probe_artifacts", "artifact hash mismatch")
    cases_raw = (folder / "cases.json").read_bytes()
    cases = _inputs(cases_raw)
    protocol = _protocol(cases)
    if receipt["schema_version"] != (2 if protocol == V6_PROTOCOL else 1):
        raise I.StageRefused("probe_receipt", "receipt schema differs from the authenticated input protocol")
    _require_parser(protocol)
    config = I._json((folder / "config.json").read_bytes())
    spec, generation, memory_limit = _settings(config)
    _, system, policy = J._policy()
    input_basis = {"cases_file_sha256": hashlib.sha256(cases_raw).hexdigest(),
                   "config_sha256": I._digest(config),
                   "rendered_input_sha256": I._digest([{"system": system, "user": p} for p in _render(cases)]),
                   "policy_identity": policy, "implementation_sha256": _code_pins(),
                   "protocol": protocol,
                   "thresholds": _thresholds(protocol),
                   "case_provenance_sha256": I._digest(I._json(cases_raw).get("provenance")),
                   "parser_schema_version": _parser_schema(protocol),
                   **({"evaluation_scope": V6_EVALUATION_SCOPE} if protocol == V6_PROTOCOL else {})}
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
    for case, row in zip(cases, rows):
        if not isinstance(row, dict) or not isinstance(row.get("raw"), dict):
            raise I.StageRefused("probe_rows", "invalid raw row")
        I._check_rows([row["raw"]], [case["id"]], spec)
        if I._digest(row) != I._digest(_analysis(case, row["raw"])):
            raise I.StageRefused("probe_rows", "row order, hashes or analysis changed")
    returned_rows, returned_batches, terminal_events = [], 0, []
    for line in (folder / "backend_batches.jsonl").read_bytes().splitlines():
        event = I._json(line)
        if not isinstance(event, dict) or event.get("call_id") != call["call_id"]:
            raise I.StageRefused("probe_raw_log", "raw event has wrong call identity")
        if event.get("event") == "batch_returned":
            offset = returned_batches * spec["batch_size"]
            batch_cases = cases[offset:offset + spec["batch_size"]]
            ids = [c["id"] for c in batch_cases]
            if (terminal_events or event.get("batch_index") != returned_batches or
                    event.get("offset") != offset or event.get("case_ids") != ids or
                    not isinstance(event.get("output"), list) or len(event["output"]) != len(ids)):
                raise I.StageRefused("probe_raw_log", "raw backend batch order/count differs")
            enriched = []
            for cid, raw_row in zip(ids, event["output"]):
                if not isinstance(raw_row, dict):
                    raise I.StageRefused("probe_raw_log", "invalid raw backend row")
                enriched.append({**raw_row, "id": cid})
            I._check_rows(enriched, ids, spec)
            returned_rows.extend(enriched)
            returned_batches += 1
        elif event.get("event") == "call_error":
            terminal_events.append(event.get("status"))
        else:
            raise I.StageRefused("probe_raw_log", "failed or unknown event cannot have a terminal receipt")
    if [r["raw"] for r in rows] != returned_rows[:len(rows)] or len(rows) > len(returned_rows):
        raise I.StageRefused("probe_raw_log", "analysed output differs from preserved backend output")
    expected_terminal = [] if basis.get("status") == "complete" else [basis.get("status")]
    if terminal_events != expected_terminal:
        raise I.StageRefused("probe_raw_log", "terminal status differs from raw call events")
    runtime = basis.get("runtime_model")
    if runtime is not None:
        J._check_runtime(runtime, spec)
        if I._family(runtime["model_type"]) == I._family(generation["family"]):
            raise I.StageRefused("self_judging", "stored probe is self-judging")
    elif rows or basis.get("status") == "complete":
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
        expected_batches = (len(cases) + spec["batch_size"] - 1) // spec["batch_size"]
        if len(rows) != len(cases) or metrics["generation_batches"] != expected_batches:
            raise I.StageRefused("probe_incomplete", "complete status requires all cases and actual batches")
        if memory_limit is not None and metrics["peak_reserved_bytes"] > memory_limit:
            raise I.StageRefused("probe_metrics", "completed probe exceeded memory budget")
    if {p.name: I._file_sha(p) for p in folder.iterdir()} != before:
        raise I.StageRefused("probe_snapshot_changed", "probe artifacts changed during read")
    if _code_pins() != input_basis["implementation_sha256"] or J._policy()[1:] != (system, policy):
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
    raw_text_equal = complete and [r["raw"]["text"] for r in left] == [r["raw"]["text"] for r in right]
    real = (a["runtime_model"] is not None and a["runtime_model"].get("backend") == "transformers" and
            not a["call"]["backend_injected"] and not b["call"]["backend_injected"] and
            a["runtime_model"]["judge_runtime"].get("test_only") is False)
    thresholds = a["summary"]["thresholds_passed"] and b["summary"]["thresholds_passed"]
    eligible = (a["input_basis"].get("protocol") == V6_PROTOCOL and
                a["input_basis"].get("parser_schema_version") == "judge_structured_v6" and
                a["summary"].get("approval_eligible") is True and b["summary"].get("approval_eligible") is True)
    verified = bool(eligible and real and complete and thresholds and raw_text_equal and row_hashes_equal)
    return {"schema_version": 1, "exit_code": 0 if verified else 2,
            "first_call_id": a["call"]["call_id"], "second_call_id": b["call"]["call_id"],
            "first_receipt_sha256": first_sha, "second_receipt_sha256": second_sha,
            "complete": bool(complete), "raw_text_deterministic": bool(raw_text_equal),
            "raw_row_hashes_equal": bool(row_hashes_equal), "thresholds_passed": bool(thresholds),
            "real_backend_pair": bool(real), "gpu_model_verified": verified,
            "protocol": a["input_basis"]["protocol"], "approval_eligible": bool(eligible),
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
            J._policy()[2] != first["identity_basis"]["input_basis"]["policy_identity"]):
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
            report.get("protocol") != V6_PROTOCOL or report.get("approval_eligible") is not True):
        raise I.StageRefused("probe_not_approved", "two real, un-injected passing v6 probes are required")
    a, b = first["identity_basis"], second["identity_basis"]
    common = {key: a["input_basis"]["implementation_sha256"][key]
              for key in ("infer.py", "judge_identity.py")}
    return {"kind": "two-probe-judge-approval", "protocol": V6_PROTOCOL,
            "thresholds": V6_THRESHOLDS, "comparison": report,
            "evaluation_scope": V6_EVALUATION_SCOPE,
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
            J._policy()[2] != expected["input_basis"]["policy_identity"]):
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
