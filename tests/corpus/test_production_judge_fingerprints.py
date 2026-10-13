"""파서 선택 지문의 CPU 계약. 판별력·실제 모델 시험이 아니다."""

import sys
from types import ModuleType

import pytest

from corpus.production import judge_fingerprints as F

COMMON = '''from __future__ import annotations
import json
def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate")
        result[key] = value
    return result
def _reject_constant(value):
    raise ValueError("nonfinite")
def parse_json(raw):
    return json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
def unrelated_report():
    return "old report"
'''

QUALITY = '''from __future__ import annotations
from fingerprint_common_fixture import parse_json, canonical_bytes
JUDGE_KEYS = ("ok", "reason")
def _text(row):
    return row["text"]
def _judge(row):
    value = parse_json(_text(row))
    if tuple(sorted(value)) != JUDGE_KEYS:
        raise ValueError("keys")
    return value
def audit_report():
    return "old audit"
'''


@pytest.fixture
def modules(tmp_path, monkeypatch):
    def load(name, text):
        path = tmp_path / (name + ".py")
        path.write_text(text, encoding="utf-8")
        module = ModuleType(name)
        module.__file__, module.__package__ = str(path), ""
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(text, str(path), "exec", dont_inherit=True), vars(module))
        return module

    def pair(common=COMMON, quality=QUALITY):
        common_module = load("fingerprint_common_fixture", common)
        quality_module = load("fingerprint_quality_fixture", quality)
        return common_module, quality_module

    return pair


def identity(pair):
    common, quality = pair
    return F._parser_identity(quality, common)


def test_actual_parser_closure_is_selected_not_whole_consumer():
    report = F.parser_identity()
    names = set(report["symbols"])
    assert "corpus.production.quality:_judge" in names
    assert "corpus.production.quality:JUDGE_KEYS" in names
    assert "corpus.production.common:_unique_keys" in names
    assert "corpus.production.common:_reject_constant" in names
    assert "corpus.production.common:canonical_bytes" in names
    assert not any("audit_run" in name or "write_json_new" in name for name in names)


def test_unrelated_report_and_line_changes_leave_identity(modules):
    before = identity(modules())
    after = identity(modules(COMMON.replace("old report", "new report"),
                             "\n\n" + QUALITY.replace("old audit", "new audit")))
    assert before == after


@pytest.mark.parametrize("common,quality", [
    (COMMON.replace('raise ValueError("duplicate")', 'raise ValueError("changed")'), QUALITY),
    (COMMON.replace('raise ValueError("nonfinite")', 'return value'), QUALITY),
    (COMMON.replace("sort_keys=True", "sort_keys=False"), QUALITY),
    (COMMON, QUALITY.replace('("ok", "reason")', '("ok", "reason", "extra")')),
    (COMMON, QUALITY.replace('return row["text"]', 'return row.get("text", "")')),
    (COMMON, QUALITY.replace("parse_json(_text(row))", "parse_json(_text(row).strip())")),
])
def test_every_parser_dependency_changes_identity(modules, common, quality):
    before = identity(modules())
    assert identity(modules(common, quality)) != before


def test_unused_import_alias_does_not_change_identity(modules):
    before = identity(modules())
    changed = QUALITY.replace("parse_json, canonical_bytes", "parse_json, canonical_bytes as unused")
    assert identity(modules(quality=changed)) == before


def test_used_import_alias_is_resolved_and_bound(modules):
    before = identity(modules())
    changed = QUALITY.replace("import parse_json,", "import parse_json as decode,")
    changed = changed.replace("parse_json(_text(row))", "decode(_text(row))")
    report = identity(modules(quality=changed))
    assert report != before
    assert "fingerprint_quality_fixture:decode" in report["symbols"]
    assert "fingerprint_common_fixture:_unique_keys" in report["symbols"]


def test_loaded_function_with_changed_file_is_refused(modules):
    pair = modules()
    F.Path(pair[1].__file__).write_text(QUALITY.replace('return row["text"]', 'return "changed"'), encoding="utf-8")
    with pytest.raises(F.FingerprintRefused, match="loaded parser code"):
        identity(pair)


def test_loaded_constant_and_import_replacement_are_refused(modules):
    pair = modules()
    pair[1].JUDGE_KEYS = ("changed",)
    with pytest.raises(F.FingerprintRefused, match="constant differs"):
        identity(pair)
    pair = modules()
    pair[1].parse_json = lambda text: {}
    with pytest.raises(F.FingerprintRefused, match="import differs"):
        identity(pair)


def test_helper_monkeypatch_and_builtin_override_are_refused(modules):
    pair = modules()
    pair[1]._text = lambda row: "changed"
    with pytest.raises(F.FingerprintRefused):
        identity(pair)
    pair = modules()
    pair[1].tuple = lambda value: value
    with pytest.raises(F.FingerprintRefused, match="replaced parser global"):
        identity(pair)


@pytest.mark.parametrize("replacement", [
    'return globals()["parse_json"](_text(row))',
    'import json\n    return json.loads(_text(row))',
])
def test_dynamic_lookup_or_local_import_fails_closed(modules, replacement):
    changed = QUALITY.replace("value = parse_json(_text(row))", replacement)
    with pytest.raises(F.FingerprintRefused, match="dynamic"):
        identity(modules(quality=changed))


def test_duplicate_binding_is_not_silently_selected(modules):
    changed = QUALITY + '\nJUDGE_KEYS = ("ok", "reason")\n'
    with pytest.raises(F.FingerprintRefused, match="ambiguous"):
        identity(modules(quality=changed))


def test_source_changed_during_read_is_refused(modules, monkeypatch):
    pair = modules()
    original = F._Closure.visit
    changed = False

    def mutate(self, module, name):
        nonlocal changed
        result = original(self, module, name)
        if not changed:
            changed = True
            F.Path(pair[1].__file__).write_text(QUALITY + "\n# concurrent change\n", encoding="utf-8")
        return result

    monkeypatch.setattr(F._Closure, "visit", mutate)
    with pytest.raises(F.FingerprintRefused, match="source changed"):
        identity(pair)


LEAK = r'''from __future__ import annotations
import json
import re
import unicodedata
from collections.abc import Mapping
from fingerprint_common_fixture import canonical_bytes
from fingerprint_infer_fixture import GENERATE_SYSTEM
SYSTEM_WINDOW = 4
TEMPLATE_WINDOW = 6
_LEAK_MARKERS = ("system:", "user:")
def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield json.dumps(value)
def _windows(text, size):
    t = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    return {t[i:i + size] for i in range(len(t) - size + 1)}
_SYSTEM_WINDOWS = _windows(GENERATE_SYSTEM, SYSTEM_WINDOW)
def instruction_template(task):
    prompt = task.get("prompt") or ""
    pieces = sorted({s for s in _strings(task.get("evidence")) if len(s) >= 8}
                    | {canonical_bytes(task.get("locked_target")).decode("utf-8")}, key=len, reverse=True)
    for p in pieces:
        prompt = prompt.replace(p, " ")
    return prompt
def leak_problems(text, task):
    reasons = []
    if any(m.casefold() in text.casefold() for m in _LEAK_MARKERS):
        reasons.append("marker")
    if _windows(text, SYSTEM_WINDOW) & _SYSTEM_WINDOWS:
        reasons.append("system")
    if _windows(text, TEMPLATE_WINDOW) & _windows(instruction_template(task), TEMPLATE_WINDOW):
        reasons.append("prompt")
    return reasons
def audit_report():
    return "unrelated"
'''
INFER = '''GENERATE_SYSTEM = "abcdef ghijklmnop"
def unrelated_generation():
    return "unrelated"
'''


@pytest.fixture
def leak_modules(tmp_path, monkeypatch):
    def pair(quality=LEAK, common=COMMON, infer=INFER):
        modules = []
        for name, text in (("fingerprint_common_fixture", common),
                           ("fingerprint_infer_fixture", infer),
                           ("fingerprint_quality_fixture", quality)):
            path = tmp_path / (name + ".py")
            path.write_text(text, encoding="utf-8")
            module = ModuleType(name)
            module.__file__, module.__package__ = str(path), ""
            monkeypatch.setitem(sys.modules, name, module)
            exec(compile(text, str(path), "exec", dont_inherit=True), vars(module))
            modules.append(module)
        common_module, infer_module, quality_module = modules
        return quality_module, common_module, infer_module
    return pair


def test_actual_leak_closure_includes_cache_and_prompt_not_whole_infer():
    report = F.leak_identity()
    symbols = report["symbols"]
    for name in ("leak_problems", "_strings", "_windows", "instruction_template",
                 "_SYSTEM_WINDOWS", "_LEAK_MARKERS", "SYSTEM_WINDOW", "TEMPLATE_WINDOW"):
        assert "corpus.production.quality:" + name in symbols
    assert "corpus.production.infer:GENERATE_SYSTEM" in symbols
    assert "corpus.production.common:canonical_bytes" in symbols
    assert not any("audit_run" in name or "read_stage_rows" in name for name in symbols)


def test_leak_unrelated_reporting_generation_changes_are_ignored(leak_modules):
    before = F._leak_identity(*leak_modules())
    after = F._leak_identity(*leak_modules(
        quality=LEAK.replace('return "unrelated"', 'return "new report"'),
        infer="\n\n" + INFER.replace('return "unrelated"', 'return "new generator"')))
    assert before == after


@pytest.mark.parametrize("quality,common,infer", [
    (LEAK.replace('("system:", "user:")', '("system:", "assistant:")'), COMMON, INFER),
    (LEAK.replace("SYSTEM_WINDOW = 4", "SYSTEM_WINDOW = 5"), COMMON, INFER),
    (LEAK.replace("TEMPLATE_WINDOW = 6", "TEMPLATE_WINDOW = 7"), COMMON, INFER),
    (LEAK.replace('reasons.append("prompt")', 'reasons.append("new_prompt")'), COMMON, INFER),
    (LEAK.replace("if len(s) >= 8", "if len(s) >= 9"), COMMON, INFER),
    (LEAK, COMMON.replace("sort_keys=True", "sort_keys=False"), INFER),
    (LEAK, COMMON, INFER.replace("abcdef ghijklmnop", "changed generation instructions")),
])
def test_leak_selected_dependencies_change_identity(leak_modules, quality, common, infer):
    before = F._leak_identity(*leak_modules())
    assert F._leak_identity(*leak_modules(quality, common, infer)) != before


def test_leak_stale_computed_cache_is_refused(leak_modules):
    modules = leak_modules()
    modules[0]._SYSTEM_WINDOWS = {"not the computed cache"}
    with pytest.raises(F.FingerprintRefused, match="cache differs"):
        F._leak_identity(*modules)


def test_unknown_initializer_is_not_executed(leak_modules):
    modules = leak_modules()
    called = []
    modules[0]._unexpected = lambda: called.append(True)
    changed = LEAK.replace("_SYSTEM_WINDOWS = _windows(GENERATE_SYSTEM, SYSTEM_WINDOW)",
                           "_SYSTEM_WINDOWS = _unexpected()")
    F.Path(modules[0].__file__).write_text(changed, encoding="utf-8")
    with pytest.raises(F.FingerprintRefused, match="initializer"):
        F._leak_identity(*modules)
    assert called == []


def test_unknown_windows_body_is_not_evaluated(leak_modules):
    changed = LEAK.replace('normalize("NFC", text)', 'normalize("NFKC", text)')
    with pytest.raises(F.FingerprintRefused, match="function body"):
        F._leak_identity(*leak_modules(quality=changed))


def test_replaced_leak_stdlib_import_is_refused(leak_modules):
    modules = leak_modules()
    modules[0].Mapping = dict
    with pytest.raises(F.FingerprintRefused, match="import differs"):
        F._leak_identity(*modules)


def test_leak_dependency_is_not_implicitly_allowed_in_parser(leak_modules):
    modules = leak_modules()
    with pytest.raises(F.FingerprintRefused, match="unsupported parser import"):
        F._Closure(modules).result(((modules[0], "_strings"),))


# 신원 연결만 시험하는 합성 B API다. 실제 구조화 파서/의미 판별력의 대체물이 아니다.
V5 = '''
JUDGE_V5_SCHEMA_VERSION = "judge_structured_v5"
V5_KEYS = ("grounded", "language_ok", "reason", "scope_ok")
def _derive_v5(value):
    return {key: value[key] for key in V5_KEYS}
def parse_judge_v5(task, row):
    value = parse_json(row["text"])
    return {"errors": [], "rejected": [], "verdict": _derive_v5(value), "flags": []}
'''


def test_legacy_identity_does_not_claim_structured_capability(modules):
    common, quality = modules()
    report = F._parser_identity(quality, common)
    assert report["capabilities"] == {"legacy_four_keys": True, "judge_structured_v5": False}
    with pytest.raises(F.FingerprintRefused, match="not implemented"):
        F._require_v5_parser(quality, common)


def test_v5_parser_and_helper_and_marker_are_bound(modules):
    common, quality = modules(quality=QUALITY + V5)
    report = F._require_v5_parser(quality, common)
    assert report["capabilities"]["judge_structured_v5"] is True
    for name in ("parse_judge_v5", "_derive_v5", "V5_KEYS", "JUDGE_V5_SCHEMA_VERSION"):
        assert "fingerprint_quality_fixture:" + name in report["symbols"]
    assert report == F._parser_identity(quality, common)


@pytest.mark.parametrize("v5", [
    'JUDGE_V5_SCHEMA_VERSION = "judge_structured_v5"\n',
    V5.replace('JUDGE_V5_SCHEMA_VERSION = "judge_structured_v5"', ''),
    V5.replace('"judge_structured_v5"', '"not_v5"'),
    'JUDGE_V5_SCHEMA_VERSION = "judge_structured_v5"\nparse_judge_v5 = None\n',
])
def test_partial_or_wrong_v5_capability_is_refused(modules, v5):
    with pytest.raises(F.FingerprintRefused, match="capability"):
        identity(modules(quality=QUALITY + v5))


def test_v5_parser_helper_changes_identity_without_whole_quality_pin(modules):
    before = identity(modules(quality=QUALITY + V5))
    changed = V5.replace("value[key] for key", "str(value[key]) for key")
    assert identity(modules(quality=QUALITY + changed)) != before
    unrelated = QUALITY.replace("old audit", "new unrelated audit")
    assert identity(modules(quality=unrelated + V5)) == before


def test_v5_dynamic_closure_has_no_legacy_fallback(modules):
    changed = V5.replace('value = parse_json(row["text"])', 'value = globals()["parse_json"](row["text"])')
    with pytest.raises(F.FingerprintRefused, match="dynamic"):
        identity(modules(quality=QUALITY + changed))


def test_v5_loaded_function_replacement_is_refused(modules):
    pair = modules(quality=QUALITY + V5)
    pair[1].parse_judge_v5 = lambda task, row: {}
    with pytest.raises(F.FingerprintRefused, match="replaced|differs"):
        identity(pair)


# v6 지문 연결의 합성 API. B의 실제 판독·형식 정규화 의미 시험은 아니다.
V6 = '''
JUDGE_V6_SCHEMA_VERSION = "judge_structured_v6"
V6_KEYS = ("names_match", "counts_match", "positions_match")
def _unfence_v6(text):
    return text[7:-3] if text.startswith("```json") and text.endswith("```") else text
def _boolean_v6(value):
    return {"true": True, "false": False}[value] if type(value) is str else value
def parse_judge_v6(task, row):
    answers = parse_json(_unfence_v6(row["text"]))
    return {"schema": JUDGE_V6_SCHEMA_VERSION,
            "grounded": all(_boolean_v6(answers[key]) for key in V6_KEYS)}
'''


def test_v6_missing_fails_closed_without_v5_fallback(modules):
    common, quality = modules(quality=QUALITY + V5)
    report = F._parser_identity(quality, common, "judge_structured_v6")
    assert report["selected_schema"] == "judge_structured_v6"
    assert report["capabilities"] == {"judge_structured_v6": False}
    assert "fingerprint_quality_fixture:parse_judge_v5" not in report["symbols"]
    with pytest.raises(F.FingerprintRefused, match="not implemented"):
        F._require_v6_parser(quality, common)
    assert F._require_v5_parser(quality, common)["capabilities"]["judge_structured_v5"] is True


def test_v6_selected_closure_binds_schema_helpers_and_json(modules):
    common, quality = modules(quality=QUALITY + V5 + V6)
    report = F._require_v6_parser(quality, common)
    assert report["selected_schema"] == "judge_structured_v6"
    assert report["capabilities"] == {"judge_structured_v6": True}
    symbols = report["symbols"]
    for name in ("JUDGE_V6_SCHEMA_VERSION", "V6_KEYS", "_unfence_v6", "_boolean_v6", "parse_judge_v6"):
        assert "fingerprint_quality_fixture:" + name in symbols
    assert "fingerprint_common_fixture:parse_json" in symbols
    assert "fingerprint_common_fixture:_unique_keys" in symbols
    assert "fingerprint_common_fixture:_reject_constant" in symbols
    assert "fingerprint_quality_fixture:parse_judge_v5" not in symbols
    assert report == F._parser_identity(quality, common, "judge_structured_v6")


def test_v6_normalization_change_changes_only_v6_identity(modules):
    first = modules(quality=QUALITY + V5 + V6)
    v5 = F._parser_identity(first[1], first[0])
    v6 = F._require_v6_parser(first[1], first[0])
    changed = V6.replace('value] if type(value) is str', 'value.lower()] if type(value) is str')
    second = modules(quality=QUALITY + V5 + changed)
    assert F._parser_identity(second[1], second[0]) == v5
    assert F._require_v6_parser(second[1], second[0]) != v6


@pytest.mark.parametrize("v6", [
    'JUDGE_V6_SCHEMA_VERSION = "judge_structured_v6"\n',
    V6.replace('JUDGE_V6_SCHEMA_VERSION = "judge_structured_v6"', ''),
    V6.replace('"judge_structured_v6"', '"judge_structured_v5"'),
    'JUDGE_V6_SCHEMA_VERSION = "judge_structured_v6"\nparse_judge_v6 = None\n',
])
def test_partial_or_wrong_v6_capability_is_refused(modules, v6):
    common, quality = modules(quality=QUALITY + V5 + v6)
    with pytest.raises(F.FingerprintRefused, match="structured-v6 parser capability"):
        F._require_v6_parser(quality, common)


def test_v6_loaded_normalizer_replacement_is_refused(modules):
    common, quality = modules(quality=QUALITY + V5 + V6)
    quality._boolean_v6 = lambda value: True
    with pytest.raises(F.FingerprintRefused, match="replaced|differs"):
        F._require_v6_parser(quality, common)


@pytest.mark.parametrize("v6", [
    'JUDGE_V6_SCHEMA_VERSION = "judge_structured_v6"\nparse_judge_v6 = parse_judge_v5\n',
    V6.replace('answers = parse_json(_unfence_v6(row["text"]))',
               'answers = parse_judge_v5(task, row)'),
])
def test_v6_parser_cannot_alias_or_call_v5_parser(modules, v6):
    common, quality = modules(quality=QUALITY + V5 + v6)
    with pytest.raises(F.FingerprintRefused, match="falls back to v5"):
        F._require_v6_parser(quality, common)
