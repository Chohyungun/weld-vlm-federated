"""Selected v7 identity is independent of report/release code, not semantics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import corpus.production
from corpus.production.common import canonical_bytes
from corpus.production import judge_fingerprints as F, judge_policy as P, quality as Q


def test_policy_v7_identity_is_equal_in_three_hash_seed_processes():
    """Use fresh interpreters and the real selected parser, including an A-tip overlay."""
    root = Path(__file__).resolve().parents[2]
    code = """
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import corpus.production
corpus.production.__path__[:] = json.loads(sys.argv[2])
from corpus.production import judge_policy as P, judge_grounded_v7 as G
from corpus.production.common import canonical_bytes
identity = P.policy_identity_v7()
print(json.dumps({'policy_sha256': hashlib.sha256(canonical_bytes(identity)).hexdigest(),
                  'implementation_sha256': identity['implementation']['closure_sha256'],
                  'parser_ready': identity['structured_parser_ready'],
                  'names': list(G.NAMES)}, ensure_ascii=False, sort_keys=True))
"""
    outputs = []
    for seed in ("0", "1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "-c", code, str(root),
             json.dumps(list(corpus.production.__path__))],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == outputs[2], outputs
    assert outputs[0]["policy_sha256"] == hashlib.sha256(canonical_bytes(P.policy_identity_v7())).hexdigest()
    assert outputs[0]["parser_ready"] is P.model_parser_ready_v7()
    assert outputs[0]["names"] == sorted(outputs[0]["names"], key=lambda name: (-len(name), name))


def test_policy_v7_actual_parser_closure():
    if not hasattr(Q, "parse_judge_v7"):
        pytest.skip("A-tip consumer overlay is required until the combined gate")
    result = P.policy_identity_v7()
    assert result["structured_parser_ready"] is True
    assert result["parser_implementation"] == F.require_v7_parser()
    assert result["parser_implementation"]["selected_schema"] == "judge_structured_v7"
    assert all(not key.endswith((":parse_judge_v5", ":parse_judge_v6"))
               for key in result["parser_implementation"]["symbols"])


def test_unrelated_consumer_function_not_part_of_v7_identity(monkeypatch):
    if not hasattr(Q, "parse_judge_v7"):
        pytest.skip("requires combined A-tip parser")
    before = P.policy_identity_v7()
    monkeypatch.setattr(Q, "audit_run", lambda *a, **k: None)
    assert P.policy_identity_v7() == before


def test_replaced_v7_policy_function_refused(monkeypatch):
    monkeypatch.setattr(P, "evaluate_v7", lambda *a, **k: {})
    with pytest.raises(F.FingerprintRefused, match="loaded policy code"):
        F.policy_implementation_v7()


def test_replaced_canonical_grounded_refused(monkeypatch):
    from corpus.production import judge_grounded_v7 as G
    monkeypatch.setattr(G, "grounded_codes", lambda *a: [])
    with pytest.raises(F.FingerprintRefused, match="loaded policy code"):
        F.policy_implementation_v7()


def test_replaced_canonical_normal_rules_refused(monkeypatch):
    from corpus.production import judge_grounded_v7 as G
    monkeypatch.setattr(G, "normal_codes", lambda *a: [])
    with pytest.raises(F.FingerprintRefused, match="loaded policy code"):
        F.policy_implementation_v7()


def test_serializer_alias_replacement_refused(monkeypatch):
    monkeypatch.setattr(P, "canonical_bytes", lambda value: b"{}")
    with pytest.raises(F.FingerprintRefused, match="dependency binding"):
        F.policy_implementation_v7()


def test_rule_runtime_mutation_changes_identity(monkeypatch):
    before = F.policy_implementation_v7()
    monkeypatch.setattr(P, "V7_VERDICT_GAP", ("different-rule",))
    assert F.policy_implementation_v7()["closure_sha256"] != before["closure_sha256"]


def test_missing_v7_parser_is_not_v6_fallback(monkeypatch):
    monkeypatch.delattr(Q, "parse_judge_v7", raising=False)
    monkeypatch.delattr(Q, "JUDGE_V7_SCHEMA_VERSION", raising=False)
    result = F.parser_identity("judge_structured_v7")
    assert result["capabilities"]["judge_structured_v7"] is False
    with pytest.raises(F.FingerprintRefused):
        F.require_v7_parser()


def test_forged_cached_label_wrapper_refused(monkeypatch):
    def wrapper(*args):
        return ("not-a-label",)
    wrapper.__wrapped__ = P._load_defect_names.__wrapped__
    monkeypatch.setattr(P, "_load_defect_names", wrapper)
    with pytest.raises(F.FingerprintRefused, match="decorated policy symbol"):
        F.policy_implementation_v7()


def test_replaced_live_label_loader_refused(monkeypatch):
    from data import label_map
    monkeypatch.setattr(label_map, "load_label_map", lambda *a: None)
    with pytest.raises(F.FingerprintRefused, match="label loader"):
        F.policy_implementation_v7()
