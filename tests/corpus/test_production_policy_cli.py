"""새 정책 CLI 연결만 검사한다. 실제 모델이나 실물 명부는 읽지 않는다."""

import json

import pytest

from corpus.production.__main__ import main
from corpus.production.common import write_json_new


def test_regeneration_forwards_parent_config(tmp_path, monkeypatch, capsys):
    from corpus.production import generation_policy
    config, path = {"recipe": "parent"}, tmp_path / "config.json"
    write_json_new(path, config)
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"run_dir": "new"}

    monkeypatch.setattr(generation_policy, "prepare_regeneration", capture)
    assert main(["prepare-regeneration", "--parent", str(tmp_path / "parent"),
                 "--run", str(tmp_path / "new"), "--config", str(path)]) == 0
    assert calls == [((tmp_path / "parent", tmp_path / "new", config), {})]
    assert json.loads(capsys.readouterr().out) == {"run_dir": "new"}
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("override", [False, True])
def test_separate_judge_forwards_explicit_path_and_model_spec(tmp_path, monkeypatch, override):
    from corpus.production import infer
    config, spec = {"judge": {"model_id": "old"}}, {"model_id": "new"}
    path, spec_path = tmp_path / "config.json", tmp_path / "judge.json"
    write_json_new(path, config)
    write_json_new(spec_path, spec)
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"exit_code": 4}

    monkeypatch.setattr(infer, "run_judge", capture)
    argv = ["judge-separate", "--run", str(tmp_path / "run"), "--config", str(path),
            "--judge-dir", str(tmp_path / "new_judge"), "--max-shards", "1",
            "--probe-approval", str(tmp_path / "approval"), "--probe-approval-sha256", "a" * 64]
    if override:
        argv += ["--judge-config", str(spec_path)]
    assert main(argv) == 4
    assert calls == [((tmp_path / "run", config, tmp_path / "new_judge"),
                      {"judge_config": spec if override else None, "max_shards": 1,
                       "probe_approval": tmp_path / "approval", "probe_approval_sha256": "a" * 64})]


@pytest.mark.parametrize("explicit", [False, True])
def test_audit_selects_only_explicit_judge_source(tmp_path, monkeypatch, explicit):
    from corpus.production import infer, quality
    path = tmp_path / "config.json"
    write_json_new(path, {})
    selected = {"identity_sha256": "new", "rows": []}
    calls, reads = [], []

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return selected

    def audit(*args, **kwargs):
        calls.append((args, kwargs))
        return {"complete": False}

    monkeypatch.setattr(infer, "read_judge_source", read)
    monkeypatch.setattr(quality, "audit_run", audit)
    argv = ["audit", "--run", str(tmp_path / "run"), "--config", str(path),
            "--output", str(tmp_path / "audit")]
    if explicit:
        argv += ["--judge-dir", str(tmp_path / "judge"),
                 "--expected-judge-identity-sha256", "a" * 64,
                 "--expected-probe-approval-sha256", "b" * 64]
    assert main(argv) == 0
    assert calls == [((tmp_path / "run", {}, tmp_path / "audit"),
                      {"judge_source": selected} if explicit else {})]
    assert reads == ([((tmp_path / "run", {}, tmp_path / "judge"),
                       {"expected_judge_identity_sha256": "a" * 64,
                        "expected_probe_approval_sha256": "b" * 64})] if explicit else [])


def test_explicit_missing_judge_never_calls_legacy_audit(tmp_path, monkeypatch, capsys):
    from corpus.production import infer, quality
    path = tmp_path / "config.json"
    write_json_new(path, {})

    def missing(*args, **kwargs):
        raise ValueError("missing selected JUDGE.json")

    def forbidden(*args, **kwargs):
        pytest.fail("explicit judge must not fall back to legacy")

    monkeypatch.setattr(infer, "read_judge_source", missing)
    monkeypatch.setattr(quality, "audit_run", forbidden)
    assert main(["audit", "--run", str(tmp_path / "run"), "--config", str(path),
                 "--output", str(tmp_path / "audit"), "--judge-dir", str(tmp_path / "judge")]) == 2
    assert "missing selected" in capsys.readouterr().err


def test_legacy_consumer_cannot_silently_accept_separate_source(tmp_path, monkeypatch, capsys):
    from corpus.production import infer, quality
    path = tmp_path / "config.json"
    write_json_new(path, {})

    def legacy(run_dir, config, output):
        pytest.fail("legacy consumer must not run for a separate judge")

    def forbidden(*args):
        pytest.fail("refuse the unsupported consumer before reading real shards")

    monkeypatch.setattr(quality, "audit_run", legacy)
    monkeypatch.setattr(infer, "read_judge_source", forbidden)
    assert main(["audit", "--run", str(tmp_path / "run"), "--config", str(path),
                 "--output", str(tmp_path / "audit"), "--judge-dir", str(tmp_path / "judge")]) == 2
    assert "not integrated" in capsys.readouterr().err


def test_audit_never_promotes_source_self_hash_to_external_pin(tmp_path, monkeypatch, capsys):
    from corpus.production import infer, quality
    path = tmp_path / "config.json"
    write_json_new(path, {})
    calls = []
    def requires_external(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("judge_source_pin_required")
    def forbidden(*args, **kwargs):
        pytest.fail("unapproved source must not reach audit")
    monkeypatch.setattr(infer, "read_judge_source", requires_external)
    monkeypatch.setattr(quality, "audit_run", forbidden)
    assert main(["audit", "--run", str(tmp_path / "run"), "--config", str(path),
                 "--output", str(tmp_path / "audit"), "--judge-dir", str(tmp_path / "judge")]) == 2
    assert calls == [{"expected_judge_identity_sha256": None, "expected_probe_approval_sha256": None}]
    assert "judge_source_pin_required" in capsys.readouterr().err


def test_external_pin_without_explicit_source_never_uses_legacy_audit(tmp_path, monkeypatch, capsys):
    from corpus.production import quality
    path = tmp_path / "config.json"
    write_json_new(path, {})
    def forbidden(*args, **kwargs):
        pytest.fail("external pins must not be silently ignored")
    monkeypatch.setattr(quality, "audit_run", forbidden)
    assert main(["audit", "--run", str(tmp_path / "run"), "--config", str(path),
                 "--output", str(tmp_path / "audit"),
                 "--expected-probe-approval-sha256", "b" * 64]) == 2
    assert "explicit --judge-dir" in capsys.readouterr().err


@pytest.mark.parametrize("stage,module_name,function,options", [
    ("judge-cases", "judge_cases", "write_cases", ("--nplus", "--output")),
    ("judge-rules-check", "judge_probe", "check_rules", ("--cases",)),
    ("compare-judge-probes", "judge_probe", "compare_probes", ("--first", "--second")),
    ("seal-judge-probe-approval", "judge_probe", "seal_probe_approval", ("--first", "--second", "--output")),
])
def test_cpu_probe_commands_forward_only_named_paths(tmp_path, monkeypatch, stage, module_name, function, options):
    import importlib
    module = importlib.import_module(f"corpus.production.{module_name}")
    paths = [tmp_path / str(i) for i in range(len(options))]
    calls = []

    def capture(*args):
        calls.append(args)
        return {"exit_code": 0}

    monkeypatch.setattr(module, function, capture)
    argv = [stage]
    for option, path in zip(options, paths):
        argv += [option, str(path)]
    assert main(argv) == 0
    assert calls == [tuple(paths)]


def test_actual_probe_requires_config_and_forwards_once(tmp_path, monkeypatch):
    from corpus.production import judge_probe
    path = tmp_path / "config.json"
    write_json_new(path, {"judge": {"model_id": "test"}})
    calls = []

    def capture(*args):
        calls.append(args)
        return {"exit_code": 3, "complete": False}

    monkeypatch.setattr(judge_probe, "run_probe", capture)
    assert main(["judge-probe", "--cases", str(tmp_path / "cases.json"),
                 "--config", str(path), "--output", str(tmp_path / "probe")]) == 3
    assert calls == [(tmp_path / "cases.json", {"judge": {"model_id": "test"}}, tmp_path / "probe")]


def test_v5_cases_forwards_all_four_original_paths_and_external_pins(tmp_path, monkeypatch):
    from corpus.production import judge_cases
    groups = {"N+": "nplus", "D+": "dplus", "ADV1": "adv1", "ADV2": "adv2"}
    paths = {group: tmp_path / (name + ".json") for group, name in groups.items()}
    pins = {group: str(i + 1) * 64 for i, group in enumerate(groups)}
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"protocol": "judge_v5_g150", "cases": 150}
    monkeypatch.setattr(judge_cases, "write_v5_cases", capture)
    argv = ["judge-cases-v5", "--output", str(tmp_path / "bundle.json")]
    for group, name in groups.items():
        argv += ["--" + name, str(paths[group]), "--" + name + "-sha256", pins[group]]
    assert main(argv) == 0
    assert calls == [((paths, tmp_path / "bundle.json"), {"expected_source_sha256": pins})]
    assert not (tmp_path / "bundle.json").exists()


def test_v5_cases_does_not_invent_missing_external_pins(tmp_path):
    argv = ["judge-cases-v5", "--output", str(tmp_path / "bundle.json")]
    for name in ("nplus", "dplus", "adv1", "adv2"):
        argv += ["--" + name, str(tmp_path / (name + ".json"))]
    with pytest.raises(SystemExit):
        main(argv)
    assert not (tmp_path / "bundle.json").exists()
