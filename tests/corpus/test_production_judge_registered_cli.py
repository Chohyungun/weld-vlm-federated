"""CPU CLI forwarding only; no real approval, fixture, backend or GPU access."""
import json

import pytest

from corpus.production import __main__ as cli
from corpus.production import judge_cases, judge_probe


def pins():
    return {"expected_manifest_sha256": "1" * 64,
            "expected_protocol": "judge_v7_g150_a_v2",
            "expected_approval_scope": "adjustment_operational_only",
            "expected_source_sha256": {g: str(i) * 64 for i, g in enumerate(("N+", "D+", "ADV1", "ADV2"), 2)}}


def pin_args():
    p = pins()
    result = ["--expected-manifest-sha256", p["expected_manifest_sha256"],
              "--expected-protocol", p["expected_protocol"],
              "--expected-approval-scope", p["expected_approval_scope"]]
    for group, name in zip(("N+", "D+", "ADV1", "ADV2"), ("nplus", "dplus", "adv1", "adv2")):
        result.extend(["--" + name + "-sha256", p["expected_source_sha256"][group]])
    return result


@pytest.mark.parametrize("stage", ["judge-probe", "judge-rules-check"])
def test_registered_probe_and_rules_forward_external_pins(tmp_path, monkeypatch, stage):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({}), encoding="utf-8")
    calls = []
    def receive(*args, **kwargs):
        calls.append((args, kwargs))
        return {"exit_code": 0}
    monkeypatch.setattr(judge_probe, "run_probe" if stage == "judge-probe" else "check_rules", receive)
    args = [stage, "--cases", str(tmp_path / "unread.json"), *pin_args()]
    if stage == "judge-probe":
        args.extend(["--config", str(config), "--output", str(tmp_path / "out")])
    assert cli.main(args) == 0
    assert len(calls) == 1 and calls[0][1] == pins()
    assert not (tmp_path / "out").exists()


def test_partial_registration_refuses_before_dispatch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("partial pins must fail before dispatch")
    monkeypatch.setattr(judge_probe, "check_rules", forbidden)
    assert cli.main(["judge-rules-check", "--cases", str(tmp_path / "unread.json"),
                     "--expected-protocol", "judge_v7_g150_a_v2"]) == 2


def test_registered_builder_preserves_external_pins_and_paths(tmp_path, monkeypatch):
    calls = []
    def receive(*args, **kwargs):
        calls.append((args, kwargs))
        return {}
    monkeypatch.setattr(judge_cases, "write_v7_registered_cases", receive)
    args = ["judge-cases-v7-registered", "--manifest", str(tmp_path / "manifest.json"),
            "--output", str(tmp_path / "out.json")]
    flags = pin_args()
    flags[0] = "--manifest-sha256"
    args.extend(flags)
    for name in ("nplus", "dplus", "adv1", "adv2"):
        args.extend(["--" + name, str(tmp_path / (name + ".json"))])
    assert cli.main(args) == 0
    assert calls[0][1] == {**pins(), "manifest_path": tmp_path / "manifest.json"}
    assert calls[0][0][0]["N+"] == tmp_path / "nplus.json"
    assert calls[0][0][1] == tmp_path / "out.json"
