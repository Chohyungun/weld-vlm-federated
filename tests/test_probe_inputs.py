"""새 페어 혼입·해시 불일치를 모델 적재 전에 거부하는 CPU 시험."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.probe_fakes import write_probe_snapshot
from vlm import probe_inputs as P

ROOT = Path(__file__).resolve().parents[1]
HDIR = ROOT / "docs/dev_log/2026-10-19-본실험복구"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HDIR / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_relocated_snapshot_uses_explicit_expected_digest(tmp_path):
    pairs, digest = write_probe_snapshot(tmp_path / "v2", [{"synthetic": True}])
    got = P.verify_probe_pairs(pairs, digest)
    assert got["snapshot_digest"] == digest and got["verified"]
    assert got["pairs_sha256"] == hashlib.sha256(pairs.read_bytes()).hexdigest()
    assert got["expected_digest_source"] == "argument"


@pytest.mark.parametrize("member", ["pairs.jsonl", "PAIRS_META.json", "counts.json", "discarded.jsonl"])
def test_changed_member_is_rejected_even_if_snapshot_is_unchanged(tmp_path, member):
    pairs, digest = write_probe_snapshot(tmp_path)
    (tmp_path / member).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="해시"):
        P.verify_probe_pairs(pairs, digest)


@pytest.mark.parametrize("value", [None, "", "A" * 64, "x" * 64, "0" * 64])
def test_new_path_requires_external_well_formed_matching_digest(tmp_path, value):
    pairs, _ = write_probe_snapshot(tmp_path)
    with pytest.raises(SystemExit):
        P.verify_probe_pairs(pairs, value)


@pytest.mark.parametrize("change", ["missing", "duplicate_digest", "duplicate_member", "escape", "missing_pairs"])
def test_snapshot_malformed_or_missing_does_not_pass(tmp_path, change):
    pairs, digest = write_probe_snapshot(tmp_path)
    snap = pairs.with_name("SNAPSHOT.sha256")
    text = snap.read_text()
    if change == "missing":
        snap.unlink()
    else:
        if change == "duplicate_digest":
            text += f"# snapshot_digest {digest}\n"
        elif change == "duplicate_member":
            text += text.splitlines()[0] + "\n"
        elif change == "escape":
            text = text.replace("pairs.jsonl", "../pairs.jsonl")
        else:
            text = text.replace("pairs.jsonl", "other.jsonl")
        snap.write_text(text)
    with pytest.raises(SystemExit):
        P.verify_probe_pairs(pairs, digest)


def test_legacy_defaults_only_use_matching_config_not_snapshot_self_claim(tmp_path, monkeypatch):
    from vlm import uni_config as U

    pairs, digest = write_probe_snapshot(tmp_path)
    monkeypatch.setattr(P, "LEGACY_PAIRS", str(pairs))
    cfg = SimpleNamespace(pairs_path=str(pairs), pairs_digest=digest)
    monkeypatch.setattr(U, "load_uni_config", lambda: cfg)
    assert P.verify_probe_pairs(pairs)["expected_digest_source"] == "legacy_config"
    with pytest.raises(SystemExit, match="명시"):
        P.verify_probe_pairs(pairs, require_digest=True)
    cfg.pairs_path = str(tmp_path / "other.jsonl")
    with pytest.raises(SystemExit, match="경로가 다르다"):
        P.verify_probe_pairs(pairs)


@pytest.mark.parametrize("script", ["04_암기시험_C.py", "04b_overlay_C.py", "05_손실분해_C.py"])
def test_each_entry_rejects_mismatch_before_any_model_or_pair_load(tmp_path, monkeypatch, script):
    from vlm import pilot_vlm as PV

    monkeypatch.chdir(ROOT)
    pairs, digest = write_probe_snapshot(tmp_path / "v2")
    pairs.write_bytes(b"tampered")
    mod = load(script)
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("입력 거부 전에 적재했다")

    monkeypatch.setattr(PV, "load_pairs", forbidden)
    monkeypatch.setattr(PV, "resolve_model_loader", forbidden)
    if script.startswith("04_"):
        monkeypatch.setattr(mod, "stub_loader", forbidden)
        args = ["--stage", "1", "--run-name", "probe", "--out-root", str(tmp_path / "out"), "--stub"]
    elif script.startswith("04b"):
        args = ["--pick-stage1", "--name", "probe", "--out-root", str(tmp_path / "out"), "--stub"]
    else:
        args = ["--out", str(tmp_path / "out.json"), "--stub"]
    with pytest.raises(SystemExit, match="해시"):
        mod.main([*args, "--pairs", str(pairs), "--pairs-digest", digest])
    assert not called and not (tmp_path / "out").exists() and not (tmp_path / "out.json").exists()


def test_overlay_reads_selected_dataset_and_records_identity(tmp_path, monkeypatch):
    from tests.uni_fakes import make_rows

    monkeypatch.chdir(ROOT)
    rows = [{**r, "width_px": 64, "height_px": 32} for r in make_rows(tmp_path / "img", "C1", 1)]
    pairs, digest = write_probe_snapshot(tmp_path / "v2", rows)
    ids = tmp_path / "ids.txt"
    ids.write_text(rows[0]["image_id"] + "\n")
    out = tmp_path / "out"
    mod = load("04b_overlay_C.py")
    assert mod.main(["--rows-file", str(ids), "--name", "probe", "--out-root", str(out), "--stub",
                     "--pairs", str(pairs), "--pairs-digest", digest]) == 0
    result = json.loads((out / "probe/summary.json").read_bytes())
    assert result["pairs_input"]["snapshot_digest"] == digest and result["n"] == 1
    assert (out / "probe/00.png").is_file()
