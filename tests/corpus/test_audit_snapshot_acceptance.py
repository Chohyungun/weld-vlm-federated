"""채택 보고서와 스냅샷 소비자의 연결. GPU 없이 실제 파일 경로를 왕복한다."""

import json
from pathlib import Path

import pytest

from corpus.generate import run_cycle_corpus as cycle
from corpus.generate import snapshot_cycle_pilot as snapshot


def _write_cycle(tmp_path, monkeypatch, canonical, *, empty=False, mixed=False):
    cfg = cycle.load_config()
    cfg["judges"]["canonical"] = canonical
    records = []
    for i in range(5):
        rec = {"sample_id": f"sample-{i}", "axis": cycle.AXIS_CLAUSE,
               "text": f"기준 문장 {i}", "stage0_pass": i != 4,
               "stage0_reasons": []}
        if i != 4:
            rec["stage1_pass"] = i != 3
        if i < 3:
            for candidate in cfg["judges"]["candidates"]:
                cid = candidate["id"]
                rec[f"judge_{cid}_pass"] = i == (0 if cid == "deepseek" else 1) or i == 2
                rec[f"judge_{cid}_parse_ok"] = i != 2
        records.append(rec)
    if empty:
        records = []
    qa = []
    if mixed:
        records += [
            {"sample_id": f"remedy-{i}", "axis": cycle.AXIS_REMEDY,
             "text": "조치", "stage0_pass": i == 0} for i in range(2)
        ]
        qa = [{"sample_id": f"qa-{i}", "text": "질문과 답", "stage0_pass": i == 0}
              for i in range(2)]
    monkeypatch.setattr(cycle, "git_commit", lambda: "test-only")
    cycle.write_report(records, qa, cfg["judges"]["candidates"], cfg, {}, tmp_path)
    return json.loads((tmp_path / "cycle_corpus_report.json").read_text(encoding="utf-8"))


def _recompute(tmp_path):
    return snapshot.recompute_axes(snapshot.build_evidence(tmp_path))


@pytest.mark.parametrize("canonical", [None, "deepseek", "phi"])
@pytest.mark.parametrize("empty", [False, True])
def test_report_to_evidence_to_seal_roundtrip(tmp_path, monkeypatch, capsys, canonical, empty):
    _write_cycle(tmp_path, monkeypatch, canonical, empty=empty)
    items = snapshot.build_evidence(tmp_path)
    header = next(r for r in items if r["kind"] == snapshot.POLICY_KIND)
    assert "n_accepted" not in json.dumps(header) and "n_pending" not in json.dumps(header)
    assert all(r["judge"] == "-" for r in items if r["kind"] != snapshot.POLICY_KIND)
    got = snapshot.recompute_axes(items)
    loaded = [json.loads(line) for line in snapshot.render_evidence(items).splitlines()]
    assert snapshot.recompute_axes(loaded) == got
    axis = got[f"axis:{cycle.AXIS_CLAUSE}"]
    assert axis["n_in"] == (0 if empty else 5)
    assert axis["n_accepted"] == (0 if empty or canonical is None else 1)
    assert axis["n_pending"] == (3 if not empty and canonical is None else 0)
    if canonical is not None:
        assert axis["stages"]["stage2_canonical"]["n_pass"] == (0 if empty else 1)
        assert axis["stages"]["stage2_canonical"]["n_in"] == (0 if empty else 3)
    assert snapshot.crosscheck(got, tmp_path) == []
    # 조치·QA가 0건인 축도 보고서와 연결돼야 한다.
    assert got[f"axis:{cycle.AXIS_REMEDY}"]["n_in"] == 0
    assert got["axis:QA"]["n_in"] == 0
    monkeypatch.setattr("sys.argv", ["snapshot", "--dir", str(tmp_path)])
    assert snapshot.main() == 0
    stdout = capsys.readouterr().out
    if canonical is None:
        assert f"채택 0 / 보류 {0 if empty else 3} / 정본 미선정" in stdout
    summary = json.loads((tmp_path / snapshot.SUMMARY).read_text(encoding="utf-8"))
    assert "정책 헤더 1건" in summary["_meta"]["evidence_count_note"]
    saved = {name: (tmp_path / name).read_bytes() for name in snapshot.DERIVED}
    monkeypatch.setattr("sys.argv", ["snapshot", "--dir", str(tmp_path), "--check"])
    assert snapshot.main() == 0
    assert saved == {name: (tmp_path / name).read_bytes() for name in snapshot.DERIVED}


@pytest.mark.parametrize("field", ["n_accepted", "n_pending", "canonical_n_in"])
def test_report_count_tampering_is_rejected(tmp_path, monkeypatch, field):
    report = _write_cycle(tmp_path, monkeypatch, "deepseek")
    assert snapshot.crosscheck(_recompute(tmp_path), tmp_path) == []
    axis = report["axes"][cycle.AXIS_CLAUSE]
    if field == "canonical_n_in":
        axis["stages"]["stage2_canonical"]["n_in"] += 1
    else:
        axis[field] += 1
    cycle.write_json(tmp_path / "cycle_corpus_report.json", report)
    assert snapshot.crosscheck(_recompute(tmp_path), tmp_path)


def test_actual_membership_cannot_override_canonical_flags(tmp_path, monkeypatch):
    _write_cycle(tmp_path, monkeypatch, "deepseek")
    assert snapshot.crosscheck(_recompute(tmp_path), tmp_path) == []
    accepted = tmp_path / "reasoning_accepted.jsonl"
    (tmp_path / "reasoning_pending.jsonl").write_bytes(accepted.read_bytes())
    accepted.write_bytes(b"")
    try:
        problems = snapshot.crosscheck(_recompute(tmp_path), tmp_path)
    except ValueError:
        return  # 모순된 항목은 근거 생성 단계에서 거부해도 된다.
    assert problems, "보고서의 수치를 복사해 잘못된 실제 배출 파일을 통과시키면 안 된다"


@pytest.mark.parametrize("canonical", [None, "deepseek", "phi"])
def test_remedy_and_qa_members_are_not_lost(tmp_path, monkeypatch, canonical):
    _write_cycle(tmp_path, monkeypatch, canonical, mixed=True)
    got = _recompute(tmp_path)
    assert snapshot.crosscheck(got, tmp_path) == []
    assert got[f"axis:{cycle.AXIS_REMEDY}"]["n_in"] == 2
    assert got["qa"]["n_in"] == 2 and got["qa"]["n_accepted"] == 1


@pytest.mark.parametrize("value", [True, -1, "1", None])
def test_invalid_count_types_are_not_accepted(tmp_path, monkeypatch, value):
    report = _write_cycle(tmp_path, monkeypatch, "deepseek")
    assert snapshot.crosscheck(_recompute(tmp_path), tmp_path) == []
    report["axes"][cycle.AXIS_CLAUSE]["n_accepted"] = value
    cycle.write_json(tmp_path / "cycle_corpus_report.json", report)
    assert snapshot.crosscheck(_recompute(tmp_path), tmp_path)


@pytest.mark.parametrize("fault", ["missing_pending", "duplicate", "invalid_flag", "swap"])
def test_invalid_evidence_is_not_sealed(tmp_path, monkeypatch, fault):
    _write_cycle(tmp_path, monkeypatch, None)
    assert snapshot.crosscheck(_recompute(tmp_path), tmp_path) == []
    pending = tmp_path / "reasoning_pending.jsonl"
    if fault == "missing_pending":
        pending.unlink()  # 이 시험이 만든 임시 파일만 제거한다.
    else:
        rows = [json.loads(line) for line in pending.read_text(encoding="utf-8").splitlines()]
        if fault == "duplicate":
            rows.append(dict(rows[0]))
        elif fault == "invalid_flag":
            rows[0]["judge_deepseek_parse_ok"] = "false"
        else:
            discarded = tmp_path / "discarded.jsonl"
            rejects = [json.loads(line) for line in discarded.read_text(encoding="utf-8").splitlines()]
            rows[0], rejects[0] = rejects[0], rows[0]
            cycle.write_jsonl(discarded, rejects)
        cycle.write_jsonl(pending, rows)
    monkeypatch.setattr("sys.argv", ["snapshot", "--dir", str(tmp_path)])
    assert snapshot.main() == 3
    assert not (tmp_path / snapshot.SNAPSHOT).exists()


@pytest.mark.parametrize("version", ["cycle_pilot", "cycle_pilot_v2"])
def test_legacy_evidence_and_summary_bytes_are_unchanged(version):
    folder = Path(__file__).resolve().parents[2] / "corpus" / "generate" / version
    evidence_bytes = (folder / snapshot.EVIDENCE).read_bytes()
    items = [json.loads(line) for line in evidence_bytes.decode("utf-8").splitlines() if line]
    assert snapshot.render_evidence(items).encode("utf-8") == evidence_bytes
    recompute = snapshot.recompute if version == "cycle_pilot" else snapshot.recompute_axes
    got = recompute(items)
    old_summary = (folder / snapshot.SUMMARY).read_bytes()
    summary = json.loads(old_summary)
    assert got == summary["recomputed"]
    entries = []
    for line in (folder / snapshot.SNAPSHOT).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            entries.append((digest, name))
    assert snapshot.render_summary(got, entries, [], len(items)).encode("utf-8") == old_summary
