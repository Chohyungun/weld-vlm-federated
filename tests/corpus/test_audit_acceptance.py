"""정본 검증기 판정이 채택 파일에 반영되고 사람 표본지는 블라인드인지 확인한다."""

import json

import pytest

from corpus.generate import run_cycle_corpus as cycle
from corpus.validate import judge_labels


def records():
    return [{"sample_id": f"s{i}", "axis": cycle.AXIS_CLAUSE, "text": "기준 문장",
             "stage0_pass": True, "stage1_pass": True,
             "judge_deepseek_pass": i == 0, "judge_deepseek_parse_ok": True}
            for i in range(2)]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("canonical", [None, "deepseek"])
def test_written_membership_and_report_follow_canonical(tmp_path, monkeypatch, canonical):
    cfg = cycle.load_config()
    cfg["judges"]["canonical"] = canonical
    monkeypatch.setattr(cycle, "git_commit", lambda: "test")
    cycle.write_report(records(), [], cfg["judges"]["candidates"], cfg, {}, tmp_path)
    accepted = read(tmp_path / "reasoning_accepted.jsonl")
    pending = read(tmp_path / "reasoning_pending.jsonl")
    discarded = read(tmp_path / "discarded.jsonl")
    report = json.loads((tmp_path / "cycle_corpus_report.json").read_text(encoding="utf-8"))
    axis = report["axes"][cycle.AXIS_CLAUSE]
    assert axis["n_accepted"] == len(accepted)
    if canonical is None:
        assert not accepted and len(pending) == 2 and not discarded
        assert axis["acceptance_status"] == "pending_canonical"
        assert axis["validated_by"] == "rule"
        assert axis["end_to_end_pass_rate"] is None
    else:
        assert [r["sample_id"] for r in accepted] == ["s0"]
        assert [r["sample_id"] for r in discarded] == ["s1"]
        assert not pending and axis["end_to_end_pass_rate"] == 0.5
        assert axis["stages"]["stage2_canonical"]["n_pass"] == 1


@pytest.mark.parametrize("case", ["not_run", "missing", "wrong_type"])
def test_canonical_missing_or_invalid_does_not_publish(tmp_path, monkeypatch, case):
    cfg = cycle.load_config()
    cfg["judges"]["canonical"] = "deepseek"
    recs = records()
    cands = cfg["judges"]["candidates"] if case != "not_run" else []
    if case != "not_run":
        recs[0]["judge_deepseek_pass"] = None if case == "missing" else "false"
    monkeypatch.setattr(cycle, "git_commit", lambda: "test")
    with pytest.raises(ValueError):
        cycle.write_report(recs, [], cands, cfg, {}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_human_sheet_omits_judge_information_and_block_order():
    cfg = judge_labels.load_cfg()
    recs = [{"sample_id": f"s{i:03}", "axis": cfg["labeling"]["strata"][0][i % 2],
             "judge_deepseek_pass": i % 4 < 2, "text": "문장"} for i in range(160)]
    sheet = judge_labels.build_sheet(recs, cfg, "deepseek")
    assert sheet == judge_labels.build_sheet(recs, cfg, "deepseek")
    assert "judge_pass" not in json.dumps(sheet) and "judge_fail" not in json.dumps(sheet)
    assert all("stratum" not in r for r in sheet)
    source = {r["sample_id"]: r for r in recs}
    groups = [judge_labels._stratum(source[r["sample_id"]], "deepseek") for r in sheet]
    assert len(set(groups)) == 4
    assert sum(a != b for a, b in zip(groups, groups[1:])) > 3
