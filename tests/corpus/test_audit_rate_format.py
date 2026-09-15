"""새 보고서의 비율 표기와 미검증 상태를 함께 확인한다."""

import json

import pytest

from corpus.generate import run_cycle_corpus as cycle
from corpus.generate import snapshot_cycle_pilot as snapshot


@pytest.mark.parametrize(
    ("canonical", "n_items", "expected"),
    [("deepseek", 3, 0.3333), ("deepseek", 0, None), (None, 3, None)],
)
def test_end_to_end_rate_rounding_preserves_pending(
    tmp_path, monkeypatch, canonical, n_items, expected,
):
    cfg = cycle.load_config()
    cfg["judges"]["canonical"] = canonical
    monkeypatch.setattr(cycle, "git_commit", lambda: "test")
    recs = [
        {"sample_id": f"s{i}", "axis": cycle.AXIS_CLAUSE, "text": "기준 문장",
         "stage0_pass": True, "stage1_pass": True,
         "judge_deepseek_pass": i == 0, "judge_deepseek_parse_ok": True}
        for i in range(n_items)
    ]
    for i, rec in enumerate(recs):
        for candidate in cfg["judges"]["candidates"]:
            rec[f"judge_{candidate['id']}_pass"] = i == 0
            rec[f"judge_{candidate['id']}_parse_ok"] = True
    cycle.write_report(recs, [], cfg["judges"]["candidates"], cfg, {}, tmp_path)
    report = json.loads((tmp_path / "cycle_corpus_report.json").read_text(encoding="utf-8"))
    axis = report["axes"][cycle.AXIS_CLAUSE]
    assert axis["end_to_end_pass_rate"] == expected
    assert axis["n_accepted"] == (int(n_items > 0) if canonical else 0)
    assert axis["n_pending"] == (0 if canonical else n_items)
    evidence = snapshot.build_evidence(tmp_path)
    assert snapshot.crosscheck(snapshot.recompute_axes(evidence), tmp_path) == []
    monkeypatch.setattr("sys.argv", ["snapshot", "--dir", str(tmp_path)])
    assert snapshot.main() == 0
    saved = {name: (tmp_path / name).read_bytes() for name in snapshot.DERIVED}
    monkeypatch.setattr("sys.argv", ["snapshot", "--dir", str(tmp_path), "--check"])
    assert snapshot.main() == 0
    assert saved == {name: (tmp_path / name).read_bytes() for name in snapshot.DERIVED}
