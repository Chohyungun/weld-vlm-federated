"""cycle_pilot 근거 보존 회귀 방지 — 74번 감사 P5.

원본 jsonl 은 `.gitignore` 로 미추적이라 워크트리를 정리하면 사라진다. 통과율
0.94 / 0.345 의 근거는 추적된 축약본만으로 재계산돼야 하고, 그 값이 보고서와
어긋나면 조용히 지나가면 안 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.generate import snapshot_cycle_pilot as S

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "corpus/generate/cycle_pilot/EVIDENCE.jsonl"
SUMMARY = REPO / "corpus/generate/cycle_pilot/EVIDENCE_SUMMARY.json"
SNAPSHOT = REPO / "corpus/generate/cycle_pilot/SNAPSHOT.sha256"
CRLF = bytes([13, 10])


def _items() -> list[dict]:
    return [json.loads(x) for x in EVIDENCE.read_text(encoding="utf-8").splitlines() if x]


def test_근거_삼종이_추적본으로_존재한다():
    for p in (EVIDENCE, SUMMARY, SNAPSHOT):
        assert p.exists(), f"{p.name} 이 없다 — snapshot_cycle_pilot 실행"
        assert CRLF not in p.read_bytes(), f"{p.name} 이 CRLF 다"


def test_통과율은_축약본만으로_재계산된다():
    """원본 jsonl 이 없어도 성립해야 한다 — 그게 이 축약본의 존재 이유다."""
    got = S.recompute(_items())
    saved = json.loads(SUMMARY.read_text(encoding="utf-8"))["recomputed"]
    assert got == saved
    assert got["qa"]["pass_rate"] == 0.94
    assert got["reasoning:deepseek"]["end_to_end_rate"] == 0.345
    assert got["reasoning:phi"]["stage2_judge"]["pass_rate"] == 0.9551


def test_보고서_대조가_기록돼_있다():
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["crosscheck_vs_reports"] == "일치"


def test_축약본은_생성문_전문을_담지_않는다():
    """추적본이라 용량과 전재 양쪽에서 전문을 담으면 안 된다. 대조는 sha256 으로 한다."""
    for r in _items():
        assert "text" not in r
        assert len(r["text_sha256"]) == 64


def test_원본과_한건씩_대조된다():
    """원본이 워크트리에 남아 있는 동안은 실제로 대조된다 (없으면 건너뛴다)."""
    src = REPO / "corpus/generate/cycle_pilot/reasoning_accepted.jsonl"
    if not src.exists():
        pytest.skip("원본이 워크트리에 없다 — 드라이브 보관분")
    want = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            want[r["sample_id"]] = S.sha256_bytes(r["text"].encode("utf-8"))
    got = {r["sample_id"]: r["text_sha256"] for r in _items()
           if r["source_file"] == "reasoning_accepted.jsonl"}
    assert got == want


def test_대조_불일치는_조용히_지나가지_않는다(tmp_path):
    """보고서 수치가 축약본과 어긋나면 사유가 나와야 한다 — 안 그러면 틀린 값이 고정된다."""
    (tmp_path / "cycle_corpus_report.json").write_text(
        json.dumps({"reasoning": {"stage0": {"n_pass": 999, "n_fail": 1},
                                  "stage2_judge": {"n_pass": 1, "n_fail": 1},
                                  "n_accepted": 1},
                    "qa": {"n_accepted": 1}}, ensure_ascii=False),
        encoding="utf-8")
    problems = S.crosscheck(S.recompute(_items()), tmp_path)
    assert any("n_pass" in p for p in problems)
    assert any("_phi_report.json: 대조 대상 없음" == p for p in problems)
