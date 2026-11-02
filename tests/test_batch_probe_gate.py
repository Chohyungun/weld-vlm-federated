"""묶음 탐침 소비자(`scripts/probe/batch_probe_gate.py`) — 종료 코드가 아니라 JSON 의 신원 · 완전성 · 동치 · 잔여 묶음 · 메모리로 정한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe import batch_probe_gate as G

HEAD, AD = "c" * 40, "a" * 64


def probe(n=21, n8_status="ok", rate=1.0, peak=8.5, **over) -> dict:
    p = {"code_commit": HEAD, "adapter_sha256": AD, "rows_sha256": "r" * 64, "n_rows": n,
         "passes": {"1": {"status": "ok", "records": {f"i{k}": {} for k in range(n)}, "peak_mem_gb": 3.9},
                    "8": {"status": n8_status, "peak_mem_gb": peak,
                          "compare_vs_batch1": {k: rate for k in G.RATES}}}}
    p.update(over)
    return p


def test_passes_only_when_everything_holds():
    assert G.problems_of(probe(), batch_size=8, adapter_sha256=AD, head=HEAD, rows_sha256="r" * 64, gpu_total_gb=40) == []


@pytest.mark.parametrize(("p", "kw", "frag"), [
    (probe(n8_status="oom"), {}, "ok 가 아니다"),
    (probe(rate=0.99), {}, "일치율"),
    (probe(n=24), {}, "배수"),
    (probe(n=8), {}, "최소"),
    (probe(peak=30.0), {"gpu_total_gb": 40}, "메모리"),
    (probe(code_commit="d" * 40), {}, "HEAD"),
    (probe(adapter_sha256="b" * 64), {}, "adapter"),
    (probe(), {"rows_sha256": "x" * 64}, "rows_sha256"),
])
def test_blocks_on_each_failure(p, kw, frag):
    bad = G.problems_of(p, batch_size=8, adapter_sha256=AD, head=HEAD, **kw)
    assert bad and any(frag in b for b in bad)


def test_cli_exit_code_and_receipt(tmp_path, monkeypatch):
    class R:
        stdout = HEAD + "\n"

    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: R())
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(probe()), encoding="utf-8")
    bad.write_text(json.dumps(probe(n8_status="oom")), encoding="utf-8")
    base = ["--batch-size", "8", "--adapter-sha256", AD, "--gpu-total-gb", "40"]
    assert G.main(["--probe", str(bad), *base, "--out", str(tmp_path / "b.json")]) == 2
    assert not (tmp_path / "b.json").exists()                             # 차단이면 판정 파일 없음
    assert G.main(["--probe", str(good), *base, "--out", str(tmp_path / "g.json")]) == 0
    assert json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))["pass"] is True
    assert G.main(["--probe", str(good), *base, "--out", str(tmp_path / "g.json")]) == 2   # 덮지 않는다
