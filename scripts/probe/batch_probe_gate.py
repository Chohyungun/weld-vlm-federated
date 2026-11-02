"""묶음 탐침 소비자 — `scripts/probe/batch_equivalence.py` 의 JSON 을 읽어 **허용 · 차단**을 정한다(Codex 10-31 F5 · P1-2).

탐침은 보고 도구라 메모리 부족(`status=oom`) · 토큰 불일치에도 종료 코드 0 으로 끝난다. 실행 감싸개는 종료 코드가 아니라 이 소비자의
판정으로 다음 단계를 연다. 판정은 아래가 **모두** 맞을 때만 통과다.

| 항목 | 조건 |
|---|---|
| 신원 | `code_commit` = 지금 체크아웃 HEAD · `adapter_sha256` = 기대 어댑터 · `rows_sha256` = 기대 목록(주면) |
| 완전성 | 배치 1 과 등록 배치 N 이 모두 `status == "ok"` · 배치 1 의 기록 수 = `n_rows` ≥ 최소 행 수 |
| 동치 | 배치 N 의 `compare_vs_batch1.tokens_equal_rate == 1.0`(텍스트 · 파싱 · 멈춤도 1.0) |
| 잔여 묶음 | `n_rows % N != 0` — 등록 크기보다 작은 마지막 묶음이 실제로 지나갔다(재개 직후 묶음의 대역) |
| 메모리 | 배치 N 의 최대 할당 메모리 ≤ GPU 전체 메모리 × 상한 비율(기본 2/3) |

통과면 `BATCH_PROBE.json`(판정 · 근거 sha)을 배타 생성하고 0, 아니면 사유를 찍고 2. 판정 파일이 이미 있으면 덮지 않는다(2).

    python -X utf8 -B scripts/probe/batch_probe_gate.py --probe <batch_equivalence.json> --batch-size 8 \
        --adapter-sha256 <hex> [--rows-sha256 <hex>] [--min-rows 16] [--gpu-total-gb 40] [--mem-ratio 0.667] --out <BATCH_PROBE.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATES = ("tokens_equal_rate", "text_equal_rate", "parse_equal_rate", "stop_equal_rate")


def problems_of(probe: dict, *, batch_size: int, adapter_sha256: str, head: str, rows_sha256: str | None = None,
                min_rows: int = 16, gpu_total_gb: float | None = None, mem_ratio: float = 2 / 3) -> list[str]:
    """탐침 JSON → 어긋남 목록(비면 통과). 순수 함수 — 시험이 직접 부른다."""
    out: list[str] = []
    if not isinstance(probe, dict):
        return ["탐침 JSON 이 객체가 아니다"]
    if probe.get("code_commit") != head:
        out.append(f"code_commit {probe.get('code_commit')!r} 가 체크아웃 HEAD {head!r} 와 다르다")
    if probe.get("adapter_sha256") != adapter_sha256:
        out.append("adapter_sha256 이 기대 어댑터와 다르다")
    if rows_sha256 is not None and probe.get("rows_sha256") != rows_sha256:
        out.append("rows_sha256 이 기대 목록과 다르다")
    n = probe.get("n_rows")
    passes = probe.get("passes") if isinstance(probe.get("passes"), dict) else {}
    one, pn = passes.get("1"), passes.get(str(batch_size))
    if type(n) is not int or n < min_rows:
        out.append(f"n_rows {n!r} 가 최소 {min_rows} 보다 작다")
    if not isinstance(one, dict) or one.get("status") != "ok" or len(one.get("records") or {}) != n:
        out.append("배치 1 이 ok 가 아니거나 기록 수가 n_rows 와 다르다")
    if batch_size < 2:
        out.append("등록 배치가 2 이상이 아니다 — 묶음 탐침의 대상이 아니다")
    elif not isinstance(pn, dict) or pn.get("status") != "ok":
        out.append(f"배치 {batch_size} 가 ok 가 아니다(status {None if not isinstance(pn, dict) else pn.get('status')!r})")
    else:
        cmp = pn.get("compare_vs_batch1") if isinstance(pn.get("compare_vs_batch1"), dict) else {}
        bad = [k for k in RATES if cmp.get(k) != 1.0]
        if bad:
            out.append(f"배치 {batch_size} 의 배치 1 대비 일치율이 1.0 이 아니다: {bad}")
        if type(n) is int and n % batch_size == 0:
            out.append(f"n_rows {n} 가 배치 {batch_size} 의 배수다 — 등록 크기보다 작은 잔여 묶음이 지나가지 않았다")
        peak = pn.get("peak_mem_gb")
        if gpu_total_gb is not None and (not isinstance(peak, (int, float)) or peak > gpu_total_gb * mem_ratio):
            out.append(f"배치 {batch_size} 의 최대 할당 메모리 {peak!r} GB 가 {gpu_total_gb} × {mem_ratio:.3f} 을 넘는다")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--adapter-sha256", required=True)
    ap.add_argument("--rows-sha256", default=None)
    ap.add_argument("--min-rows", type=int, default=16)
    ap.add_argument("--gpu-total-gb", type=float, default=None)
    ap.add_argument("--mem-ratio", type=float, default=2 / 3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"판정 파일이 이미 있다 — 덮지 않는다: {a.out}", file=sys.stderr)
        return 2
    raw = a.probe.read_bytes()
    try:
        probe = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        probe = None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=False).stdout.strip()
    bad = problems_of(probe, batch_size=a.batch_size, adapter_sha256=a.adapter_sha256, head=head, rows_sha256=a.rows_sha256,
                      min_rows=a.min_rows, gpu_total_gb=a.gpu_total_gb, mem_ratio=a.mem_ratio)
    if bad:
        for b in bad:
            print(f"[batch_probe] 차단 — {b}", file=sys.stderr)
        return 2
    rec = {"kind": "batch_probe_gate_v1", "pass": True, "probe_sha256": hashlib.sha256(raw).hexdigest(), "head": head,
           "batch_size": a.batch_size, "n_rows": probe["n_rows"], "adapter_sha256": a.adapter_sha256,
           "peak_mem_gb": probe["passes"][str(a.batch_size)]["peak_mem_gb"], "gpu_total_gb": a.gpu_total_gb,
           "env": probe.get("env")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("pass", "batch_size", "n_rows", "peak_mem_gb")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
