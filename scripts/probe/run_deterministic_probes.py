"""P1 규격 항등성 · P4 인코딩 지문을 실물 타일에 실행한다.

    uv run python scripts/probe/run_deterministic_probes.py \
        --manifest data/interim/manifest_v1/manifest.csv \
        --root . --out outputs/probe/deterministic_v1.json

**헤더 판독값을 쓰고 매니페스트 선언값을 믿지 않는다.** RIAWELC 에서 라벨이 224 라고
적혀 있는데 실제 파일이 227 이었던 전례가 있다. 라벨을 믿으면 규격이 통일됐다고 보고하면서
실제로는 안 통일된 상태가 된다.

100% 아니면 머지 금지다. 검사 대상이 0장이면 통과로 처리하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.probes.deterministic import (
    compare_header_to_manifest,
    gate,
    p1_spec_identity,
    p4_encoding_fingerprint,
    read_header,
)


def load_manifest(path: Path, modality: str | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if modality:
        rows = [r for r in rows if r.get("modality") == modality]
    return rows


def read_all(rows: list[dict[str, str]], root: Path, workers: int):
    """헤더를 병렬로 읽는다. 읽기 전용이며 원본을 건드리지 않는다."""
    def one(r):
        p = root / r["rel_path"]
        try:
            return read_header(p, image_id=r["image_id"]), None
        except Exception as exc:                       # noqa: BLE001
            return None, f'{r["image_id"]}: {type(exc).__name__} {exc}'

    headers, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for h, err in pool.map(one, rows):
            (errors if err else headers).append(err or h)
    return headers, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--modality", default="RT")
    ap.add_argument("--out", default="outputs/probe/deterministic_v1.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    rows = load_manifest(Path(args.manifest), args.modality)
    print(f"매니페스트 {len(rows)}행 (modality={args.modality})")
    if not rows:
        print("검사 대상 0장 — 통과로 처리하지 않는다")
        return 1

    headers, errors = read_all(rows, root, args.workers)
    print(f"헤더 판독 성공 {len(headers)} / 실패 {len(errors)}")

    manifest_wh = {
        r["image_id"]: (int(r["width_px"]), int(r["height_px"]))
        for r in rows
        if r.get("width_px") and r.get("height_px")
    }

    p1 = p1_spec_identity(headers)
    cross = compare_header_to_manifest(headers, manifest_wh)
    p4 = p4_encoding_fingerprint(headers)
    results = [p1, cross, p4]
    ok, verdict = gate(results)
    if errors:
        ok = False
        verdict += f" / 헤더 판독 실패 {len(errors)}건"

    fingerprints = Counter(h.encoding_fingerprint for h in headers)
    payload = {
        "manifest": str(args.manifest),
        "modality": args.modality,
        "n_manifest_rows": len(rows),
        "n_headers_read": len(headers),
        "n_read_errors": len(errors),
        "read_errors_head": errors[:10],
        "probes": [r.as_dict() for r in results],
        "encoding_fingerprints": [
            {"fingerprint": list(fp), "count": n} for fp, n in fingerprints.most_common()
        ],
        "gate_passed": ok,
        "gate_verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    for r in results:
        print(f"  [{r.probe}] {'통과' if r.passed else '실패'} — {r.detail}")
    print(f"\n{verdict}")
    print(f"결과 저장: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
