"""근사 중복 2a단계 — 전 영상(모델이 보는 판: tiles_v1_histmatch)의 64x36 그레이 축소본. 읽기 전용, 출력은 미추적 경로."""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_workspace" / "2026-09-21-neardup"
W, H = 64, 36
csv.field_size_limit(10_000_000)


def thumb(rel: str) -> np.ndarray:
    with Image.open(ROOT / rel) as im:
        im.draft("L", (W * 4, H * 4))                 # JPEG DCT 단계 축소 — 디코드가 빨라진다
        return np.asarray(im.convert("L").resize((W, H), Image.BOX), dtype=np.uint8)


def main() -> int:
    rows = []
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["image_id"], r["rel_path"]))
    n = len(rows)
    out = np.zeros((n, H, W), dtype=np.uint8)
    t0 = time.perf_counter()
    with Pool(4) as pool:
        for i, a in enumerate(pool.imap(thumb, [r[1] for r in rows], chunksize=64)):
            out[i] = a
            if i % 5000 == 0:
                print(f"  {i:,}/{n:,}  {time.perf_counter() - t0:.0f}s", flush=True)
    np.save(OUT / "thumbs_histmatch_64x36.npy", out)
    (OUT / "thumbs_ids.txt").write_text("\n".join(r[0] for r in rows) + "\n", encoding="utf-8")
    print(f"완료 {n:,}장 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
