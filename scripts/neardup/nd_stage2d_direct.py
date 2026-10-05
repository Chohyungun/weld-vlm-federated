"""근사 중복 2d단계(직행판) — 기본 특징 단계를 건너뛰고 두 생성기 합집합 전체에 정밀 특징을 단다. 규모가 큰 train·val 안쪽용.

사용: python nd_stage2d_direct.py <tag> [workers]   출력 feat3d_<tag>.csv (열: 쌍 정보 + 정밀 특징)
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage2c_v2 as base
import nd_stage2d_refine as refine

W = refine.W


def main() -> int:
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows = base.union(tag)
    print(f"[{tag}] 합집합 {len(rows):,}쌍 — 전부 정밀 특징", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=refine._init) as pool, (W / f"feat3d_{tag}.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["image_id_a", "image_id_b", "hamming", "thumb_rmsd", "same_group", "in_phash", "in_thumb"] + refine.HEAD)
        for k, f in enumerate(pool.imap(refine.features, [(r[0], r[1]) for r in rows], chunksize=16)):
            w.writerow(rows[k] + f)
            if k % 4000 == 0:
                print(f"  {k:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
