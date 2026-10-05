"""근사 중복 2g단계 — 2d 의 사전 거름이 뺀 교차 후보 **전량**에 정밀 특징과 블록 일치를 단다.

왜: 2d 는 가능성이 낮은 쌍을 걸러 내고 그쪽에서는 무작위 600쌍만 쟀다. 걸러진 쪽에도 같은 그림이 있었다
(정합 봉우리는 무디지만 블록은 전역에서 맞는, 심하게 평활화된 사본). 특징이 없으면 어떤 규칙도 적용되지 않는다 —
문턱을 낮출 일이 아니라 전량을 재야 닫힌다(정의 검토의 지적).

사용: python nd_stage2g_rest.py <tag> [workers]   입력 feat2_<tag>.csv · feat3_<tag>.csv → 출력 feat3_rest_<tag>.csv · feat5_rest_<tag>.csv
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage2d_refine as refine
import nd_stage2f_blocks as blocks

W = refine.W


def _init() -> None:
    refine._init()
    blocks._init()


def both(pair):
    f = refine.features(pair)
    dx, dy = int(f[refine.HEAD.index("rd_dx")]), int(f[refine.HEAD.index("rd_dy")])
    return f, blocks.features((pair[0], pair[1], dx, dy))


def main() -> int:
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    done = {(r["image_id_a"], r["image_id_b"]) for r in csv.DictReader((W / f"feat3_{tag}.csv").open(encoding="utf-8")) if r["selected"] == "1"}
    with (W / f"feat2_{tag}.csv").open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        head = rd.fieldnames
        rows = [r for r in rd if r.get("bp_std_b") and (r["image_id_a"], r["image_id_b"]) not in done]
    rows.sort(key=lambda r: (r["image_id_a"], r["image_id_b"]))
    print(f"[{tag}] 걸러졌던 쪽 {len(rows):,}쌍 — 전량 정밀 특징 + 블록 일치", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init) as pool, \
            (W / f"feat3_rest_{tag}.csv").open("w", encoding="utf-8", newline="") as o3, \
            (W / f"feat5_rest_{tag}.csv").open("w", encoding="utf-8", newline="") as o5:
        w3, w5 = csv.writer(o3), csv.writer(o5)
        w3.writerow([*head, *refine.HEAD])
        w5.writerow(["image_id_a", "image_id_b", *blocks.HEAD])
        for k, (f, b) in enumerate(pool.imap(both, [(r["image_id_a"], r["image_id_b"]) for r in rows], chunksize=16)):
            w3.writerow([rows[k][c] for c in head] + f)
            w5.writerow([rows[k]["image_id_a"], rows[k]["image_id_b"], *b])
            if k % 2000 == 0:
                print(f"  {k:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
