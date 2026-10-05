"""근사 중복 3c단계 — 세 번째 생성기가 **새로** 찾은 교차 쌍(앞의 두 생성기 합집합 밖)에 정밀 특징을 단다.

재현율의 근거: 새 쌍 가운데 같은 그림(rd_sharp 기준)이 몇 쌍인가 = 앞의 두 생성기가 놓친 양.
사용: python nd_stage3c_refine_new.py [workers] [max_pairs]   입력 bpcand_new_cross.csv → 출력 feat3_new_cross.csv
쌍이 max_pairs 를 넘으면 기술자 상관이 높은 순으로 자르고, 자른 사실을 찍는다(조용히 자르지 않는다).
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage2d_refine as refine

W = refine.W


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    rows = list(csv.DictReader((W / "bpcand_new_cross.csv").open(encoding="utf-8")))
    rows.sort(key=lambda r: -float(r["bpdesc_corr"]))
    if len(rows) > cap:
        print(f"!! 새 쌍 {len(rows):,} 이 상한 {cap:,} 을 넘는다 — 기술자 상관 상위 {cap:,} 만 본다(하한 상관 {rows[cap - 1]['bpdesc_corr']})", flush=True)
        rows = rows[:cap]
    rows.sort(key=lambda r: (r["image_id_a"], r["image_id_b"]))
    print(f"새 쌍 {len(rows):,}", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=refine._init) as pool, (W / "feat3_new_cross.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["image_id_a", "image_id_b", "bpdesc_corr"] + refine.HEAD)
        for k, f in enumerate(pool.imap(refine.features, [(r["image_id_a"], r["image_id_b"]) for r in rows], chunksize=16)):
            w.writerow([rows[k]["image_id_a"], rows[k]["image_id_b"], rows[k]["bpdesc_corr"]] + f)
            if k % 4000 == 0:
                print(f"  {k:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
