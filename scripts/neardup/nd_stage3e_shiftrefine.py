"""근사 중복 3e단계 — G4(큰 이동)가 새로 올린 후보에 정밀 특징과 블록 일치를 단다.

G4 는 문턱을 낮게 잡아 잡음을 많이 물고 온다(무작위 쌍의 0.08% 가 0.45 를 넘는다). 여기서 두 번 거른다.

1. **이동이 작은 쌍은 버린다.** 약 34px(2칸) 미만은 앞의 세 생성기가 이미 덮는 영역이라 G4 로 다시 볼 이유가 없다.
2. **봉우리가 낮은 쌍은 버린다.** 기본 0.6 — 거친 기술자 수준에서 이만큼 맞지 않으면 화소 수준에서 볼 것이 없다.

남은 쌍에만 화소 특징을 단다. 걸러 낸 쪽에서 무작위 대조 표본도 함께 계산해 **거른 자리에 같은 그림이 없는지** 본다.

사용: python nd_stage3e_shiftrefine.py [봉우리 문턱] [최소 이동 칸] [워커]    기본 0.6 · 2칸 · 2워커
      워커를 올리기 전에 여유 메모리를 먼저 본다(하나가 약 0.7GB 를 쓴다).
출력: feat3_shift_cross.csv · feat5_shift_cross.csv — 전부 _workspace(무시 규칙 대상).
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage2d_refine as refine
import nd_stage2f_blocks as blocks

W = refine.W
CELL_PX = 17                      # 64x36 기술자의 한 칸이 안쪽 상자에서 차지하는 폭
CONTROL = 400                     # 걸러 낸 쪽의 무작위 대조 표본


def _init() -> None:
    refine._init()
    blocks._init()


def both(pair):
    f = refine.features(pair)
    dx, dy = int(f[refine.HEAD.index("rd_dx")]), int(f[refine.HEAD.index("rd_dy")])
    return f, blocks.features((pair[0], pair[1], dx, dy))


def main() -> int:
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    min_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rows = list(csv.DictReader((W / "shiftcand_cross.csv").open(encoding="utf-8")))
    for r in rows:
        r["cells"] = max(abs(int(r["cell_dx"])), abs(int(r["cell_dy"])))
    def passes(r: dict) -> bool:
        return r["cells"] >= min_cells and float(r["shift_peak"]) >= tau

    keep = [r for r in rows if passes(r)]
    rest = [r for r in rows if not passes(r)]
    rng = np.random.default_rng(20260921)
    ctrl = [rest[i] for i in rng.choice(len(rest), size=min(CONTROL, len(rest)), replace=False)] if rest else []
    todo = keep + ctrl
    for r in keep:
        r["selected"] = "1"
    for r in ctrl:
        r["selected"] = "0"
    print(f"G4 후보 {len(rows):,}쌍 → 이동 {min_cells}칸(약 {min_cells * CELL_PX}px) 이상 · 봉우리 ≥ {tau} 인 {len(keep):,}쌍 "
          f"+ 걸러 낸 {len(rest):,}쌍에서 뽑은 대조 {len(ctrl):,}쌍 · 워커 {workers}", flush=True)
    head = ["image_id_a", "image_id_b", "shift_peak", "cell_dx", "cell_dy", "cells", "selected"]
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init) as pool, \
            (W / "feat3_shift_cross.csv").open("w", encoding="utf-8", newline="") as o3, \
            (W / "feat5_shift_cross.csv").open("w", encoding="utf-8", newline="") as o5:
        w3, w5 = csv.writer(o3), csv.writer(o5)
        w3.writerow([*head, *refine.HEAD])
        w5.writerow(["image_id_a", "image_id_b", *blocks.HEAD])
        for k, (f, b) in enumerate(pool.imap(both, [(r["image_id_a"], r["image_id_b"]) for r in todo], chunksize=8)):
            w3.writerow([todo[k][c] for c in head] + f)
            w5.writerow([todo[k]["image_id_a"], todo[k]["image_id_b"], *b])
            if k % 200 == 0:
                print(f"  {k:,}/{len(todo):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(todo):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
