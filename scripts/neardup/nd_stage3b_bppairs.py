"""근사 중복 3b단계 — 세 번째 후보 생성기(밝기 변환 불변 거친 구조 기술자)의 전 쌍 상관. 교차 쌍만.

1) 이미 아는 같은 그림 쌍(feat3 의 rd_sharp>=0.5)과 대조 쌍에서 기술자 상관의 분포를 재 문턱을 고른다.
2) 전 교차 쌍(6.2억)에서 상관 >= 문턱인 쌍을 뽑고, 앞의 두 생성기 합집합에 **없던** 쌍을 따로 적는다.
출력은 _workspace(무시 규칙 대상).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
csv.field_size_limit(10_000_000)

ids = (W / "thumbs_ids.txt").read_text(encoding="utf-8").split("\n")[:-1]
pos = {k: i for i, k in enumerate(ids)}
D = np.load(W / "bpdesc_64x36.npy").reshape(len(ids), -1).astype(np.float32)
D -= D.mean(axis=1, keepdims=True)
nrm = np.linalg.norm(D, axis=1)
flat = nrm < 1e-3
nrm[flat] = 1.0
D /= nrm[:, None]
D[flat] = 0
print(f"기술자 {D.shape} · 구조가 없는(평탄) 영상 {int(flat.sum()):,}장", flush=True)

rows = list(csv.DictReader((W / "feat3_cross_eval_vs_trainval.csv").open(encoding="utf-8")))
known = {(r["image_id_a"], r["image_id_b"]) for r in rows}
union_all = set()
for name in ("cand_cross_eval_vs_trainval.csv", "thumbcand_cross_eval_vs_trainval.csv"):
    with (W / name).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            union_all.add((r["image_id_a"], r["image_id_b"]))
c = np.array([float(D[pos[r["image_id_a"]]] @ D[pos[r["image_id_b"]]]) for r in rows])
sharp = np.array([float(r["rd_sharp"]) for r in rows])
sel = np.array([r["selected"] == "1" for r in rows])
same, ctrl = sel & (sharp >= 0.5), ~sel
print("같은 그림(rd_sharp>=0.5) 쌍의 기술자 상관 분위 [1,5,10,25,50]:", np.round(np.percentile(c[same], [1, 5, 10, 25, 50]), 3).tolist(), flush=True)
print("대조 쌍의 기술자 상관 분위 [50,90,99,최대]:", np.round(np.percentile(c[ctrl], [50, 90, 99, 100]), 3).tolist(), flush=True)
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else float(np.round(np.percentile(c[same], 5), 2))
print(f"문턱 {TAU} — 아는 같은 그림의 {100 * (c[same] >= TAU).mean():.1f}% 를 덮는다 · 대조의 {100 * (c[ctrl] >= TAU).mean():.2f}% 가 넘는다", flush=True)

split = {}
with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        split[r["image_id"]] = r["split"]
sp = np.array([split[i] for i in ids])
ev, tv = np.flatnonzero(sp == "eval"), np.flatnonzero(sp != "eval")
B = D[tv]
hist = np.zeros(40, dtype=np.int64)
new, n_all = [], 0
t0 = time.perf_counter()
for s in range(0, len(ev), 512):
    ai = ev[s:s + 512]
    cc = D[ai] @ B.T
    hist += np.histogram(cc, bins=np.linspace(0, 1, 41))[0]
    i, j = np.nonzero(cc >= TAU)
    n_all += len(i)
    for x, y in zip(i, j):
        k = (ids[ai[x]], ids[tv[y]])
        if k not in union_all:
            new.append((k[0], k[1], float(cc[x, y])))
    if (s // 512) % 8 == 0:
        print(f"  {s + len(ai):,}/{len(ev):,} 문턱 이상 {n_all:,} · 그중 새 쌍 {len(new):,} {time.perf_counter() - t0:.0f}s", flush=True)
with (W / "bpcand_new_cross.csv").open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image_id_a", "image_id_b", "bpdesc_corr"])
    w.writerows([(a, b, f"{v:.5f}") for a, b, v in new])
(W / "stage3b_summary.json").write_text(json.dumps({"tau": TAU, "n_ge_tau": n_all, "n_new": len(new), "hist_0_1_step_0.025": hist.tolist()}), encoding="utf-8")
print(f"완료 — 문턱 이상 {n_all:,}쌍 · 앞의 두 생성기에 없던 새 쌍 {len(new):,}", flush=True)
