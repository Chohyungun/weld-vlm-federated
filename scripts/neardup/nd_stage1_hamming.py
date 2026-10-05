"""근사 중복 1단계 — 256비트 지각 해시의 전수 해밍 거리 분포 (교차·같은 쪽 안). 읽기 전용, 출력은 미추적 경로.

식별자가 든 후보 목록은 _workspace 아래(무시 규칙 대상)에만 쓴다. 화면에는 건수만 찍는다.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_workspace" / "2026-09-21-neardup"
OUT.mkdir(parents=True, exist_ok=True)
CAND_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 48          # 후보로 남길 최대 거리(넉넉히)
csv.field_size_limit(10_000_000)

rows = []
with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        rows.append((r["image_id"], r["split"], r["phash_hex"], r["group_id"]))
ids = np.array([r[0] for r in rows])
split = np.array([r[1] for r in rows])
gid = np.array([r[3] for r in rows])
H = np.array([[int(r[2][i:i + 16], 16) for i in range(0, 64, 16)] for r in rows], dtype=np.uint64)   # (N,4)
ev = np.flatnonzero(split == "eval")
tv = np.flatnonzero(split != "eval")
print(f"eval {len(ev):,} · train+val {len(tv):,} · 후보 최대 거리 {CAND_MAX}", flush=True)


def scan(a_idx: np.ndarray, b_idx: np.ndarray, same: bool, tag: str):
    """a×b 전 쌍의 거리 히스토그램과 d<=CAND_MAX 후보. same=True 면 i<j 만 센다."""
    hist = np.zeros(257, dtype=np.int64)
    cand = []
    A, B = H[a_idx], H[b_idx]
    t0 = time.perf_counter()
    step = 96
    for s in range(0, len(A), step):
        x = A[s:s + step]                                           # (k,4)
        d = np.zeros((len(x), len(B)), dtype=np.uint16)
        for w in range(4):
            d += np.bitwise_count(x[:, w:w + 1] ^ B[None, :, w]).astype(np.uint16)
        if same:
            ii = np.arange(s, s + len(x))[:, None]
            jj = np.arange(len(B))[None, :]
            mask = jj > ii
            hist += np.bincount(d[mask].ravel(), minlength=257)[:257]
            sel = np.argwhere(mask & (d <= CAND_MAX))
        else:
            hist += np.bincount(d.ravel(), minlength=257)[:257]
            sel = np.argwhere(d <= CAND_MAX)
        for i, j in sel:
            cand.append((int(a_idx[s + i]), int(b_idx[j]), int(d[i, j])))
        if (s // step) % 20 == 0:
            print(f"  [{tag}] {s + len(x):,}/{len(A):,}  후보 {len(cand):,}  {time.perf_counter() - t0:.0f}s", flush=True)
    return hist, cand


result = {}
for tag, a, b, same in (("cross_eval_vs_trainval", ev, tv, False), ("within_eval", ev, ev, True), ("within_trainval", tv, tv, True)):
    hist, cand = scan(a, b, same, tag)
    result[tag] = {"n_pairs": int(hist.sum()), "hist": hist.tolist(), "n_cand": len(cand)}
    same_group = sum(1 for i, j, _ in cand if gid[i] == gid[j])
    result[tag]["cand_same_group"] = same_group
    with (OUT / f"cand_{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id_a", "image_id_b", "hamming", "same_group"])
        for i, j, d in cand:
            w.writerow([ids[i], ids[j], d, int(gid[i] == gid[j])])
    cum = np.cumsum(hist)
    print(f"[{tag}] 전 쌍 {hist.sum():,} · 후보(d<={CAND_MAX}) {len(cand):,} (같은 묶음 {same_group:,})", flush=True)
    print("   누적 쌍 수: " + " · ".join(f"<={t}: {int(cum[t]):,}" for t in (0, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64) if t <= 256), flush=True)
(OUT / "stage1_hamming_hist.json").write_text(json.dumps(result), encoding="utf-8")
print("완료", flush=True)
