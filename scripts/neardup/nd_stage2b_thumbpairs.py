"""근사 중복 2b단계(2판) — 축소 영상 안쪽 영역(테두리 제외)의 전 쌍 RMS 차이. 지각 해시와 독립인 두 번째 후보 생성기.

기준은 rmsd <= RMAX 하나다(축소본 상관은 용접부 영상끼리 원래 높아 분별력이 없다 — 실측).
식별자가 든 파일은 _workspace(무시 규칙 대상)에만 쓴다.
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
RMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
LIMIT = 3_000_000
csv.field_size_limit(10_000_000)

ids = (OUT / "thumbs_ids.txt").read_text(encoding="utf-8").split("\n")[:-1]
T = np.load(OUT / "thumbs_histmatch_64x36.npy")[:, 3:33, 6:58].reshape(len(ids), -1).astype(np.float32)
n_pix = T.shape[1]
sq = (T.astype(np.float64) ** 2).sum(axis=1)

meta = {}
with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        meta[r["image_id"]] = (r["split"], r["phash_hex"], r["group_id"])
split = np.array([meta[i][0] for i in ids])
gid = np.array([meta[i][2] for i in ids])
PH = np.array([[int(meta[i][1][k:k + 16], 16) for k in range(0, 64, 16)] for i in ids], dtype=np.uint64)
ev, tv = np.flatnonzero(split == "eval"), np.flatnonzero(split != "eval")
print(f"축소본 안쪽 {n_pix}화소 · 기준 rmsd<={RMAX}", flush=True)
RB = np.arange(0, 31, 1.0)


def scan(a_idx, b_idx, same, tag):
    hr = np.zeros(len(RB) - 1, dtype=np.int64)
    ca, cb, cr = [], [], []
    B, Bsq = T[b_idx], sq[b_idx]
    t0 = time.perf_counter()
    step = 512
    for s in range(0, len(a_idx), step):
        ai = a_idx[s:s + step]
        dot = (T[ai] @ B.T).astype(np.float64)
        rmsd = np.sqrt(np.maximum((sq[ai][:, None] + Bsq[None, :] - 2 * dot) / n_pix, 0))
        if same:
            tri = np.arange(len(b_idx))[None, :] > np.arange(s, s + len(ai))[:, None]
            hr += np.histogram(rmsd[tri], bins=RB)[0]
            keep = tri & (rmsd <= RMAX)
        else:
            hr += np.histogram(rmsd, bins=RB)[0]
            keep = rmsd <= RMAX
        i, j = np.nonzero(keep)
        ca.append(ai[i]); cb.append(b_idx[j]); cr.append(rmsd[i, j])
        if sum(len(x) for x in ca) > LIMIT:
            raise SystemExit(f"[{tag}] 후보가 {LIMIT:,} 을 넘었다 — 문턱을 낮춰라")
        if (s // step) % 20 == 0:
            print(f"  [{tag}] {s + len(ai):,}/{len(a_idx):,} 후보 {sum(len(x) for x in ca):,} {time.perf_counter() - t0:.0f}s", flush=True)
    ca, cb, cr = np.concatenate(ca), np.concatenate(cb), np.concatenate(cr)
    ham = np.zeros(len(ca), dtype=np.int64)
    for w in range(4):
        ham += np.bitwise_count(PH[ca, w] ^ PH[cb, w]).astype(np.int64)
    return hr, ca, cb, cr, ham


res = {}
for tag, a, b, same in (("cross_eval_vs_trainval", ev, tv, False), ("within_eval", ev, ev, True), ("within_trainval", tv, tv, True)):
    hr, ca, cb, cr, ham = scan(a, b, same, tag)
    res[tag] = {"rmsd_hist_0_to_30_step_1": hr.tolist(), "n_cand": len(ca)}
    with (OUT / f"thumbcand_{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id_a", "image_id_b", "thumb_rmsd", "hamming", "same_group"])
        for x, y, r, h in zip(ca, cb, cr, ham):
            w.writerow([ids[x], ids[y], f"{r:.4f}", int(h), int(gid[x] == gid[y])])
    print(f"[{tag}] 후보 {len(ca):,} · 같은 묶음 {int((gid[ca] == gid[cb]).sum()):,}", flush=True)
    for rmax in (1.0, 2.0, 3.0, 4.0):
        m = cr <= rmax
        print(f"   rmsd<={rmax}: {int(m.sum()):,}쌍 · 해밍<=16 {int((ham[m] <= 16).sum()):,} · 17~48 {int(((ham[m] > 16) & (ham[m] <= 48)).sum()):,} · >48 {int((ham[m] > 48).sum()):,}", flush=True)
(OUT / "stage2b_thumb_hist.json").write_text(json.dumps(res), encoding="utf-8")
print("완료", flush=True)
