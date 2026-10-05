"""근사 중복 2c단계(2판) — 두 후보 생성기의 합집합에 전 해상도 화소 특징을 단다. 테두리(상수 채움)를 빼고 안쪽 상자만 본다.

사용: python nd_stage2c_v2.py <tag> [workers]      tag ∈ cross_eval_vs_trainval | within_eval | within_trainval
입력: cand_<tag>.csv(지각 해시 d<=48) ∪ thumbcand_<tag>.csv(축소본 rmsd<=4). 출력: feat2_<tag>.csv — 전부 _workspace(무시 규칙 대상).

특징
  mad · rmsd · corr · p99_absdiff · frac_gt8      밝기 차이와 상관(전 해상도)
  grad_corr                                        화소 단위 기울기 상관 — 화소 잡음(압축 이력)에 민감하다
  bp_corr                                          대역 통과(가우시안 차, 약 3~16px 규모) 상관 — 결함 크기 구조를 본다
  bp_peak · bp_dx · bp_dy                          대역 통과 영상의 교차상관 최댓값과 그 이동(전 해상도 px)
  std_a/b · bp_std_a/b                             밝기·구조의 에너지(평탄한 영상은 판정 근거가 약하다)
"""
from __future__ import annotations

import csv
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
X0, Y0, X1, Y1 = 102, 58, 1178, 662
csv.field_size_limit(10_000_000)
REL: dict[str, str] = {}


def _init() -> None:
    cv2.setNumThreads(1)
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            REL[r["image_id"]] = r["rel_path"]


@lru_cache(maxsize=96)
def load(iid: str):
    with Image.open(ROOT / REL[iid]) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32)[Y0:Y1, X0:X1]
    h = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
    bp = cv2.GaussianBlur(h, (0, 0), 1.5) - cv2.GaussianBlur(h, (0, 0), 8.0)
    bp -= bp.mean()
    g = np.abs(np.diff(a, axis=1))[:-1] + np.abs(np.diff(a, axis=0))[:, :-1]
    return a, g, bp, np.fft.rfft2(bp)


def _corr(a, b) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def features(pair):
    a, ga, bpa, fa = load(pair[0])
    b, gb, bpb, fb = load(pair[1])
    d = np.abs(a - b)
    na, nb = float(np.sqrt((bpa * bpa).sum())), float(np.sqrt((bpb * bpb).sum()))
    if na > 0 and nb > 0:
        xc = np.fft.irfft2(fa * np.conj(fb), s=bpa.shape) / (na * nb)
        k = int(np.argmax(xc))
        py, px = divmod(k, xc.shape[1])
        dy = py if py <= xc.shape[0] // 2 else py - xc.shape[0]
        dx = px if px <= xc.shape[1] // 2 else px - xc.shape[1]
        peak, bp0 = float(xc[py, px]), float(xc[0, 0])
    else:
        peak, bp0, dx, dy = 0.0, 0.0, 0, 0
    return [round(float(d.mean()), 4), round(float(np.sqrt((d * d).mean())), 4), round(_corr(a, b), 6),
            round(float(np.percentile(d, 99)), 2), round(float((d > 8).mean()), 5),
            round(_corr(ga, gb), 6), round(bp0, 6), round(peak, 6), dx * 2, dy * 2,
            round(float(a.std()), 2), round(float(b.std()), 2), round(float(bpa.std()), 3), round(float(bpb.std()), 3)]


HEAD = ["mad", "rmsd", "corr", "p99_absdiff", "frac_gt8", "grad_corr", "bp_corr", "bp_peak", "bp_dx", "bp_dy",
        "std_a", "std_b", "bp_std_a", "bp_std_b"]


def union(tag: str):
    ids = (W / "thumbs_ids.txt").read_text(encoding="utf-8").split("\n")[:-1]
    pos = {k: i for i, k in enumerate(ids)}
    T = np.load(W / "thumbs_histmatch_64x36.npy")[:, 3:33, 6:58].reshape(len(ids), -1).astype(np.float64)
    pairs: dict[tuple[str, str], dict] = {}
    with (W / f"cand_{tag}.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            pairs[(r["image_id_a"], r["image_id_b"])] = {"hamming": int(r["hamming"]), "same_group": r["same_group"], "in_phash": 1, "in_thumb": 0}
    with (W / f"thumbcand_{tag}.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            k = (r["image_id_a"], r["image_id_b"])
            k = k if k in pairs or (k[1], k[0]) not in pairs else (k[1], k[0])
            p = pairs.setdefault(k, {"hamming": int(r["hamming"]), "same_group": r["same_group"], "in_phash": 0, "in_thumb": 0})
            p["in_thumb"] = 1
    rows = []
    for (x, y), p in pairs.items():
        t = float(np.sqrt(((T[pos[x]] - T[pos[y]]) ** 2).mean()))
        rows.append([x, y, p["hamming"], f"{t:.4f}", p["same_group"], p["in_phash"], p["in_thumb"]])
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def main() -> int:
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows = union(tag)
    n_both = sum(1 for r in rows if r[5] and r[6])
    print(f"[{tag}] 합집합 {len(rows):,}쌍 · 해시만 {sum(1 for r in rows if r[5] and not r[6]):,} · 축소본만 {sum(1 for r in rows if r[6] and not r[5]):,} · 둘 다 {n_both:,}", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init) as pool, (W / f"feat2_{tag}.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["image_id_a", "image_id_b", "hamming", "thumb_rmsd", "same_group", "in_phash", "in_thumb"] + HEAD)
        for k, f in enumerate(pool.imap(features, [(r[0], r[1]) for r in rows], chunksize=32)):
            w.writerow(rows[k] + f)
            if k % 4000 == 0:
                print(f"  {k:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
