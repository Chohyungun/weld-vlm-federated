"""근사 중복 3d단계 — 네 번째 후보 생성기(G4). **큰 이동(겹치는 크롭)** 을 본다.

G1·G2·G3 은 약 32px 을 넘는 이동을 못 본다(10번 문서 §1-다). 여기서는 G3 의 거친 구조 기술자(64x36, 한 칸 약 17px)를
푸리에 변환으로 **모든 이동에서 한 번에** 맞춰 본다: 원형 교차상관의 최댓값과 그 이동. 이동 s 에서 겹치지 않는 부분은 잡음으로
섞이므로 봉우리는 대략 (겹침 비율 x 실제 상관)이다.

사용:
  python nd_stage3d_shiftscan.py pilot            아는 같은 그림(이동 크기별)과 무작위 쌍에서 봉우리 분포를 재 문턱을 고른다
  python nd_stage3d_shiftscan.py scan <문턱>      eval x train·val 전수. 이미 확실히 걸린 eval(rd_sharp>=0.5 · blk_frac>=0.5)은 건너뛴다
출력: shiftcand_cross.csv (앞의 세 생성기에 없던 쌍만) · stage3d_summary.json — 전부 _workspace(무시 규칙 대상).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import fft as sfft

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
csv.field_size_limit(10_000_000)
GH, GW = 36, 64
CHUNK = 8192
WORKERS = 2          # 여유 메모리·게이트와 함께 돈다. 올리기 전에 여유 메모리를 먼저 본다


def load_desc():
    ids = (W / "thumbs_ids.txt").read_text(encoding="utf-8").split("\n")[:-1]
    D = np.load(W / "bpdesc_64x36.npy").astype(np.float32)
    D -= D.mean(axis=(1, 2), keepdims=True)
    nrm = np.sqrt((D * D).sum(axis=(1, 2)))
    flat = nrm < 1e-3
    nrm[flat] = 1.0
    D /= nrm[:, None, None]
    D[flat] = 0
    return ids, D, flat


def peaks(fa: np.ndarray, FB: np.ndarray):
    """기술자 하나(의 rfft2)와 여러 기술자의 원형 교차상관 → (봉우리, 세로 이동 칸, 가로 이동 칸). 이동은 a 기준 b 의 위치."""
    val = np.empty(len(FB), dtype=np.float32)
    arg = np.empty(len(FB), dtype=np.int32)
    for s in range(0, len(FB), CHUNK):
        xc = sfft.irfft2(fa[None] * np.conj(FB[s:s + CHUNK]), s=(GH, GW), workers=WORKERS).reshape(-1, GH * GW)
        arg[s:s + CHUNK] = xc.argmax(axis=1)
        val[s:s + CHUNK] = xc[np.arange(len(xc)), arg[s:s + CHUNK]]
    py, px = np.divmod(arg, GW)
    dy = np.where(py <= GH // 2, py, py - GH)
    dx = np.where(px <= GW // 2, px, px - GW)
    return val, dy, dx


def known_pairs() -> set:
    known = set()
    for name in ("cand_cross_eval_vs_trainval.csv", "thumbcand_cross_eval_vs_trainval.csv", "bpcand_new_cross.csv"):
        with (W / name).open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                known.add((r["image_id_a"], r["image_id_b"]))
    return known


def pilot(ids, D, F, pos) -> None:
    rows = [r for r in csv.DictReader((W / "feat3_cross_eval_vs_trainval.csv").open(encoding="utf-8")) if r["selected"] == "1"]
    rows += list(csv.DictReader((W / "feat3_new_cross.csv").open(encoding="utf-8")))
    same = [r for r in rows if float(r["rd_sharp"]) >= 0.5]
    bands = [(0, 0), (1, 8), (9, 16), (17, 32), (33, 10_000)]
    print("아는 같은 그림(rd_sharp>=0.5) — 이동 크기별 [쌍 수 · G3 상관(이동 0) 중앙 · G4 봉우리 5·25·50 분위]", flush=True)
    for lo, hi in bands:
        sel = [r for r in same if lo <= max(abs(int(float(r["rd_dx"]))), abs(int(float(r["rd_dy"])))) <= hi]
        if not sel:
            continue
        g3, g4 = [], []
        for r in sel[:1500]:
            a, b = pos[r["image_id_a"]], pos[r["image_id_b"]]
            g3.append(float((D[a] * D[b]).sum()))
            g4.append(float(peaks(F[a], F[b:b + 1])[0][0]))
        print(f"  이동 {lo}~{hi}px: {len(sel):,}쌍 · G3 {np.median(g3):.3f} · G4 {np.round(np.percentile(g4, [5, 25, 50]), 3).tolist()}", flush=True)
    rng = np.random.default_rng(20260921)
    sp = split_of(ids)
    ev, tv = np.flatnonzero(sp == "eval"), np.flatnonzero(sp != "eval")
    t0 = time.perf_counter()
    mx = []
    for a in rng.choice(ev, 40, replace=False):
        v, _, _ = peaks(F[a], F[tv])
        mx.append(v)
    mx = np.concatenate(mx)
    print(f"무작위 eval 40장 x train·val 전량({len(mx):,}쌍)의 G4 봉우리 분위 [50, 99, 99.9, 99.99, 최대]: "
          f"{np.round(np.percentile(mx, [50, 99, 99.9, 99.99, 100]), 3).tolist()} · eval 한 장에 {(time.perf_counter() - t0) / 40:.2f}s", flush=True)
    for t in (0.30, 0.35, 0.40, 0.45, 0.50):
        print(f"  문턱 {t}: 무작위 쌍의 {100 * (mx >= t).mean():.4f}% → 전수에서 약 {int((mx >= t).mean() * len(ev) * len(tv)):,}쌍", flush=True)


def split_of(ids) -> np.ndarray:
    split = {}
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            split[r["image_id"]] = r["split"]
    return np.array([split[i] for i in ids])


def scan(ids, F, pos, tau: float) -> None:
    sp = split_of(ids)
    ev, tv = np.flatnonzero(sp == "eval"), np.flatnonzero(sp != "eval")
    blk = {(r["image_id_a"], r["image_id_b"]): float(r["blk_frac"]) for name in ("feat5_cross_eval_vs_trainval.csv", "feat5_new_cross.csv")
           for r in csv.DictReader((W / name).open(encoding="utf-8"))}
    sure = set()
    for name in ("feat3_cross_eval_vs_trainval.csv", "feat3_new_cross.csv"):
        for r in csv.DictReader((W / name).open(encoding="utf-8")):
            if float(r["rd_sharp"]) >= 0.5 and blk.get((r["image_id_a"], r["image_id_b"]), 0.0) >= 0.5:
                sure.add(r["image_id_a"])
    known = known_pairs()
    # 중간 저장: 끝까지 가야 결과가 남는 구조면 한 번 멈출 때 전부 잃는다(09-21 에 그랬다).
    # 이미 본 eval 은 건너뛰고, 찾은 쌍과 히스토그램은 100장마다 디스크에 내린다.
    done_path, cand_path = W / "shiftscan_done.txt", W / "shiftcand_cross.csv"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    todo = [a for a in ev if ids[a] not in sure and ids[a] not in done]
    hist = np.zeros(40, dtype=np.int64)
    n_all = 0
    if done and (W / "stage3d_summary.json").exists():
        prev = json.loads((W / "stage3d_summary.json").read_text(encoding="utf-8"))
        hist += np.array(prev["hist_0_1_step_0.025"], dtype=np.int64)
        n_all = prev["n_ge_tau"]
    fresh = not cand_path.exists() or not done
    print(f"eval {len(ev):,}장 가운데 확실히 걸린 {len(sure):,}장과 지난번에 본 {len(done):,}장을 건너뛰고 "
          f"{len(todo):,}장을 train·val {len(tv):,}장과 맞춘다 · 문턱 {tau} · 스레드 {WORKERS}", flush=True)
    FB = F[tv]
    n_new = 0
    t0 = time.perf_counter()

    def flush(rows: list, seen: list) -> None:
        nonlocal n_new
        with cand_path.open("a", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerows(rows)
        n_new += len(rows)
        done.update(seen)
        done_path.write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")
        (W / "stage3d_summary.json").write_text(json.dumps({
            "tau": tau, "n_eval_scanned": len(done), "n_eval_skipped_sure": len(sure), "n_ge_tau": n_all,
            "n_new_so_far": n_new, "complete": len(done) + len(sure) >= len(ev),
            "hist_0_1_step_0.025": hist.tolist()}, ensure_ascii=False), encoding="utf-8")

    if fresh:
        with cand_path.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(["image_id_a", "image_id_b", "shift_peak", "cell_dx", "cell_dy"])
    rows, seen = [], []
    for k, a in enumerate(todo, 1):
        v, dy, dx = peaks(F[a], FB)
        hist += np.histogram(v, bins=np.linspace(0, 1, 41))[0]
        for j in np.flatnonzero(v >= tau):
            n_all += 1
            key = (ids[a], ids[tv[j]])
            if key not in known:
                rows.append((key[0], key[1], f"{v[j]:.5f}", int(dx[j]), int(dy[j])))
        seen.append(ids[a])
        if k % 100 == 0:
            flush(rows, seen)
            rows, seen = [], []
            print(f"  {k:,}/{len(todo):,} 문턱 이상 {n_all:,} · 그중 새 쌍 {n_new:,} {time.perf_counter() - t0:.0f}s", flush=True)
    flush(rows, seen)
    print(f"완료 — 문턱 이상 {n_all:,}쌍 · 앞의 세 생성기에 없던 새 쌍 {n_new:,} · {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> int:
    mode = sys.argv[1]
    ids, D, flat = load_desc()
    pos = {k: i for i, k in enumerate(ids)}
    print(f"기술자 {D.shape} · 평탄 {int(flat.sum()):,}장", flush=True)
    F = sfft.rfft2(D, workers=WORKERS).astype(np.complex64)
    if mode == "pilot":
        pilot(ids, D, F, pos)
    else:
        scan(ids, F, pos, float(sys.argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
