"""VT 근사 중복 — 기술자 · 후보 생성기(R-10) · 국소 특징 검증 점수 (4판 5-4).

    uv run python -X utf8 -m data.vt.neardup features    # 전 장의 기술자 · 창 기술자 · ORB (병렬도 2 이하, 이어 하기)
    uv run python -X utf8 -m data.vt.neardup candidates  # G-id · G-nn · G-x
    uv run python -X utf8 -m data.vt.neardup verify      # 후보마다 ORB 정합 + RANSAC 정합점 수

회차 1 은 **후보와 점수까지** 낸다. 문턱 t · k 는 분할 회차에 판독 짝으로 정한다(2판 3-4).
제외할 장(라벨 공간 밖 · 화소 꼴 제외)도 넣는다 — 뺀 사진이 두 성분을 잇는 다리일 수 있다(1판 5-2).
산출은 장 단위 값이라 스테이징(`<staging>/neardup/`)에만 둔다.

기술자(설정 `neardup`) — 1/8 해독 회색 → `INTER_AREA` 로 줄임 → 행 평균 · 열 평균 제거 → 가우시안 차(σ 둘) → L2 정규화.
세로 정상 사진은 반시계 90° 로 돌린 뒤 잰다(화소 꼴과 같은 프레임).
검증 — 두 장의 작업 프레임(결함 1/2 해독 · 정상 1/4 해독)에서 ORB · 해밍 최근접 둘의 비 · `estimateAffinePartial2D`
RANSAC. 두 장의 점을 원천 화소로 옮긴 뒤 맞추고, 재투영 문턱은 **두 작업 프레임 가운데 거친 쪽의 화소**로 잰다
(설정 4 화소 × 거친 쪽 축소 배율) — 짝의 순서(작은 id 가 어느 쪽인지)에 달리지 않는다. 배율은 원천 화소의 비다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from data.vt.config import REPO_ROOT, load_config, repo_path
from data.vt.guard import Guard, assert_vt_writable
from data.vt.labels import zip_member_index

GEN_ID, GEN_NN, GEN_X, GEN_RAND = "id", "nn", "x", "rand"


# ------------------------------------------------------------------------------------------
# 기술자
# ------------------------------------------------------------------------------------------
def coarse_desc(gray: np.ndarray, size: tuple[int, int], sigmas: tuple[float, float]) -> np.ndarray:
    """거친 구조 기술자. `size` 는 (폭, 높이)."""
    x = cv2.resize(gray.astype(np.float32), tuple(size), interpolation=cv2.INTER_AREA)
    x = x - x.mean(axis=1, keepdims=True)
    x = x - x.mean(axis=0, keepdims=True)
    d = gaussian_filter(x, sigmas[0], mode="nearest") - gaussian_filter(x, sigmas[1], mode="nearest")
    v = d.ravel().astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def window_boxes(w: int, h: int, scales: list[float], tile: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """배율마다 창(타일 / 배율)을 간격(창의 절반)으로 깐다. 마지막 창은 경계에 붙인다. 원천 화소 좌표."""
    out = []
    for s in scales:
        ww, wh = round(tile[0] / s), round(tile[1] / s)
        if ww > w or wh > h:
            continue
        sx, sy = max(1, ww // 2), max(1, wh // 2)
        xs = list(range(0, w - ww + 1, sx))
        ys = list(range(0, h - wh + 1, sy))
        if xs[-1] != w - ww:
            xs.append(w - ww)
        if ys[-1] != h - wh:
            ys.append(h - wh)
        out += [(x, y, x + ww, y + wh) for y in ys for x in xs]
    return out


def _gray_draft(raw: bytes, reduce: int, rot90_k: int) -> tuple[np.ndarray, int, int]:
    """축소 해독 회색과 (돌린 뒤) 원천 크기."""
    with Image.open(io.BytesIO(raw)) as im:
        W, H = im.size
        im.draft("L", (max(1, W // reduce), max(1, H // reduce)))
        g = np.asarray(im.convert("L"))
    if rot90_k:
        g = np.ascontiguousarray(np.rot90(g, k=rot90_k))
        W, H = H, W
    return g, W, H


def image_features(raw: bytes, *, is_normal: bool, rot90_k: int, cfg: dict) -> dict:
    nd = cfg["neardup"]
    tile = tuple(cfg["tile"]["tile_size"])
    g8, W, H = _gray_draft(raw, 8, rot90_k)
    out = {"desc64": coarse_desc(g8, tuple(nd["descriptor_size"]), tuple(nd["dog_sigmas"]))}
    xsize = tuple(nd["cross"]["descriptor_size"])
    fy, fx = g8.shape[0] / H, g8.shape[1] / W
    if is_normal:
        boxes = window_boxes(W, H, nd["cross"]["window_scales"], tile)
        wd = []
        for (x0, y0, x1, y1) in boxes:
            a, c = round(y0 * fy), round(x0 * fx)
            b, d = max(round(y1 * fy), a + 1), max(round(x1 * fx), c + 1)
            wd.append(coarse_desc(g8[a:b, c:d], xsize, tuple(nd["dog_sigmas"])))
        out["win_desc"] = np.stack(wd).astype(np.float16) if wd else np.zeros((0, xsize[0] * xsize[1]), np.float16)
        out["win_box"] = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    else:
        out["desc32"] = coarse_desc(g8, xsize, tuple(nd["dog_sigmas"]))
    v = nd["verify"]
    red = int(v["normal_reduce"] if is_normal else v["defect_reduce"])
    gw, W2, H2 = _gray_draft(raw, red, rot90_k)
    out["frame_wh"] = (W2, H2)                       # 원천 화소의 프레임 크기(돌린 뒤)
    orb = cv2.ORB_create(nfeatures=int(v["orb_features"]))
    kp, de = orb.detectAndCompute(gw, None)
    out["orb_xy"] = np.asarray([k.pt for k in kp], dtype=np.float32).reshape(-1, 2)
    out["orb_desc"] = de if de is not None else np.zeros((0, 32), np.uint8)
    out["orb_f"] = W2 / gw.shape[1]                 # 원천 화소 / 작업 프레임 화소
    return out


# ------------------------------------------------------------------------------------------
# 후보
# ------------------------------------------------------------------------------------------
def id_pairs(folder_of: dict[int, str], span: int) -> set[tuple[int, int]]:
    """G-id — 같은 폴더 안 id 차 1 ~ span."""
    by: dict[str, set[int]] = defaultdict(set)
    for i, f in folder_of.items():
        by[f].add(i)
    out = set()
    for ids in by.values():
        for a in ids:
            for d in range(1, span + 1):
                if a + d in ids:
                    out.add((a, a + d))
    return out


def nn_pairs(ids: np.ndarray, desc: np.ndarray, k: int, block: int = 2048) -> set[tuple[int, int]]:
    """G-nn — 코사인 최근접 상위 k(자기 제외). 짝은 (작은 id, 큰 id)."""
    X = desc.astype(np.float32)
    out = set()
    n = len(ids)
    kk = min(k, n - 1)
    if kk <= 0:
        return out
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        top = np.argsort(-S, axis=1, kind="stable")[:, :kk]     # 동률은 자리(입력 순서) 오름차순 — 결정적이다
        for r, row in enumerate(top):
            a = int(ids[s + r])
            for j in row:
                b = int(ids[j])
                out.add((min(a, b), max(a, b)))
    return out


def random_pairs(folder_of: dict[int, str], folders: list[str], seed: int, per_folder: int) -> set[tuple[int, int]]:
    """G-rand — 같은 폴더의 무작위 짝(대조군). 실물의 "다른 장면" 점수 분포를 낸다(06b I-4). 결정적이다."""
    import random

    rng = random.Random(seed)
    out: set[tuple[int, int]] = set()
    for f in folders:
        ids = sorted(i for i, g in folder_of.items() if g == f)
        if len(ids) < 2:
            continue
        got: set[tuple[int, int]] = set()
        tries = 0
        while len(got) < per_folder and tries < per_folder * 20:
            a, b = rng.sample(ids, 2)
            got.add((min(a, b), max(a, b)))
            tries += 1
        out |= got
    return out


class CrossTopK:
    """G-x — 결함 크롭마다 정상 사진 창의 코사인 상위 k. 같은 사진의 창은 하나(가장 높은 창)로 센다."""

    def __init__(self, q: np.ndarray, k: int) -> None:
        self.q = q.astype(np.float32)                   # 결함 × 차원
        n = len(q)
        self.k = k
        self.score = np.full((n, k), -np.inf, np.float32)
        self.photo = np.full((n, k), -1, np.int64)
        self.win = np.full((n, k), -1, np.int64)

    def update(self, photo_ids: np.ndarray, win_desc: np.ndarray, win_owner: np.ndarray) -> None:
        """사진 여럿의 창을 한 번에. `win_owner` 는 창마다 `photo_ids` 의 자리(정렬된 연속 구간)."""
        if len(win_desc) == 0:
            return
        S = win_desc.astype(np.float32) @ self.q.T     # 창 × 결함
        starts = np.flatnonzero(np.r_[True, win_owner[1:] != win_owner[:-1]])
        best = np.maximum.reduceat(S, starts, axis=0)  # 사진 × 결함
        arg = np.empty_like(best, dtype=np.int64)
        for i, (a, b) in enumerate(zip(starts, list(starts[1:]) + [len(S)], strict=True)):
            arg[i] = np.argmax(S[a:b], axis=0)         # 사진 안의 창 번호
        owners = photo_ids[win_owner[starts]]
        cs = np.concatenate([self.score, best.T], axis=1)
        cp = np.concatenate([self.photo, np.broadcast_to(owners, (len(self.q), len(owners)))], axis=1)
        cw = np.concatenate([self.win, arg.T], axis=1)
        order = np.argsort(-cs, axis=1, kind="stable")[:, :self.k]
        rows = np.arange(len(self.q))[:, None]
        self.score, self.photo, self.win = cs[rows, order], cp[rows, order], cw[rows, order]


# ------------------------------------------------------------------------------------------
# 검증
# ------------------------------------------------------------------------------------------
def verify_pair(xy_a, de_a, f_a, xy_b, de_b, f_b, *, ratio: float, reproj: float,
                wh_a: tuple[int, int] | None = None, wh_b: tuple[int, int] | None = None) -> dict:
    """ORB 정합 + RANSAC. **질의는 A 다**(짝의 작은 id). 값은 원천 화소다.

    - `inliers` 정합점 수 · `scale` A → B 변환의 배율(B 에서 본 A 의 크기 비)
    - `cover_a` · `cover_b` 정합점이 각 장에서 덮는 경계 상자의 면적 몫 — 글자처럼 한 곳에 몰린 정합과 화면 전체에
      퍼진 정합을 뒤에서 가를 재료(06b I-4). 프레임 크기를 주지 않으면 비운다
    - `tx` · `ty` A → B 변환의 이동(B 프레임의 원천 화소)
    """
    out = {"matches": 0, "inliers": 0, "scale": "", "cover_a": "", "cover_b": "", "tx": "", "ty": ""}
    if len(de_a) < 2 or len(de_b) < 2:
        return out
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for m, n in (p for p in bf.knnMatch(de_a, de_b, k=2) if len(p) == 2) if m.distance < ratio * n.distance]
    out["matches"] = len(good)
    if len(good) < 3:
        return out
    pa = np.float32([xy_a[m.queryIdx] for m in good]) * np.float32(f_a)     # 원천 화소
    pb = np.float32([xy_b[m.trainIdx] for m in good]) * np.float32(f_b)
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                         ransacReprojThreshold=reproj * max(f_a, f_b))
    if M is None or inl is None:
        return out
    out["inliers"] = int(inl.sum())
    out["scale"] = round(float(np.hypot(M[0, 0], M[1, 0])), 4)
    out["tx"], out["ty"] = round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)
    m = inl.ravel().astype(bool)
    for side, pts, wh in (("cover_a", pa[m], wh_a), ("cover_b", pb[m], wh_b)):
        if wh is not None and len(pts):
            span = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
            out[side] = round(float(span) / float(wh[0] * wh[1]), 4)
    return out


# ------------------------------------------------------------------------------------------
# 실행
# ------------------------------------------------------------------------------------------
def _paths(cfg: dict) -> dict:
    st = repo_path(cfg["staging_root"])
    return {"staging": st, "nd": st / "neardup", "chunks": st / "neardup" / "chunks",
            "raw": repo_path(cfg["copy"]["raw_zip_dir"])}


def _rot(d: dict, cfg: dict) -> int:
    return int(cfg["tile"]["portrait_normal_rot90_k"]) if d["is_normal"] and d["h"] > d["w"] else 0


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _input_hashes(P: dict) -> dict:
    """후보 · 대조의 결속 — 덩어리 메타 · 라벨 파싱 산출 · 이 모듈의 바이트 · OpenCV 판(06c N-4)."""
    return {"chunks_meta_sha256": _sha(P["chunks"] / "_meta.json"),
            "labels_jsonl_sha256": _sha(P["staging"] / "parse" / "labels.jsonl"),
            "module_sha256": _sha(Path(__file__)), "opencv": cv2.__version__}


def bind_meta(path: Path, meta: dict) -> None:
    """이어 하기 폴더를 입력(설정 · 입력 해시)에 묶는다. 이미 다른 값으로 묶였으면 멈춘다."""
    text = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"이어 하던 산출이 다른 입력으로 만들어졌다 — 그 폴더를 옆 이름으로 비켜 두고(지우지 않는다) "
                             f"다시 돈다: {path.parent}")
        return
    path.write_text(text, encoding="utf-8")


def _feature_chunk(args: tuple) -> str:
    zpath, rows, cfg, out_path = args
    cv2.setNumThreads(1)
    feats = []
    with zipfile.ZipFile(zpath) as zf:
        idx = zip_member_index(zf)                     # 같은 줄기 멤버가 둘이면 멈춘다(06b m-12 · 06c N-6)
        for d in rows:
            feats.append(image_features(zf.read(idx[d["file_stem"]]), is_normal=d["is_normal"], rot90_k=_rot(d, cfg),
                                        cfg=cfg))
    ids = np.asarray([d["vt_id"] for d in rows], np.int64)
    kp_n = np.asarray([len(f["orb_desc"]) for f in feats], np.int64)
    pack = {"ids": ids, "is_normal": np.asarray([d["is_normal"] for d in rows]),
            "desc64": np.stack([f["desc64"] for f in feats]).astype(np.float16),
            "orb_n": kp_n, "orb_f": np.asarray([f["orb_f"] for f in feats], np.float32),
            "frame_wh": np.asarray([f["frame_wh"] for f in feats], np.int32).reshape(-1, 2),
            "orb_xy": np.concatenate([f["orb_xy"] for f in feats]) if kp_n.sum() else np.zeros((0, 2), np.float32),
            "orb_desc": np.concatenate([f["orb_desc"] for f in feats]) if kp_n.sum() else np.zeros((0, 32), np.uint8)}
    nrm = [f for f in feats if "win_desc" in f]
    if nrm:
        pack["win_n"] = np.asarray([len(f["win_desc"]) for f in nrm], np.int64)
        pack["win_desc"] = np.concatenate([f["win_desc"] for f in nrm])
        pack["win_box"] = np.concatenate([f["win_box"] for f in nrm])
    dfc = [f for f in feats if "desc32" in f]
    if dfc:
        pack["desc32"] = np.stack([f["desc32"] for f in dfc]).astype(np.float16)
    tmp = Path(out_path + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **pack)
    os.replace(tmp, out_path)
    return out_path


def cmd_features(cfg: dict, workers: int, chunk: int) -> int:
    from data.vt.build_r1 import load_labels

    if workers > 2:
        raise SystemExit("병렬도는 2 이하다")
    from data.vt.build_r1 import require_copy_verified

    P = _paths(cfg)
    assert_vt_writable(P["chunks"], allowed_roots=[P["staging"]], repo_root=REPO_ROOT)
    require_copy_verified(P["staging"])
    P["chunks"].mkdir(parents=True, exist_ok=True)
    # 덩어리를 입력 · 코드에 묶는다 — 라벨 파싱 산출 · 복사 기록 · 이 모듈의 바이트 · 해독기 판(06b I-7 ④)
    import PIL

    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    bind_meta(P["chunks"] / "_meta.json", {
        "neardup": cfg["neardup"], "tile": cfg["tile"], "chunk": chunk,
        "labels_jsonl_sha256": _sha(P["staging"] / "parse" / "labels.jsonl"),
        "copy_record_sha256": _sha(P["staging"] / "copy" / "copy_record.jsonl"),
        "module_sha256": _sha(Path(__file__)), "opencv": cv2.__version__, "pillow": PIL.__version__})
    labs = load_labels(P["staging"])
    by_zip: dict[str, list] = defaultdict(list)
    for d in labs:
        by_zip[d["src_zip"]].append(d)
    tasks = []
    for z, rows in sorted(by_zip.items()):
        rows.sort(key=lambda d: d["vt_id"])
        for i in range(0, len(rows), chunk):
            out = P["chunks"] / f"{Path(z).stem}__{i // chunk:04d}.npz"
            if not out.exists():
                tasks.append((str(P["raw"] / z), rows[i:i + chunk], cfg, str(out)))
    logf = (P["nd"] / "features_log.jsonl").open("a", encoding="utf-8", newline="\n")

    def log(d):
        d = {"t": time.strftime("%H:%M:%S"), **d}
        logf.write(json.dumps(d, ensure_ascii=False) + "\n")
        logf.flush()
        print(json.dumps(d, ensure_ascii=False), flush=True)

    guard = Guard.from_config(cfg, REPO_ROOT, log=log, check_c=False)     # E: 사본만 읽는다
    log({"start": True, "todo_chunks": len(tasks)})
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending, n = [], 0
        for t in tasks:
            guard.wait()
            pending.append(ex.submit(_feature_chunk, t))
            while len(pending) >= workers:
                pending.pop(0).result()
                n += 1
                log({"chunks_done": n, "of": len(tasks), "seconds": round(time.monotonic() - t0, 1)})
        for f in pending:
            f.result()
            n += 1
    log({"chunks_done": n, "end": True, "seconds": round(time.monotonic() - t0, 1)})
    return merge_features(cfg)


def merge_features(cfg: dict) -> int:
    """덩어리 파일을 큰 배열로 — 덩어리 순서 그대로 쓴다(자리는 `ids.npy` 로 찾는다).

    창 기술자는 합치지 않는다(후보 단계가 덩어리마다 읽는다). ORB 는 메모리 사상 파일에 덩어리마다 채운다 —
    검증 단계의 두 작업자가 같은 파일을 메모리 사상으로 함께 읽는다.
    """
    P = _paths(cfg)
    assert_vt_writable(P["nd"], allowed_roots=[P["staging"]], repo_root=REPO_ROOT)
    files = sorted(P["chunks"].glob("*.npz"))
    small = []
    for f in files:
        with np.load(f) as z:
            small.append({"ids": z["ids"], "is_normal": z["is_normal"], "orb_n": z["orb_n"]})
    ids = np.concatenate([x["ids"] for x in small])
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("같은 id 가 두 덩어리에 있다")
    kp_n = np.concatenate([x["orb_n"] for x in small])
    off = np.concatenate([[0], np.cumsum(kp_n)]).astype(np.int64)
    nd = P["nd"]
    xy = np.lib.format.open_memmap(nd / "orb_xy.npy", mode="w+", dtype=np.float32, shape=(int(off[-1]), 2))
    de = np.lib.format.open_memmap(nd / "orb_desc.npy", mode="w+", dtype=np.uint8, shape=(int(off[-1]), 32))
    d64, f_all, d_ids, d32, wh_all = [], [], [], [], []
    pos, n_win = 0, 0
    for f, x in zip(files, small, strict=True):
        with np.load(f) as z:
            k = int(z["orb_n"].sum())
            xy[pos:pos + k] = z["orb_xy"]
            de[pos:pos + k] = z["orb_desc"]
            pos += k
            d64.append(z["desc64"])
            f_all.append(z["orb_f"])
            wh_all.append(z["frame_wh"])
            if "desc32" in z:
                d_ids.append(z["ids"][~z["is_normal"]])
                d32.append(z["desc32"])
            if "win_n" in z:
                n_win += int(z["win_n"].sum())
    xy.flush()
    de.flush()
    del xy, de
    np.save(nd / "ids.npy", ids)
    np.save(nd / "is_normal.npy", np.concatenate([x["is_normal"] for x in small]))
    np.save(nd / "desc64.npy", np.concatenate(d64))
    np.save(nd / "orb_off.npy", off)
    np.save(nd / "orb_f.npy", np.concatenate(f_all))
    np.save(nd / "frame_wh.npy", np.concatenate(wh_all))
    np.save(nd / "defect_ids.npy", np.concatenate(d_ids))
    np.save(nd / "defect_desc32.npy", np.concatenate(d32))
    meta = {"n": len(ids), "n_defect": int(sum(len(v) for v in d_ids)), "orb_kp_total": int(off[-1]),
            "orb_kp_zero_images": int((kp_n == 0).sum()), "windows_total": n_win, "chunks": len(files)}
    (nd / "features_meta.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


def cmd_candidates(cfg: dict, threads: int, photos_per_step: int = 8) -> int:
    import csv

    from threadpoolctl import threadpool_limits

    from data.vt.build_r1 import load_labels
    from data.vt.records import read_csv_checked

    P = _paths(cfg)
    assert_vt_writable(P["nd"], allowed_roots=[P["staging"]], repo_root=REPO_ROOT)
    Guard.from_config(cfg, REPO_ROOT, log=lambda d: print(json.dumps(d, ensure_ascii=False)), check_c=False).wait()
    nd = cfg["neardup"]
    labs = load_labels(P["staging"])
    folder_of = {d["vt_id"]: d["folder"] for d in labs}
    excluded = {d["vt_id"] for d in labs if d["label_excluded"]}
    summ_path = P["staging"] / "records" / "plan_summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        excl_rows = read_csv_checked(P["staging"] / "records" / "exclusions.csv", summ["records"]["exclusions.csv"])
        excluded |= {int(r["image_id"].split(":")[1]) for r in excl_rows}
    ids = np.load(P["nd"] / "ids.npy")
    if set(ids.tolist()) != set(folder_of):
        raise RuntimeError("기술자의 장 집합이 라벨 전수와 다르다")
    is_n = np.load(P["nd"] / "is_normal.npy")
    d64 = np.load(P["nd"] / "desc64.npy")
    gens: dict[tuple[int, int], set[str]] = defaultdict(set)
    for pr in id_pairs(folder_of, int(nd["id_span"])):
        gens[pr].add(GEN_ID)
    rc = nd["random_control"]
    for pr in random_pairs(folder_of, list(cfg["labels"]["folders"]), int(rc["seed"]), int(rc["per_folder"])):
        gens[pr].add(GEN_RAND)
    with threadpool_limits(threads):
        for mask in (~is_n, is_n):
            for pr in nn_pairs(ids[mask], d64[mask], int(nd["nn_k"])):
                gens[pr].add(GEN_NN)
        # G-x — 사진 몇 장씩 나눠 곱한다(창 × 결함 행렬이 메모리에 들게)
        d_ids = np.load(P["nd"] / "defect_ids.npy")
        xt = CrossTopK(np.load(P["nd"] / "defect_desc32.npy"), int(nd["cross"]["top_k"]))
        boxes: dict[int, np.ndarray] = {}
        for f in sorted(P["chunks"].glob("*.npz")):
            with np.load(f) as z:
                if "win_n" not in z:
                    continue
                nid, wn, wd, wb = z["ids"][z["is_normal"]], z["win_n"], z["win_desc"], z["win_box"]
            woff = np.concatenate([[0], np.cumsum(wn)])
            for s0 in range(0, len(nid), photos_per_step):
                s1 = min(s0 + photos_per_step, len(nid))
                rows = slice(int(woff[s0]), int(woff[s1]))
                owner = np.repeat(np.arange(s1 - s0), wn[s0:s1])
                xt.update(nid[s0:s1], wd[rows], owner)
                for j in range(s0, s1):
                    boxes[int(nid[j])] = wb[int(woff[j]):int(woff[j + 1])]
    x_rows = []
    for r, did in enumerate(d_ids):
        for sc, pid, w in zip(xt.score[r], xt.photo[r], xt.win[r], strict=True):
            if pid < 0:
                continue
            gens[(min(int(did), int(pid)), max(int(did), int(pid)))].add(GEN_X)
            box = boxes[int(pid)][int(w)]
            x_rows.append({"defect_id": int(did), "normal_id": int(pid), "cos": round(float(sc), 4),
                           "win": ",".join(str(int(v)) for v in box)})
    with (P["nd"] / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["a_id", "b_id", "gens"])
        for (a, b), g in sorted(gens.items()):
            w.writerow([a, b, "+".join(sorted(g))])
    with (P["nd"] / "cross_windows.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["defect_id", "normal_id", "cos", "win"], lineterminator="\n")
        w.writeheader()
        w.writerows(x_rows)
    cnt: dict[str, int] = defaultdict(int)
    for g in gens.values():
        for x in g:
            cnt[x] += 1
    meta = {"inputs": _input_hashes(P), "pairs": len(gens), "by_generator": dict(cnt),
            "images_excluded": len(excluded),
            "pairs_touching_excluded": sum(1 for (a, b) in gens if a in excluded or b in excluded),
            "excluded_images_in_some_pair": len({i for pr in gens for i in pr} & excluded),
            "cross_rows": len(x_rows)}
    (P["nd"] / "candidates_meta.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


_SHARED: dict = {}


def _verify_init(nd_dir: str) -> None:
    cv2.setNumThreads(1)
    d = Path(nd_dir)
    ids = np.load(d / "ids.npy")
    _SHARED.update({"pos": {int(v): i for i, v in enumerate(ids)}, "off": np.load(d / "orb_off.npy"),
                    "xy": np.load(d / "orb_xy.npy", mmap_mode="r"), "de": np.load(d / "orb_desc.npy", mmap_mode="r"),
                    "f": np.load(d / "orb_f.npy"), "wh": np.load(d / "frame_wh.npy")})


def _verify_chunk(args: tuple) -> str:
    pairs, ratio, reproj, out_path = args
    S = _SHARED
    rows = []
    for a, b, g in pairs:
        ia, ib = S["pos"][a], S["pos"][b]
        sa, ea = S["off"][ia], S["off"][ia + 1]
        sb, eb = S["off"][ib], S["off"][ib + 1]
        r = verify_pair(np.asarray(S["xy"][sa:ea]), np.asarray(S["de"][sa:ea]), float(S["f"][ia]),
                        np.asarray(S["xy"][sb:eb]), np.asarray(S["de"][sb:eb]), float(S["f"][ib]),
                        ratio=ratio, reproj=reproj, wh_a=tuple(S["wh"][ia]), wh_b=tuple(S["wh"][ib]))
        rows.append(f"{a},{b},{g},{r['matches']},{r['inliers']},{r['scale']},{r['cover_a']},{r['cover_b']},"
                    f"{r['tx']},{r['ty']}\n")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(rows)
    os.replace(tmp, out_path)
    return out_path


def cmd_verify(cfg: dict, workers: int, chunk: int) -> int:
    import csv

    if workers > 2:
        raise SystemExit("병렬도는 2 이하다")
    P = _paths(cfg)
    v = cfg["neardup"]["verify"]
    vd = P["nd"] / "verify_chunks"
    assert_vt_writable(vd, allowed_roots=[P["staging"]], repo_root=REPO_ROOT)
    vd.mkdir(parents=True, exist_ok=True)
    cand_path = P["nd"] / "candidates.csv"
    cand_sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
    feat_sha = hashlib.sha256((P["nd"] / "orb_off.npy").read_bytes()).hexdigest()
    bind_meta(vd / "_meta.json", {"candidates_sha256": cand_sha, "orb_off_sha256": feat_sha, "chunk": chunk,
                                  "verify": v, "module_sha256": _sha(Path(__file__)), "opencv": cv2.__version__})
    with cand_path.open(encoding="utf-8") as fh:
        pairs = [(int(r["a_id"]), int(r["b_id"]), r["gens"]) for r in csv.DictReader(fh)]
    cmeta = json.loads((P["nd"] / "candidates_meta.json").read_text(encoding="utf-8"))
    if len(pairs) != cmeta["pairs"]:
        raise SystemExit(f"후보 줄 수 {len(pairs)} 이 후보 메타의 {cmeta['pairs']} 와 다르다")
    tasks = []
    for i in range(0, len(pairs), chunk):
        out = vd / f"v{i // chunk:05d}.csv"
        if not out.exists():
            tasks.append((pairs[i:i + chunk], float(v["ratio"]), float(v["ransac_reproj_px"]), str(out)))
    logf = (P["nd"] / "verify_log.jsonl").open("a", encoding="utf-8", newline="\n")

    def log(d):
        d = {"t": time.strftime("%H:%M:%S"), **d}
        logf.write(json.dumps(d, ensure_ascii=False) + "\n")
        logf.flush()
        print(json.dumps(d, ensure_ascii=False), flush=True)

    guard = Guard.from_config(cfg, REPO_ROOT, log=log, check_c=False)     # E: 사본의 특징만 읽는다
    log({"start": True, "pairs": len(pairs), "todo_chunks": len(tasks)})
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_verify_init, initargs=(str(P["nd"]),)) as ex:
        pending, n = [], 0
        for t in tasks:
            guard.wait()
            pending.append(ex.submit(_verify_chunk, t))
            while len(pending) >= workers:
                pending.pop(0).result()
                n += 1
                if n % 10 == 0:
                    log({"chunks_done": n, "of": len(tasks), "seconds": round(time.monotonic() - t0, 1)})
        for f in pending:
            f.result()
            n += 1
    files = [vd / f"v{i // chunk:05d}.csv" for i in range(0, len(pairs), chunk)]
    body = "".join(f.read_text(encoding="utf-8") for f in files)
    got = [tuple(line.split(",")[:3]) for line in body.splitlines()]
    if got != [(str(a), str(b), g) for a, b, g in pairs]:
        raise SystemExit("점수의 짝 순서가 후보와 다르다 — verify_chunks 를 옆 이름으로 비켜 두고(지우지 않는다) 다시 돈다")
    with (P["nd"] / "scores.csv").open("w", encoding="utf-8", newline="") as out:
        out.write("a_id,b_id,gens,matches,inliers,scale,cover_a,cover_b,tx,ty\n")
        out.write(body)
    (P["nd"] / "scores_meta.json").write_text(json.dumps({
        "candidates_sha256": cand_sha, "rows": len(got),
        "how_to_read": {
            "query": "a_id(짝의 작은 id)가 질의다 — 정합 · RANSAC 은 A → B 방향이다",
            "scale": "A → B 변환의 배율(B 에서 본 A 의 크기 비) — 원천 화소",
            "cover_a/cover_b": "정합점의 경계 상자 면적 / 그 장의 프레임 면적(원천 화소, 세로 정상은 돌린 뒤)",
            "tx/ty": "A → B 변환의 이동 — B 프레임의 원천 화소",
            "reproj": "RANSAC 재투영 문턱 = 설정 ransac_reproj_px × 두 장 가운데 큰 축소 배율(원천 화소)",
            "gens": "id · nn · x · rand(같은 폴더 무작위 대조군) 의 합 — 한 짝이 여럿에서 나올 수 있다",
            "not_a_threshold": "점수는 간선이 아니다 — 문턱 t 는 분할 회차에 판독 짝으로 정한다"}},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log({"end": True, "chunks": len(files), "pairs": len(got), "candidates_sha256": cand_sha,
         "seconds": round(time.monotonic() - t0, 1)})
    return 0


# ------------------------------------------------------------------------------------------
# 대조 — 점수를 읽기 전에 등록한 두 대조(6판). 후보 · 검증을 바꾸지 않는 별도 단계다
# ------------------------------------------------------------------------------------------
def positive_pair_features(raw: bytes, *, is_normal: bool, cfg: dict) -> tuple[dict, dict]:
    """R-8b 와 같은 꼴의 옮긴 창 짝(원본의 frac 크기 창 둘, shift 만큼 옮긴 원점, 품질 q)의 특징 둘."""
    from data.vt.r8 import synth_pair

    r8 = cfg["r8"]
    a, b = synth_pair(raw, float(r8["synth_window_frac"]), float(r8["synth_shift_frac"]), int(r8["synth_quality"]))
    out = []
    for x in (a, b):
        with Image.open(io.BytesIO(x)) as im:
            w, h = im.size
        k = int(cfg["tile"]["portrait_normal_rot90_k"]) if is_normal and h > w else 0
        out.append(image_features(x, is_normal=is_normal, rot90_k=k, cfg=cfg))
    return out[0], out[1]


def control_judgement(pos: list[int], rand: list[int], rule: dict) -> dict:
    """한 폴더의 판정 — 무작위 짝의 q 백분위보다 정합점이 큰 합성 겹침의 몫이 문턱 이상이면 양성 대조가 선다(6판)."""
    if not pos or not rand:
        return {"stands": None, "why": "짝이 없다"}
    r = sorted(rand)
    floor = r[min(len(r) - 1, int(float(rule["rand_percentile"]) * len(r)))]
    share = sum(x > floor for x in pos) / len(pos)
    return {"stands": share >= float(rule["min_share"]), "rand_floor": floor, "share_above": round(share, 4),
            "pos_median": sorted(pos)[len(pos) // 2], "n_pos": len(pos), "n_rand": len(r)}


def cmd_control(cfg: dict) -> int:
    import csv
    import random

    from data.vt.build_r1 import load_labels, require_copy_verified

    P = _paths(cfg)
    out = P["nd"] / "control"
    assert_vt_writable(out, allowed_roots=[P["staging"]], repo_root=REPO_ROOT)
    require_copy_verified(P["staging"])
    Guard.from_config(cfg, REPO_ROOT, log=lambda d: print(json.dumps(d, ensure_ascii=False)), check_c=False).wait()
    cv2.setNumThreads(1)
    if out.exists():
        raise SystemExit(f"대조 산출이 이미 있다 — 옆 이름으로 비켜 두고(지우지 않는다) 다시 돈다: {out}")
    out.mkdir(parents=True)
    nc, r8, v = cfg["neardup_control"], cfg["r8"], cfg["neardup"]["verify"]
    labs = load_labels(P["staging"])
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for d in labs:
        by_folder[d["folder"]].append(d)
    # (가) ORB 양성 대조 — R-8b 가 고른 그 장들(같은 순서 키 · 같은 수)
    pos_rows = []
    for folder, rows in by_folder.items():
        pick = sorted(rows, key=lambda d: hashlib.sha256(f"{r8['synth_rank_salt']}{d['image_id']}".encode()).hexdigest())
        pick = pick[: int(r8["synth_per_folder"])]
        by_zip: dict[str, list[dict]] = defaultdict(list)
        for d in pick:
            by_zip[d["src_zip"]].append(d)
        for z, zr in sorted(by_zip.items()):
            with zipfile.ZipFile(P["raw"] / z) as zf:
                idx = zip_member_index(zf)
                for d in zr:
                    fa, fb = positive_pair_features(zf.read(idx[d["file_stem"]]), is_normal=d["is_normal"], cfg=cfg)
                    r = verify_pair(fa["orb_xy"], fa["orb_desc"], fa["orb_f"], fb["orb_xy"], fb["orb_desc"], fb["orb_f"],
                                    ratio=float(v["ratio"]), reproj=float(v["ransac_reproj_px"]),
                                    wh_a=fa["frame_wh"], wh_b=fb["frame_wh"])
                    pos_rows.append({"folder": folder, "image_id": d["image_id"], **r})
        print(json.dumps({"positive_done": folder, "n": len(pick)}, ensure_ascii=False), flush=True)
    # (나) 결함–정상 무작위 짝 — G-x 의 바닥(결함 크롭 대 정상 사진, 이미 단 특징으로)
    _verify_init(str(P["nd"]))
    S = _SHARED
    rng = random.Random(int(nc["cross_random"]["seed"]))
    normals = sorted(d["vt_id"] for d in labs if d["is_normal"])
    cross_rows = []
    for folder, rows in by_folder.items():
        if rows[0]["is_normal"]:
            continue
        ids = sorted(d["vt_id"] for d in rows)
        for _ in range(int(nc["cross_random"]["per_defect_folder"])):
            a, b = rng.choice(ids), rng.choice(normals)
            ia, ib = S["pos"][a], S["pos"][b]
            sa, ea, sb, eb = S["off"][ia], S["off"][ia + 1], S["off"][ib], S["off"][ib + 1]
            r = verify_pair(np.asarray(S["xy"][sa:ea]), np.asarray(S["de"][sa:ea]), float(S["f"][ia]),
                            np.asarray(S["xy"][sb:eb]), np.asarray(S["de"][sb:eb]), float(S["f"][ib]),
                            ratio=float(v["ratio"]), reproj=float(v["ransac_reproj_px"]),
                            wh_a=tuple(S["wh"][ia]), wh_b=tuple(S["wh"][ib]))
            cross_rows.append({"folder": folder, "a_id": a, "b_id": b, **r})
    cols = ["matches", "inliers", "scale", "cover_a", "cover_b", "tx", "ty"]
    for name, rows, head in (("positive.csv", pos_rows, ["folder", "image_id"]),
                             ("cross_random.csv", cross_rows, ["folder", "a_id", "b_id"])):
        with (out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head + cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    meta = {"inputs": _input_hashes(P), "rule": nc["positive_rule"], "cross_random": nc["cross_random"],
            "positive": {"selection": "R-8b 와 같다 — sha256(synth_rank_salt + image_id) 오름차순 앞 synth_per_folder",
                         "pairs": len(pos_rows)}, "cross_random_pairs": len(cross_rows),
            "sha256": {n: _sha(out / n) for n in ("positive.csv", "cross_random.csv")}}
    # 판정 — 검증 점수가 있으면 폴더마다 G-rand 의 바닥과 맞댄다(6판의 규칙)
    sp = P["nd"] / "scores.csv"
    if sp.exists():
        folder_of = {d["vt_id"]: d["folder"] for d in labs}
        rand: dict[str, list[int]] = defaultdict(list)
        with sp.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if "rand" in r["gens"].split("+"):
                    rand[folder_of[int(r["a_id"])]].append(int(r["inliers"]))
        meta["judgement"] = {f: control_judgement([int(r["inliers"]) for r in pos_rows if r["folder"] == f],
                                                  rand.get(f, []), nc["positive_rule"]) for f in by_folder}
        meta["scores_sha256"] = _sha(sp)
    (out / "control_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "inputs"}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["features", "merge", "candidates", "verify", "control"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.stage == "features":
        return cmd_features(cfg, args.workers, args.chunk)
    if args.stage == "merge":
        return merge_features(cfg)
    if args.stage == "candidates":
        return cmd_candidates(cfg, args.threads)
    if args.stage == "control":
        return cmd_control(cfg)
    return cmd_verify(cfg, args.workers, args.chunk * 20)


if __name__ == "__main__":
    sys.exit(main())
