"""R-8 — 이웃 화소 측정의 양성 대조 (4판 4절, 02d N-16). 원천 E: 사본을 읽기만 한다.

    uv run python -X utf8 -m data.vt.r8 [--prior <3판의 neighbor_pixels.json>]

- **측정 함수는 3판 1-3 과 같다**(`feat` — 1/8 해독 회색 9×8 의 가로 차 64 비트 · 회색 64×36 상관, 무작위 짝은
  같은 폴더 · 같은 크기, 고정 시드 20261004). 결함 폴더 넷은 3판과 같은 순서 · 같은 난수 흐름으로 다시 재어 3판의
  값이 다시 나오는지 맞댄다(`--prior`). 정상 폴더는 새 난수 흐름(같은 시드)으로 잰다 — R-8a.
- R-8b — 폴더마다 `sha256(salt + image_id)` 오름차순 앞 N 장에서 원본의 90 % 창 둘(원점과, 5 % 옮긴 원점)을
  품질 90 으로 다시 인코딩한 짝. 같은 `feat`.
- 판정은 R-5 그대로 — 짝의 차분 해시 거리 중앙값이 그 폴더 무작위 짝의 절반 이하이면 "닮았다".

장 단위 값은 내지 않는다. 폴더별 집계만 스테이징의 `r8/r8.json` 에 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data.vt.config import REPO_ROOT, load_config, repo_path
from data.vt.guard import assert_vt_writable
from data.vt.labels import zip_member_index


def defect_order(cfg: dict) -> list[str]:
    """결함 폴더 — 설정의 순서(3판 1-3 이 잰 순서와 같다: 기공 · 용입부족 · 융합불량 · 언더컷)."""
    return [f for f in cfg["labels"]["folders"] if f != cfg["labels"]["normal_folder"]]


def feat(raw: bytes) -> dict:
    """3판 1-3 의 측정 함수(실측 스크립트 `vt_long_runs.feat`)와 같은 계산."""
    im = Image.open(io.BytesIO(raw))
    im.draft("L", (im.size[0] // 8, im.size[1] // 8))
    g = im.convert("L")
    small = np.asarray(g.resize((64, 36), Image.BILINEAR), dtype=np.float32)
    d9 = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.float32)
    dh = (d9[:, 1:] > d9[:, :-1]).flatten()
    return {"sha": hashlib.sha256(raw).hexdigest(), "small": small, "dhash": dh, "size": im.size}


def pair_stats(pairs, F) -> dict:
    """3판 1-3 의 집계와 같다."""
    dh = [int((F[a]["dhash"] != F[b]["dhash"]).sum()) for a, b in pairs]
    cs = []
    for a, b in pairs:
        x, y = F[a]["small"].ravel(), F[b]["small"].ravel()
        if x.std() > 0 and y.std() > 0:
            cs.append(float(np.corrcoef(x, y)[0, 1]))
    return {"n_pairs": len(pairs), "dhash_median": float(np.median(dh)) if dh else None,
            "dhash_le_5_share": round(sum(d <= 5 for d in dh) / max(1, len(dh)), 4),
            "corr_median": round(float(np.median(cs)), 4) if cs else None,
            "corr_ge_0_9_share": round(sum(c >= 0.9 for c in cs) / max(1, len(cs)), 4),
            "same_bytes": sum(F[a]["sha"] == F[b]["sha"] for a, b in pairs)}


def neighbor_and_random(F: dict, rng: random.Random) -> tuple[list, list]:
    """이웃(id 차 1)과 같은 크기 무작위 짝 — 3판 1-3 의 절차 그대로(난수 소비 순서까지)."""
    ids = sorted(F)
    nb = [(a, a + 1) for a in ids if a + 1 in F]
    by_size: dict = {}
    for i in ids:
        by_size.setdefault(F[i]["size"], []).append(i)
    rand = []
    for a, _ in nb:
        b = rng.choice(by_size[F[a]["size"]])
        if b != a:
            rand.append((a, b))
    return nb, rand


def similar(s_pairs: dict, s_rand: dict) -> bool:
    return (s_pairs["dhash_median"] is not None and s_rand["dhash_median"] is not None
            and s_pairs["dhash_median"] <= s_rand["dhash_median"] / 2)


def synth_pair(raw: bytes, frac: float, shift: float, quality: int) -> tuple[bytes, bytes]:
    """원본의 `frac` 크기 창 둘 — 원점과, 가로 · 세로로 `shift` 만큼 옮긴 원점. 품질 `quality` 로 다시 인코딩."""
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        W, H = im.size
        ww, wh = int(W * frac), int(H * frac)
        dx, dy = int(W * shift), int(H * shift)
        out = []
        for x0, y0 in ((0, 0), (dx, dy)):
            buf = io.BytesIO()
            im.crop((x0, y0, x0 + ww, y0 + wh)).save(buf, format="JPEG", quality=quality)
            out.append(buf.getvalue())
    return out[0], out[1]


def consequence(folders: dict, normal_folder: str, defect_folders: list[str]) -> str:
    """귀결 문장 — 4판 4절과 5판 2절의 보완. "닮지 않았다" 는 "거의 같은 구도가 아니다" 까지만 뜻한다."""
    n = folders[normal_folder]
    if n["neighbors_similar"]:
        failed = [f for f in defect_folders if not folders[f]["synthetic_similar"]]
        if not failed:
            return "양성 대조가 선다 — 정상 바닥이 화소 근거를 얻고, 결함 폴더 넷의 '닮지 않았다' 를 근거로 쓴다"
        return ("양성 대조가 선다 — 다만 합성 겹침을 잡지 못한 결함 폴더(" + " · ".join(failed)
                + ")의 '닮지 않았다' 는 근거로 쓰지 않는다(5판 2절)")
    if n["synthetic_similar"]:
        return ("결함 폴더의 '닮지 않았다' 를 근거로 쓰지 않고 4판 9절 1 로 올린다 — "
                "측정은 겹친 창을 잡지만 정상의 이웃은 거의 같은 구도가 아니다")
    return "결함 폴더의 '닮지 않았다' 를 근거로 쓰지 않고 4판 9절 1 로 올린다 — 측정이 겹침을 잡지 못한다"


def main(argv: list[str] | None = None) -> int:
    from data.vt.build_r1 import load_labels

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", type=Path, default=None, help="3판의 neighbor_pixels.json — 결함 넷의 재현 대조")
    args = ap.parse_args(argv)
    cfg = load_config()
    r8 = cfg["r8"]
    from data.vt.build_r1 import require_copy_verified
    from data.vt.guard import Guard

    staging = repo_path(cfg["staging_root"])
    out_dir = staging / "r8"
    assert_vt_writable(out_dir, allowed_roots=[staging], repo_root=REPO_ROOT)
    require_copy_verified(staging)
    Guard.from_config(cfg, REPO_ROOT, log=lambda d: print(json.dumps(d, ensure_ascii=False)), check_c=False).wait()
    DEFECT_ORDER = defect_order(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = repo_path(cfg["copy"]["raw_zip_dir"])
    labs = load_labels(staging)
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for d in labs:
        by_folder[d["folder"]].append(d)
    t0 = time.monotonic()
    feats: dict[str, dict[int, dict]] = {}
    synth_raw: dict[str, list[tuple[bytes, bytes]]] = defaultdict(list)
    for folder, rows in by_folder.items():
        pick = sorted(rows, key=lambda d: hashlib.sha256(f"{r8['synth_rank_salt']}{d['image_id']}".encode()).hexdigest())
        pick_ids = {d["vt_id"] for d in pick[: int(r8["synth_per_folder"])]}
        F = {}
        by_zip: dict[str, list[dict]] = defaultdict(list)
        for d in rows:
            by_zip[d["src_zip"]].append(d)
        for z, zr in sorted(by_zip.items()):
            with zipfile.ZipFile(raw_dir / z) as zf:
                idx = zip_member_index(zf)               # 같은 줄기 멤버가 둘이면 멈춘다(06b m-12)
                for d in zr:
                    raw = zf.read(idx[d["file_stem"]])
                    F[d["vt_id"]] = feat(raw)
                    if d["vt_id"] in pick_ids:
                        synth_raw[folder].append(synth_pair(raw, float(r8["synth_window_frac"]),
                                                            float(r8["synth_shift_frac"]), int(r8["synth_quality"])))
        feats[folder] = F
        print(json.dumps({"folder": folder, "n": len(F), "seconds": round(time.monotonic() - t0, 1)},
                         ensure_ascii=False), flush=True)

    res = {"rule": "짝의 차분 해시 거리 중앙값 ≤ 그 폴더 무작위 짝의 절반이면 닮았다 (R-5 · R-8)", "folders": {}}
    rng_defect = random.Random(int(r8["random_seed"]))       # 3판 1-3 과 같은 흐름 — 결함 넷을 그 순서로
    order = DEFECT_ORDER + [f for f in feats if f not in DEFECT_ORDER]
    for folder in order:
        F = feats[folder]
        rng = rng_defect if folder in DEFECT_ORDER else random.Random(int(r8["random_seed"]))
        nb, rand = neighbor_and_random(F, rng)
        s_nb, s_rd = pair_stats(nb, F), pair_stats(rand, F)
        SF, sp = {}, []
        for k, (ra, rb) in enumerate(synth_raw[folder]):
            SF[2 * k], SF[2 * k + 1] = feat(ra), feat(rb)
            sp.append((2 * k, 2 * k + 1))
        s_sy = pair_stats(sp, SF)
        res["folders"][folder] = {
            "n_images": len(F), "neighbors": s_nb, "random_same_size": s_rd, "synthetic_overlap": s_sy,
            "neighbors_similar": similar(s_nb, s_rd), "synthetic_similar": similar(s_sy, s_rd)}
    nf = cfg["labels"]["normal_folder"]
    n = res["folders"][nf]
    res["R-8a_normal_neighbors_similar"] = n["neighbors_similar"]
    res["R-8b_synthetic_similar_by_folder"] = {f: v["synthetic_similar"] for f, v in res["folders"].items()}
    res["consequence"] = consequence(res["folders"], nf, DEFECT_ORDER)
    # 폴더마다 — 그 폴더의 "닮지 않았다" 를 근거로 쓸 수 있는가: R-8a 가 서고 그 폴더의 R-8b 도 설 때만(5판 2절)
    res["defect_not_similar_usable"] = {f: bool(n["neighbors_similar"] and res["folders"][f]["synthetic_similar"])
                                        for f in DEFECT_ORDER}
    if args.prior:
        prior = json.loads(args.prior.read_text(encoding="utf-8"))
        rep = {}
        for f in DEFECT_ORDER:
            a, b = prior["folders"][f], res["folders"][f]
            rep[f] = all(a[k] == b[k] for k in ("neighbors", "random_same_size"))
        res["reproduces_3rd_edition"] = rep
        res["prior_sha256"] = hashlib.sha256(args.prior.read_bytes()).hexdigest()
    res["seconds"] = round(time.monotonic() - t0, 1)
    (out_dir / "r8.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                     newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "folders"}, ensure_ascii=False))
    # 3판 측정이 다시 나오지 않으면 같은 함수가 아니다 — 0 으로 끝내지 않는다(06b I-5)
    return 3 if args.prior and not all(res["reproduces_3rd_edition"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
