"""근사 중복 1b단계 — 후보 문턱의 재현율. 알려진 영상에 변환을 가해 프로젝트의 지각 해시가 얼마나 움직이는지 잰다.

해시는 매니페스트와 같은 판(tiles_v1)에서, 같은 함수(compute_phash)로 계산한다. 변환본은 세션 스크래치에만 쓴다.
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.dedup.phash import compute_phash

OUT = ROOT / "_workspace" / "2026-09-21-neardup"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
csv.field_size_limit(10_000_000)
random.seed(20260921)

prov = {}
with (ROOT / "data/interim/manifest_v1/tiles.csv").open(encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        prov[r["image_id"]] = r["provenance"]
rows = []
with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        rows.append((r["image_id"], r["rel_path"].replace("tiles_v1_histmatch", "tiles_v1"), r["phash_hex"],
                     prov.get(r["image_id"], "?"), r["material"]))
# 출처 x 재질로 층화
strata: dict[tuple, list] = {}
for r in rows:
    strata.setdefault((r[3], r[4]), []).append(r)
sample = []
for k, v in sorted(strata.items()):
    sample += random.sample(v, min(len(v), max(10, round(N * len(v) / len(rows)))))
print(f"표본 {len(sample)}장 · 층 {[(k, len(v)) for k, v in sorted(strata.items())]}", flush=True)


def jpeg(im: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("L")


def resize_back(im: Image.Image, s: float) -> Image.Image:
    w, h = im.size
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def shift(im: Image.Image, dx: int, dy: int) -> Image.Image:
    a = np.asarray(im)
    return Image.fromarray(np.roll(np.roll(a, dy, axis=0), dx, axis=1))


def lut(im: Image.Image, f) -> Image.Image:
    a = np.asarray(im).astype(np.float64)
    return Image.fromarray(np.clip(f(a), 0, 255).round().astype(np.uint8))


TRANSFORMS = {
    "재저장 q95": lambda im: jpeg(im, 95),
    "재압축 q90": lambda im: jpeg(im, 90),
    "재압축 q75": lambda im: jpeg(im, 75),
    "재압축 q60": lambda im: jpeg(im, 60),
    "재압축 q40": lambda im: jpeg(im, 40),
    "재압축 q20": lambda im: jpeg(im, 20),
    "리사이즈 0.75 왕복": lambda im: resize_back(im, 0.75),
    "리사이즈 0.5 왕복": lambda im: resize_back(im, 0.5),
    "리사이즈 0.25 왕복": lambda im: resize_back(im, 0.25),
    "리사이즈 0.5 왕복 + q75": lambda im: jpeg(resize_back(im, 0.5), 75),
    "밝기 +10": lambda im: lut(im, lambda a: a + 10),
    "밝기 -25": lambda im: lut(im, lambda a: a - 25),
    "대비 x1.15": lambda im: lut(im, lambda a: (a - 128) * 1.15 + 128),
    "감마 0.8": lambda im: lut(im, lambda a: 255 * (a / 255) ** 0.8),
    "감마 1.25": lambda im: lut(im, lambda a: 255 * (a / 255) ** 1.25),
    "이동 1px": lambda im: shift(im, 1, 1),
    "이동 4px": lambda im: shift(im, 4, 2),
    "이동 16px": lambda im: shift(im, 16, 8),
    "이동 64px": lambda im: shift(im, 64, 0),
}


def ham(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


res: dict[str, list[int]] = {k: [] for k in TRANSFORMS}
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / "t.jpg"
    for n, (_iid, rel, hx, _pv, _mat) in enumerate(sample):
        with Image.open(ROOT / rel) as im0:
            im = im0.convert("L")
        for name, f in TRANSFORMS.items():
            f(im).save(tmp, "JPEG", quality=95)
            res[name].append(ham(compute_phash(tmp), hx))
        if n % 50 == 0:
            print(f"  {n}/{len(sample)}", flush=True)

summary = {}
print(f"{'변환':24s} 중앙  p95  p99  최대 | 덮는 비율 d<=16  <=24  <=32  <=48")
for name, d in res.items():
    a = np.array(d)
    summary[name] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                     "max": int(a.max()), **{f"cover_le{t}": float((a <= t).mean()) for t in (16, 24, 32, 48)}}
    s = summary[name]
    print(f"{name:24s} {s['median']:4.0f} {s['p95']:4.0f} {s['p99']:4.0f} {s['max']:5d} | "
          f"{s['cover_le16']:6.3f} {s['cover_le24']:5.3f} {s['cover_le32']:5.3f} {s['cover_le48']:5.3f}", flush=True)
(OUT / "stage1b_recall_transforms.json").write_text(json.dumps({"n": len(sample), "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
