"""근사 중복 4단계 — 원인. 같은 그림으로 확정된 쌍의 **원본 파일**을 읽어(읽기 전용) 무엇이 같은지 본다.

사용: python nd_stage4_rawcause.py <특징 csv> <행 번호들(쉼표)>
원본은 data/raw/aihub71761/_zips 의 zip 구성원이다. 식별자·파일명은 화면에 찍지 않는다(성질만 찍는다).
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.measure_tiling_geometry import read_labels
from scripts.run_tiling import index_zip_members

csv.field_size_limit(10_000_000)


def jpeg_quality_hint(im: Image.Image) -> str:
    q = getattr(im, "quantization", None)
    if not q:
        return "?"
    t = q.get(0)
    return f"q0sum={int(sum(t))}" if t else "?"


def main() -> int:
    feat, want = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")]
    rows = list(csv.DictReader(feat.open(encoding="utf-8")))
    source = "aihub71761"                                  # 매니페스트 image_id 의 출처 접두
    recs = {f"{source}:{r.image_id}": r for r in read_labels(ROOT / "data/interim/aihub_labels") if r.modality == "RT"}
    members = index_zip_members(ROOT / "data/raw/aihub71761/_zips")
    print(f"라벨 {len(recs):,} · 원본 구성원 {len(members):,}", flush=True)

    def raw(iid: str):
        rec = recs[iid]
        zp, name = members[Path(rec.file_name).stem]
        with zipfile.ZipFile(zp) as z:
            data = z.read(name)
        im = Image.open(io.BytesIO(data))
        arr = np.asarray(im.convert("L"), dtype=np.float32)
        return rec, zp.name, data, im, arr

    agg = Counter()
    for idx in want:
        r = rows[idx]
        ra, za, da, ima, a = raw(r["image_id_a"])
        rb, zb, db, imb, b = raw(r["image_id_b"])
        same_bytes = hashlib.sha256(da).digest() == hashlib.sha256(db).digest()
        same_size = a.shape == b.shape
        stem_a, stem_b = Path(ra.file_name).stem, Path(rb.file_name).stem
        # 파일명 꼬리 숫자와 접두가 같은지(값은 찍지 않는다)
        pre_a, pre_b = stem_a.rsplit("_", 1)[0], stem_b.rsplit("_", 1)[0]
        line = (f"#{idx}: 원본 바이트 동일 {same_bytes} · 크기 {a.shape[::-1]} vs {b.shape[::-1]} · 같은 zip {za == zb} · "
                f"파일명 접두 같음 {pre_a == pre_b} · 정상여부 {ra.is_normal}/{rb.is_normal} · 결함종 {ra.cases}/{rb.cases} · "
                f"JPEG 양자화 {jpeg_quality_hint(ima)}/{jpeg_quality_hint(imb)} · 모드 {ima.mode}/{imb.mode}")
        if same_size:
            d = np.abs(a - b)
            ca = a - a.mean(); cb = b - b.mean()
            corr = float((ca * cb).sum() / np.sqrt((ca * ca).sum() * (cb * cb).sum()))
            line += f" · 원본 mad {d.mean():.2f} corr {corr:.4f} · 평균밝기 {a.mean():.0f}/{b.mean():.0f} · 표준편차 {a.std():.0f}/{b.std():.0f}"
        print(line, flush=True)
        agg[(same_bytes, same_size, za == zb, ra.is_normal, rb.is_normal)] += 1
    print("요약 (원본바이트동일, 크기같음, 같은zip, 정상A, 정상B):", dict(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
