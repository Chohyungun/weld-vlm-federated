"""함정 10 재확인 — **영상 안에 인쇄된 정보**를 눈으로 읽을 수 있게 모은다.

왜 원본을 읽나. 우리 파이프라인은 테두리 8%(가로 102px · 세로 58px)를 상수로 덮는다. 필름에 인쇄된 식별 문자열·
두께 표기·IQI 는 대개 가장자리에 찍히므로 **처리본에서는 이미 지워져 있을 수 있다.** 그래서 라벨 zip 이 아니라
**원본 이미지 zip 구성원**을 직접 열고, 전 프레임의 위·아래 띠를 잘라 붙인다.

무엇을 찾나. 두께(예: `12.6UM`), 화소당 실치수로 이어질 눈금, IQI(투과도계) 와이어, 촬영 조건 표기.
하나라도 실제로 읽히면 치수 기준 판정의 길이 열린다. 안 읽히면 "원리적으로 불가" 를 실측으로 다시 닫는다.

사용: python scripts/metadata/film_text_probe.py [장수] [출력 폴더] [띠 비율]
      기본 24장 · _workspace/2026-09-21-neardup/montage/filmtext · 0.12
원본은 **읽기만** 한다. 그림에는 표본 번호와 출처·재질만 적고 식별자·파일명은 적지 않는다.
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.measure_tiling_geometry import read_labels
from scripts.run_tiling import index_zip_members

BAND = 0.12
PER_SHEET = 6
csv.field_size_limit(10_000_000)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "_workspace/2026-09-21-neardup/montage/filmtext"
    band = float(sys.argv[3]) if len(sys.argv) > 3 else BAND

    man = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline=""))}
    prov = {r["image_id"]: r["provenance"] for r in csv.DictReader((ROOT / "data/interim/manifest_v1/tiles.csv").open(encoding="utf-8", newline=""))}
    recs = {f"aihub71761:{r.image_id}": r for r in read_labels(ROOT / "data/interim/aihub_labels") if r.modality == "RT"}
    members = index_zip_members(ROOT / "data/raw/aihub71761/_zips")
    print(f"라벨 {len(recs):,} · 원본 구성원 {len(members):,}", flush=True)

    # 출처 × 재질로 층화해 고르게 뽑는다 — 한 계열에만 인쇄가 있을 수 있다
    strata = defaultdict(list)
    for iid, m in man.items():
        if iid in recs:
            strata[f"{prov.get(iid, '?')}/{m['material']}"].append(iid)
    rng = np.random.default_rng(20260921)
    picked = []
    per = max(1, n // max(1, len(strata)))
    for k in sorted(strata):
        pool = strata[k]
        idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        picked += [(k, pool[i]) for i in idx]
    picked = picked[:n]
    print(f"층 {len(strata)}개에서 {len(picked)}장", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    made = 0
    for s in range(0, len(picked), PER_SHEET):
        chunk = picked[s:s + PER_SHEET]
        panels = []
        for key, iid in chunk:
            rec = recs[iid]
            hit = members.get(Path(rec.file_name).stem)
            if not hit:
                continue
            zp, name = hit
            with zipfile.ZipFile(zp) as z:
                data = z.read(name)
            with Image.open(io.BytesIO(data)) as im:
                a = np.asarray(im.convert("L"), dtype=np.uint8)
            h = a.shape[0]
            b = int(h * band)
            panels.append((np.vstack([a[:b], np.full((4, a.shape[1]), 255, np.uint8), a[h - b:]]),
                           f"#{made} {key} · 전 프레임 {a.shape[1]}x{a.shape[0]} · 위 {b}px + 아래 {b}px"))
            made += 1
        if not panels:
            continue
        w = max(p[0].shape[1] for p in panels)
        ht = sum(p[0].shape[0] + 18 for p in panels)
        sheet = Image.new("L", (w, ht), 255)
        dr = ImageDraw.Draw(sheet)
        y = 0
        for arr, cap in panels:
            sheet.paste(Image.fromarray(arr), (0, y + 16))
            dr.text((4, y + 3), cap, fill=0)
            y += arr.shape[0] + 18
        sheet.save(outdir / f"filmtext_{s // PER_SHEET:02d}.png")
    print(f"{made}장 → {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
