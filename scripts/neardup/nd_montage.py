"""근사 중복 — 쌍 몽타주. 한 줄에 [영상 A 안쪽 | 영상 B 안쪽 | 절대차 x8] 을 놓고 특징을 적는다.

사용: python nd_montage.py <특징 csv> <출력 폴더> <이름> <쌍 번호들(쉼표)> [--align]
  쌍 번호는 특징 csv 의 0-기준 행 번호.
  `--align` 은 기록된 이동(`rd_dx`·`rd_dy`, 없으면 `cell_dx`·`cell_dy` × 17px)만큼 B 를 밀어 맞춘 뒤 차이를 낸다.
  **이동이 큰 쌍은 이 옵션 없이는 차이 판이 아무것도 말해 주지 않는다** — 안 맞춘 두 영상의 차이일 뿐이다.

식별자는 그림에 적지 않는다(행 번호만). 출력은 _workspace(무시 규칙 대상).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
X0, Y0, X1, Y1 = 102, 58, 1178, 662
PW, PH = 538, 302
ROWS_PER_SHEET = 6
csv.field_size_limit(10_000_000)

rel = {}
with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        rel[r["image_id"]] = r["rel_path"]


def interior(iid: str) -> np.ndarray:
    with Image.open(ROOT / rel[iid]) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)[Y0:Y1, X0:X1]


def shift_of(r: dict) -> tuple[int, int]:
    """기록된 이동(전 해상도 px). 정밀 특징이 있으면 그 값을, G4 후보면 칸 단위 이동을 px 로 바꾼다."""
    if r.get("rd_dx"):
        return int(float(r["rd_dx"])), int(float(r["rd_dy"]))
    if r.get("cell_dx"):
        return int(r["cell_dx"]) * 17, int(r["cell_dy"]) * 17
    return 0, 0


def align(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """a[y, x] 와 b[y - dy, x - dx] 가 같은 자리가 되게 둘 다 자른다(2d 의 `_overlap` 과 같은 규약)."""
    h, w = a.shape
    if abs(dx) >= w or abs(dy) >= h:
        return a, b
    ya0, ya1, xa0, xa1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    return a[ya0:ya1, xa0:xa1], b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]


def main() -> int:
    feat, outdir, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    want = [int(x) for x in sys.argv[4].split(",") if x]
    do_align = "--align" in sys.argv[5:]
    rows = list(csv.DictReader(feat.open(encoding="utf-8")))
    outdir.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(want), ROWS_PER_SHEET):
        chunk = want[s:s + ROWS_PER_SHEET]
        sheet = Image.new("L", (PW * 3 + 8, (PH + 22) * len(chunk)), 255)
        dr = ImageDraw.Draw(sheet)
        for k, idx in enumerate(chunk):
            r = rows[idx]
            a, b = interior(r["image_id_a"]), interior(r["image_id_b"])
            if do_align:
                a, b = align(a, b, *shift_of(r))
            d = np.clip(np.abs(a - b) * 8, 0, 255)
            y = k * (PH + 22)
            for j, arr in enumerate((a, b, d)):
                sheet.paste(Image.fromarray(arr.astype(np.uint8)).resize((PW, PH), Image.BILINEAR), (j * (PW + 4), y + 20))
            def g(k, d="?", _r=r):
                return _r.get(k, d)

            def num(k, fmt=".3f", _r=r):
                v = _r.get(k)
                return format(float(v), fmt) if v not in (None, "") else "?"

            dx, dy = shift_of(r)
            cap = (f"#{idx}  ham={g('hamming')}  mad={num('mad', '.2f')}  corr={num('corr', '.4f')}  "
                   f"sharp={num('rd_sharp')}  blk={num('blk_frac')}  himin={num('hi_min')}  txtgap={num('txt_gap')}  "
                   f"이동=({dx},{dy})  [A | B | |A-B|x8{' · 이동 보정' if do_align else ''}]")
            if r.get("note"):                              # 표본의 성격을 그림에 적는다(식별자는 적지 않는다)
                cap += f"   {r['note']}"
            dr.text((4, y + 4), cap, fill=0)
        sheet.save(outdir / f"{name}_{s // ROWS_PER_SHEET:02d}.png")
    print(f"{name}: {len(want)}쌍 → {(len(want) + ROWS_PER_SHEET - 1) // ROWS_PER_SHEET}장", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
