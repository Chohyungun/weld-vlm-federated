"""라벨이 갈린 쌍의 **결함 자리를 원해상도로 잘라** 나란히 본다.

왜. 전체 프레임 몽타주로는 기공 같은 작은 결함이 보이지 않아 "한쪽 어노테이션이 빠진 것" 인지
"정말 경계라 판단이 갈린 것" 인지 가릴 수 없다. 표시가 있는 쪽의 상자를 가져와, 정합 이동만큼 옮겨
**양쪽 같은 자리**를 원해상도로 잘라 놓으면 눈으로 답할 수 있다.

한 행에 [표시된 쪽 | 표시 없는 쪽 | 차이x6] 를 놓고 상자를 그린다. 그림에는 행 번호와 라벨만 적는다.

사용: python nd_defectcrop.py <쌍 csv> <출력 폴더> <이름> <행 번호들(쉼표)> [여백 px]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
X0, Y0 = 102, 58                       # 안쪽 상자의 왼쪽 위 — bbox 는 전 프레임 좌표다
MARGIN = 60
SCALE = 2                              # 작은 상자는 키워서 본다
ROWS = 5
csv.field_size_limit(10_000_000)

REL: dict[str, str] = {}
DEF: dict[str, list[tuple]] = {}


def _load_tables() -> None:
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            REL[r["image_id"]] = r["rel_path"]
    with (ROOT / "data/interim/manifest_v1/annotations.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            xs = [r["bbox_x1_px"], r["bbox_y1_px"], r["bbox_x2_px"], r["bbox_y2_px"]]
            if any(v == "" for v in xs):
                continue                           # 상자가 비어 있는 행은 자를 자리가 없다
            DEF.setdefault(r["image_id"], []).append((*(int(v) for v in xs), r["defect_type"]))


def frame(iid: str) -> np.ndarray:
    with Image.open(ROOT / REL[iid]) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def crop(a: np.ndarray, box, dx: int = 0, dy: int = 0, margin: int = MARGIN):
    """bbox 를 (dx, dy) 만큼 옮겨 여백과 함께 자른다. 화면 밖이면 잘라 맞춘다."""
    x1, y1, x2, y2 = box[:4]
    x1, x2 = x1 + dx - margin, x2 + dx + margin
    y1, y2 = y1 + dy - margin, y2 + dy + margin
    h, w = a.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return a[y1:y2, x1:x2], (x1, y1)


def main() -> int:
    feat, outdir, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    want = [int(x) for x in sys.argv[4].split(",") if x]
    margin = int(sys.argv[5]) if len(sys.argv) > 5 else MARGIN
    _load_tables()
    rows = list(csv.DictReader(feat.open(encoding="utf-8")))
    outdir.mkdir(parents=True, exist_ok=True)
    made = 0
    sheet_rows: list = []

    def flush(k: int) -> None:
        if not sheet_rows:
            return
        wpx = max(r[0].shape[1] for r in sheet_rows) * 3 * SCALE + 16
        hpx = sum(r[0].shape[0] * SCALE + 22 for r in sheet_rows)
        sheet = Image.new("L", (wpx, hpx), 255)
        dr = ImageDraw.Draw(sheet)
        y = 0
        for imgs, cap in sheet_rows:
            ch, cw = imgs.shape[0], imgs.shape[1]
            for j, arr in enumerate(_panels[cap]):
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((cw * SCALE, ch * SCALE), Image.NEAREST)
                sheet.paste(im, (j * (cw * SCALE + 8), y + 20))
            dr.text((4, y + 4), cap, fill=0)
            y += ch * SCALE + 22
        sheet.save(outdir / f"{name}_{k:02d}.png")

    _panels: dict[str, list] = {}
    for s in range(0, len(want), ROWS):
        sheet_rows.clear()
        _panels.clear()
        for idx in want[s:s + ROWS]:
            r = rows[idx]
            ia, ib = r["image_id_a"], r["image_id_b"]
            # 표시가 있는 쪽을 왼쪽에 둔다
            if DEF.get(ia) and not DEF.get(ib):
                src, dst, flip = ia, ib, False
            elif DEF.get(ib) and not DEF.get(ia):
                src, dst, flip = ib, ia, True
            elif DEF.get(ia):
                src, dst, flip = ia, ib, False
            else:
                continue
            boxes = DEF[src]
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            A, B = frame(src), frame(dst)
            dx, dy = int(float(r.get("rd_dx") or 0)), int(float(r.get("rd_dy") or 0))
            if flip:
                dx, dy = -dx, -dy
            ca, _ = crop(A, box, 0, 0, margin)
            cb, _ = crop(B, box, -dx, -dy, margin)
            hh, ww = min(ca.shape[0], cb.shape[0]), min(ca.shape[1], cb.shape[1])
            ca, cb = ca[:hh, :ww], cb[:hh, :ww]
            cap = (f"#{idx} {box[4]} 상자{len(boxes)}개 · 이동({dx},{dy}) · "
                   f"{'표시 있는 쪽이 B' if flip else '표시 있는 쪽이 A'} | 왼쪽=표시 있음 · 가운데=표시 없음 · 오른쪽=차이x6")
            _panels[cap] = [ca, cb, np.abs(ca - cb) * 6]
            sheet_rows.append((ca, cap))
            made += 1
        flush(s // ROWS)
    print(f"{name}: {made}쌍 → {(made + ROWS - 1) // ROWS}장", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
