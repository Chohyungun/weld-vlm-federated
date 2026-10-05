"""생성기의 합성 보정 — **판독 전에 합성으로** 잰다(미니스펙 2판 3-1 (가) · (다)). 값을 고르지 않는다 — 분포와 곡선만 낸다.

(가) G3 합성 양성: 정의상 같은 사진인 짝 — 저장 타일에 재압축 · 리사이즈 · 밝기 · 대비 · 감마 · 작은 이동을 건 것.
변환 목록은 v1 1b 재현율 실험(`scripts/neardup/nd_stage1b_recall.py` 의 TRANSFORMS)과 같다. 그 모듈은 들여오면 실행되므로
여기에 같은 정의를 옮겼다(시험이 두 목록의 이름 · 순서를 대조한다). 변환본은 1b 처럼 품질 95 로 저장한 뒤 기술자를 낸다.

(다) 겹침 곡선: 같은 파노라마에서 가로로 s px 떨어진 두 원점을 실물 사슬로 잘라(겹침 1 − s/1076), G3 · G4 · P 탐색의 값과
P 탐색이 참 이동(± 1 칸)을 찾았는지를 겹침마다 적는다.
"""
from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image

from data.nd2 import desc_scan as S
from data.nd2 import recut as R


def _jpeg(im: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("L")


def _resize_back(im: Image.Image, s: float) -> Image.Image:
    w, h = im.size
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def _shift(im: Image.Image, dx: int, dy: int) -> Image.Image:
    a = np.asarray(im)
    return Image.fromarray(np.roll(np.roll(a, dy, axis=0), dx, axis=1))


def _lut(im: Image.Image, f) -> Image.Image:
    a = np.asarray(im).astype(np.float64)
    return Image.fromarray(np.clip(f(a), 0, 255).round().astype(np.uint8))


TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "재저장 q95": lambda im: _jpeg(im, 95),
    "재압축 q90": lambda im: _jpeg(im, 90),
    "재압축 q75": lambda im: _jpeg(im, 75),
    "재압축 q60": lambda im: _jpeg(im, 60),
    "재압축 q40": lambda im: _jpeg(im, 40),
    "재압축 q20": lambda im: _jpeg(im, 20),
    "리사이즈 0.75 왕복": lambda im: _resize_back(im, 0.75),
    "리사이즈 0.5 왕복": lambda im: _resize_back(im, 0.5),
    "리사이즈 0.25 왕복": lambda im: _resize_back(im, 0.25),
    "리사이즈 0.5 왕복 + q75": lambda im: _jpeg(_resize_back(im, 0.5), 75),
    "밝기 +10": lambda im: _lut(im, lambda a: a + 10),
    "밝기 -25": lambda im: _lut(im, lambda a: a - 25),
    "대비 x1.15": lambda im: _lut(im, lambda a: (a - 128) * 1.15 + 128),
    "감마 0.8": lambda im: _lut(im, lambda a: 255 * (a / 255) ** 0.8),
    "감마 1.25": lambda im: _lut(im, lambda a: 255 * (a / 255) ** 1.25),
    "이동 1px": lambda im: _shift(im, 1, 1),
    "이동 4px": lambda im: _shift(im, 4, 2),
    "이동 16px": lambda im: _shift(im, 16, 8),
    "이동 64px": lambda im: _shift(im, 64, 0),
}
#: G3(영 이동 상관)의 "작은 이동" 이 아닌 것 — 문턱 제안의 5 백분위에서 뺀다(값은 따로 적는다)
LARGE = ("이동 64px",)


def g3_positive(jpeg: bytes) -> dict[str, float]:
    """저장 타일 바이트 하나 → 변환마다 (원본 기술자 · 변환본 기술자) 의 G3 상관."""
    with Image.open(io.BytesIO(jpeg)) as im0:
        im = im0.convert("L")
    d0 = R.desc_bytes(jpeg)
    out = {}
    for name, f in TRANSFORMS.items():
        b = io.BytesIO()
        f(im).save(b, "JPEG", quality=95)
        dn = S.normalize(np.stack([d0, R.desc_bytes(b.getvalue())]))
        out[name] = float(dn[0].ravel() @ dn[1].ravel())
    return out


def overlap_shifts(overlaps: list[float]) -> list[int]:
    """겹침(안쪽 상자 가로 1076 기준) → 가로 이동 px."""
    return [round((1.0 - o) * (R.X1 - R.X0)) for o in overlaps]


def overlap_curve_one(im_l: Image.Image, y0: int, overlaps: list[float], *, quality: int, fill: int,
                      ref_cdf: np.ndarray, p_min_overlap: float) -> list[dict]:
    """파노라마 하나(L) → 겹침마다 두 원점(0 과 s)을 실물 사슬로 잘라 생성기 값. 폭이 모자라는 겹침은 건너뛴다."""
    rows = []
    base = None
    for ov, s in zip(overlaps, overlap_shifts(overlaps), strict=True):
        if s + R.TILE_W > im_l.width:
            continue
        if base is None:
            base = R.desc_bytes(R.chain(im_l, (0, y0, R.TILE_W, y0 + R.TILE_H), quality=quality, fill=fill, ref_cdf=ref_cdf))
        d1 = R.desc_bytes(R.chain(im_l, (s, y0, s + R.TILE_W, y0 + R.TILE_H), quality=quality, fill=fill, ref_cdf=ref_cdf))
        dn = S.normalize(np.stack([base, d1]))
        F = S.spectra(dn, workers=1)
        g4, g4y, g4x = S.g4_peaks(F[0], F[1:], workers=1)
        p, px, py, _ = S.psearch(base, d1, p_min_overlap)
        true_cell = s / ((R.X1 - R.X0) / 64)                        # base[y, x] = d1[y, x − s] → dx = +s
        rows.append({"overlap": ov, "shift_px": s, "g3": float(dn[0].ravel() @ dn[1].ravel()),
                     "g4": float(g4[0]), "g4_dx": int(g4x[0]), "g4_dy": int(g4y[0]),
                     "p": (float(p) if np.isfinite(p) else None), "p_dx": px, "p_dy": py,
                     "p_shift_ok": bool(abs(px - true_cell) <= 1.0 and abs(py) <= 1)})
    return rows
