"""타일 모원본 다시 자르기 — 실물 사슬을 메모리에서 그대로 거친다(미니스펙 2판 3-1 (라)).

사슬: 원천 파노라마(읽기만) → L · 품질 74 재인코딩(`tiling.encode_tile` 과 같은 일) → 테두리 마스킹
(`run_border_mask.mask_and_encode` 를 **그대로** 부른다) → 학습 풀 기준 히스토그램 정합(`run_hist_match.build_lut` 와
`stage_apply` 의 저장 줄) → JPEG → `draft` 해독 → 3a 의 기술자(`nd_stage3a_bpdesc.desc` 와 같은 줄).
다시 자른 타일은 디스크에 쓰지 않는다 — 실물 함수가 경로에 쓰는 자리에는 메모리 받개를 준다.

같은 일임은 두 겹으로 본다. 합성 시험이 실물 함수(`encode_tile` · `desc`)와 바이트 · 배열이 같음을 보이고,
실물에서는 자기 확인(`self_check`)이 실제 원점으로 다시 만든 정합 JPEG 의 sha256 을 매니페스트와 대조한다.
"""
from __future__ import annotations

import hashlib
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from data.convert.tiling import candidate_origins
from scripts.run_border_mask import mask_and_encode
from scripts.run_hist_match import build_lut

TILE_W, TILE_H, STRIDE_X, ALIGN = 1280, 720, 640, 8
#: 3a 기술자의 안쪽 상자(1280 × 720 기준) — `scripts/neardup/nd_stage3a_bpdesc.py` 와 같다
X0, Y0, X1, Y1 = 102, 58, 1178, 662


class _MemParent:
    def mkdir(self, *a, **k) -> None:
        return None


class MemSink:
    """`dst.parent.mkdir(...)` · `dst.write_bytes(data)` 만 받는 메모리 받개 — 실물 함수가 디스크 대신 여기에 쓴다."""

    def __init__(self) -> None:
        self.parent = _MemParent()
        self.data: bytes | None = None

    def write_bytes(self, data: bytes) -> int:
        self.data = bytes(data)
        return len(data)


def origins(width: int) -> list[int]:
    return candidate_origins(width, TILE_W, STRIDE_X, ALIGN)


def encode_crop(im_l: Image.Image, box: tuple[int, int, int, int], *, quality: int) -> bytes:
    """`tiling.encode_tile` 의 잘라 저장하는 줄(mode L · progressive False · optimize False) — 해독은 한 번만 한다."""
    buf = BytesIO()
    im_l.crop(box).save(buf, format="JPEG", quality=quality, progressive=False, optimize=False)
    return buf.getvalue()


def mask_bytes(tile_jpeg: bytes, *, fill: int, quality: int) -> bytes:
    sink = MemSink()
    mask_and_encode(BytesIO(tile_jpeg), sink, fill, quality)     # 실물 함수 그대로
    assert sink.data is not None
    return sink.data


def histmatch_bytes(masked_jpeg: bytes, ref_cdf: np.ndarray, *, quality: int) -> bytes:
    """`run_hist_match.stage_apply` 의 정합 · 저장 줄과 같다."""
    with Image.open(BytesIO(masked_jpeg)) as im:
        arr = np.asarray(im.convert("L"))
    lut = build_lut(np.bincount(arr.ravel(), minlength=256), ref_cdf)
    buf = BytesIO()
    Image.fromarray(lut[arr], mode="L").save(buf, format="JPEG", quality=quality, progressive=False, optimize=False)
    return buf.getvalue()


def desc_bytes(jpeg: bytes) -> np.ndarray:
    """`nd_stage3a_bpdesc.desc` 와 같은 줄 — 경로 대신 바이트를 받는다."""
    cv2.setNumThreads(1)
    with Image.open(BytesIO(jpeg)) as im:
        im.draft("L", (640, 360))
        a = np.asarray(im.convert("L"), dtype=np.float32)
    sy, sx = a.shape[0] / 720.0, a.shape[1] / 1280.0
    a = a[round(Y0 * sy):round(Y1 * sy), round(X0 * sx):round(X1 * sx)]
    bp = cv2.GaussianBlur(a, (0, 0), 4.0 * sx * 2) - cv2.GaussianBlur(a, (0, 0), 16.0 * sx * 2)
    bp -= bp.mean(axis=1, keepdims=True)
    bp -= bp.mean(axis=0, keepdims=True)
    return cv2.resize(bp, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def chain(im_l: Image.Image, box: tuple[int, int, int, int], *, quality: int, fill: int, ref_cdf: np.ndarray) -> bytes:
    """잘라 재인코딩 → 마스킹 → 정합까지. 정합 JPEG 바이트(저장 타일과 같아야 하는 것)."""
    return histmatch_bytes(mask_bytes(encode_crop(im_l, box, quality=quality), fill=fill, quality=quality),
                           ref_cdf, quality=quality)


def recut(raw: bytes, y0: int, *, quality: int, fill: int, ref_cdf: np.ndarray,
          want_sha_at: int | None = None) -> tuple[list[int], np.ndarray, str | None]:
    """파노라마 원천 바이트 → (원점 목록, 원점마다의 기술자 (n, 36, 64), 실제 원점의 정합 JPEG sha256).

    세로 원점은 저장 타일의 상자에서 받는다(밴드 중심 — `plan_tile` 이 정한 값).
    """
    with Image.open(BytesIO(raw)) as im:
        im_l = im.convert("L")
    xs = origins(im_l.width)
    out = np.empty((len(xs), 36, 64), dtype=np.float32)
    sha = None
    for k, x0 in enumerate(xs):
        jpg = chain(im_l, (x0, y0, x0 + TILE_W, y0 + TILE_H), quality=quality, fill=fill, ref_cdf=ref_cdf)
        out[k] = desc_bytes(jpg)
        if want_sha_at is not None and x0 == want_sha_at:
            sha = hashlib.sha256(jpg).hexdigest()
    return xs, out, sha


def pair_overlap(o_prime: int, o_y: int) -> float:
    """평가 타일 E 가 파노라마 Y 의 원점 o′ 과 맞을 때, E 와 Y 의 저장 타일(원점 oY)의 안쪽 상자 겹침(가로만)."""
    return max(0.0, 1.0 - abs(o_prime - o_y) / (X1 - X0))
