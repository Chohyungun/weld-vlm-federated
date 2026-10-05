"""간선 v2 — 모원본 다시 자르기가 실물 함수와 같은 바이트 · 배열을 내는지(합성 파노라마, 실데이터를 읽지 않는다)."""
from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from data.convert.tiling import candidate_origins, encode_tile
from data.nd2 import recut as R
from scripts.neardup import nd_stage3a_bpdesc as bpdesc
from scripts.run_border_mask import mask_and_encode


def _pano(w: int, h: int = 900, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    a = (rng.normal(size=(h // 8, w // 8)) * 30 + 120)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.mark.parametrize("width", [1280, 1500, 2565, 4000])
def test_원점_집합이_실물_함수와_같다(width):
    assert R.origins(width) == candidate_origins(width, 1280, 640, 8)
    assert R.origins(width)[-1] + 1280 <= width


def test_잘라_재인코딩이_실물과_같은_바이트(tmp_path):
    raw = _pano(2600)
    box = (640, 96, 1920, 816)
    sha, _, _ = encode_tile(BytesIO(raw), box, tmp_path / "t.jpg", mode="L", quality=74, progressive=False, optimize=False)
    with Image.open(BytesIO(raw)) as im:
        mine = R.encode_crop(im.convert("L"), box, quality=74)
    assert hashlib.sha256(mine).hexdigest() == sha


def test_마스킹이_실물과_같은_바이트(tmp_path):
    raw = _pano(1280, 720, seed=3)
    sha, _, _ = mask_and_encode(BytesIO(raw), tmp_path / "m.jpg", 114, 74)
    assert hashlib.sha256(R.mask_bytes(raw, fill=114, quality=74)).hexdigest() == sha


def test_기술자가_3a_와_같다(tmp_path):
    raw = _pano(1280, 720, seed=4)
    p = tmp_path / "d.jpg"
    p.write_bytes(raw)
    np.testing.assert_array_equal(R.desc_bytes(raw), bpdesc.desc(str(p)))     # 절대 경로는 ROOT 를 무시한다


def test_정합의_사상은_단조이고_기준_분포로_간다():
    ref = np.ones(256)
    ref_cdf = np.cumsum(ref) / ref.sum()
    raw = _pano(1280, 720, seed=5)
    out = R.histmatch_bytes(R.mask_bytes(raw, fill=114, quality=74), ref_cdf, quality=74)
    with Image.open(BytesIO(out)) as im:
        a = np.asarray(im.convert("L"))
    assert a.shape == (720, 1280)
    assert 100 < a.mean() < 155                                                # 균등 기준이면 평균이 가운데 근처


def test_다시_자르기는_실제_원점의_정합_해시를_낸다():
    ref_cdf = np.cumsum(np.ones(256)) / 256
    raw = _pano(2565, 800, seed=6)
    xs, D, sha = R.recut(raw, 40, quality=74, fill=114, ref_cdf=ref_cdf, want_sha_at=640)
    assert xs == candidate_origins(2565, 1280, 640, 8)
    assert D.shape == (len(xs), 36, 64)
    with Image.open(BytesIO(raw)) as im:
        want = R.chain(im.convert("L"), (640, 40, 1920, 760), quality=74, fill=114, ref_cdf=ref_cdf)
    assert sha == hashlib.sha256(want).hexdigest()
    np.testing.assert_array_equal(D[xs.index(640)], R.desc_bytes(want))


def test_메모리_받개는_디스크에_쓰지_않는다(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = _pano(1280, 720, seed=7)
    R.mask_bytes(raw, fill=114, quality=74)
    assert list(tmp_path.iterdir()) == []


def test_짝_겹침():
    assert R.pair_overlap(640, 640) == 1.0
    assert R.pair_overlap(640, 0) == pytest.approx(1 - 640 / 1076)
    assert R.pair_overlap(1280, 0) == 0.0
