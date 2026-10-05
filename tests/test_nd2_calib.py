"""간선 v2 — 합성 보정의 정의(합성 그림, 실데이터를 읽지 않는다)."""
from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from data.nd2 import calib as C


def test_변환_목록이_v1_1b_와_같다():
    src = Path(__file__).resolve().parents[1] / "scripts/neardup/nd_stage1b_recall.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TRANSFORMS":
            names = [k.value for k in node.value.keys]
    assert names == list(C.TRANSFORMS)


def _tile(seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(size=(90, 160)) * 30 + 120, 0, 255).astype(np.uint8)
    im = Image.fromarray(a, mode="L").resize((1280, 720), Image.BICUBIC)
    b = BytesIO()
    im.save(b, "JPEG", quality=74)
    return b.getvalue()


def test_합성_양성은_재저장에서_높고_큰_이동에서_낮다():
    r = C.g3_positive(_tile())
    assert set(r) == set(C.TRANSFORMS)
    assert r["재저장 q95"] > 0.99
    assert r["이동 64px"] < r["이동 4px"]


def test_겹침_이동():
    assert C.overlap_shifts([0.9, 0.5, 0.2]) == [108, 538, 861]


def test_겹침_곡선은_P_탐색이_참_이동을_찾는다():
    rng = np.random.default_rng(1)
    a = np.clip(rng.normal(size=(100, 300)) * 30 + 120, 0, 255).astype(np.uint8)
    im = Image.fromarray(a, mode="L").resize((2400, 800), Image.BICUBIC)
    ref = np.cumsum(np.ones(256)) / 256
    rows = C.overlap_curve_one(im, 40, [0.9, 0.5, 0.3], quality=74, fill=114, ref_cdf=ref, p_min_overlap=0.2)
    assert [r["overlap"] for r in rows] == [0.9, 0.5, 0.3]
    assert all(r["p_shift_ok"] for r in rows)
    assert rows[0]["g4"] > rows[-1]["g4"]                        # G4 는 겹침과 함께 준다
