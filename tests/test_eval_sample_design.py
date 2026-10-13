"""평가 층화 표본의 설계 규칙(25번 §2) — 두 구현이 맞춰야 할 의미를 합성 묶음으로 고정한다.

실물 매니페스트를 읽지 않는다. 공식 시드로 실물 표본을 뽑지 않는다.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe import eval_sample_ci_estimate as E  # noqa: E402


@pytest.mark.parametrize(("n_def", "n_img", "want"), [
    (0, 5, "0"), (1, 10, "1"), (2, 2, "1"),          # m = 0.1 · 1.0
    (3, 2, "2-3"), (14, 10, "1"), (15, 10, "2-3"),   # 1.5 경계는 2-3 쪽
    (34, 10, "2-3"), (35, 10, "4+"), (7, 2, "4+"),   # 3.5 경계는 4+ 쪽
])
def test_묶음_구간은_정수_비교로(n_def, n_img, want) -> None:
    assert E.group_bucket(n_def, n_img) == want


def test_결함이_있으면_0_구간이_아니다() -> None:
    assert all(E.group_bucket(k, 50) != "0" for k in range(1, 200))


@pytest.mark.parametrize(("G_h", "n", "want"), [
    (400, 3000, 96), (9, 3000, 2), (1, 3000, 1), (2, 2000, 2), (81, 2000, 13), (213, 4000, 68),
    (400, 12461, 400), (400, 20000, 400),
])
def test_배분_식(G_h, n, want) -> None:
    assert E.allocation(G_h, n) == want


def test_배분은_크기에_단조() -> None:
    for G_h in range(1, 500, 7):
        vals = [E.allocation(G_h, n) for n in (2000, 3000, 4000, 12461)]
        assert vals == sorted(vals) and vals[-1] == G_h


def _groups(k: int) -> list[dict]:
    out = []
    for i in range(k):
        out.append({"group_id": f"grp_{i:04d}", "material": "ST" if i % 5 else "AL", "n_def": (i * 7) % 23,
                    "n_img": 1 + i % 9, "rare": i % 11 == 0})
    return out


def test_같은_시드에서_작은_표본은_큰_표본의_부분집합() -> None:
    gs = _groups(600)
    seed = "test_seed|synthetic"
    picks = {}
    for n in (2000, 3000, 4000, 12461):
        sel = E.draw(gs, "P", n, seed)
        picks[n] = {h: c for h, (c, _) in sel.items()}
    for small, big in ((2000, 3000), (3000, 4000), (4000, 12461)):
        for h, c in picks[small].items():
            assert c == picks[big][h][:len(c)], (small, big, h)   # 접두


def test_순서는_시드와_묶음_id_의_sha256() -> None:
    gs = _groups(60)
    sel = E.draw(gs, "P", 12461, "s")
    for h, (chosen, G_h) in sel.items():
        keys = [hashlib.sha256(f"s|{gs[i]['group_id']}".encode("utf-8")).hexdigest() for i in chosen]
        assert keys == sorted(keys) and len(chosen) == G_h


def test_시드가_다르면_다른_표본() -> None:
    gs = _groups(600)
    a = E.draw(gs, "P", 3000, "a")
    b = E.draw(gs, "P", 3000, "b")
    assert any(a[h][0] != b[h][0] for h in a)
    assert {h: len(c) for h, (c, _) in a.items()} == {h: len(c) for h, (c, _) in b.items()}   # 층별 묶음 수는 시드와 무관


def test_복제_시드는_공식_시드와_다른_꼴() -> None:
    assert not E.OFFICIAL_SEED.startswith("ci_estimate|")
    assert E.OFFICIAL_SEED.startswith("eval_sample_v1|")
