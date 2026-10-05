"""대조 재현 — 재현 대상과 **같은 추첨인지**를 먼저 맞댄다.

앞 판은 재현 대상의 산출물을 읽지 않았다. 스냅샷이나 행 순서가 어긋나면 다른 추첨의 구성을
조용히 냈다. 여기서는 맞대는 넷(칸 서명 · 매니페스트 · 제외 목록 · 씨앗)과 추첨 수를 하나씩 어긋나게 한다.
"""

from __future__ import annotations

import pytest

from scripts.probe import neardup_control_replay as rp

SIG, MAN, EXC = "s" * 64, "m" * 64, "e" * 64


def artifact(**over) -> dict:
    ctl = {"computed": True, "cells_signature": SIG, "rng_seed": 20260825, "n_draws": 100}
    ctl.update(over.pop("ctl", {}))
    art = {"composition_control": ctl, "exclusion_list": {"sha256": EXC},
           "per_seed": {"1": {"inputs": {"manifest.csv": MAN}},
                        "2": {"inputs": {"manifest.csv": MAN}}}}
    art.update(over)
    return art


def check(art=None, **over):
    kw = {"signature": SIG, "manifest_sha256": MAN, "exclusion_sha256": EXC,
          "rng_seed": 20260825, "draws": 100}
    kw.update(over)
    return rp.check_against_rescore(art or artifact(), **kw)


def test_같은_입력이면_맞댄_값을_돌려준다() -> None:
    got = check()
    assert got["cells_signature"] == SIG and got["rescore_n_draws"] == 100


def test_추첨_수가_적으면_받는다() -> None:
    """앞에서부터 같은 추첨이다."""
    assert check(draws=20)["rescore_n_draws"] == 100


@pytest.mark.parametrize(("over", "why"), [
    ({"signature": "x" * 64}, "칸 서명"),
    ({"manifest_sha256": "x" * 64}, "매니페스트"),
    ({"exclusion_sha256": "x" * 64}, "제외 목록"),
    ({"rng_seed": 1}, "씨앗"),
    ({"draws": 101}, "추첨 수"),
])
def test_하나라도_다르면_멈춘다(over, why: str) -> None:
    with pytest.raises(rp.ReplayMismatch, match=why):
        check(**over)


def test_시드마다_매니페스트가_다르면_멈춘다() -> None:
    art = artifact()
    art["per_seed"]["2"]["inputs"]["manifest.csv"] = "y" * 64
    with pytest.raises(rp.ReplayMismatch, match="매니페스트"):
        check(art)


def test_대조_추첨이_없는_산출물은_재현_대상이_아니다() -> None:
    with pytest.raises(rp.ReplayMismatch, match="대조 추첨이 없다"):
        check(artifact(ctl={"computed": False}))


def test_사유를_한꺼번에_낸다() -> None:
    with pytest.raises(rp.ReplayMismatch) as exc:
        check(signature="x" * 64, rng_seed=1)
    assert "칸 서명" in str(exc.value) and "씨앗" in str(exc.value)


def test_퍼짐의_백분위_수준을_싣는다() -> None:
    """`lo`·`hi` 가 구간처럼 읽히지 않게."""
    got = rp.spread([0.1, 0.2, 0.3])
    assert got["lo_hi_percentiles"] == [2.5, 97.5]
