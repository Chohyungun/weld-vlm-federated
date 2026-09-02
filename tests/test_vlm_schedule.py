"""통합형 학습률 스케줄 시험 — 총괄 판정 4 (2026-09-02) / 80번 F12·C12.

통합형은 상수 1e-4 였고 회계 CSV 의 `lr` 이 9행 전부 0.0001 이었다. 검출은 전역 오프셋
cosine 을 정확히 이행하고 있었으므로 **두 칸의 대칭이 깨진 상태**였고, 그 차이가 RQ2
비교에 그대로 섞인다.

여기서 고정하는 것은 셋이다.
1. 라운드 경계를 넘어 **하나의** cosine 으로 이어지는가 (리셋되지 않는가).
2. 검출이 쓰는 `one_cycle` 코사인과 **같은 식**인가.
3. 총 스텝 예산에 라운드 길이를 넣는 실수를 하면 어떻게 보이는가(이빨).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm.schedule import LRF, cosine_lr, global_step  # noqa: E402

LR0 = 1e-4


def test_시작은_lr0_이고_끝은_lr0_곱하기_lrf_다():
    total = 300
    assert cosine_lr(LR0, 0, total) == pytest.approx(LR0)
    assert cosine_lr(LR0, total, total) == pytest.approx(LR0 * LRF)


def test_단조_감소한다():
    total = 300
    vals = [cosine_lr(LR0, s, total) for s in range(total + 1)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "cosine 이 단조가 아니다"


def test_ultralytics_one_cycle_과_같은_식이다():
    """검출과 **같은 모양**이어야 '대칭 회복'이 말이 된다."""
    total = 100
    for s in (0, 7, 33, 66, 100):
        x = s / total
        expect = LR0 * (((1 - math.cos(math.pi * x)) / 2) * (LRF - 1) + 1)
        assert cosine_lr(LR0, s, total) == pytest.approx(expect, rel=1e-12)


def test_전역_오프셋이_라운드를_넘어_이어진다():
    """R=3·라운드당 100스텝이면 r1 의 첫 스텝은 전역 100 이다."""
    assert global_step(0, 0, 100) == 0
    assert global_step(1, 0, 100) == 100
    assert global_step(2, 50, 100) == 250

    total = 300
    seq = [cosine_lr(LR0, global_step(r, s, 100), total)
           for r in range(3) for s in range(100)]
    assert all(a >= b for a, b in zip(seq, seq[1:])), (
        "라운드 경계에서 학습률이 올라갔다 — 스케줄이 리셋된 것이다"
    )


def test_라운드_길이를_총예산으로_주면_톱니가_된다__이빨():
    """흔한 실수를 시험이 잡는다. 이렇게 하면 라운드마다 lr0 으로 되돌아간다."""
    saw = [cosine_lr(LR0, s, 100) for _ in range(3) for s in range(100)]
    boundaries = [saw[i] < saw[i + 1] for i in range(len(saw) - 1)]
    assert any(boundaries), "톱니가 안 생기면 이 시험이 아무것도 구분하지 못한다"


def test_warmup_은_기본으로_꺼져_있다():
    """검출은 warmup 3 epoch 을 쓴다. 판정 4 의 문언에는 warmup 이 없어 켜지 않았다 —
    이 비대칭은 82번에 판정 요청으로 올렸다. 기본값이 바뀌면 여기서 드러난다."""
    assert cosine_lr(LR0, 0, 100) == pytest.approx(LR0)
    assert cosine_lr(LR0, 0, 100, warmup_steps=10) < LR0


def test_학습_루프가_스케줄을_실제로_부른다():
    """상수 lr 로 되돌아가는 회귀를 막는다 — F12 가 바로 그 상태였다."""
    src = Path("vlm/pilot_vlm.py").read_text(encoding="utf-8")
    assert "cosine_lr(" in src and "global_step(" in src
    assert 'grp["lr"] = lr_now' in src, "옵티마이저에 반영되지 않으면 계산만 하는 것이다"
    assert "total_step_budget" in src


def test_총_스텝_예산이_R_을_탄다():
    """`num_rounds` 를 안 받으면 라운드마다 리셋된다."""
    import inspect

    from vlm.pilot_vlm import train_rounds

    assert "num_rounds" in inspect.signature(train_rounds).parameters
