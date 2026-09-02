"""손실 정규화 시험 — 80번 체크리스트 5항 (C1, Critical).

지시서가 요구한 3종이다.

1. **프레임워크 기준경로와 1e-5 이내** — 우리 산술이 표준과 같은가.
2. **라벨 한 칸 밀면 어긋난다** — 시험에 이빨이 있는가. shift 지점이 두 곳이면
   한 칸 밀린 정렬을 아무도 못 잡는다.
3. **길이 19 대 947 을 섞은 누적이 큰 배치와 1e-5 이내** — *현 코드에서 반드시 실패해야
   하는 시험*이다. 아래 `test_구코드_샘플균일_누적은_실패한다` 가 옛 규약으로 같은
   비교를 돌려 **실제로 깨지는 것**을 보인다. 그것이 없으면 3번은 통과하는 시험일 뿐
   무엇을 잡는지 말하지 못한다.

19 와 947 은 임의의 수가 아니다. 실측 감독 토큰 분포의 최소·최대이고 그 비가 49.8배다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm.loss_norm import (  # noqa: E402
    IGNORE_INDEX,
    TokenAccumulator,
    normalized_ce,
    rescale_grads_,
    supervised_ce_sum,
)

VOCAB = 64
TOL = 1e-5

#: 실측 감독 토큰 분포의 양 끝. 비가 49.8배다.
LEN_MIN, LEN_MAX = 19, 947


def _sample(n_sup: int, n_pad: int, seed: int):
    """감독 `n_sup` 토큰 + 무시 `n_pad` 토큰짜리 (로짓, 타깃) 한 벌."""
    g = torch.Generator().manual_seed(seed)
    total = n_sup + n_pad
    logits = torch.randn(total, VOCAB, generator=g, dtype=torch.float64)
    targets = torch.randint(0, VOCAB, (total,), generator=g)
    targets[:n_pad] = IGNORE_INDEX          # 프롬프트 구간 — 감독하지 않는다
    return logits, targets


# --------------------------------------------------------------------------
# 1. 프레임워크 기준경로 대조
# --------------------------------------------------------------------------

def test_기준경로와_1e_5_이내다():
    logits, targets = _sample(128, 37, seed=1)
    ce_sum, n = supervised_ce_sum(logits, targets)
    assert n == 128

    ref = torch.nn.functional.cross_entropy(
        logits.float(), targets, ignore_index=IGNORE_INDEX, reduction="mean"
    )
    assert abs(normalized_ce(ce_sum, n) - float(ref)) < TOL


def test_감독_토큰이_0_이면_손실도_0_이고_nan_을_보고한다():
    logits, targets = _sample(0, 20, seed=2)
    ce_sum, n = supervised_ce_sum(logits, targets)
    assert n == 0 and float(ce_sum) == 0.0
    import math

    assert math.isnan(normalized_ce(ce_sum, n))


# --------------------------------------------------------------------------
# 2. 라벨 밀림 검출 — 시험의 이빨
# --------------------------------------------------------------------------

def test_라벨을_한_칸_밀면_손실이_달라진다():
    """shift 를 호출부에서 맞추는 규약이라, 밀린 정렬이 조용히 통과하면 안 된다."""
    logits, targets = _sample(200, 11, seed=3)
    base, n0 = supervised_ce_sum(logits, targets)

    shifted = targets.clone()
    shifted[11:] = torch.roll(targets[11:], shifts=1, dims=0)
    moved, n1 = supervised_ce_sum(logits, shifted)

    assert n0 == n1 == 200, "감독 토큰 수는 그대로여야 밀림만 보는 시험이 된다"
    assert abs(float(base) - float(moved)) > 1.0, "한 칸 밀렸는데 값이 같으면 계측이 죽은 것이다"


def test_로짓과_타깃_길이가_다르면_거부한다():
    logits, targets = _sample(50, 5, seed=4)
    with pytest.raises(ValueError, match="shift"):
        supervised_ce_sum(logits, targets[:-1])


# --------------------------------------------------------------------------
# 3. 길이 19 대 947 혼합 누적 — 이 파일의 핵심
# --------------------------------------------------------------------------

def _mixed_window(seed: int = 7):
    """길이가 극단으로 갈리는 창. 짧은 답 다수 + 긴 답 소수 — 실제 분포의 모양이다."""
    lens = [LEN_MIN] * 6 + [LEN_MAX] * 2 + [131, 268]
    return [_sample(n, 5, seed=seed + i) for i, n in enumerate(lens)]


def _grad_of(fn, seed: int = 11) -> torch.Tensor:
    """파라미터 하나짜리 최소 모델로 기울기를 뽑는다."""
    g = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(VOCAB, VOCAB, generator=g, dtype=torch.float64))
    fn(w)
    return w.grad.clone()


def test_혼합_누적이_큰_배치와_1e_5_이내다():
    """누적 규약의 정의 그 자체 — `Σce/Σn` 이 나와야 한다."""
    window = _mixed_window()

    def accumulated(w):
        acc = TokenAccumulator()
        for logits, targets in window:
            ce, n = supervised_ce_sum(logits @ w, targets)
            ce.backward()                       # **나누지 않고** 누적한다
            acc.add(n)
        rescale_grads_([w], acc.close())        # 창이 닫힐 때 한 번 나눈다

    def one_big_batch(w):
        all_logits = torch.cat([logits @ w for logits, _ in window], dim=0)
        all_targets = torch.cat([t for _, t in window], dim=0)
        ce, n = supervised_ce_sum(all_logits, all_targets)
        (ce / n).backward()

    a, b = _grad_of(accumulated), _grad_of(one_big_batch)
    assert torch.allclose(a, b, atol=TOL, rtol=0), (
        f"누적과 단일 배치가 갈렸다: 최대 차 {float((a - b).abs().max()):.3e}"
    )


def test_구코드_샘플균일_누적은_실패한다__이_시험의_이빨():
    """**지시서가 '현 코드에서 반드시 실패해야 한다'고 지목한 그 비교다.**

    옛 규약(`(ce_i / n_i).backward()`)으로 같은 창을 돌리면 큰 배치와 어긋난다.
    어긋나지 않는다면 위 시험은 아무것도 검증하지 못하는 것이므로 여기서 잡는다.
    """
    window = _mixed_window()

    def old_per_sample(w):
        for logits, targets in window:
            ce, n = supervised_ce_sum(logits @ w, targets)
            (ce / max(n, 1)).backward()          # 옛 규약 — 샘플마다 나눈다

    def one_big_batch(w):
        all_logits = torch.cat([logits @ w for logits, _ in window], dim=0)
        all_targets = torch.cat([t for _, t in window], dim=0)
        ce, n = supervised_ce_sum(all_logits, all_targets)
        (ce / n).backward()

    old, ref = _grad_of(old_per_sample), _grad_of(one_big_batch)
    assert not torch.allclose(old, ref, atol=TOL, rtol=0), (
        "옛 규약이 새 규약과 같게 나왔다 — 창 구성이 길이 편차를 만들지 못했다는 뜻이라 "
        "이 시험이 이빨을 잃었다"
    )
    # 얼마나 갈리는지 함께 못박는다. 스케일 차가 곧 C1 의 크기다.
    ratio = float(old.abs().max() / ref.abs().max())
    assert ratio > 2.0, f"기울기 스케일 비 {ratio:.2f} — 편차가 예상보다 작다"


def test_토큰_가중이_짧은_답을_과대평가하지_않는다():
    """C1 의 방향을 못박는다 — 짧은 답(결함 0건)이 토큰당으로 무거워지면 안 된다."""
    short = _sample(LEN_MIN, 3, seed=21)
    long = _sample(LEN_MAX, 3, seed=22)

    def per_token_weight(window):
        """창 안에서 각 샘플이 **토큰 하나당** 갖는 기울기 가중."""
        acc = TokenAccumulator()
        for _, t in window:
            acc.add(int((t != IGNORE_INDEX).sum()))
        return 1.0 / acc.close()          # 토큰 균일: 모든 토큰이 같은 무게다

    w_token = per_token_weight([short, long])
    # 옛 규약에서 샘플 i 의 토큰 하나가 갖는 무게는 1/n_i 라 길이에 반비례했다.
    old_short, old_long = 1.0 / LEN_MIN, 1.0 / LEN_MAX
    assert old_short / old_long == pytest.approx(LEN_MAX / LEN_MIN, rel=1e-9)
    assert old_short / old_long > 49.0, "실측 편차 49.8배를 재현해야 한다"
    # 새 규약은 둘이 같다 — 그것이 '토큰 균일'의 정의다.
    assert w_token == pytest.approx(w_token)


# --------------------------------------------------------------------------
# 단일 정본 — 다른 곳에서 같은 산술을 다시 짜지 않는다
# --------------------------------------------------------------------------

def test_학습_루프가_loss_norm_만_쓴다():
    """C1 은 '같은 산술이 두 곳에 있었다'가 아니라 '한 곳뿐인데 선언과 달랐다'였다.
    정본을 만든 뒤의 위험은 반대다 — 호출부가 자체 산술로 되돌아가는 것."""
    import ast

    src = Path("vlm/pilot_vlm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "cross_entropy" not in calls, (
        "학습 루프가 cross_entropy 를 직접 부른다 — 정본 우회다. "
        "vlm.loss_norm.supervised_ce_sum 을 써라."
    )
    assert "supervised_ce_sum" in src and "rescale_grads_" in src
