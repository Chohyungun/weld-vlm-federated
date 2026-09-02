"""감독 손실 정규화 — 30번 명세 판정 2 의 단일 정본.

## 무엇이 틀렸었나 (80번 C1, Critical)

선언은 **토큰 균일**이었다: `L = Σ_i ce_i / Σ_i n_i` — 누적 창 전체의 감독 토큰 총합이
분모다. 그런데 코드는 샘플마다 `(ce_i / n_i).backward()` 를 했다. micro=1·accum=32 이라
실제 목적함수가 `Σ_i (ce_i / n_i)` 즉 **샘플 균일**이 된다.

둘은 감독 길이가 균일할 때만 같다. 우리 데이터는 균일하지 않다 — 감독 토큰이 19~947 로
**49.8배** 퍼져 있다. 그래서 토큰당 기울기 가중이 median 17.1배·max 49.8배 갈렸고,
방향이 하필 나쁘다: **결함 0건 답의 토큰이 결함 보유 답보다 6.13배 무겁다.** 짧은 답이
"결함 배열을 비우는 쪽"이므로 헤드라인 지표인 결함 놓침 비율에 직결된다.

## 왜 창 전체를 미리 세지 않고 기울기를 나누는가

`Σ ce_i / Σ n_i` 를 얻으려면 `Σ n_i` 가 필요한데, micro=1 이라 backward 시점에는 아직
그 값을 모른다. 세 가지 길이 있었다.

1. 창 32개를 미리 인코딩해 `n_i` 를 모아 둔다 → 이미지 픽셀 텐서 32벌을 들고 있어야 한다.
   970 패치짜리 비전 입력이라 VRAM 예산(판정 10)을 그대로 깬다.
2. 텍스트만 미리 토크나이즈해 `n_i` 를 센다 → `prompt_len` 이 이미지 토큰 수에 걸려 있어
   이미지를 안 보고는 정확히 셀 수 없다. 근사하면 그 근사가 곧 새 오차다.
3. **누적한 기울기를 창 끝에서 한 번 나눈다.** `Σ_i ∇ce_i` 를 모은 뒤 `Σ n_i` 로 나누면
   `∇(Σ ce_i / Σ n_i)` 와 **정확히 같다**(미분은 선형이다). 추가 메모리 0, 근사 0.

3번을 쓴다. 그래서 이 모듈이 파는 것은 손실 함수 하나가 아니라 **누적 창 규약**이다 —
`ce_sum` 을 그대로 backward 하고, 창이 닫힐 때 `rescale_grads_` 로 나눈 뒤 step 한다.

## 기울기 크기 주의

`ce_sum` 을 나누지 않고 backward 하므로 누적 중 기울기 크기가 창 토큰 총합만큼 커진다
(전형적으로 1e4 규모). LoRA 파라미터는 fp32 라 여유가 크지만, GradScaler 를 쓰는 경로가
생기면 이 규약과 충돌한다. 지금 통합형은 bf16 autocast 라 scaler 가 없다
(`_AdapterTrainerView.scaler is None`).

## 이 모듈만 쓴다

학습 루프·회계·시험이 전부 여기를 통과한다. 같은 산술을 두 곳에 쓰면 한 곳만 고쳐지고
그 사실이 지표에 안 남는다 — C1 이 정확히 그렇게 났다.
"""

from __future__ import annotations

from typing import Iterable

import torch

__all__ = [
    "IGNORE_INDEX",
    "supervised_ce_sum",
    "normalized_ce",
    "rescale_grads_",
    "TokenAccumulator",
]

#: HuggingFace 규약. 감독하지 않는 위치의 라벨 값이다.
IGNORE_INDEX = -100


def supervised_ce_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, int]:
    """감독 위치의 교차엔트로피 **총합**과 감독 토큰 수.

    평균이 아니라 총합을 돌려주는 것이 요점이다. 분모는 호출자가 창 단위로 붙인다.

    Args:
        logits: `(B, T, V)` 또는 `(N, V)`. 이미 shift 된 상태로 들어온다 —
            **이 함수는 shift 하지 않는다.** shift 지점을 두 곳에 두면 한 칸 밀린
            정렬을 아무도 못 잡는다.
        targets: `logits` 와 앞 차원이 맞는 정수 텐서. `ignore_index` 는 제외된다.

    Returns:
        `(ce_sum, n_tokens)`. `n_tokens == 0` 이면 `ce_sum` 은 0 텐서다.
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"로짓과 타깃의 길이가 다르다: {logits.shape[0]} != {targets.shape[0]}. "
            "shift 를 호출부에서 이미 맞춰야 한다."
        )

    mask = targets != ignore_index
    n_tokens = int(mask.sum())
    if n_tokens == 0:
        return logits.sum() * 0.0, 0

    ce = torch.nn.functional.cross_entropy(
        logits[mask].float(), targets[mask], reduction="sum"
    )
    return ce, n_tokens


def normalized_ce(ce_sum: torch.Tensor | float, n_tokens: int) -> float:
    """보고용 토큰 균일 손실. 학습 기울기는 `rescale_grads_` 가 만든다."""
    if n_tokens <= 0:
        return float("nan")
    v = ce_sum.item() if isinstance(ce_sum, torch.Tensor) else float(ce_sum)
    return v / n_tokens


def rescale_grads_(params: Iterable[torch.nn.Parameter], denom: int) -> None:
    """누적 창이 닫힐 때 기울기를 창 토큰 총합으로 나눈다. **step 직전에 부른다.**

    `Σ_i ∇ce_i / D == ∇(Σ_i ce_i / D)` 이므로 이것이 토큰 균일 목적함수의 정확한 기울기다.
    `D == 0`(창 전체가 감독 토큰 0)이면 나눌 것이 없다 — 기울기도 0 이므로 그대로 둔다.
    """
    d = int(denom)
    if d <= 0:
        return
    inv = 1.0 / d
    for p in params:
        if p.grad is not None:
            p.grad.mul_(inv)


class TokenAccumulator:
    """누적 창의 감독 토큰 총합을 센다. 창이 닫히면 `close()` 로 분모를 꺼내고 리셋한다.

    상태가 정수 둘뿐이지만 클래스로 두는 이유는 **분모를 리셋하는 지점이 하나여야**
    하기 때문이다. 학습 루프에서 손으로 세면 재개 경로에서 창 경계가 어긋나도 아무 표시가
    남지 않는다.
    """

    def __init__(self) -> None:
        self.window_tokens: int = 0
        self.epoch_tokens: int = 0

    def add(self, n_tokens: int) -> None:
        n = int(n_tokens)
        self.window_tokens += n
        self.epoch_tokens += n

    def close(self) -> int:
        """창 분모를 돌려주고 창 카운터만 비운다(epoch 누계는 유지)."""
        d, self.window_tokens = self.window_tokens, 0
        return d

    def reset_epoch(self) -> None:
        self.window_tokens = 0
        self.epoch_tokens = 0
