"""통합형 학습률 스케줄 — 전역 오프셋 cosine (총괄 판정 4, 2026-09-02).

## 무엇이 비대칭이었나 (80번 F12·C12)

검출 칸은 `total_epochs` 를 전역 예산으로 두고 `start_epoch = round_idx × E` 로 옮겨
**하나의 cosine 을 라운드에 걸쳐 이어 간다.** 통합형은 그 대응물이 아예 없었다 — 상수
1e-4 였고 회계 CSV 의 `lr` 이 9행 전부 0.0001 이다. 두 칸을 나란히 놓고 RQ2 를 읽는데
학습률 궤적이 다르면 그 차이가 구조 차이에 섞인다.

## 스텝 단위로 잇는다

명세 문장이 "각 클라이언트 총 스텝 예산 전체에 걸친 단일 cosine 을 전역 스텝 오프셋으로
이어 간다"이다. epoch 이 아니라 **스텝**이다. 클라이언트마다 표본 수가 달라 epoch당
스텝이 다르므로, epoch 으로 이으면 클라이언트별 궤적이 갈린다.

    전역 스텝 = round_idx × (라운드당 스텝) + 라운드 내 스텝

## Ultralytics 와 같은 모양

검출이 쓰는 `one_cycle` 코사인을 그대로 옮겼다. 두 칸의 스케줄이 **같은 식**이어야
"대칭 회복"이 말이 된다.

    lr(x) = lr0 × [ ((1 − cos(π·x)) / 2) × (lrf − 1) + 1 ],  x = 진행률 ∈ [0, 1]

x=0 에서 lr0, x=1 에서 lr0·lrf 다.

## warmup 은 넣지 않았다 — 잔여 비대칭으로 보고한다

검출 `FIXED_OVERRIDES` 는 `warmup_epochs: 3.0` 을 쓴다. 총괄 판정 4 의 문언은
"전역 오프셋 cosine" 이고 warmup 은 언급이 없다. warmup 을 임의로 넣으면 학습 궤적을
바꾸는 5칸 공통 고정 변경을 트랙이 단독으로 하는 셈이라 기본값 0 으로 두고
`warmup_steps` 인자만 열어 둔다. 이 비대칭은 82번에 판정 요청으로 올린다.
"""

from __future__ import annotations

import math

__all__ = ["cosine_lr", "global_step", "LRF"]

#: 최종 학습률 비. 검출 `FIXED_OVERRIDES["lrf"]` 와 같은 값이다.
LRF = 0.01


def global_step(round_idx: int, step_in_round: int, steps_per_round: int) -> int:
    """전역 스텝. 라운드 경계를 넘어 하나의 스케줄로 잇는 좌표다."""
    return int(round_idx) * int(steps_per_round) + int(step_in_round)


def cosine_lr(
    lr0: float,
    step: int,
    total_steps: int,
    *,
    lrf: float = LRF,
    warmup_steps: int = 0,
) -> float:
    """전역 스텝 `step` 에서의 학습률.

    Args:
        step: 0-기반 전역 스텝.
        total_steps: 클라이언트의 **총 스텝 예산**(R × 라운드당 스텝). 라운드 길이가 아니다.
            여기에 라운드 길이를 넣으면 라운드마다 스케줄이 리셋돼 검출과 다시 어긋난다.
        warmup_steps: 0 이면 warmup 없음(기본). 켜면 0 → lr0 선형 상승.
    """
    total = max(int(total_steps), 1)
    s = min(max(int(step), 0), total)

    if warmup_steps > 0 and s < warmup_steps:
        return float(lr0) * (s + 1) / float(warmup_steps)

    x = s / total
    factor = ((1.0 - math.cos(math.pi * x)) / 2.0) * (lrf - 1.0) + 1.0
    return float(lr0) * factor
