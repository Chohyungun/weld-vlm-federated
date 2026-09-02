"""연합 집계 — 표본 수 가중 FedAvg와 BatchNorm 버퍼 정책 (지정 함정 구간 #2).

## BatchNorm 을 왜 함께 평균하는가

C1·C2(강재)와 C3(알루미늄)는 입력 분포가 실제로 다르다. FedAvg 로 BN 의 running 통계까지
평균하면 글로벌 모델이 흔들릴 수 있고, 알려진 처방은 BN 을 로컬에 남기는 FedBN 이다.
**그러나 이 연구는 글로벌 모델 하나를 글로벌 평가셋으로 채점하는 설계라 로컬 BN 을 남길
수 없다.** 그래서 기본안은 BN 버퍼 포함 전체 가중 평균으로 가고, 그 선택과 근거를 논문에
한 문장으로 남긴다.

대신 흔들림을 감지할 진단만 붙인다. 진단은 **관측 전용**이며 어떤 신호도 실행 중 동작을
바꾸지 않는다. 런 도중 지표를 보고 개입하면 그 순간 사전 등록의 의미가 사라진다.

## `num_batches_tracked` 를 평균하지 않는 이유

정수 카운터다. 가중 평균하면 반올림이 끼어들고, 그 값은 어느 클라이언트의 실제 관측
횟수도 아니게 된다. momentum 이 설정돼 있으면 이 값은 정규화 연산에 쓰이지도 않는다.
결정론적으로 최댓값을 통과시켜 "가장 많이 본 클라이언트 기준"으로 맞춘다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch

from detection import serialize

__all__ = ["AggregationResult", "weighted_fedavg", "bn_divergence"]


@dataclass
class AggregationResult:
    """집계 산출물과 진단.

    `client_norms`·`global_norm`은 집계 직전·직후 파라미터 norm 이다. 순서 뒤섞임이나
    dtype 사고가 나면 이 값이 눈에 띄게 튀므로, 라운드 곡선으로 보면 조기에 잡힌다.
    """

    ndarrays: list[np.ndarray]
    client_norms: list[float] = field(default_factory=list)
    global_norm: float = 0.0
    total_examples: int = 0
    bn_buffer_divergence: float = 0.0
    missing_variance_ratio: float = 0.0


def weighted_fedavg(
    client_arrays: Sequence[Sequence[np.ndarray]],
    num_examples: Sequence[int],
    keys: Sequence[str],
    reference: Mapping[str, torch.Tensor],
) -> AggregationResult:
    """표본 수 가중 FedAvg.

    Args:
        client_arrays: 클라이언트별 ndarray 리스트. 전부 `keys` 순서를 따라야 한다.
        num_examples: 클라이언트별 학습 표본 수 (C1:C2:C3 ≈ 32k:16k:8k).
        reference: 기준 모델의 state_dict. dtype·shape 검사에 쓴다.

    Raises:
        SerializeError: 키·shape·dtype 이 어긋나면 집계하지 않고 실패한다.
        ValueError: 클라이언트가 없거나 표본 수 합이 0이면 실패한다.
    """
    if not client_arrays:
        raise ValueError("집계할 클라이언트가 없다. accept_failures=False 인데 여기 도달했다면 서버 배선 오류다.")
    if len(client_arrays) != len(num_examples):
        raise ValueError(f"클라이언트 수와 표본 수가 다르다: {len(client_arrays)} != {len(num_examples)}")

    total = int(sum(num_examples))
    if total <= 0:
        raise ValueError("표본 수 합이 0이다. 가중치를 만들 수 없다.")

    # 무결성 검사를 먼저 — 어긋난 상태로 평균을 내면 조용히 오염된 모델이 나온다.
    for arrays in client_arrays:
        serialize.assert_compatible(arrays, keys, reference)

    # G2-2 ① 유한성 전수. NaN 한 원소가 섞이면 평균 전체가 NaN 이 되고 지표에는
    # `global_l2 = nan` 만 남는다 — 어느 클라이언트의 어느 텐서인지 알 수 없다.
    # 실측에서 정확히 그 상태가 났다(80번 F3).
    for ci, arrays in enumerate(client_arrays):
        for k, arr in zip(keys, arrays):
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"클라이언트 {ci} 의 [{k}] 에 유한하지 않은 값 {n_bad}개. "
                    "집계하면 글로벌이 통째로 NaN 이 되고 출처를 잃는다."
                )

    weights = np.array([n / total for n in num_examples], dtype=np.float64)
    # G2-2 ② 가중치 합. 재정규화 사고(실패 클라이언트를 빼고 평균)를 산술로 잡는다.
    if abs(float(weights.sum()) - 1.0) > 1e-9:
        raise ValueError(
            f"가중치 합이 1 이 아니다: {float(weights.sum()):.12f}. "
            "R×E=N 등가가 흔적 없이 깨지는 경로다."
        )
    out: list[np.ndarray] = []

    for i, k in enumerate(keys):
        stack = [c[i] for c in client_arrays]
        if np.issubdtype(stack[0].dtype, np.floating):
            acc = np.zeros(stack[0].shape, dtype=np.float64)
            for w, arr in zip(weights, stack):
                acc += w * arr.astype(np.float64, copy=False)
            out.append(acc.astype(np.float32, copy=False))
        else:
            # 정수 버퍼 — 평균 금지. 결정론적 최댓값 통과.
            acc = stack[0]
            for arr in stack[1:]:
                acc = np.maximum(acc, arr)
            out.append(np.array(acc, copy=True))

    client_norms = [serialize.params_l2_norm(a) for a in client_arrays]
    global_norm = serialize.params_l2_norm(out)
    # G2-2 ③ 출력 norm 불변식. 볼록 가중 평균은 ‖Σ w_k x_k‖ ≤ Σ w_k‖x_k‖ ≤ max‖x_k‖ 다.
    # 넘으면 평균이 아니라 다른 산술이 돈 것이다(부호·순서·중복 누적 사고가 여기 걸린다).
    ceiling = max(client_norms) * (1.0 + 1e-6)
    if global_norm > ceiling:
        raise ValueError(
            f"집계 결과 norm {global_norm:.6f} 이 클라이언트 최대 {max(client_norms):.6f} 를 "
            "넘는다. 볼록 가중 평균에서는 일어날 수 없다 — 평균이 아닌 산술이 돌았다."
        )

    return AggregationResult(
        ndarrays=out,
        client_norms=client_norms,
        global_norm=global_norm,
        total_examples=total,
        bn_buffer_divergence=bn_divergence(client_arrays, keys),
        missing_variance_ratio=_missing_variance_ratio(client_arrays, num_examples, keys),
    )


def bn_divergence(client_arrays: Sequence[Sequence[np.ndarray]], keys: Sequence[str]) -> float:
    """클라이언트 간 BN running 통계의 평균 거리. 관측 전용 진단.

    강재 클라이언트와 알루미늄 클라이언트의 입력 분포 차이가 이 값으로 드러난다.
    라운드에 걸쳐 커지면 BN 기인 불안정을 의심할 근거가 되지만, 그 판단은 런이 끝난 뒤
    사전 등록한 트리거로 한다.
    """
    idx = [i for i, k in enumerate(keys) if k.endswith("running_mean") or k.endswith("running_var")]
    if not idx or len(client_arrays) < 2:
        return 0.0
    dists: list[float] = []
    for i in idx:
        vecs = [np.asarray(c[i], dtype=np.float64).ravel() for c in client_arrays]
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                dists.append(float(np.linalg.norm(vecs[a] - vecs[b])))
    return float(np.mean(dists)) if dists else 0.0


def _missing_variance_ratio(
    client_arrays: Sequence[Sequence[np.ndarray]],
    num_examples: Sequence[int],
    keys: Sequence[str],
) -> float:
    """가중 평균이 놓치는 분산 성분의 비율. 관측 전용 진단.

    분산의 가중 평균은 진짜 풀링 분산이 아니다 — 클라이언트 간 **평균 차이** 항이 빠진다.
    풀링 분산은 `Σw(σ² + μ²) − (Σwμ)²` 인데 FedAvg 는 `Σwσ²` 만 남기므로, 그 차이가
    전체에서 차지하는 비율이 이 값이다. 0 에 가까우면 BN 평균이 안전하다는 신호이고,
    커지면 클라이언트 분포가 실제로 갈라져 있다는 뜻이다.
    """
    mean_idx = {k: i for i, k in enumerate(keys) if k.endswith("running_mean")}
    var_idx = {k.replace("running_var", "running_mean"): i for i, k in enumerate(keys) if k.endswith("running_var")}
    common = sorted(set(mean_idx) & set(var_idx))
    if not common or len(client_arrays) < 2:
        return 0.0

    total = int(sum(num_examples)) or 1
    w = np.array([n / total for n in num_examples], dtype=np.float64)

    missing, pooled = 0.0, 0.0
    for k in common:
        mi, vi = mean_idx[k], var_idx[k]
        mus = np.stack([np.asarray(c[mi], dtype=np.float64).ravel() for c in client_arrays])
        vars_ = np.stack([np.asarray(c[vi], dtype=np.float64).ravel() for c in client_arrays])
        mu_bar = np.tensordot(w, mus, axes=(0, 0))
        between = float(np.sum(np.tensordot(w, (mus - mu_bar) ** 2, axes=(0, 0))))
        within = float(np.sum(np.tensordot(w, vars_, axes=(0, 0))))
        missing += between
        pooled += within + between
    return float(missing / pooled) if pooled > 0 else 0.0
