"""검출 모델 가중치 교환 규약 — 직렬화·복원·무결성 검사.

연합학습에서 서버와 클라이언트가 주고받는 것은 `state_dict`가 아니라 **순서가 고정된
ndarray 리스트**다. 순서가 한 번이라도 어긋나면 엉뚱한 텐서끼리 평균되는데, 그렇게
망가진 모델도 학습은 계속 돌아가고 지표만 조용히 나빠진다. 그래서 이 모듈의 실제 목적은
직렬화가 아니라 **어긋남을 즉시 실패로 만드는 것**이다.

교환 규약(스펙 §4-1 확정)
  - 부동소수 텐서는 **fp32로 승격**해 보낸다. fp16 왕복은 라운드마다 집계 정밀도를 깎는다.
  - 정수 버퍼(`num_batches_tracked` 등)는 **원 dtype을 유지**한다. fp32로 바꾸면 큰 값에서
    정밀도를 잃고, 되돌릴 때 반올림이 끼어든다.
  - EMA 가중치가 아니라 **raw 가중치**를 보낸다. 클라이언트마다 다른 평활 상태를 평균하면
    다음 라운드의 출발점이 어떤 단일 학습 궤적과도 대응하지 않는다.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "SerializeError",
    "canonical_keys",
    "keys_digest",
    "state_dict_to_ndarrays",
    "ndarrays_to_state_dict",
    "assert_compatible",
    "params_l2_norm",
    "payload_nbytes",
    "tensor_digest",
]


class SerializeError(RuntimeError):
    """교환 규약 위반. 복구를 시도하지 않고 즉시 실패한다."""


def canonical_keys(state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    """정본 키 리스트. `state_dict`의 삽입 순서를 그대로 쓴다.

    정렬하지 않는 이유는 PyTorch의 `state_dict` 순서가 모듈 등록 순서로 결정론적이고,
    정렬하면 오히려 원본 구조와 대응이 끊겨 사람이 눈으로 대조하기 어려워지기 때문이다.
    순서가 바뀌었는지는 `keys_digest`로 검사한다.
    """
    return list(state_dict.keys())


def keys_digest(keys: Sequence[str]) -> str:
    """키 리스트(순서 포함)의 sha256. 실험 시작 시 1회 고정해 configs에 기록한다."""
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def state_dict_to_ndarrays(
    state_dict: Mapping[str, torch.Tensor], keys: Sequence[str]
) -> list[np.ndarray]:
    """`state_dict`를 정본 키 순서의 ndarray 리스트로 변환한다."""
    missing = [k for k in keys if k not in state_dict]
    if missing:
        raise SerializeError(f"정본 키가 state_dict에 없다: {missing[:5]} (총 {len(missing)}개)")
    extra = [k for k in state_dict if k not in set(keys)]
    if extra:
        raise SerializeError(f"정본 키에 없는 항목이 state_dict에 있다: {extra[:5]} (총 {len(extra)}개)")

    out: list[np.ndarray] = []
    for k in keys:
        t = state_dict[k].detach().cpu()
        # 부동소수는 fp32 승격, 정수·불리언은 원 dtype 유지
        arr = t.to(torch.float32).numpy() if t.is_floating_point() else t.numpy()
        # `np.ascontiguousarray`를 쓰지 않는다 — 0차원 텐서를 (1,)로 만들어 버린다.
        # BatchNorm 의 `num_batches_tracked`가 정확히 0차원이라 복원 시 shape 불일치로 터진다.
        out.append(np.array(arr, copy=True, order="C"))
    return out


def ndarrays_to_state_dict(
    arrays: Sequence[np.ndarray],
    keys: Sequence[str],
    reference: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """ndarray 리스트를 `state_dict`로 복원한다. dtype·shape는 `reference`를 따른다.

    `reference`는 주입 대상 모델의 현재 `state_dict`다. 여기서 dtype을 되찾기 때문에
    fp32로 오간 값이 원래 정밀도로 돌아간다.
    """
    if len(arrays) != len(keys):
        raise SerializeError(f"배열 수와 키 수가 다르다: {len(arrays)} != {len(keys)}")

    sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, arr in zip(keys, arrays):
        ref = reference.get(k)
        if ref is None:
            raise SerializeError(f"복원 대상 모델에 없는 키다: {k}")
        if tuple(arr.shape) != tuple(ref.shape):
            raise SerializeError(f"shape 불일치 [{k}]: 수신 {tuple(arr.shape)} != 기대 {tuple(ref.shape)}")
        sd[k] = torch.as_tensor(arr).to(dtype=ref.dtype)
    return sd


def assert_compatible(
    arrays: Sequence[np.ndarray], keys: Sequence[str], reference: Mapping[str, torch.Tensor]
) -> None:
    """집계 직전·직후에 거는 무결성 검사 3종 — 키·shape·dtype.

    조용한 오염을 잡는 저비용 안전장치다. 위반은 경고가 아니라 예외로 올린다.

    **dtype 은 계열이 아니라 정확 일치로 본다** (80번 F3 / G2-1). 이전 판은
    "부동소수 계열이 서로 맞는가"만 봐서 fp16 페이로드가 그대로 통과했다 — 실측에서
    글로벌 최대 편차 5.44e-04 가 났고 아무 검사도 발화하지 않았다. 교환 규약이
    "부동소수는 fp32" 라고 못박고 있으므로 그대로 검사한다.
    """
    if len(arrays) != len(keys):
        raise SerializeError(f"배열 수 불일치: {len(arrays)} != {len(keys)}")
    for k, arr in zip(keys, arrays):
        ref = reference.get(k)
        if ref is None:
            raise SerializeError(f"기준 모델에 없는 키: {k}")
        if tuple(arr.shape) != tuple(ref.shape):
            raise SerializeError(f"shape 불일치 [{k}]: {tuple(arr.shape)} != {tuple(ref.shape)}")
        expect_float = ref.is_floating_point()
        got_float = np.issubdtype(arr.dtype, np.floating)
        if expect_float != got_float:
            raise SerializeError(
                f"dtype 계열 불일치 [{k}]: 수신 {arr.dtype}, 기준 {ref.dtype}. "
                "부동소수는 fp32, 정수 버퍼는 원 dtype이어야 한다."
            )
        if expect_float:
            # 계열이 아니라 정확 일치. fp16 이 통과하던 구멍을 막는다.
            if arr.dtype != np.float32:
                raise SerializeError(
                    f"부동소수 dtype 불일치 [{k}]: 수신 {arr.dtype}, 규약 float32. "
                    "fp16 페이로드는 글로벌 최대 편차 5.44e-04 를 남기고 조용히 통과했다."
                )
        else:
            ref_np = torch.empty(0, dtype=ref.dtype).numpy().dtype
            if arr.dtype != ref_np:
                raise SerializeError(
                    f"정수 dtype 불일치 [{k}]: 수신 {arr.dtype}, 기준 {ref_np}. "
                    "정수 버퍼는 원 dtype 그대로여야 한다(카운터를 평균하지 않는다)."
                )


def params_l2_norm(arrays: Sequence[np.ndarray]) -> float:
    """부동소수 파라미터 전체의 L2 norm.

    집계 직전 클라이언트별로, 집계 직후 글로벌에 대해 기록한다. 순서 뒤섞임이나 dtype
    사고가 나면 이 값이 눈에 띄게 튀므로, 라운드 곡선으로 보면 사고를 조기에 잡는다.
    정수 버퍼는 학습된 값이 아니라 카운터라 제외한다.
    """
    total = 0.0
    for arr in arrays:
        if np.issubdtype(arr.dtype, np.floating):
            a = arr.astype(np.float64, copy=False)
            total += float(np.dot(a.ravel(), a.ravel()))
    return float(np.sqrt(total))


def payload_nbytes(arrays: Sequence[np.ndarray]) -> int:
    """직렬화 페이로드 크기(바이트). 통신량 결과표의 실측 입력이다."""
    return int(sum(a.nbytes for a in arrays))


def tensor_digest(arrays: Sequence[np.ndarray], sample: int = 5) -> list[float]:
    """대표 텐서의 L2 norm + **전 텐서 합산 norm**. 주입 증빙이다.

    서버가 보낸 값과 클라이언트가 주입 후 계산한 값이 같아야 한다. 다르면 주입이
    조용히 무력화된 것이다 — Ultralytics는 조건에 따라 사전학습 가중치를 덮어쓸 수 있다.

    ## 표집이 한쪽 눈을 감고 있었다 (80번 C4 / G2-4)

    이전 판은 `step = len(floats) // sample` 로 등간격 표집했다. LoRA 어댑터는
    `lora_A`·`lora_B` 가 **교대로** 배열되는데 372 텐서에서 step 이 74(짝수)라 뽑힌 5개가
    **전부 `lora_A`** 였다. `lora_B` 만 누락되는 부분 주입은 원리적으로 안 잡혔다.

    두 가지로 고친다.
    1. step 을 홀수로 강제해 교대 배열에서 A·B 가 함께 뽑히게 한다.
    2. 마지막 원소로 **전 텐서 L2 합**을 붙인다. 표집이 무엇을 놓치든 전체가 바뀌면
       이 값이 움직인다. 비용은 표집과 같은 자릿수다.
    """
    floats = [a for a in arrays if np.issubdtype(a.dtype, np.floating)]
    if not floats:
        return []
    step = max(1, len(floats) // sample)
    if step % 2 == 0 and len(floats) > step:
        step += 1                      # 교대 배열에서 한 계열만 뽑히는 것을 막는다
    picked = floats[::step][:sample]
    digest = [float(np.linalg.norm(a.astype(np.float64, copy=False))) for a in picked]
    digest.append(params_l2_norm(floats))     # 전 텐서 — 표집이 놓쳐도 여기서 잡힌다
    return digest
