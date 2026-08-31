"""검출 칸 공통 초기 가중치 — 동일 출발 증명의 구현.

다섯 칸 비교가 성립하려면 세 검출 칸이 **같은 가중치에서 출발**해야 한다. Ultralytics 는
run 마다 사전학습 가중치를 부분 로드하고 헤드(nc 불일치 층)를 난수 초기화하는데, 이 난수가
run 마다 다르면 "같은 출발점" 주장이 깨진다.

여기서는 stock 과 같은 구성 경로(DetectionModel + 부분 로드)를 **시드를 박고 1회** 수행해
state_dict 를 뽑는다. 로컬·중앙·연합 전 run 이 이 ndarray 를 주입받아 출발하고, 주입 증빙
다이제스트로 사후 대조한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from detection import serialize

__all__ = ["build_initial_weights"]


def build_initial_weights(
    *,
    pretrained: str = "yolo11n.pt",
    nc: int,
    seed: int,
    cache_path: str | Path | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, torch.Tensor]]:
    """(초기 ndarray, 정본 키, 기준 state_dict) 를 돌려준다.

    stock 트레이너의 `get_model` 과 같은 경로다: 모델 yaml 로 nc 에 맞는 구조를 만들고
    사전학습 가중치를 교집합만 로드한다. 헤드 난수 초기화가 `seed` 로 고정되므로
    같은 (pretrained, nc, seed) 는 항상 같은 가중치를 낸다.

    cache_path 를 주면 npz 로 저장·재사용한다. 같은 run 의 서버·클라이언트가 각자
    이 함수를 불러도 결과가 같지만, 캐시를 쓰면 대조가 파일 해시 하나로 끝난다.
    """
    if cache_path is not None and Path(cache_path).exists():
        loaded = np.load(cache_path)
        arrays = [loaded[k] for k in loaded.files]
        model = _build(pretrained, nc, seed)  # 키·기준 dtype 은 구조에서 온다
        ref = model.state_dict()
        keys = serialize.canonical_keys(ref)
        serialize.assert_compatible(arrays, keys, ref)
        return arrays, keys, ref

    model = _build(pretrained, nc, seed)
    ref = model.state_dict()
    keys = serialize.canonical_keys(ref)
    arrays = serialize.state_dict_to_ndarrays(ref, keys)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **{k: a for k, a in zip(keys, arrays)})
    return arrays, keys, ref


def _build(pretrained: str, nc: int, seed: int):
    # ultralytics 8.4 는 attempt_load_one_weight 가 없고 load_checkpoint 를 쓴다.
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    torch.manual_seed(seed)  # 헤드 난수 초기화 고정 — 동일 출발 증명의 핵심
    weights, _ = load_checkpoint(pretrained)
    cfg = weights.yaml if hasattr(weights, "yaml") else pretrained.replace(".pt", ".yaml")
    model = DetectionModel(cfg, nc=nc, verbose=False)
    model.load(weights)
    return model
