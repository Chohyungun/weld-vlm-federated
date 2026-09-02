"""통합형 칸 공통 초기 어댑터 — 동일 출발 증명의 VLM 쪽 구현.

검출 칸에는 `detection/init_weights.py` 의 `initial.npz` + 라운드별 `injection_digest`
로 "세 칸이 같은 가중치에서 출발했다"는 사후 증명이 있었다. **통합형에는 대응물이
아예 없었고**, 그 빈자리에서 ⑦ r0 사고가 났다(74번 감사 C-1). 이 모듈이 그 대칭을
맞춘다 — 같은 형태의 파일 산출물과 같은 형태의 다이제스트를 낸다.

## 왜 시드 고정만으로 끝내지 않는가

`fl.seeding.seeded()` 로 `get_peft_model` 을 감싸면 A 는 결정론적으로 같아진다. 그것이
1차 방어다. 그러나 그것은 **"peft 의 초기화가 호출마다 같은 난수를 같은 순서로
소비한다"에 기대는 증명**이라, peft 버전이 바뀌면 조용히 깨질 수 있고 사후 대조 수단도
없다. 그래서 검출과 같은 2차 방어를 둔다:

1. 시드를 박고 **1회** 만들어 `adapter_initial.npz` 로 떨군다.
2. 모든 클라이언트가 r0 에서 그 파일을 **주입받아** 출발한다.
3. 주입 직후 다이제스트를 회계에 남긴다. 세 클라이언트 값이 같아야 한다.

3번이 있으면 1·2번이 무력화돼도 사후에 드러난다. 검출의 `injection_digest` 와 같은 구조다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from detection import serialize

__all__ = [
    "build_initial_adapter",
    "assert_same_start",
    "assert_injected_matches",
    "adapter_proof",
    "InitProof",
]


class InitProof(dict):
    """초기 어댑터 증빙 한 벌 — `keys_digest`·`tensor_digest`·`l2`.

    dict 를 그대로 쓰는 이유는 회계 CSV·JSON·원자 로그 세 곳에 그대로 실려야 하기
    때문이다. 별도 타입을 만들면 직렬화 지점마다 변환 코드가 붙는다.
    """


def adapter_proof(arrays: list[np.ndarray], keys: list[str]) -> InitProof:
    """어댑터 한 벌에서 증빙을 뽑는다. 주입 전후·클라이언트 간 대조의 단위다."""
    return InitProof(
        keys_digest=serialize.keys_digest(keys),
        tensor_digest=serialize.tensor_digest(arrays),
        l2=serialize.params_l2_norm(arrays),
        n_tensors=len(arrays),
    )


def build_initial_adapter(
    *,
    model_id: str | None = None,
    seed: int,
    cache_path: str | Path | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, torch.Tensor]]:
    """(초기 어댑터 ndarray, 정본 키, 기준 state_dict) 를 돌려준다.

    `detection.init_weights.build_initial_weights` 와 시그니처·반환·캐시 규약을 일부러
    맞췄다. 두 칸의 동일 출발 증명이 같은 모양이어야 감사가 한 번에 읽힌다.

    peft 는 `lora_B` 를 0 으로, `lora_A` 를 난수로 놓는다. 즉 고정해야 하는 것은 A
    하나지만, 저장·주입은 어댑터 전체로 한다 — 부분 주입은 "무엇이 주입됐는가"를
    다시 사람이 판별해야 하는 상태를 만든다.
    """
    from vlm.pilot_vlm import MODEL_ID

    mid = model_id or MODEL_ID

    if cache_path is not None and Path(cache_path).exists():
        # **캐시를 조건 없이 믿지 않는다.** 다른 모델·다른 시드로 만든 파일을 조용히
        # 재사용하면 "동일 출발"이 파일 이름 하나에 걸리게 된다 — 이번 사고와 같은
        # 종류의 침묵이다. 곁의 proof 가 신원을 들고 있으므로 대조한다.
        proof_p = Path(cache_path).with_suffix(".proof.json")
        if proof_p.exists():
            meta = json.loads(proof_p.read_text(encoding="utf-8"))
            if int(meta.get("seed", -1)) != int(seed) or meta.get("model_id") != mid:
                raise RuntimeError(
                    f"초기 어댑터 캐시의 신원이 다르다: 캐시 "
                    f"(model={meta.get('model_id')}, seed={meta.get('seed')}) != "
                    f"요청 (model={mid}, seed={seed}). {cache_path} 를 지우고 다시 만들어라."
                )
        loaded = np.load(cache_path)
        keys = list(loaded.files)
        arrays = [loaded[k] for k in keys]
        return arrays, keys, {}

    from peft import get_peft_model_state_dict

    from vlm.pilot_vlm import _load_model

    model, _ = _load_model(mid, init_seed=seed)
    sd = get_peft_model_state_dict(model)
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)
    ref = {k: v.detach().cpu() for k, v in sd.items()}
    del model
    torch.cuda.empty_cache()

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **{k: a for k, a in zip(keys, arrays)})
        p.with_suffix(".proof.json").write_text(
            json.dumps(
                {"model_id": mid, "seed": int(seed), **adapter_proof(arrays, keys)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return arrays, keys, ref


def assert_injected_matches(sent: list[np.ndarray], keys: list[str],
                            after: InitProof | dict, *, who: str) -> None:
    """G2-5 — 주입이 **서버가 보낸 것과 같은지** 대조한다.

    이전 구조는 클라이언트끼리만 비교했다. 세 클라이언트 모두에서 주입이 no-op 이면
    셋의 증빙이 똑같으므로 그대로 통과한다(80번 C4 ③). 기준은 옆 클라이언트가 아니라
    **서버가 보낸 페이로드**여야 한다.

    `set_peft_model_state_dict` 는 내부적으로 `strict=False` 라 키가 안 맞아도 조용히
    넘어간다. 그 반환값을 믿는 대신 주입 후 상태를 다시 읽어 여기서 대조한다.
    """
    want = adapter_proof(sent, keys)
    bad = []
    if want["keys_digest"] != after.get("keys_digest"):
        bad.append(f"keys_digest {after.get('keys_digest', '')[:12]} != 서버 {want['keys_digest'][:12]}")
    if round(want["l2"], 9) != round(float(after.get("l2", -1.0)), 9):
        bad.append(f"l2 {after.get('l2')} != 서버 {want['l2']}")
    if [round(x, 9) for x in want["tensor_digest"]] != \
       [round(x, 9) for x in after.get("tensor_digest", [])]:
        bad.append(f"tensor_digest {after.get('tensor_digest')} != 서버 {want['tensor_digest']}")
    if bad:
        raise RuntimeError(
            f"{who}: 주입이 서버 페이로드와 다르다 — set_peft_model_state_dict 가 "
            "조용히 무시했을 수 있다(strict=False):\n  " + "\n  ".join(bad)
        )


def assert_same_start(proofs: dict[int, InitProof | dict]) -> None:
    """클라이언트별 r0 초기 어댑터 증빙이 전부 같은지 검사한다.

    **런타임 가드다. 시험이 아니라 실행 경로에 건다.** 74번 C-1 은 시험이 없어서 난
    사고가 아니라, 사고가 나도 산출물에 아무 흔적이 남지 않아서 10시간을 다 쓴 뒤에야
    드러난 사고다. r0 집계 직전에 여기서 죽는 편이 낫다.
    """
    if len(proofs) < 2:
        return
    items = sorted(proofs.items())
    ref_c, ref = items[0]
    bad: list[str] = []
    for c, p in items[1:]:
        if p.get("keys_digest") != ref.get("keys_digest"):
            bad.append(f"c{c}: keys_digest {p.get('keys_digest', '')[:12]} != "
                       f"c{ref_c} {ref.get('keys_digest', '')[:12]}")
        if [round(x, 9) for x in p.get("tensor_digest", [])] != \
           [round(x, 9) for x in ref.get("tensor_digest", [])]:
            bad.append(f"c{c}: tensor_digest {p.get('tensor_digest')} != "
                       f"c{ref_c} {ref.get('tensor_digest')}")
        # G2-4 — 계산해 두고 비교하지 않던 값(80번 F14). 한 줄에 전 텐서를 덮는다.
        if round(float(p.get("l2", -1.0)), 9) != round(float(ref.get("l2", -2.0)), 9):
            bad.append(f"c{c}: l2 {p.get('l2')} != c{ref_c} {ref.get('l2')}")
        if int(p.get("n_tensors", -1)) != int(ref.get("n_tensors", -2)):
            bad.append(f"c{c}: n_tensors {p.get('n_tensors')} != c{ref_c} {ref.get('n_tensors')}")
    if bad:
        raise RuntimeError(
            "r0 초기 어댑터가 클라이언트마다 다르다 — 함정 #3(독립 난수 상쇄) 재발이다:\n  "
            + "\n  ".join(bad)
        )
