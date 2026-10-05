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
    "read_initial_cache",
    "init_adapter_digest",
    "InitCacheRejected",
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


class InitCacheRejected(ValueError):
    """초기 어댑터 캐시를 이 목적에서 받을 수 없다. 캐시를 읽기만 하고 고치지 않는다."""


def init_adapter_digest(arrays: list[np.ndarray]) -> str:
    """정본 키 순서로 배열 바이트를 이은 sha256 — 학습 설정 원문 · 등록의 `init_adapter_digest`.

    `adapter_proof` 의 `tensor_digest`(표집한 노름 목록)와 다른 양이다. 둘 다 meta 에 싣는다
    (리허설 2판 §2-7 의 `init_adapter_digest` · `init_adapter_tensor_digest`).
    """
    import hashlib

    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


#: proof 에 실린 구현 식별자가 가져야 하는 필드(`vlm/seams.impl_id`). 본실험은 비어 있으면 받지 않는다.
IMPL_ID_REQUIRED = ("seam", "module", "qualname", "source_path", "blob_sha1")


def _check_proof_binding(proof: dict, arrays: list[np.ndarray], keys: list[str], *, required: bool,
                         where: str) -> None:
    """proof 의 세 증빙(`init_adapter_digest` · 키 · 텐서)을 읽은 배열에서 다시 내 맞댄다."""
    got = adapter_proof(arrays, keys)
    want = {"init_adapter_digest": init_adapter_digest(arrays), "keys_digest": got["keys_digest"],
            "tensor_digest": got["tensor_digest"], "n_tensors": got["n_tensors"]}
    for k, v in want.items():
        if k not in proof:
            if required:
                raise InitCacheRejected(f"본실험은 {k} 가 없는 proof 를 받지 않는다: {where}")
            continue
        if json.loads(json.dumps(proof[k])) != json.loads(json.dumps(v)):
            raise InitCacheRejected(f"초기 어댑터 캐시의 배열이 proof 의 {k} 와 다르다 — 파일이 바뀌었다: {where}")


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
    revision: str | None = None,
    purpose: str = "main",
    model_loader=None,
    standin_allowed: bool = False,
) -> tuple[list[np.ndarray], list[str], dict[str, torch.Tensor]]:
    """(초기 어댑터 ndarray, 정본 키, 기준 state_dict) 를 돌려준다.

    `detection.init_weights.build_initial_weights` 와 시그니처·반환·캐시 규약을 일부러
    맞췄다. 두 칸의 동일 출발 증명이 같은 모양이어야 감사가 한 번에 읽힌다.

    peft 는 `lora_B` 를 0 으로, `lora_A` 를 난수로 놓는다. 즉 고정해야 하는 것은 A
    하나지만, 저장·주입은 어댑터 전체로 한다 — 부분 주입은 "무엇이 주입됐는가"를
    다시 사람이 판별해야 하는 상태를 만든다.

    `revision` 은 모델 판이다. proof 에 싣고 캐시 대조에 넣는다 — 이름이 같고 판이 다른 모델로 만든
    캐시를 받지 않는다. 판을 적지 않던 옛 proof 는 `None` 으로 읽힌다.

    `purpose` · `model_loader` 는 모델 적재 이음새다(리허설 2판 §1-4). 적재기는 **캐시를 보기 전에**
    목적과 맞댄다. proof 에 목적과 구현 식별자를 싣고, **본실험은 proof 가 없거나 목적이 `main` 이
    아니거나 승인되지 않은 구현으로 만든 캐시를 받지 않는다**(X-16). 옛 proof 는 목적이 없어 본실험에서
    거부된다 — 파일럿 캐시를 본실험이 조용히 물려받지 않게 한다. 캐시 파일은 고치지 않는다.
    """
    from vlm.pilot_vlm import MODEL_ID, resolve_model_loader

    mid = model_id or MODEL_ID
    loader, impl_ids = resolve_model_loader(model_loader, purpose=purpose,
                                            standin_allowed=standin_allowed)

    if cache_path is not None and Path(cache_path).exists():
        arrays, keys = read_initial_cache(cache_path, model_id=mid, seed=seed, revision=revision, purpose=purpose)
        return arrays, keys, {}

    from peft import get_peft_model_state_dict

    model, _ = loader(mid, init_seed=seed, revision=revision)
    sd = get_peft_model_state_dict(model)
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)
    ref = {k: v.detach().cpu() for k, v in sd.items()}
    del model
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()

    if cache_path is not None:
        # proof 먼저, 배열(완료 표지)을 마지막에 — 둘 다 tmp → fsync → 교체다. 도중에 죽으면 배열이 없어 다음 호출이
        # 다시 만든다(찢긴 배열 · 찢긴 proof 가 캐시로 읽히지 않는다). 다섯 칸이 이 파일 하나에서 출발한다.
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        proof = json.dumps(
            {"model_id": mid, "revision": revision, "seed": int(seed), "purpose": purpose,
             "impl_ids": impl_ids, "init_adapter_digest": init_adapter_digest(arrays),
             **adapter_proof(arrays, keys)},
            ensure_ascii=False, indent=2).encode("utf-8")
        _durable_replace(p.with_suffix(".proof.json"), lambda fh: fh.write(proof))
        _durable_replace(p, lambda fh: np.savez(fh, **{k: a for k, a in zip(keys, arrays)}))
    return arrays, keys, ref


def _durable_replace(path: Path, write) -> None:
    """`write(fh)` 로 tmp 에 쓰고 fsync 한 뒤 `os.replace` 로 제 이름을 준다."""
    import os

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_initial_cache(cache_path: str | Path, *, model_id: str, seed: int, revision: str | None,
                       purpose: str) -> tuple[list[np.ndarray], list[str]]:
    """공통 초기 어댑터 캐시를 **읽기만** 한다 — 모델을 올리지 않는다. 규칙은 `build_initial_adapter` 의 캐시 분기와 같다.

    실행기가 연합 칸의 앞 산출물을 되짚을 때(초기 어댑터 digest 의 대조) 같은 규칙으로 읽으려고 따로 꺼냈다.
    """
    # **캐시를 조건 없이 믿지 않는다.** 다른 모델·다른 시드로 만든 파일을 조용히
    # 재사용하면 "동일 출발"이 파일 이름 하나에 걸리게 된다 — 이번 사고와 같은
    # 종류의 침묵이다. 곁의 proof 가 신원을 들고 있으므로 대조한다.
    proof_p = Path(cache_path).with_suffix(".proof.json")
    mid = model_id
    if purpose == "main":
        if not proof_p.exists():
            raise InitCacheRejected(
                f"본실험은 proof 없는 초기 어댑터 캐시를 받지 않는다: {cache_path}")
        pm = json.loads(proof_p.read_text(encoding="utf-8"))
        ids = pm.get("impl_ids")
        if pm.get("purpose") != "main" or not ids or not all(i.get("approved") for i in ids):
            raise InitCacheRejected(
                f"본실험이 받을 수 없는 초기 어댑터 캐시다(목적 {pm.get('purpose')!r}, "
                f"승인 구현 {[i.get('approved') for i in ids or []]}): {cache_path}. "
                "본실험 경로에서 새로 만든다.")
        bad_ids = [i for i in ids if any(not i.get(k) for k in IMPL_ID_REQUIRED)]
        if bad_ids:
            raise InitCacheRejected(
                f"proof 의 구현 식별자에 빈 필드가 있다(필수 {list(IMPL_ID_REQUIRED)}): {cache_path}")
    if proof_p.exists():
        meta = json.loads(proof_p.read_text(encoding="utf-8"))
        if (int(meta.get("seed", -1)) != int(seed) or meta.get("model_id") != mid
                or meta.get("revision") != revision):
            raise RuntimeError(
                f"초기 어댑터 캐시의 신원이 다르다: 캐시 "
                f"(model={meta.get('model_id')}, revision={meta.get('revision')}, seed={meta.get('seed')}) != "
                f"요청 (model={mid}, revision={revision}, seed={seed}). {cache_path} 를 지우고 다시 만들어라."
            )
    with np.load(cache_path) as loaded:
        keys = list(loaded.files)
        arrays = [loaded[k] for k in keys]
    if proof_p.exists():
        # proof 를 **읽은 배열과** 결속한다 — 배열 파일만 바뀌면 옛 proof 가 그대로 통과해 새 배열이 공통
        # 출발점이 된다. 본실험은 세 증빙이 모두 있어야 하고, 다른 목적은 있는 증빙을 모두 맞댄다.
        _check_proof_binding(json.loads(proof_p.read_text(encoding="utf-8")), arrays, keys,
                             required=(purpose == "main"), where=str(cache_path))
    return arrays, keys


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
