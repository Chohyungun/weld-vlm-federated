"""모델 생성기의 실제 구현 — HF 적재 · 학습 어댑터 주입 · greedy 1회(리허설 2판 §2 · 07번 §13-3).

| 단계 | 무엇 |
|---|---|
| 적재 | 학습과 같은 적재기(`vlm.pilot_vlm._load_model` — 4bit nf4 · bf16 계산 · 비병합 LoRA, 같은 모델 판 · 프로세서 인자) |
| 어댑터 | 학습이 낸 `adapter_last.npz` 의 키 순서 그대로 peft 상태로 주입하고, 주입 뒤 상태가 파일과 같은지 다시 본다 |
| 입력 | 학습과 같은 렌더 — 사용자 차례에 이미지 · 프롬프트, 생성 접두(`add_generation_prompt=True`), 같은 템플릿 인자 |
| 생성 | `do_sample=False` · `num_beams=1` · 배치 1 · 등록된 채움 방향 · `max_new_tokens` 는 생성 설정의 한도. 다른 디코딩은 받지 않는다 |
| 지연 | `model.generate` 호출 구간만(CUDA 면 앞뒤로 동기화) — 파일럿 export 와 같다 |
| 멈춤 | 마지막 새 토큰이 종료 토큰이면 `eos`(한도에 닿는 자리여도), 아니고 한도에 닿았으면 `length`. 둘 다 아니면 **`stop_undetermined` 를 적고 계속한다**(2판 §2-4 · 계약 §16-2 m-2 — 평가 쪽 검증기가 그 묶음을 거부한다) |
| 모델 적재기 | `resolve_model_loader` 로 **부르기 전에** 목적과 맞댄다 — 본실험은 대역 적재기(인자로 오든 모듈 속성을 바꿔 끼웠든)를 받지 않는다 |

생성문의 파싱 · 역변환은 export 가 한다(`vlm.gen_parse` — 한 함수). **실물 검증은 GPU 창의 몫이다** — 합성 시험은 같은 꼴의 대역 모델로 돈다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

__all__ = ["GREEDY", "load_model_generator"]

GREEDY = {"do_sample": False, "num_beams": 1}


def _eos_ids(model: Any, tok: Any) -> set[int]:
    out: set[int] = set()
    for src in (getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                getattr(tok, "eos_token_id", None)):
        if isinstance(src, int):
            out.add(src)
        elif isinstance(src, (list, tuple)):
            out.update(int(x) for x in src)
    if not out:
        raise RuntimeError("종료 토큰을 모델 · 토크나이저에서 찾지 못했다 — 멈춤을 정할 수 없다")
    return out


def _inject(model: Any, adapter_path: Any) -> list[str]:
    """`adapter_last.npz` 를 peft 상태로 주입하고 주입이 먹었는지 다시 본다. 돌려주는 값은 키다."""
    import numpy as np
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    from detection import serialize

    with np.load(adapter_path) as z:
        keys = list(z.files)
        arrays = [z[k] for k in keys]
    ref = get_peft_model_state_dict(model)
    have = serialize.canonical_keys(ref)
    if sorted(have) != sorted(keys):
        raise RuntimeError(f"어댑터 파일의 키가 모델의 어댑터 키와 다르다 — 파일 {len(keys)} · 모델 {len(have)}")
    set_peft_model_state_dict(model, serialize.ndarrays_to_state_dict(arrays, keys, ref))
    after = serialize.state_dict_to_ndarrays(get_peft_model_state_dict(model), keys)
    for k, a, b in zip(keys, arrays, after):
        if a.shape != b.shape or not np.array_equal(a.astype(b.dtype), b):
            raise RuntimeError(f"어댑터 주입이 먹지 않았다: {k}")
    return keys


def load_model_generator(gen_cfg: Any, *, model_loader: Callable | None = None):
    """`GenerationConfig` → 생성기 `(이미지, image_id) → GenOut`. `model_loader` 는 시험의 대역 자리다(파이썬 인자로만)."""
    import torch

    from vlm.export_run import GenOut
    from vlm.pilot_vlm import _template_kwargs, resolve_model_loader

    if gen_cfg.mode != "model":
        raise ValueError(f"모델 생성기는 모델 모드에서만 연다: {gen_cfg.mode!r}")
    if dict(gen_cfg.decoding or {}) != GREEDY:
        raise ValueError(f"greedy 1회({GREEDY})가 아닌 디코딩은 받지 않는다: {gen_cfg.decoding!r}")
    if gen_cfg.batch_size != 1:
        raise ValueError(f"생성 배치는 1 이다: {gen_cfg.batch_size!r}")
    if not gen_cfg.adapter_path:
        raise ValueError("모델 모드에 어댑터 경로가 없다")
    # 모델 적재기의 승인 — 부르기 전에(2판 §1-4). 진단의 커밋된 영수증은 export 가 이음새 단계에서 먼저 막는다.
    loader, loader_ids = resolve_model_loader(model_loader, purpose=getattr(gen_cfg, "purpose", "main"),
                                              standin_allowed=True)
    model, proc = loader(gen_cfg.model_id, init_seed=0, revision=gen_cfg.model_revision,
                         processor_kwargs=dict(gen_cfg.processor_kwargs or {}))
    _inject(model, gen_cfg.adapter_path)
    model.eval()
    tok = proc.tokenizer
    tok.padding_side = str(gen_cfg.padding_side)
    eos = _eos_ids(model, tok)
    kwargs = _template_kwargs(gen_cfg.chat_template_kwargs)
    patch = int(proc.image_processor.patch_size)
    limit = int(gen_cfg.max_new_tokens)
    dev = next(model.parameters()).device

    def generate(img, image_id: str) -> GenOut:
        user = {"role": "user", "content": [{"type": "image", "image": img.convert("RGB")},
                                            {"type": "text", "text": gen_cfg.prompt_text}]}
        enc = proc.apply_chat_template([user], tokenize=True, return_dict=True, return_tensors="pt",
                                       add_generation_prompt=True, **kwargs)
        enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in dict(enc).items()}
        n_in = int(enc["input_ids"].shape[1])
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, **GREEDY, max_new_tokens=limit, return_dict_in_generate=True,
                                 output_scores=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) * 1000.0
        new = [int(t) for t in out.sequences[0, n_in:].tolist()]
        if new and new[-1] in eos:
            stop = "eos"
        elif len(new) >= limit:
            stop = "length"
        else:
            stop = "stop_undetermined"           # 기록하고 계속한다 — 거부는 평가 쪽 검증기의 몫이다(STOP_UNDETERMINED)
        logprobs = [float(torch.log_softmax(s[0].float(), dim=-1)[t]) for s, t in zip(out.scores, new)]
        grid = enc.get("image_grid_thw")
        t, h, w = (int(v) for v in grid[0].tolist())
        return GenOut(text=tok.decode(new, skip_special_tokens=True), gen_stop=stop, n_new_tokens=len(new),
                      model_input_wh=(w * patch, h * patch), grid_thw=(t, h, w), token_ids=new,
                      token_logprobs=logprobs, latency_ms=ms)

    generate.impl_ids = {"model_loader": loader_ids[0]}     # 실제로 부른 적재기 — export 가 이음새 단계의 기록과 맞댄다
    return generate
