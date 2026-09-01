"""⑥⑦ 통합형 파일럿 학습 — Qwen3.5-0.8B QLoRA 4bit.

파일럿 전용 배선이다. 함정 겨냥 규약은 그대로 지킨다:

- **좌표는 `vlm/coords.py` 만 통과한다** (함정 #4). 타깃 bbox 는 원본 픽셀 →
  `to_model`(NORM_1000) → `quantize` 로 만들고, 모델 좌표는 어떤 파일에도 저장하지 않는다.
- **어댑터 교환은 fp32 행렬별 가중 평균** (함정 #3). 집계는 검출과 같은
  `fl.aggregate.weighted_fedavg` 를 쓴다 — LoRA A·B 도 float 텐서라 같은 산술이다.
- **교환 폐포** (30번 명세 G2-3): 학습되는 파라미터 집합과 교환 페이로드 키 집합의
  완전 일치를 라운드 1에서 검사한다.
- 조기 종료 없음 · 손실 정규화는 감독 토큰 총합 분모(30번 명세 판정 2) ·
  LoRA dropout 0(판정 9) · micro=1/accum=32(판정 10) · 프롬프트 파일 고정.

타깃은 30번 명세 4-1 의 단일 JSON 이다. 코퍼스 담당의 `target_text`(산문 판정문)는
파일럿 학습 타깃에 쓰지 않는다 — 명세가 타깃 형식을 JSON 하나로 확정했고, clause_only
축에서 verdict 는 "판정불가"가 스키마 정합값이다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from detection import serialize
from detection.round_runner import derive_seed
from vlm.coords import CoordCfg, ImageGeom, quantize, to_model

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT_PATH = Path("vlm/prompts/unified_pilot_v1.txt")
PAIRS_PATH = Path("data/processed/pairs_pilot_v1/pairs.jsonl")
COORD_CFG = CoordCfg(coord_space="NORM_1000")
MICRO_BATCH = 1          # 판정 10 — micro=2 는 사다리에 없다
GRAD_ACCUM = 32          # 유효 배치 32
LR = 1e-4
#: 언어부 linear 접미사 — 실물 named_modules 덤프에서 확정(2026-08-31 프로브).
#: SSM/linear-attention 프로젝션(in_proj_*, out_proj)을 포함한다. 'all-linear' 금지.
TARGET_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
]


def build_target(row: dict, geom: ImageGeom) -> str:
    """스켈레톤 → 학습 타깃 JSON. 좌표는 coords 모듈만 통과한다."""
    defects = []
    for d in row["skeleton"]["defects"]:
        box = quantize(to_model(d["bbox_px"], geom, COORD_CFG))
        defects.append({"iso_code": str(d["type"]), "bbox_2d": list(box)})
    verdict = row["skeleton"].get("verdict") or "판정불가"   # clause_only 축의 정합값
    clauses = row["skeleton"].get("clauses") or []
    return json.dumps(
        {"defects": defects, "verdict": verdict, "cited_clauses": clauses},
        ensure_ascii=False, separators=(",", ":"),
    )


def load_pairs(split: str, client: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(PAIRS_PATH, encoding="utf-8")]
    out = [r for r in rows if r["split"] == split and (client is None or r["client"] == client)]
    if not out:
        raise ValueError(f"페어 0건: split={split} client={client}")
    return out


def _load_model(model_id: str | None = None):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    mid = model_id or MODEL_ID
    proc = AutoProcessor.from_pretrained(mid)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        mid, quantization_config=bnb, device_map={"": 0}
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_SUFFIXES,
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    # 비전 어댑터 0건 확인 — 붙었다면 동결 원칙 위반이므로 즉시 실패
    vis = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if vis:
        raise RuntimeError(f"비전 인코더에 어댑터가 붙었다: {vis[:3]}")
    return model, proc


class _StepView:
    """`ResumeCheckpointer` 가 읽는 스텝 카운터 모양."""

    def __init__(self, n: int) -> None:
        self.n = int(n)


class _AdapterTrainerView:
    """통합형 학습 루프를 재개 체크포인터에 물리는 어댑터.

    체크포인터는 트레이너 모양(`model`·`optimizer`·`epoch`·`start_epoch`·`device`)을
    기대한다. 통합형은 자체 루프라 그 모양이 없다. 루프를 트레이너로 바꾸는 대신
    **필요한 다섯 개만 노출하는 얇은 뷰**를 둔다 — 재개 하나 때문에 학습 루프를
    프레임워크 모양으로 접을 이유가 없다.

    `start_epoch=0` 으로 고정하는 이유: 통합형은 라운드가 곧 전체 epoch 구간이라
    저장되는 `epochs_ran_in_round` 가 그대로 누적 epoch 수가 된다.
    """

    def __init__(self, model, optimizer, epoch: int = 0) -> None:
        self.model = model
        self.optimizer = optimizer
        self.epoch = int(epoch)
        self.start_epoch = 0
        self.scaler = None          # bf16 autocast 라 GradScaler 를 쓰지 않는다
        self.train_loader = None    # 로더가 없다 — 셔플은 `random.Random(seed+ep)` 다
        self.device = torch.device("cuda")


def _encode(proc, row: dict, prompt: str):
    from PIL import Image

    img = Image.open(row["image_path"]).convert("RGB")
    geom = ImageGeom(orig_w=img.size[0], orig_h=img.size[1])
    target = build_target(row, geom)
    user = {"role": "user", "content": [{"type": "image", "image": img},
                                        {"type": "text", "text": prompt}]}
    full = proc.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": target}]}],
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    prompt_only = proc.apply_chat_template(
        [user], tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    labels = full["input_ids"].clone()
    prompt_len = int(prompt_only["input_ids"].shape[1])
    labels[:, :prompt_len] = -100   # 감독은 타깃 토큰만
    return full, labels, prompt_len


def train_rounds(
    *,
    rows: list[dict],
    epochs: int,
    round_idx: int,
    client_idx: int,
    base_seed: int,
    adapter_in: list[np.ndarray] | None = None,
    adapter_keys: list[str] | None = None,
    log_cb=None,
    model_id: str | None = None,
    supervised_logits_only: bool = True,
    resume_dir: str | None = None,
    run_id: str = "",
) -> tuple[list[np.ndarray], list[str], dict[str, Any], dict]:
    """한 라운드(⑥은 라운드 1개 = 전체 epoch). 어댑터 fp32 ndarray 를 돌려준다.

    Args:
        model_id: 크기-시간 곡선 프로브용 덮어쓰기. 본실험은 항상 기본값을 쓴다.
        supervised_logits_only: 판정 11 이행 스위치. `False` 는 **이행 전후 비교를
            재기 위해서만** 쓴다 — 전 위치 × vocab 로짓을 물질화한다.
        resume_dir: 재개 전용 체크포인트 디렉터리(어댑터·옵티마이저·epoch·RNG).
            ⑥ 은 파일럿에서도 10.2시간짜리 단일 런이고 본실험은 칸당 수 주다.
            `best` 금지 규칙과 무관하다 — 채점 대상이 아니라 재개용이다.
        run_id: 재개 신원의 일부.
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    model, proc = _load_model(model_id)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    adapter_sd = get_peft_model_state_dict(model)
    keys = adapter_keys or serialize.canonical_keys(adapter_sd)

    # G2-3 교환 폐포 — 학습되는 것과 교환되는 것이 완전히 같은가.
    # peft 어댑터 sd 키는 "...lora_A.weight", named_parameters 는 "...lora_A.default.weight".
    trainable = {n.replace(".default.", ".") .removeprefix("base_model.model.")
                 for n, p in model.named_parameters() if p.requires_grad}
    payload = {k.removeprefix("base_model.model.") for k in keys}
    from fl.client_vlm import adapter_exchange_contract
    ok, fails = adapter_exchange_contract(sorted(trainable), sorted(payload))
    if not ok:
        raise RuntimeError("교환 폐포 등식 실패 (G2-3):\n  " + "\n  ".join(fails))

    if adapter_in is not None:
        ref = get_peft_model_state_dict(model)
        sd = serialize.ndarrays_to_state_dict(adapter_in, keys, ref)
        set_peft_model_state_dict(model, sd)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    torch.cuda.reset_peak_memory_stats()
    model.train()

    seed = derive_seed(base_seed, round_idx, client_idx)
    steps = 0
    supervised_total = 0
    import time
    t0 = time.perf_counter()

    # -- 재개 --------------------------------------------------------------
    # ⑥ 는 10.2시간, 본실험은 칸당 수 주짜리 **단일 런**이다. 검출보다 재개가 더 절실하다.
    # 통합형은 검출과 달리 **정확히 이어진다** — 셔플이 `random.Random(seed + ep)` 라
    # 이력이 아니라 `(seed, epoch)` 의 함수이고, 누적 경계도 `(j+1) % GRAD_ACCUM` 이라
    # epoch 안에서 닫힌다. 프레임워크 지역 변수에 걸린 상태가 없다.
    ckpt = None
    start_ep = 0
    if resume_dir is not None:
        from detection.resume import (ResumeCheckpointer, ResumeIdentity,
                                      apply_resume, latest_resume)

        ident = ResumeIdentity(
            run_id=str(run_id), round_idx=int(round_idx), client_idx=int(client_idx),
            seed=int(seed), total_epochs=int(epochs), local_epochs=int(epochs),
            model=str(model_id or MODEL_ID), data=str(PAIRS_PATH),
        )
        state = latest_resume(resume_dir, identity=ident)
        if state is not None:
            view = _AdapterTrainerView(model, opt)
            apply_resume(
                view, state,
                state_dict_fn=lambda tr: get_peft_model_state_dict(tr.model),
                load_state_dict_fn=set_peft_model_state_dict,
            )
            start_ep = state.next_epoch
            steps = state.optimizer_steps
            supervised_total = state.payload.get("supervised_tokens", 0)
            print(f"[resume] epoch {state.epoch_done} 까지 완료 → epoch {start_ep} 부터 이어 간다",
                  flush=True)
        ckpt = ResumeCheckpointer(
            resume_dir, identity=ident,
            state_dict_fn=lambda tr: get_peft_model_state_dict(tr.model),
        )

    for ep in range(start_ep, epochs):
        order = list(range(len(rows)))
        random.Random(seed + ep).shuffle(order)
        ce_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        tok_cnt = 0
        for j, idx in enumerate(order):
            enc, labels, prompt_len = _encode(proc, rows[idx], prompt)
            enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
            labels = labels.to("cuda")
            # 판정 11 — 감독 위치의 로짓만 물질화한다. 감독 구간이 접미(prompt 뒤 전부)라
            # `logits_to_keep` 정수 슬라이스로 정확히 겹친다. 0 을 주면 전 위치를 뽑는다.
            n_keep = int(labels.shape[1] - prompt_len + 1) if supervised_logits_only else 0
            out = model(**enc, logits_to_keep=n_keep)
            # 판정 2 — shift 후 감독 토큰 총합 분모. HF 평균 loss 를 쓰지 않는다.
            logits = out.logits[:, :-1]
            # 남긴 로짓 j 는 절대 위치 T-n_keep+j 를 예측하므로 타깃은 그 다음 토큰이다.
            tgt = labels[:, 1:] if n_keep == 0 else labels[:, -(n_keep - 1):]
            mask = tgt != -100
            ce = torch.nn.functional.cross_entropy(
                logits[mask].float(), tgt[mask], reduction="sum"
            )
            n_tok = int(mask.sum())
            (ce / max(n_tok, 1)).backward()   # micro=1 이라 샘플 정규화 = 토큰 총합/토큰 수
            ce_sum += ce.detach(); tok_cnt += n_tok
            if (j + 1) % GRAD_ACCUM == 0 or (j + 1) == len(order):
                opt.step(); opt.zero_grad(); steps += 1
        supervised_total += tok_cnt
        if log_cb:
            log_cb(ep, float(ce_sum.item() / max(tok_cnt, 1)), steps, time.perf_counter() - t0)
        if ckpt is not None:
            # `epoch=ep, start_epoch=0` 이라 저장되는 값이 곧 **누적치**다 —
            # epochs_ran_in_round = ep+1, optimizer_steps = steps.
            ckpt.step_counter = _StepView(steps)
            ckpt.extra = {"supervised_tokens": supervised_total}
            ckpt(_AdapterTrainerView(model, opt, epoch=ep))

    final_sd = get_peft_model_state_dict(model)
    arrays = serialize.state_dict_to_ndarrays(final_sd, keys)
    metrics = {
        "optimizer_steps": steps,
        "supervised_tokens": supervised_total,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "payload_bytes": serialize.payload_nbytes(arrays),
        "param_l2": serialize.params_l2_norm(arrays),
        "wall_s": time.perf_counter() - t0,
        "seed": seed,
    }
    ref_sd = {k: v.detach().cpu() for k, v in final_sd.items()}
    del model
    torch.cuda.empty_cache()
    return arrays, keys, metrics, ref_sd
