"""⑥⑦ 통합형 파일럿 학습 — Qwen3.5-0.8B QLoRA 4bit.

파일럿 전용 배선이다. 함정 겨냥 규약은 그대로 지킨다:

- **좌표는 `vlm/coords.py` 만 통과한다** (함정 #4). 타깃 bbox 는 원본 픽셀 →
  `to_model`(ABS_ORIG) → `quantize` 로 만들고, 모델 좌표는 어떤 파일에도 저장하지 않는다.
  ABS_ORIG 에서 정변환은 항등이지만 **경로는 그대로 유지한다** — 규약이 다시 바뀌어도
  호출부가 아니라 설정값 하나만 움직이게 하기 위해서다.
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
from fl.seeding import seeded, shared_init_seed
from vlm.coords import CoordCfg, ImageGeom, quantize, to_model
from vlm.loss_norm import TokenAccumulator, normalized_ce, rescale_grads_, supervised_ce_sum
from vlm.schedule import LRF, cosine_lr, global_step

MODEL_ID = "Qwen/Qwen3.5-0.8B"
#: 프롬프트는 다섯 칸 공통 고정 항목이라 **한 글자도 달라선 안 된다**(개발규약 3-3).
#: 그래서 좌표 문장을 고칠 때 v1 을 덮어쓰지 않고 새 파일을 만들었다 — v1 로 돈 파일럿
#: 산출물과 v2 로 돌 본실험을 `prompt_sha256` 으로 구분할 수 있어야 한다.
PROMPT_PATH = Path("vlm/prompts/unified_v2_absorig.txt")
PAIRS_PATH = Path("data/processed/pairs_pilot_v1/pairs.jsonl")
#: **ABS_ORIG** — 카나리아-1 실측(75번 §5)에서 0.8B·4B 두 모델이 판별 가능한 3장 전부에서
#: 절대 원본 픽셀로 답했다. 총괄 판정 1(2026-09-02)로 전환 확정. 개발규약 3-8("학습 타깃은
#: 채택 모델의 네이티브 좌표계를 따른다")의 이행이며, 라벨측 왕복 IoU 손실도 사라진다
#: (NORM_1000 은 실페어 4,560 박스에서 median 0.98119·IoU<0.95 가 379건, ABS_ORIG 는 전부 1.0).
COORD_CFG = CoordCfg(coord_space="ABS_ORIG")
#: LoRA A 초기화 기본 시드. 파일럿 상수(`scripts/pilot_c.py:BASE_SEED`)와 같은 값이며
#: 검출 칸 `build_initial_weights(seed=BASE_SEED)` 와도 같다 — 두 칸의 "동일 출발"이
#: 같은 상수에서 나와야 사후 대조가 한 번에 된다. 호출부가 명시하면 그 값이 이긴다.
DEFAULT_INIT_SEED = 20260828
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


def _load_model(model_id: str | None = None, *, init_seed: int = DEFAULT_INIT_SEED):
    """QLoRA 4bit 모델 + LoRA 어댑터. **어댑터 초기화는 반드시 시드 아래에서 일어난다.**

    peft 는 `lora_B` 를 0 으로, `lora_A` 를 난수로 놓는다. `init_seed` 를 고정하지 않으면
    클라이언트마다 다른 A 로 출발하고, 그러면 r0 가중 평균이 "같은 기저의 평균"이 아니라
    **독립 난수의 상쇄**가 된다(74번 감사 C-1 · 함정 #3). 실제로 그렇게 났다.

    `seed_all` 이 아니라 `seeded()` 컨텍스트를 쓰는 이유는 `fl/seeding.py` 에 적었다 —
    초기화 한 번 때문에 그 뒤 학습 전체의 난수 흐름이 호출 순서에 묶이면 안 된다.
    """
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
    with seeded(shared_init_seed(init_seed)):
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
    init_seed: int | None = None,
    num_rounds: int = 1,
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
        init_seed: LoRA A 초기화 시드. **라운드·클라이언트에 따라 달라지면 안 된다** —
            `derive_seed` 와 헷갈리지 않도록 별도 인자로 뒀다. None 이면 `base_seed`.
        num_rounds: **전역 라운드 수 R.** 학습률 cosine 이 이 값으로 총 스텝 예산을
            계산한다(판정 4). ⑥ 처럼 단일 런이면 1 이고, 그때 라운드 하나가 곧 전체
            예산이라 검출의 `total_epochs` 와 같은 역할을 한다. 여기에 1 을 넣고
            ⑦ 을 돌리면 라운드마다 스케줄이 리셋돼 검출과 다시 어긋난다.
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    # 체크리스트 18 — 게이트를 **실제로 부른다.** 통합형은 Ultralytics 를 안 쓰므로
    # cudnn 결정론에 대응물이 없었다(80번 D13). 실효값은 metrics 에 실려 나간다.
    from fl.run_gates import apply_run_gates, fingerprint_for_cell

    gates = apply_run_gates(
        cell=f"uni_r{round_idx}_c{client_idx}",
        fingerprints=[fingerprint_for_cell(f"uni_r{round_idx}_c{client_idx}",
                                           base_ckpt=str(model_id or MODEL_ID))],
    )

    model, proc = _load_model(
        model_id, init_seed=shared_init_seed(base_seed if init_seed is None else init_seed)
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    adapter_sd = get_peft_model_state_dict(model)
    keys = adapter_keys or serialize.canonical_keys(adapter_sd)

    # 주입 **전** 초기 어댑터 증빙. 세 클라이언트가 같은 A 로 출발했음을 사후에 대조하는
    # 근거이며, 검출 칸의 `injection_digest` 와 같은 역할이다(74번 감사 C-1).
    from vlm.init_adapter import adapter_proof

    init_proof = adapter_proof(serialize.state_dict_to_ndarrays(adapter_sd, keys), keys)

    # G2-3 교환 폐포 — 학습되는 것과 교환되는 것이 완전히 같은가.
    # peft 어댑터 sd 키는 "...lora_A.weight", named_parameters 는 "...lora_A.default.weight".
    trainable = {n.replace(".default.", ".") .removeprefix("base_model.model.")
                 for n, p in model.named_parameters() if p.requires_grad}
    payload = {k.removeprefix("base_model.model.") for k in keys}
    from fl.client_vlm import adapter_exchange_contract
    ok, fails = adapter_exchange_contract(sorted(trainable), sorted(payload))
    if not ok:
        raise RuntimeError("교환 폐포 등식 실패 (G2-3):\n  " + "\n  ".join(fails))

    injected_proof = None
    if adapter_in is not None:
        ref = get_peft_model_state_dict(model)
        sd = serialize.ndarrays_to_state_dict(adapter_in, keys, ref)
        set_peft_model_state_dict(model, sd)
        # 주입이 실제로 먹었는지 확인한다 — 서버가 보낸 값과 대조 가능한 형태로 남긴다.
        after = serialize.state_dict_to_ndarrays(get_peft_model_state_dict(model), keys)
        injected_proof = adapter_proof(after, keys)

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
            if "supervised_tokens" not in state.payload:
                # 기본값 0 으로 접으면 안 된다(85번 ① 부수 결함). 판정 2 에서 이 값이
                # **FedAvg 가중**이 됐으므로, 구판 체크포인트로 재개하면 재개 이전 구간의
                # 토큰이 통째로 빠진 가중이 조용히 전송된다. 시끄럽게 죽는 쪽이 맞다.
                raise RuntimeError(
                    "재개 체크포인트에 supervised_tokens 가 없다 — 판정 2(토큰 가중) 이전 "
                    "판으로 만든 상태다. 이 상태로 이어 가면 가중이 과소 전송된다. "
                    "체크포인트를 지우고 라운드를 처음부터 돌려라."
                )
            supervised_total = state.payload["supervised_tokens"]
            print(f"[resume] epoch {state.epoch_done} 까지 완료 → epoch {start_ep} 부터 이어 간다",
                  flush=True)
        ckpt = ResumeCheckpointer(
            resume_dir, identity=ident,
            state_dict_fn=lambda tr: get_peft_model_state_dict(tr.model),
        )

    # -- 누적 창·학습률 상태 -------------------------------------------------
    # `steps_per_round` 는 이 클라이언트가 한 라운드에 밟는 옵티마이저 스텝 수다.
    # `math.ceil` 인 이유: 마지막 부분 창도 step 한다(`(j+1) == len(order)` 분기).
    import math as _math

    steps_per_epoch = _math.ceil(len(rows) / GRAD_ACCUM)
    steps_per_round = steps_per_epoch * int(epochs)
    total_step_budget = steps_per_round * max(int(num_rounds), 1)
    acc = TokenAccumulator()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    steps_in_round = int(steps)          # 재개했다면 이미 밟은 스텝에서 이어 간다
    lr_trace: list[tuple[int, float]] = []

    epochs_done = int(start_ep)          # 이 라운드에서 완료한 epoch 누적 수(재개분 포함)
    for ep in range(start_ep, epochs):
        order = list(range(len(rows)))
        random.Random(seed + ep).shuffle(order)
        ce_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        tok_cnt = 0
        for j, idx in enumerate(order):
            enc, labels, prompt_len = _encode(proc, rows[idx], prompt)
            enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
            # `model(**enc, labels=...)` 는 HF 내부의 **평균** loss 를 계산한다 — 판정 2
            # (토큰 총합 분모)의 우회 채널이다. AST 시험은 명시적 `labels=` 만 잡으므로
            # dict 로 스며드는 경로는 여기서 막는다(85번 ⑧).
            assert "labels" not in enc, "labels 가 모델 입력에 스며들면 HF 평균 loss 가 돈다"
            labels = labels.to("cuda")
            # 판정 11 — 감독 위치의 로짓만 물질화한다. 감독 구간이 접미(prompt 뒤 전부)라
            # `logits_to_keep` 정수 슬라이스로 정확히 겹친다. 0 을 주면 전 위치를 뽑는다.
            n_keep = int(labels.shape[1] - prompt_len + 1) if supervised_logits_only else 0
            out = model(**enc, logits_to_keep=n_keep)
            # shift 는 여기서 한 번만 한다 — `supervised_ce_sum` 은 shift 하지 않는다.
            logits = out.logits[:, :-1]
            # 남긴 로짓 j 는 절대 위치 T-n_keep+j 를 예측하므로 타깃은 그 다음 토큰이다.
            tgt = labels[:, 1:] if n_keep == 0 else labels[:, -(n_keep - 1):]

            # 판정 2 — **토큰 균일**. `ce_sum` 을 나누지 않고 누적하고, 창이 닫힐 때
            # 기울기를 창 토큰 총합으로 한 번 나눈다. 샘플마다 나누면 목적함수가
            # 샘플 균일이 되고, 감독 길이가 19~947 로 49.8배 퍼져 있어 짧은 답(결함 0건)이
            # 토큰당 6.13배 무거워진다(80번 C1).
            ce, n_tok = supervised_ce_sum(logits, tgt)
            ce.backward()
            acc.add(n_tok)
            ce_sum += ce.detach(); tok_cnt += n_tok

            if (j + 1) % GRAD_ACCUM == 0 or (j + 1) == len(order):
                rescale_grads_(trainable_params, acc.close())
                # 판정 4 — 전역 오프셋 cosine. 라운드 경계를 넘어 하나의 스케줄로 잇는다.
                lr_now = cosine_lr(LR, global_step(round_idx, steps_in_round, steps_per_round),
                                   total_step_budget)
                for grp in opt.param_groups:
                    grp["lr"] = lr_now
                opt.step(); opt.zero_grad()
                steps += 1; steps_in_round += 1
                lr_trace.append((steps, round(lr_now, 10)))
        supervised_total += tok_cnt
        epochs_done += 1
        if log_cb:
            log_cb(ep, normalized_ce(ce_sum, tok_cnt), steps, time.perf_counter() - t0)
        if ckpt is not None:
            # `epoch=ep, start_epoch=0` 이라 저장되는 값이 곧 **누적치**다 —
            # epochs_ran_in_round = ep+1, optimizer_steps = steps.
            ckpt.step_counter = _StepView(steps)
            ckpt.extra = {"supervised_tokens": supervised_total}
            ckpt(_AdapterTrainerView(model, opt, epoch=ep))

    final_sd = get_peft_model_state_dict(model)
    arrays = serialize.state_dict_to_ndarrays(final_sd, keys)
    # 회계에 실리는 값은 **실측**이어야 한다. `epochs_ran`·`optimizer`·`lr` 을 호출부에서
    # 상수로 재구성하던 것이 74번 감사 P9 의 절반이다 — 여기서 실물을 읽어 넘긴다.
    opt_group = opt.param_groups[0]
    metrics = {
        "optimizer_steps": steps,
        "supervised_tokens": supervised_total,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "payload_bytes": serialize.payload_nbytes(arrays),
        "param_l2": serialize.params_l2_norm(arrays),
        "wall_s": time.perf_counter() - t0,
        "seed": seed,
        # -- 실측 회계 --------------------------------------------------------
        # epoch 루프 안에서 센 값이다. `epochs` 인자를 되돌려 주면 "예산을 다 돌았다"가
        # 아니라 "예산을 다 돌았다고 적었다"가 되어 검사가 공허해진다(74번 P9).
        "epochs_ran": int(epochs_done),
        "epochs_this_process": int(epochs_done - start_ep),
        "resumed_from_epoch": int(start_ep) if start_ep else None,
        "optimizer": type(opt).__name__,
        "lr": float(opt_group["lr"]),
        # 판정 4 — 상수가 아니라 궤적을 남긴다. 검출의 `lr_trace` 와 대응한다.
        "lr0": float(LR),
        "lrf": float(LRF),
        "lr_trace_head": lr_trace[:3],
        "lr_trace_tail": lr_trace[-3:],
        "steps_per_round": int(steps_per_round),
        "total_step_budget": int(total_step_budget),
        "betas": [float(b) for b in opt_group.get("betas", ())],
        "weight_decay": float(opt_group.get("weight_decay", float("nan"))),
        "init_seed": shared_init_seed(base_seed if init_seed is None else init_seed),
        "init_proof": dict(init_proof),
        "injected_proof": None if injected_proof is None else dict(injected_proof),
        # 판정 자체를 했는지를 산출물이 증명한다(G1-6).
        "gates_evaluated": gates["gates_evaluated"],
        "gate_results": gates["gate_results"],
    }
    ref_sd = {k: v.detach().cpu() for k, v in final_sd.items()}
    del model
    torch.cuda.empty_cache()
    return arrays, keys, metrics, ref_sd
