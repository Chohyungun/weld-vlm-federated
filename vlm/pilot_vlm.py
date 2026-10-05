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

import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from detection import serialize
from detection.round_runner import derive_seed
from fl.seeding import seeded, shared_init_seed
from vlm.coords import CoordCfg, ImageGeom, coord_cfg_hash, quantize, to_model
from vlm.loss_norm import TokenAccumulator, normalized_ce, rescale_grads_, supervised_ce_sum
from vlm.schedule import LRF, cosine_lr, global_step
from vlm.seams import check_seam

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
#: 학습률 예열 스텝. `cosine_lr` 에 명시해 넘기고 학습 설정 원문(`lr_schedule`)에 싣는다.
WARMUP_STEPS = 0
#: 언어부 linear 접미사 — 실물 named_modules 덤프에서 확정(2026-08-31 프로브).
#: SSM/linear-attention 프로젝션(in_proj_*, out_proj)을 포함한다. 'all-linear' 금지.
#: 재개 신원에 싣는 **정책의 판**. 값이 바뀌면 다른 프로토콜이므로 문자열도 함께 바꾼다.
#: 셔플은 `random.Random(f"{derive_seed(base, r, c)}:{ep}")` 이고, 덧셈 파생과 구분된다.
SHUFFLE_POLICY = "str-seed-v1"
#: 채팅 템플릿 인자의 기본값. **원본은 이 객체 하나다** — 학습 렌더 · 생성 접두 · 접두 지문의 모든
#: `apply_chat_template` 호출이 이 인자를 그대로 받는다(리허설 2판 §2-5 의 대응 표). 본실험은 설정의
#: `uni_chat_template_kwargs` 를 넘기고, 넘기지 않으면 이 값이다. 미지정에 기대지 않는다 — 기본값이
#: 모델마다 반대다(11번 §1).
DEFAULT_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def template_mode_of(kwargs: Mapping[str, Any]) -> str:
    """신원·학습 설정에 싣는 템플릿 모드 문자열. **인자 객체에서만 만든다** — 따로 적지 않는다.

    `{"enable_thinking": False}` → `"enable_thinking=False"`. 키를 정렬해 순서에 기대지 않는다.
    """
    return ",".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))


#: 채팅 템플릿 모드. 기본 인자에서 만든 값이다(문자열을 따로 적지 않는다).
TEMPLATE_MODE = template_mode_of(DEFAULT_CHAT_TEMPLATE_KWARGS)


def _template_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """넘긴 인자 또는 기본값. 사전이 아니면 멈춘다."""
    if kwargs is None:
        return dict(DEFAULT_CHAT_TEMPLATE_KWARGS)
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"chat_template_kwargs 가 사전이 아니다: {kwargs!r}")
    return dict(kwargs)


def load_prompt(path: str | Path | None = None) -> tuple[str, str]:
    """프롬프트 문자열과 그 해시. **해시는 모델이 받는 문자열의 UTF-8 바이트**다(리허설 2판 §2-5).

    `read_text` 는 줄끝을 LF 로 바꿔 읽는다 — 모델이 받는 것이 그 문자열이므로 그 바이트를 해시한다.
    파일의 원시 바이트가 커밋과 같은지는 export 의 작업 트리 청결 검사가 따로 본다.
    학습 · 등록 · export 가 이 함수 하나를 부른다.
    """
    text = Path(path or PROMPT_PATH).read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()

TARGET_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
]


def gen_prefix_digest(proc, chat_template_kwargs: Mapping[str, Any] | None = None) -> str:
    """생성 접두의 **판본** 지문. 템플릿이 바뀌면 값이 바뀐다.

    감독 마스킹이 생성 접두의 길이로 자르므로, 접두가 달라지면 감독 구간이 밀린다. 그 판을
    재개 신원에 실어 정책이 다른 상태로 이어 가는 것을 막는다.

    프롬프트 내용은 **넣지 않는다.** 프롬프트는 별도의 공통 고정 항목이고, 여기서 보려는 것은
    템플릿과 인자가 만드는 접두의 구조다. 그래서 고정 더미 텍스트로 렌더한다 — 이미지도 넣지
    않으므로 가볍고, 모델을 올리지 않은 상태에서도 계산된다.
    """
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    ids = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, **_template_kwargs(chat_template_kwargs),
    )["input_ids"][0].tolist()
    raw = ",".join(str(i) for i in ids).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def gen_prefix_sha256(proc, prompt_text: str,
                      chat_template_kwargs: Mapping[str, Any] | None = None) -> str:
    """곁 파일 · 등록의 `gen_prefix_sha256` — **해시할 바이트까지 정한 정의**(리허설 2판 §2-5).

    등록 프롬프트를 이미지 없이 렌더한 생성 접두의 토큰 id 열이다. 배치 차원 하나를 벗긴 1차원 정수 목록을
    `json.dumps(ids, separators=(",", ":"))` 의 UTF-8 로 해시한다. 재개 신원의 `gen_prefix_digest`
    (고정 더미 글로 렌더한 16자)와는 **다른 양이고 서로 대조하지 않는다.**
    """
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    out = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, add_generation_prompt=True,
        **_template_kwargs(chat_template_kwargs),
    )
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError(f"생성 접두가 한 줄이 아니다: 배치 {len(ids)}")
        ids = list(ids[0])
    if not all(type(i) is int for i in ids):
        raise TypeError("생성 접두의 토큰 id 가 정수가 아니다")
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("utf-8")).hexdigest()


#: 학습 타깃의 꼴. `build_target` 이 내는 문자열의 규약이다 — 키 · 순서 · 좌표 경로 · 직렬화.
#: 이 사전의 해시가 학습 설정 원문의 `target_contract_sha256` 이다. `build_target` 의 출력 꼴을 바꾸면
#: 이 사전과 `version` 을 함께 바꾼다 — 시험이 출력의 키를 이 사전과 맞대 먼저 떨어진다.
TARGET_CONTRACT: dict[str, Any] = {
    "version": 1,
    "keys": ["defects", "verdict", "cited_clauses"],
    "defect_keys": ["iso_code", "bbox_2d"],
    "bbox": "quantize(to_model(bbox_px, geom, coord_cfg))",
    "verdict_missing": "판정불가",
    "clauses_missing": [],
    "json": {"ensure_ascii": False, "separators": [",", ":"]},
}


def canonical_json(obj: Any) -> str:
    """정규화 JSON 문자열. 학습 설정 원문 · 신원 해시가 같은 규칙을 쓴다(리허설 2판 §2-5)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TARGET_CONTRACT_SHA256 = sha256_text(canonical_json(TARGET_CONTRACT))


def processor_config_sha256(proc) -> str:
    """학습 쪽 프로세서 설정의 해시 — **이미지 프로세서의 `to_dict()`** 를 정규화 JSON 으로 잰다.

    학습과 생성이 같은 입력 크기로 도는지 가르는 값이다(리허설 2판 §3 ② 다). export 도 이 함수를 부른다 —
    두 쪽이 따로 정의하면 같은 설정에서 값이 갈린다. 이미지 프로세서가 없으면 빈 사전으로 잰다.
    직렬화되지 않는 값은 `str` 로 적는다.
    """
    ip = getattr(proc, "image_processor", None)
    d = ip.to_dict() if ip is not None and hasattr(ip, "to_dict") else {}
    return sha256_text(json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                  default=str))


def train_rows_digest(rows: list[dict]) -> str:
    """이 칸이 **실제로 학습한 행의 `image_id` 수열**의 sha256(리허설 2판 §2-7).

    순서는 받은 행의 순서다(셔플 전). 문자열 목록을 정규화 JSON 으로 적어 잰다. export 가 페어의 그
    참여자 행 전체에서 이 함수로 다시 내 맞댄다 — 부분 목록으로 학습한 어댑터가 본실험에 새지 않게 한다.
    """
    return sha256_text(canonical_json([str(r["image_id"]) for r in rows]))


def build_target(row: dict, geom: ImageGeom, coord_cfg: CoordCfg | None = None) -> str:
    """스켈레톤 → 학습 타깃 JSON. 좌표는 coords 모듈만 통과한다. 규약은 인자로 받고 기본은 `ABS_ORIG`."""
    cfg = coord_cfg or COORD_CFG
    defects = []
    for d in row["skeleton"]["defects"]:
        box = quantize(to_model(d["bbox_px"], geom, cfg))
        defects.append({"iso_code": str(d["type"]), "bbox_2d": list(box)})
    verdict = row["skeleton"].get("verdict") or "판정불가"   # clause_only 축의 정합값
    clauses = row["skeleton"].get("clauses") or []
    return json.dumps(
        {"defects": defects, "verdict": verdict, "cited_clauses": clauses},
        ensure_ascii=False, separators=(",", ":"),
    )


def load_pairs(split: str, client: str | None = None, *,
               pairs_path: str | Path | None = None) -> list[dict]:
    """페어 파일에서 한 분할(과 참여자)의 행. **경로는 인자로 받는다** — 기본은 파일럿 페어다."""
    with open(pairs_path or PAIRS_PATH, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    out = [r for r in rows if r["split"] == split and (client is None or r["client"] == client)]
    if not out:
        raise ValueError(f"페어 0건: split={split} client={client}")
    return out


def _load_model(model_id: str | None = None, *, init_seed: int = DEFAULT_INIT_SEED,
                revision: str | None = None, processor_kwargs: Mapping[str, Any] | None = None):
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
    # 프로세서는 **등록된 설정으로** 연다(`min/max_pixels` 등). 기본값으로 열면 export 가 등록값으로 연
    # 프로세서와 입력 크기가 갈릴 수 있다(리허설 2판 §0-1 · 조건 ② 다). 모델 판도 인자로 고정한다.
    proc = AutoProcessor.from_pretrained(mid, revision=revision, **dict(processor_kwargs or {}))
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        mid, revision=revision, quantization_config=bnb, device_map={"": 0}
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


#: 승인된 실제 모델 적재기 — **import 시점에 잡아 둔 참조**다(리허설 2판 §1-4). 모듈 속성
#: `_load_model` 을 바꿔 끼워도 이 튜플은 원래 함수를 가리킨다.
_APPROVED_MODEL_LOADERS: tuple = (_load_model,)


def resolve_model_loader(model_loader: Callable | None = None, *, purpose: str = "main",
                         standin_allowed: bool = False) -> tuple[Callable, list[dict]]:
    """부를 모델 적재기와 그 구현 식별자. **부르기 전에** 목적과 맞댄다(`vlm/seams.py`).

    인자가 없으면 **부르는 순간의** 모듈 전역 `_load_model` 을 쓴다 — 누가 모듈 속성을 바꿔 끼웠다면
    그 대역이 여기서 잡혀 승인 목록과의 객체 대조에 걸린다.
    """
    fn = model_loader if model_loader is not None else _load_model
    return fn, [check_seam(fn, _APPROVED_MODEL_LOADERS, seam="model_loader", purpose=purpose,
                           standin_allowed=standin_allowed)]


def make_optimizer(model):
    """학습의 최적화기. 학습과 등록(학습 설정 원문의 `optimizer` 묶음을 실제 객체에서 읽는다)이 이 함수 하나로 만든다."""
    import torch

    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def _config_blocks(model, opt, *, supervised_logits_only: bool) -> dict[str, Any]:
    """학습 설정 원문의 다섯 묶음 — **실제로 만든 객체에서** 읽는다(리허설 2판 §2-5, X-08)."""
    pc = model.peft_config["default"]
    lora = {"r": int(pc.r), "alpha": int(pc.lora_alpha), "dropout": float(pc.lora_dropout),
            "bias": str(pc.bias), "target_modules": sorted(str(m) for m in (pc.target_modules or []))}
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    if qc is None:
        quant = None
    else:
        get = (lambda k, d=None: qc.get(k, d)) if isinstance(qc, dict) else (lambda k, d=None: getattr(qc, k, d))
        quant = {"load_in_4bit": bool(get("load_in_4bit", False)),
                 "quant_type": get("bnb_4bit_quant_type"),
                 "double_quant": bool(get("bnb_4bit_use_double_quant", False)),
                 "compute_dtype": str(get("bnb_4bit_compute_dtype")).removeprefix("torch.")}
    g = opt.param_groups[0]
    optimizer = {"name": type(opt).__name__, "betas": [float(b) for b in g.get("betas", ())],
                 "eps": float(g.get("eps", float("nan"))),
                 "weight_decay": float(g.get("weight_decay", float("nan")))}
    loss = {"normalization": "supervised_token_sum", "supervised_logits_only": bool(supervised_logits_only)}
    lr_schedule = {"kind": "cosine_global_offset", "warmup_steps": int(WARMUP_STEPS)}
    return {"lora": lora, "quant": quant, "optimizer": optimizer, "loss": loss, "lr_schedule": lr_schedule}


def _grid_key(enc) -> str:
    """학습 배치의 관측 격자 키 — `json.dumps([t, h, w])`(기본 구분자, 리허설 2판 §2-5 의 꼴)."""
    g = enc.get("image_grid_thw") if hasattr(enc, "get") else None
    if g is None:
        return "null"
    rows = g.tolist() if hasattr(g, "tolist") else list(g)
    return json.dumps([int(v) for v in rows[0]] if len(rows) == 1 else rows)


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
        # 재개가 옵티마이저 상태를 옮길 장치 — 모델이 있는 곳이다. 실제 적재기는 GPU 0 이고,
        # 합성 시험의 대역은 CPU 다. 고정해 두면 CPU 대역의 상태가 GPU 로 올라간다.
        self.device = next(model.parameters()).device


def _encode(proc, row: dict, prompt: str, *, chat_template_kwargs: Mapping[str, Any] | None = None,
            coord_cfg: CoordCfg | None = None):
    from PIL import Image

    img = Image.open(row["image_path"]).convert("RGB")
    geom = ImageGeom(orig_w=img.size[0], orig_h=img.size[1])
    target = build_target(row, geom, coord_cfg)
    kwargs = _template_kwargs(chat_template_kwargs)
    user = {"role": "user", "content": [{"type": "image", "image": img},
                                        {"type": "text", "text": prompt}]}
    # `enable_thinking=False` 를 **두 호출에 모두 명시한다.** 미지정에 기대면 안 된다 —
    # 템플릿의 기본값이 모델마다 반대다(0.8B·2B 는 비생각, 4B 는 생각). 아래 감독 마스킹이
    # `prompt_len` **길이로** 자르므로, 생성 접두가 학습 렌더의 토큰 접두가 아니면 경계가
    # 밀린다. 실제로 4B 에서 감독이 두 토큰 밀려 시작했다(11번 §1). 채팅 템플릿 모드는
    # 다섯 조건 공통 고정 항목이다(개발규약 3-3). 두 호출이 **같은 인자 객체**를 받는다 — 원본은 하나다.
    full = proc.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": target}]}],
        tokenize=True, return_dict=True, return_tensors="pt",
        **kwargs,
    )
    prompt_only = proc.apply_chat_template(
        [user], tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, **kwargs,
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
    model_revision: str | None = None,
    pairs_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    coord_cfg: CoordCfg | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
    model_loader: Callable | None = None,
    purpose: str = "main",
    standin_allowed: bool = False,
    tag: str = "",
    seed_index: int | None = None,
    on_identity_ok: Callable[[int, Mapping[str, Any] | None, dict[str, Any]], None] | None = None,
    ledger_cb: Callable[[dict[str, Any]], None] | None = None,
    before_ckpt_cb: Callable[[int, Any], None] | None = None,
    after_ckpt_cb: Callable[[int, Any], None] | None = None,
    resume_extra: Mapping[str, Any] | None = None,
    pairs_digest: str | None = None,
    client_tag: str = "",
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
        model_revision · pairs_path · prompt_path · chat_template_kwargs · coord_cfg · processor_kwargs:
            **본실험은 설정에서 읽어 넘긴다**(`vlm/uni_config.py`). 넘기지 않으면 파일럿 기본값이다 —
            모듈 상수를 본실험 값으로 바꾸지 않는다(미니스펙 12번 §2-1). `pairs_path` 는 재개 신원의
            `data` 에도 실린다. 행 자체는 호출부가 `load_pairs(..., pairs_path=)` 로 읽어 `rows` 로 준다.
        model_loader · purpose · standin_allowed: 모델 적재 이음새(리허설 2판 §1-4). `purpose="main"` 은
            승인된 실제 적재기만 받고, 대역이면 **부르기 전에** `SeamRejected` 로 멈춘다. 부른 구현의
            식별자는 지표의 `impl_ids` 에 실린다.
        tag · seed_index: 칸 태그와 시드 번호. 지표에 실어 산출물이 어느 칸의 것인지 말하게 한다.
        on_identity_ok(start_epoch, payload, resume_identity): **재개 신원 대조를 지난 뒤 · 상태를 적용하기 전** 한 번 부른다.
            원장의 이음매 행이 여기서 쓰인다 — 거부될 실행이 원장을 더럽히지 않는다(리허설 2판 §2-6).
            `payload` 는 이어 받을 재개 파일의 내용이고 처음부터면 None 이다. 여기서 올린 예외는 적용 전에 멈춘다.
        ledger_cb(values): epoch 마다 원장 행의 값 — `epoch`·`epochs_ran`·`optimizer_steps`·`supervised_tokens`·
            `mean_ce`·`lr`·`param_l2`·`peak_vram_gb`·`wall_s`. 체크포인트 **전에** 부른다.
        before_ckpt_cb · after_ckpt_cb(epoch, ckpt): 체크포인트 앞뒤의 훅. 재개 경로가 켜졌을 때만 불린다.
            `after_ckpt_cb` 는 **저장이 성공했을 때만** 부른다 — 체크포인터는 저장 예외를 삼키므로
            (`last_error`) 훅이 불렸다는 것이 저장 성공의 증거가 되게 한다. 실패한 저장은 지표의
            `resume_save_failures` 로 센다.
        pairs_digest · client_tag: 재개 신원(`vlm/resume_uni.UniResumeIdentity`)에 싣는 페어 내용 지문과 참여자.
        resume_extra: 재개 파일에 함께 실을 값. **덮어쓰지 않고 합친다** — `supervised_tokens` 와
            관측 격자는 이 함수가 싣는다.
    """
    # 이음새 — 모델 적재기를 **부르기 전에** 목적과 맞댄다(리허설 2판 §1-4 의 순서 3). 본실험에
    # 대역이 오면 여기서 멈추고 대역은 한 번도 불리지 않는다.
    loader, impl_ids = resolve_model_loader(model_loader, purpose=purpose, standin_allowed=standin_allowed)

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    # 체크리스트 18 — 게이트를 **실제로 부른다.** 통합형은 Ultralytics 를 안 쓰므로
    # cudnn 결정론에 대응물이 없었다(80번 D13). 실효값은 metrics 에 실려 나간다.
    from fl.run_gates import apply_run_gates, fingerprint_for_cell

    gates = apply_run_gates(
        cell=f"uni_r{round_idx}_c{client_idx}",
        fingerprints=[fingerprint_for_cell(f"uni_r{round_idx}_c{client_idx}",
                                           base_ckpt=str(model_id or MODEL_ID))],
    )

    kwargs = _template_kwargs(chat_template_kwargs)
    cfg_coord = coord_cfg or COORD_CFG
    model, proc = loader(
        model_id, init_seed=shared_init_seed(base_seed if init_seed is None else init_seed),
        revision=model_revision, processor_kwargs=processor_kwargs,
    )
    # 장치는 적재한 모델에서 읽는다. 실제 적재기는 GPU 0 에 올리고, 합성 시험의 대역은 CPU 에 둔다 —
    # CPU 에서는 CUDA 를 한 번도 건드리지 않는다.
    dev = next(model.parameters()).device
    on_cuda = dev.type == "cuda"
    prompt, prompt_sha256 = load_prompt(prompt_path)
    prefix_digest = gen_prefix_digest(proc, kwargs)

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

    opt = make_optimizer(model)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()

    seed = derive_seed(base_seed, round_idx, client_idx)
    steps = 0
    supervised_total = 0
    import time
    t0 = time.perf_counter()

    # -- 재개 --------------------------------------------------------------
    # ⑥ 는 10.2시간, 본실험은 칸당 수 주짜리 **단일 런**이다. 검출보다 재개가 더 절실하다.
    # 통합형의 셔플 순열은 `(seed, epoch)` 만의 함수여서 난수 이력에 걸려 있지 않고, 누적
    # 경계도 `(j+1) % GRAD_ACCUM` 이라 epoch 안에서 닫힌다. 프레임워크 지역 변수에 걸린
    # 상태도 없다. 이 성질은 파생을 문자열 시드로 바꾼 뒤에도 그대로다.
    #
    # **이것만으로 "정확히 이어진다" 고 쓰지 않는다.** 순열이 재현된다는 것과 전체 학습이
    # 수치적으로 동치라는 것은 다른 주장이다 — 옵티마이저·스케줄 상태의 복원, 누적 gradient,
    # 커널 비결정성이 모두 걸린다. 4B 전 구간 등가는 아직 시험되지 않았다(0.8B 범위만 봤다).
    # 등가 시험의 정의와 남은 범위는 미니스펙의 재개 절에 있다.
    ckpt = None
    start_ep = 0
    save_failures = 0
    grid_counts: dict[str, int] | None = None     # 관측 격자 — 첫 완주 epoch 에서 전 행을 한 번씩 센다
    from vlm.init_adapter import init_adapter_digest
    from vlm.resume_uni import UniResumeIdentity

    proc_sha = processor_config_sha256(proc)
    # 통합형 재개 신원 — 검출 신원의 필드에 12번 §4-1 의 필드와 프로세서 설정을 더했다. 해상도만 바꿔도
    # 중간 체크포인트를 이어 받지 않는다.
    ident = UniResumeIdentity(
        run_id=str(run_id), round_idx=int(round_idx), client_idx=int(client_idx),
        seed=int(seed), total_epochs=int(epochs), local_epochs=int(epochs),
        model=str(model_id or MODEL_ID), data=str(pairs_path or PAIRS_PATH),
        # 정책의 판. 구판 체크포인트는 이 셋이 빈 문자열이라 여기서 어긋나 거부된다.
        shuffle_policy=SHUFFLE_POLICY,
        template_mode=template_mode_of(kwargs),
        gen_prefix_digest=prefix_digest,
        cell=str(tag), client_tag=str(client_tag), num_rounds=int(num_rounds),
        pairs_digest=str(pairs_digest or ""), prompt_sha256=prompt_sha256,
        coord_cfg_hash=coord_cfg_hash(cfg_coord), target_contract_sha256=TARGET_CONTRACT_SHA256,
        input_adapter_digest=init_adapter_digest(list(adapter_in)) if adapter_in is not None else "",
        micro_batch=int(MICRO_BATCH), grad_accum=int(GRAD_ACCUM), lr0=float(LR), lrf=float(LRF),
        processor_config_sha256=proc_sha,
    )
    if resume_dir is not None:
        from detection.resume import ResumeCheckpointer, apply_resume, latest_resume
        # `latest_resume` 이 신원 불일치를 `ValueError` 로 막는다 — **모델 상태를 적용하기
        # 전이다.** 정책 판을 신원에 넣은 것이 그 차단을 정책에까지 넓힌다.
        state = latest_resume(resume_dir, identity=ident)
        if state is not None:
            # 필드 누락도 **적용 전에** 본다. 이전 판은 `apply_resume` 뒤에 검사해서,
            # 거부할 상태의 가중치가 이미 모델에 얹힌 다음 죽었다.
            if "supervised_tokens" not in state.payload:
                # 기본값 0 으로 접으면 안 된다(85번 ① 부수 결함). 판정 2 에서 이 값이
                # **FedAvg 가중**이 됐으므로, 구판 체크포인트로 재개하면 재개 이전 구간의
                # 토큰이 통째로 빠진 가중이 조용히 전송된다. 시끄럽게 죽는 쪽이 맞다.
                raise RuntimeError(
                    "재개 체크포인트에 supervised_tokens 가 없다 — 판정 2(토큰 가중) 이전 "
                    "판으로 만든 상태다. 이 상태로 이어 가면 가중이 과소 전송된다. "
                    "체크포인트를 지우고 라운드를 처음부터 돌려라."
                )
        # 이음매 — **신원 대조를 지난 뒤 · 상태를 적용하기 전**(리허설 2판 §2-6). 훅이 올린 예외는 적용 전에 멈춘다.
        if on_identity_ok is not None:
            on_identity_ok(int(state.next_epoch) if state is not None else 0,
                           None if state is None else state.payload, asdict(ident))
        if state is not None:
            # 적용 **직전**에 한 줄을 stderr 로 찍는다. 적용 뒤에 찍으면 적용 중에 죽은 실행이 흔적을 남기지 않는다.
            print(f"[resume] {resume_dir}: epoch {state.epoch_done} 까지의 상태를 적용한다 → "
                  f"epoch {state.next_epoch} 부터 이어 간다", file=sys.stderr, flush=True)
            view = _AdapterTrainerView(model, opt)
            apply_resume(
                view, state,
                state_dict_fn=lambda tr: get_peft_model_state_dict(tr.model),
                load_state_dict_fn=set_peft_model_state_dict,
            )
            start_ep = state.next_epoch
            steps = state.optimizer_steps
            supervised_total = state.payload["supervised_tokens"]
            if state.payload.get("train_input_grid_observed"):
                grid_counts = dict(state.payload["train_input_grid_observed"])
        ckpt = ResumeCheckpointer(
            resume_dir, identity=ident,
            state_dict_fn=lambda tr: get_peft_model_state_dict(tr.model),
        )
    elif on_identity_ok is not None:
        # 재개 경로가 꺼져 있으면 대조할 재개 신원이 없다 — 같은 자리(적재 뒤 · 학습 전)에서 부른다.
        on_identity_ok(0, None, asdict(ident))

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
        # 문자열 시드다. `seed + ep` 로 더하면 **등록 시드가 연속 정수여서 인접 시드가 순열을
        # 공유한다** — 시드 3개 × epoch 3개의 9조합이 만드는 순열이 9개가 아니라 5개이고,
        # 그중 하나는 세 시드가 함께 쓴다. 겹침이 조건마다 달라서(중앙·로컬만 겹치고 연합은
        # 라운드 오프셋이 갈라 준다) 시드 간 산포가 비대칭으로 잡히고, 그 산포가 회복률 분모의
        # 표준편차로 들어가 채택 관문을 헐겁게 만든다.
        #
        # `random.seed(str)` 은 내부에서 sha512 를 거치므로 이것이 `(시드, epoch)` 의 해시이고
        # `PYTHONHASHSEED` 에 의존하지 않는다(시험으로 고정). `seed` 는 이미
        # `derive_seed(base_seed, round_idx, client_idx)` 를 지난 값이다.
        random.Random(f"{seed}:{ep}").shuffle(order)
        ce_sum = torch.zeros((), device=dev, dtype=torch.float32)
        tok_cnt = 0
        grid_by_row: dict[int, str] = {}
        for j, idx in enumerate(order):
            enc, labels, prompt_len = _encode(proc, rows[idx], prompt,
                                              chat_template_kwargs=kwargs, coord_cfg=cfg_coord)
            grid_by_row[idx] = _grid_key(enc)
            enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}
            # `model(**enc, labels=...)` 는 HF 내부의 **평균** loss 를 계산한다 — 판정 2
            # (토큰 총합 분모)의 우회 채널이다. AST 시험은 명시적 `labels=` 만 잡으므로
            # dict 로 스며드는 경로는 여기서 막는다(85번 ⑧).
            assert "labels" not in enc, "labels 가 모델 입력에 스며들면 HF 평균 loss 가 돈다"
            labels = labels.to(dev)
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
                                   total_step_budget, warmup_steps=WARMUP_STEPS)
                for grp in opt.param_groups:
                    grp["lr"] = lr_now
                opt.step(); opt.zero_grad()
                steps += 1; steps_in_round += 1
                lr_trace.append((steps, round(lr_now, 10)))
        supervised_total += tok_cnt
        epochs_done += 1
        if len(grid_by_row) == len(rows):
            # epoch 하나가 모든 행을 한 번씩 지난다 — 행마다 한 번 센 건수다.
            grid_counts = dict(Counter(grid_by_row.values()))
        mean_ce = normalized_ce(ce_sum, tok_cnt)
        if log_cb:
            log_cb(ep, mean_ce, steps, time.perf_counter() - t0)
        if ledger_cb is not None:
            # 원장 행의 값. 체크포인트 **전에** 쓴다 — 그 사이에 죽으면 다음 프로세스가 이 epoch 를 다시 돌아
            # 같은 행이 둘이 되고, 이음매 행 뒤의 것이 앞의 것을 대체한다(리허설 2판 §2-6 의 겹친 행).
            ledger_cb({
                "epoch": int(ep),
                "epochs_ran": int(epochs_done),
                "optimizer_steps": int(steps),
                "supervised_tokens": int(supervised_total),
                "mean_ce": float(mean_ce),
                "lr": float(opt.param_groups[0]["lr"]),
                "param_l2": float(serialize.params_l2_norm(
                    serialize.state_dict_to_ndarrays(get_peft_model_state_dict(model), keys))),
                "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if on_cuda else 0.0,
                "wall_s": float(time.perf_counter() - t0),
            })
        if ckpt is not None:
            # `epoch=ep, start_epoch=0` 이라 저장되는 값이 곧 **누적치**다 —
            # epochs_ran_in_round = ep+1, optimizer_steps = steps.
            ckpt.step_counter = _StepView(steps)
            # 통째로 바꾸지 않고 합친다 — 호출부가 싣는 값(신원 해시 · 프로세스 시각)이 지워지지 않게 한다.
            ckpt.extra = {**dict(ckpt.extra or {}), **dict(resume_extra or {}),
                          "supervised_tokens": supervised_total,
                          "train_input_grid_observed": grid_counts}
            if before_ckpt_cb is not None:
                before_ckpt_cb(int(ep), ckpt)
            n_before = ckpt.n_saves
            ckpt.last_error = None
            ckpt(_AdapterTrainerView(model, opt, epoch=ep))
            if ckpt.last_error is None and ckpt.n_saves == n_before + 1:
                if after_ckpt_cb is not None:
                    after_ckpt_cb(int(ep), ckpt)
            else:
                # 체크포인터가 저장 예외를 삼켰다. 학습은 잇되 이 epoch 에서 재개할 수 없다는 사실을 센다.
                save_failures += 1

    final_sd = get_peft_model_state_dict(model)
    arrays = serialize.state_dict_to_ndarrays(final_sd, keys)
    # 회계에 실리는 값은 **실측**이어야 한다. `epochs_ran`·`optimizer`·`lr` 을 호출부에서
    # 상수로 재구성하던 것이 74번 감사 P9 의 절반이다 — 여기서 실물을 읽어 넘긴다.
    opt_group = opt.param_groups[0]
    metrics = {
        "optimizer_steps": steps,
        "supervised_tokens": supervised_total,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if on_cuda else 0.0,
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
        # 무엇으로 학습했는지 — 설정에서 받은 값을 그대로 싣는다(곁 파일 · meta 의 출처).
        "model_id": str(model_id or MODEL_ID),
        "model_revision": model_revision,
        "pairs_path": str(pairs_path or PAIRS_PATH),
        "prompt_sha256": prompt_sha256,
        "chat_template_kwargs": dict(kwargs),
        "template_mode": template_mode_of(kwargs),
        "coord_space": cfg_coord.coord_space,
        "coord_cfg_hash": coord_cfg_hash(cfg_coord),
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
        "processor_config_sha256": proc_sha,
        "resume_save_failures": int(save_failures),
        # 관측 격자 — 행마다 한 번 센 값별 건수. 모든 행을 센 epoch 가 없으면 None 이다.
        "train_input_grid_observed": grid_counts,
        "gen_prefix_digest": prefix_digest,
        "resume_identity": asdict(ident),
        "purpose": purpose,
        "tag": str(tag),
        "seed_index": seed_index,
        "impl_ids": impl_ids,
        "micro_batch": int(MICRO_BATCH),
        "grad_accum": int(GRAD_ACCUM),
        "shuffle_policy": SHUFFLE_POLICY,
        # 학습 설정 원문의 다섯 묶음 — 실제로 만든 객체에서 읽었다.
        "config_blocks": _config_blocks(model, opt, supervised_logits_only=supervised_logits_only),
        # 판정 자체를 했는지를 산출물이 증명한다(G1-6).
        "gates_evaluated": gates["gates_evaluated"],
        "gate_results": gates["gate_results"],
    }
    ref_sd = {k: v.detach().cpu() for k, v in final_sd.items()}
    del model
    if on_cuda:
        torch.cuda.empty_cache()
    return arrays, keys, metrics, ref_sd
