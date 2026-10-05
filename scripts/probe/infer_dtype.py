"""추론 dtype 비교 — 4bit nf4 대 bf16. 생성 단가의 지배 항이 무엇인지 가린다.

QLoRA 4bit 은 **학습**의 제약에서 온다(기울기·옵티마이저 상태가 16 GB 에 들어가야 한다).
추론에는 그 제약이 없다. 4B 를 bf16 으로 올리면 약 8 GB 이므로 16 GB 에 들어간다.
bitsandbytes nf4 는 행렬곱마다 역양자화를 하므로 생성에서 오히려 느릴 수 있다.

평가 시간이 예산의 지배 항이고(12번 9절) 그 시간이 거의 전부 디코딩이므로
(선행 0.66 s 대 토큰당 0.15 s, 11번 3-1-2) 이 배율이 예산 판정을 바꿀 수 있다.

**길이를 고정해서** 잰다 — 자유 길이로 재면 dtype 차이가 아니라 길이 분포 차이를 잰다.
어댑터는 양쪽 다 붙인다(실제 평가 경로가 그렇다).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

OUT = Path("outputs/probe_c/infer_dtype.json")
LENGTHS = [1, 64]
N_IMG = 6


def load(kind: str, model_id: str):
    """4bit 또는 bf16 으로 같은 모델·같은 어댑터 설정을 올린다."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    from fl.seeding import seeded, shared_init_seed
    from vlm.pilot_vlm import DEFAULT_INIT_SEED, TARGET_SUFFIXES

    proc = AutoProcessor.from_pretrained(model_id)
    if kind == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=TARGET_SUFFIXES)
    with seeded(shared_init_seed(DEFAULT_INIT_SEED)):
        model = get_peft_model(model, lora)
    model.eval()
    return model, proc


def measure(kind: str, model_id: str, paths: list[str], prompt: str) -> dict:
    from PIL import Image
    from transformers import GenerationConfig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model, proc = load(kind, model_id)
    load_s = round(time.perf_counter() - t_load, 2)
    after_load_gb = round(torch.cuda.memory_allocated() / 1e9, 3)
    print(f"  [{kind}] 로드 {load_s}s · 가중치 {after_load_gb} GB", flush=True)

    per_len = {}
    for length in LENGTHS:
        cfg = GenerationConfig(max_new_tokens=length, min_new_tokens=length,
                               do_sample=False, num_beams=1)
        lat = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": prompt}]}]
            enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True,
                                           enable_thinking=False)
            n_prompt = int(enc["input_ids"].shape[1])
            enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, generation_config=cfg)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            assert int(out.shape[1]) - n_prompt == length, "길이 고정이 듣지 않았다"
        per_len[length] = round(statistics.median(lat), 3)
        print(f"  [{kind}] 고정 {length}토큰: 중앙 {per_len[length]}s", flush=True)

    tok_s = round((LENGTHS[1] - LENGTHS[0]) / (per_len[LENGTHS[1]] - per_len[LENGTHS[0]]), 2)
    prefill = round(per_len[LENGTHS[0]] - 1 / tok_s, 3)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    del model
    torch.cuda.empty_cache()
    return {"dtype": kind, "로드_s": load_s, "가중치_GB": after_load_gb,
            "peak_GB": peak, "길이별_지연_s": per_len,
            "초당_토큰": tok_s, "선행_s": prefill}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-img", type=int, default=N_IMG)
    args = ap.parse_args()

    from scripts.probe.cost_4b import MODEL, _val_images
    from vlm.pilot_vlm import PROMPT_PATH

    paths = _val_images(args.n_img)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    print(f"val 이미지 {len(paths)}장 · 길이 {LENGTHS} · 어댑터 양쪽 동일", flush=True)

    rows = [measure(k, MODEL, paths, prompt) for k in ("4bit", "bf16")]
    a, b = rows[0], rows[1]
    n_eval = 12461
    rep = {
        "무엇을_쟀나": "같은 모델·같은 어댑터·같은 프롬프트에서 추론 dtype 만 바꾼 생성 단가",
        "model": MODEL, "n_img": len(paths),
        "측정": rows,
        "비교": {
            "초당_토큰_배": round(b["초당_토큰"] / a["초당_토큰"], 3),
            "선행_배": round(b["선행_s"] / a["선행_s"], 3) if a["선행_s"] > 0 else None,
            "가중치_GB_배": round(b["가중치_GB"] / a["가중치_GB"], 3),
        },
        "평가환산_5모델_h": {
            f"{k['dtype']}_L{L}": round((k["선행_s"] + L / k["초당_토큰"]) * n_eval * 5 / 3600, 1)
            for k in rows for L in (64, 128)
        },
        "주의": "학습은 4bit 이 필요하다(기울기·옵티마이저). 추론 dtype 을 바꾸면 학습과 추론의 "
              "구성이 갈리므로 다섯 조건에 같은 dtype 을 쓰고 고정 설정으로 기록해야 한다.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n비교: {rep['비교']}\n평가환산: {rep['평가환산_5모델_h']}\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
