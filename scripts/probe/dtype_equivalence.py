"""추론 dtype 을 바꾸면 **모델이 달라지는가** — 출력 토큰 수열을 그대로 맞댄다.

## 왜 속도만의 문제가 아닌가

QLoRA 는 **4bit 로 양자화된 기저** 위에서 어댑터를 학습한다. 그 어댑터를 bf16 기저에 얹으면
기저 가중치가 다르다(양자화 오차만큼). 그래서 `infer_dtype=bf16` 은 속도 선택이 아니라
**다른 모델을 채점하는 선택**일 수 있다. 속도(11번 §3-2)와 따로 재야 한다.

## 재는 방법

greedy(`do_sample=False`, 빔 1)는 결정적이다. 같은 어댑터·같은 프롬프트·같은 이미지에서
두 dtype 의 **생성 토큰 id 수열**을 맞대면 답이 바로 나온다. 같으면 속도 선택이고, 다르면
모델이 바뀌는 선택이다.

첫 불일치 위치를 함께 남긴다 — 초반에 갈리면 좌표·코드가 바뀌는 것이고, 끝에서 갈리면
서술의 꼬리만 다른 것이다. 같은 값이 아니라 **어디서 갈리는가**가 판정에 들어간다.

## 이 프로브는 경합에 무관하다

시간을 재지 않는다. 다른 트랙이 시험을 돌려도 토큰 수열은 달라지지 않으므로 단독 창이 필요 없다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

OUT = Path("outputs/probe_c/dtype_equivalence.json")
MAX_NEW = 128
N_IMG = 8


def load(kind: str, model_id: str, adapter: Path | None):
    """4bit 또는 bf16 기저에 **같은 어댑터**를 얹는다."""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    from detection import serialize
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

    loaded: dict = {"adapter": None}
    if adapter is not None:
        import numpy as np
        from peft import get_peft_model_state_dict

        ref = get_peft_model_state_dict(model)
        keys = serialize.canonical_keys(ref)
        npz = np.load(adapter)
        arrays = [npz[k] for k in npz.files]
        sd = serialize.ndarrays_to_state_dict(arrays, keys, ref)

        # 로드 **전** lora_B 의 노름. peft 는 lora_B 를 0 으로 놓으므로 0 이어야 한다.
        before = _lora_b_norm(model)
        res = set_peft_model_state_dict(model, sd)
        after = _lora_b_norm(model)

        # **조용한 실패를 막는다.** 키가 안 맞아 아무것도 얹히지 않으면 두 dtype 이 똑같이
        # 초기 어댑터가 되고, 출력이 같게 나와 "dtype 이 모델을 바꾸지 않는다" 는 거짓 결론이 된다.
        #
        # `missing_keys` 를 그대로 보면 안 된다. 이 값은 "얹은 state dict 에 없는 **모델 전체
        # 키**" 라서, LoRA 만 담은 사전을 얹으면 비전 인코더·기저 가중치가 전부 들어온다.
        # 처음에 그대로 검사했다가 정상 적재를 막았다 — 가드가 정상 입력을 막은 실제 사례다.
        # 실질 검증은 아래 `lora_B` 노름이 0 에서 양수로 바뀌는 것이고, 키 쪽은 **LoRA 키에
        # 한정한 누락**과 `unexpected_keys` 만 본다.
        missing = [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
        unexpected = list(getattr(res, "unexpected_keys", []) or [])
        if missing or unexpected:
            raise RuntimeError(
                f"어댑터의 LoRA 키가 맞지 않는다: 없음 {missing[:3]} · 남음 {unexpected[:3]}")
        if not (after > 0 and after > before):
            raise RuntimeError(
                f"어댑터가 실제로 얹히지 않았다 — lora_B 노름 {before:.6f} -> {after:.6f}. "
                "0 이면 초기 상태이므로 비교가 무의미하다"
            )
        # 적재 **뒤** 어댑터를 다시 뽑아 키별로 대조한다. 입력 배열의 노름(`param_l2`)은 같은
        # 배열의 반올림 값이라 **두 기저에 키별로 같은 값이 들어갔음을 증명하지 않는다.**
        # `lora_B` 노름이 양수라는 것도 "지정한 체크포인트를 읽었다" 는 출처 증명이 아니다.
        back = get_peft_model_state_dict(model)
        digest = _sd_digest(back)
        loaded.update(adapter=str(adapter), lora_b_norm_before=round(before, 6),
                      lora_b_norm_after=round(after, 6),
                      param_l2=round(float(serialize.params_l2_norm(arrays)), 6),
                      n_keys=len(back), reextracted_digest=digest)
    model.eval()
    return model, proc, loaded


def _sd_digest(sd) -> str:
    """키 이름·순서·값까지 담은 지문. 두 기저의 적재 결과를 이것으로 맞댄다.

    키를 **정렬하지 않는다** — 이름과 순서가 계약의 일부다. 값은 float64 로 올려 바이트를 낸다.
    """
    import hashlib

    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(v.detach().cpu().to(torch.float64).numpy().tobytes())
    return h.hexdigest()[:16]


def _lora_b_norm(model) -> float:
    # `lora_B` 파라미터 전체의 L2. 초기 어댑터는 0 이다(peft 가 0 으로 놓는다).
    tot = 0.0
    for n, prm in model.named_parameters():
        if "lora_B" in n:
            tot += float(prm.detach().float().pow(2).sum())
    return tot ** 0.5


def generate(model, proc, prompt: str, paths: list[str]) -> list[dict]:
    from PIL import Image
    from transformers import GenerationConfig

    cfg = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False, num_beams=1)
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                       return_tensors="pt", add_generation_prompt=True,
                                       enable_thinking=False)
        n_prompt = int(enc["input_ids"].shape[1])
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
        with torch.no_grad():
            ids = model.generate(**enc, generation_config=cfg)
        new = ids[0, n_prompt:].tolist()
        out.append({"path": p, "n_prompt": n_prompt, "token_ids": new,
                    "text": proc.batch_decode([new], skip_special_tokens=True)[0]})
    return out


def compare(a: list[dict], b: list[dict]) -> dict:
    rows, same = [], 0
    for x, y in zip(a, b):
        eq = x["token_ids"] == y["token_ids"]
        same += eq
        first = None
        if not eq:
            for i, (u, v) in enumerate(zip(x["token_ids"], y["token_ids"])):
                if u != v:
                    first = i
                    break
            if first is None:
                first = min(len(x["token_ids"]), len(y["token_ids"]))
        rows.append({
            "path": Path(x["path"]).name, "같은가": eq,
            "n_4bit": len(x["token_ids"]), "n_bf16": len(y["token_ids"]),
            "첫_불일치_위치": first,
            "공통_접두_비율": (round(first / max(len(x["token_ids"]), 1), 3)
                        if first is not None else 1.0),
            # **자르지 않는다.** 160자로 자른 판에서는 갈린 장의 JSON 이 잘려 필드 파싱이
            # 대부분 실패했다(8장 중 8장, 6장 중 4장). 완성 출력의 필드를 대조하지 못하면
            # "무엇이 달랐나" 에 답할 수 없다 — 첫 불일치 위치만으로 좌표·코드를 구별하면 안 된다.
            "text_4bit": x["text"] if not eq else None,
            "text_bf16": y["text"] if not eq else None,
        })
    return {"장수": len(rows), "토큰수열_같은_장수": same, "장별": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="예: Qwen/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default=None, help="npz 경로. 생략하면 초기 어댑터(lora_B=0)")
    ap.add_argument("--n-img", type=int, default=N_IMG)
    ap.add_argument("--tag", required=True, help="산출물 안의 구분 이름")
    args = ap.parse_args()

    from scripts.probe.cost_4b import _val_images
    from vlm.pilot_vlm import PROMPT_PATH

    paths = _val_images(args.n_img)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    adapter = Path(args.adapter) if args.adapter else None
    print(f"{args.tag}: {args.model} · 어댑터 {adapter or '초기(lora_B=0)'} · {len(paths)}장 · "
          f"한도 {MAX_NEW}", flush=True)

    runs, infos = {}, {}
    for kind in ("4bit", "bf16"):
        model, proc, info = load(kind, args.model, adapter)
        infos[kind] = info
        if adapter is not None:
            print(f"  [{kind}] 어댑터 확인 · lora_B 노름 {info['lora_b_norm_before']} -> "
                  f"{info['lora_b_norm_after']} · 적재 param_l2 {info['param_l2']}", flush=True)
        runs[kind] = generate(model, proc, prompt, paths)
        print(f"  [{kind}] 생성 완료 · 토큰 수 {[len(r['token_ids']) for r in runs[kind]]}", flush=True)
        del model
        torch.cuda.empty_cache()

    if adapter is not None:
        a, b = infos["4bit"], infos["bf16"]
        # 입력 배열은 같아야 한다(같은 파일을 읽었다).
        assert a["param_l2"] == b["param_l2"], "두 실행이 다른 어댑터 파일을 읽었다"
        assert a["n_keys"] == b["n_keys"], "적재된 어댑터 키 수가 다르다"
        # 적재 뒤 재추출 지문. **여기서 갈리면 같은 어댑터가 아니다.** 다만 기저 dtype 이
        # 다르면 어댑터 텐서 dtype 도 달라질 수 있어, 갈렸을 때 곧 오류로 단정하지 않고 적는다.
        same = a["reextracted_digest"] == b["reextracted_digest"]
        print(f"  적재 뒤 재추출 지문: 4bit {a['reextracted_digest']} · "
              f"bf16 {b['reextracted_digest']} · 같은가 {same}", flush=True)

    cmp = compare(runs["4bit"], runs["bf16"])
    rep = {
        "무엇을_쟀나": "같은 어댑터를 4bit 기저와 bf16 기저에 얹고 greedy 생성한 토큰 수열의 일치",
        "model": args.model, "어댑터": str(adapter) if adapter else "초기(lora_B=0)",
        "한도": MAX_NEW, "비교": cmp, "어댑터_적재_확인": infos,
        "적재_동일성_주의": ("param_l2 는 입력 배열의 노름이라 키별 동일 주입의 증명이 아니다. "
                      "적재 뒤 재추출 지문(reextracted_digest)이 그 역할을 한다. "
                      "lora_B 노름이 양수라는 것은 출처 증명이 아니다."),
        "한계": ("초기 어댑터는 lora_B=0 이라 어댑터 기여가 0 이다 — 기저 차이만 본 것이고 "
               "미세조정 뒤에는 달라질 수 있다." if adapter is None else
               "파일럿 어댑터다(논문 인용 대상 아님). 본실험 어댑터에서 같다는 보증은 아니다."),
    }
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[args.tag] = rep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  토큰 수열이 같은 장: {cmp['토큰수열_같은_장수']}/{cmp['장수']}")
    for r in cmp["장별"]:
        if not r["같은가"]:
            print(f"    {r['path']}: 첫 불일치 {r['첫_불일치_위치']} "
                  f"(공통 접두 {r['공통_접두_비율']*100:.0f}%) · 길이 {r['n_4bit']} 대 {r['n_bf16']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
