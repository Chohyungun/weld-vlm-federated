"""합성 입력으로 재는 장비 단가 — 결정표 7-4-1 의 ①(합성 이미지 · 합성 타깃 · 고정 길이).

예산표(`02_예산표_C.md`) §1 의 단가는 5060 Ti 에서 `cost_4b.py` · `infer_dtype.py` 로 쟀다. 그 둘은 파일럿 페어와
`val` 이미지를 연다 — 실제 자료다. 이 스크립트는 **실제 자료를 열지 않고** 같은 형상을 만든다.

- 이미지: 1280×720 합성(밝기 기울기 · 잡음 · 가로 띠 · 상자 자리의 어두운 얼룩). 실제 이미지가 전부 1280×720 이라
  프로세서를 거친 이미지 토큰 수가 같다(880).
- 타깃: 결함 수 k 의 합성 골격을 학습 경로의 `build_target` 으로 JSON 화한다. 유형 코드는 사상표(`configs/label_map.yaml`)에서
  읽고, 조항 식별자는 합성 문자열이다.
- 학습: `cost_4b.measure_train` 과 같은 부품(`_load_model` · `_encode` · 토큰 총합 손실 · 창 재정규화 · AdamW, micro 1 / accum 32)과 같은 셈법이다.
- 생성: 길이를 고정한다(`min_new_tokens = max_new_tokens`). 내용이 시간을 바꾸지 않는 범위는 **이 고정 길이 프로브**에 한정된다 —
  본실험의 자유 생성은 종료 시점이 입력에 따라 갈리므로 이 값이 그 시간을 보증하지 않는다(예산표 §13-2).

    uv run python scripts/probe/synth_rate.py --part train --work <합성 폴더> --out <JSON>
    uv run python scripts/probe/synth_rate.py --part gen   --work <합성 폴더> --out <JSON>

산출은 배타 생성이다 — 있으면 덮지 않는다. 절대 경로를 산출에 쓰지 않는다.

**합성 폴더(`--work`)는 이 스크립트 전용이다.** 없거나 비어 있으면 표지(`SYNTH_INPUT.json` — 합성 규칙)를 쓰고 시작한다.
비어 있지 않으면 표지가 지금 규칙과 같고 나머지가 전부 `syn_kKK_SS.png` 꼴의 일반 파일일 때만 쓴다 — 아니면 모델을 올리기 전에
거부한다(종료 코드 2). 같은 이름의 파일을 다시 쓸 때는 그 자리의 합성 그림을 다시 그려 화소가 같은지 맞댄다 — 다르면 거부한다.
그래서 임의의 기존 파일을 합성 입력으로 읽지 않는다. 산출에는 쓴 그림마다의 sha256 · 그 묶음의 해시 · 행(타깃)의 해시와,
모델 · 판 · 프로세서 설정 · 프롬프트 해시 · **실제로 적재된 판**(모델 설정의 `_commit_hash`)을 싣는다. 적재된 판이 설정의 판과
다르면 계측하지 않고 멈춘다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

import torch

IMG_W, IMG_H = 1280, 720
GEN_SEED = 20261005                  # 합성 입력 전용. 학습 시드와 무관하다

#: 학습 대표 조건 — 결함 수 k 의 비중. 평균 1.73 은 본실험 학습 분할의 평균 결함 1.735(11번 §3-1-2)에 맞춘 것이다.
#: 실제 분포의 모양은 쓰지 않았다(매니페스트를 열지 않는다) — 이 비중은 평균만 맞춘 합성 혼합이다.
MIX = {0: 30, 1: 25, 2: 20, 3: 12, 4: 8, 8: 5}           # 백분율(정수 — 최대 잔여 몫이 정수 무게를 받는다)
POINTS = (0, 2, 8)                   # 예산표 §13-1 의 세 점
WORST_K = 14                         # 5060 Ti 프로브의 최장 타깃 단발이 결함 14개였다

#: 생성 곡선 — `cost_4b.GEN_CURVE` 와 같은 (길이, 장수)
GEN_CURVE = [(1, 6), (32, 6), (64, 6), (128, 6), (256, 4), (512, 3)]
PLAN_LENGTHS, PLAN_N = (1, 65), 8    # 예산표 §13-1 의 2 — L=1 과 L=65, 8장
DTYPE_LENGTHS, DTYPE_N = (1, 64), 6  # `infer_dtype.py` 와 같다

WORK_MARK = "SYNTH_INPUT.json"
#: 합성 폴더의 규칙. 이 값이 하나라도 다르면 그 폴더의 그림은 지금 규칙으로 그린 것이 아니다.
WORK_RULE = {"kind": "synth_rate_work", "rule": "synth_v1", "gen_seed": GEN_SEED, "image": [IMG_W, IMG_H],
             "name_form": "syn_kKK_SS.png"}
WORK_NAME = re.compile(r"syn_k\d{2}_\d{2}\.png")

#: 이 실행에서 쓴 합성 그림 — 이름 → 파일 바이트의 sha256. 산출에 싣는다(경로는 싣지 않는다).
USED_IMAGES: dict[str, str] = {}


class SynthWorkRefused(RuntimeError):
    """합성 폴더나 적재 판이 계측 조건을 만족하지 않는다 — 모델을 올리기 전 또는 계측 전에 멈춘다."""

    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


def prepare_work(work: Path) -> None:
    """합성 폴더를 이 스크립트 전용으로 연다 — 없거나 비었으면 표지를 쓰고, 있으면 표지 · 이름 꼴을 맞댄다."""
    if work.is_symlink() or (work.exists() and not work.is_dir()):
        raise SynthWorkRefused("work_not_dir", f"합성 폴더가 일반 폴더가 아니다: {work.name}")
    work.mkdir(parents=True, exist_ok=True)
    entries = sorted(work.iterdir(), key=lambda q: q.name)
    mark = work / WORK_MARK
    if not entries:
        with mark.open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(WORK_RULE, ensure_ascii=False, sort_keys=True) + "\n")
        return
    if not mark.is_file() or mark.is_symlink():
        raise SynthWorkRefused("work_not_dedicated", f"비어 있지 않은 폴더에 합성 표지({WORK_MARK})가 없다 — 전용 빈 폴더에서 시작한다")
    try:
        got = json.loads(mark.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SynthWorkRefused("work_mark_bad", f"합성 표지를 읽지 못했다: {exc}") from exc
    if got != WORK_RULE:
        raise SynthWorkRefused("work_rule_mismatch", f"합성 표지의 규칙이 지금 규칙과 다르다: {got}")
    for q in entries:
        if q.name == WORK_MARK:
            continue
        if q.is_symlink() or not q.is_file() or not WORK_NAME.fullmatch(q.name):
            raise SynthWorkRefused("work_foreign_entry", f"합성 폴더에 이 스크립트가 쓰지 않는 항목이 있다: {q.name}")


def _sha_file(q: Path) -> str:
    return hashlib.sha256(q.read_bytes()).hexdigest()


def rows_sha256(rows: list[dict]) -> str:
    """행(합성 타깃)의 해시 — 그림은 경로가 아니라 이름과 파일 해시로 넣는다(폴더 위치가 해시를 바꾸지 않는다)."""
    canon = [{"image_id": r["image_id"], "image": Path(r["image_path"]).name, "image_sha256": USED_IMAGES[Path(r["image_path"]).name],
              "skeleton": r["skeleton"]} for r in rows]
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def images_sha256() -> str:
    lines = "".join(f"{h}  {n}\n" for n, h in sorted(USED_IMAGES.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def loaded_identity(model, proc, cfg) -> dict:
    """실제로 적재된 판. 모델 설정의 `_commit_hash` 가 설정의 판과 다르면 멈춘다 — 없으면 '모름'으로 싣는다."""
    conf = getattr(model, "config", None)
    got = getattr(conf, "_commit_hash", None)
    tok = getattr(proc, "tokenizer", None)
    proc_hash = (getattr(tok, "init_kwargs", None) or {}).get("_commit_hash") if tok is not None else None
    if got is not None and cfg.model_revision is not None and got != cfg.model_revision:
        raise SynthWorkRefused("loaded_revision_mismatch", f"적재된 모델 판 {got} ≠ 설정의 판 {cfg.model_revision}")
    if proc_hash is not None and cfg.model_revision is not None and proc_hash != cfg.model_revision:
        raise SynthWorkRefused("loaded_revision_mismatch", f"적재된 프로세서 판 {proc_hash} ≠ 설정의 판 {cfg.model_revision}")
    return {"model_commit_hash": got, "processor_commit_hash": proc_hash,
            "model_matches_config": None if got is None else got == cfg.model_revision,
            "processor_matches_config": None if proc_hash is None else proc_hash == cfg.model_revision}


def _env() -> dict:
    import bitsandbytes
    import transformers

    return {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__, "python": platform.python_version()}


def _codes() -> list[str]:
    import yaml

    lm = yaml.safe_load((REPO / "configs/label_map.yaml").read_text(encoding="utf-8"))
    keys = sorted(set(lm["sources"]["aihub71761"]["mapping"].values()))
    return [str(lm["defect_types"][k]["iso_code"]) for k in keys]


def _boxes(rng: random.Random, k: int) -> list[list[int]]:
    out = []
    for _ in range(k):
        w, h = rng.randint(20, 90), rng.randint(10, 40)
        x1, y1 = rng.randint(0, IMG_W - w - 1), rng.randint(290, 430 - h)
        out.append([x1, y1, x1 + w, y1 + h])
    return out


def _draw(seed: int, boxes: list[list[int]]):
    import numpy as np
    from PIL import Image

    g = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, IMG_W, dtype=np.float32)[None, :]
    a = np.full((IMG_H, IMG_W), 110.0, dtype=np.float32) + 30.0 * x
    a[280:440, :] += 40.0                                     # 용접 띠 자리
    a += g.normal(0.0, 8.0, size=a.shape).astype(np.float32)
    for x1, y1, x2, y2 in boxes:
        a[y1:y2, x1:x2] -= 50.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([a, a, a], axis=-1), mode="RGB")


def make_rows(work: Path, ks: list[int], *, tag: str, n_images_per_k: int = 16) -> list[dict]:
    """결함 수 열 `ks` 대로 합성 행을 만든다. 이미지는 k 마다 `n_images_per_k` 장을 그려 돌려 쓴다(장마다 파일에서 다시 연다)."""
    codes = _codes()
    rng = random.Random(f"{GEN_SEED}|{tag}")
    prepare_work(work)
    rows = []
    for j, k in enumerate(ks):
        slot = j % n_images_per_k
        img_rng = random.Random(f"{GEN_SEED}|img|{k}|{slot}")
        boxes = _boxes(img_rng, k)
        path = work / f"syn_k{k:02d}_{slot:02d}.png"
        drawn = _draw(GEN_SEED + 1000 * k + slot, boxes)
        if path.is_symlink():
            raise SynthWorkRefused("work_foreign_entry", f"합성 그림 자리가 링크다: {path.name}")
        if path.exists():
            import numpy as np
            from PIL import Image

            with Image.open(path) as im:
                same = im.mode == "RGB" and im.size == (IMG_W, IMG_H) and np.array_equal(np.asarray(im), np.asarray(drawn))
            if not same:
                raise SynthWorkRefused("synth_content_mismatch", f"같은 이름의 파일이 지금 규칙의 합성 그림과 다르다: {path.name}")
        else:
            drawn.save(path)
        USED_IMAGES[path.name] = _sha_file(path)
        types = [codes[rng.randrange(len(codes))] for _ in range(k)]
        rows.append({"image_id": f"syn_{tag}_{j}", "image_path": str(path),
                     "skeleton": {"defects": [{"type": t, "bbox_px": b} for t, b in zip(types, boxes)],
                                  "verdict": None, "clauses": sorted({f"SYN-{t}" for t in types})}})
    return rows


def _mix_ks(n: int) -> list[int]:
    from vlm.rehearsal_lists import largest_remainder

    share = largest_remainder(n, {str(k): w for k, w in MIX.items()})
    ks = [int(k) for k, m in share.items() for _ in range(m)]
    random.Random(f"{GEN_SEED}|mix").shuffle(ks)
    return ks


def _prompt_bytes() -> bytes:
    from vlm.pilot_vlm import PROMPT_PATH

    return (REPO / PROMPT_PATH).read_bytes()


def _cfg():
    from vlm.uni_config import load_uni_config

    return load_uni_config()


def _train_condition(model, proc, prompt, opt, params, rows, *, warmup_steps: int, label: str) -> dict:
    from vlm.loss_norm import TokenAccumulator, rescale_grads_, supervised_ce_sum
    from vlm.pilot_vlm import GRAD_ACCUM, _encode

    model.train()
    torch.cuda.reset_peak_memory_stats()
    acc = TokenAccumulator()
    per_step, per_sample = [], []
    tok_in_step, prompt_lens = 0, []
    t_step = time.perf_counter()
    for j, row in enumerate(rows):
        t_s = time.perf_counter()
        enc, labels, prompt_len = _encode(proc, row, prompt)
        prompt_lens.append(int(prompt_len))
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
        labels = labels.to("cuda")
        n_keep = int(labels.shape[1] - prompt_len + 1)
        out = model(**enc, logits_to_keep=n_keep)
        ce, n_tok = supervised_ce_sum(out.logits[:, :-1], labels[:, -(n_keep - 1):])
        ce.backward()
        acc.add(n_tok)
        tok_in_step += n_tok
        torch.cuda.synchronize()
        per_sample.append(time.perf_counter() - t_s)
        if (j + 1) % GRAD_ACCUM == 0:
            rescale_grads_(params, acc.close())
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            i = (j + 1) // GRAD_ACCUM
            per_step.append({"step": i, "wall_s": round(time.perf_counter() - t_step, 3), "supervised_tokens": tok_in_step,
                             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3), "warmup": i <= warmup_steps})
            print(f"    [{label}] step {i}/{len(rows) // GRAD_ACCUM} {per_step[-1]['wall_s']:.1f}s tok {tok_in_step}"
                  f"{' (워밍업)' if i <= warmup_steps else ''}", flush=True)
            tok_in_step = 0
            t_step = time.perf_counter()
    steady = [s for s in per_step if not s["warmup"]]
    sps = [s["wall_s"] / GRAD_ACCUM for s in steady]
    return {"label": label, "rows": len(rows), "rows_sha256": rows_sha256(rows), "warmup_steps": warmup_steps, "steps": per_step,
            "prompt_tokens": sorted(set(prompt_lens)),
            "steady": {"n_steps": len(steady), "step_s_mean": round(statistics.fmean(s["wall_s"] for s in steady), 3),
                       "per_sample_s_median": round(statistics.median(sps), 4), "per_sample_s_mean": round(statistics.fmean(sps), 4),
                       "per_sample_s_minmax": [round(min(sps), 4), round(max(sps), 4)],
                       "supervised_tokens_per_row": round(statistics.fmean(s["supervised_tokens"] for s in steady) / GRAD_ACCUM, 2),
                       "peak_vram_gb": max(s["peak_vram_gb"] for s in steady)}}


def measure_train(work: Path) -> dict:
    from vlm.loss_norm import supervised_ce_sum
    from vlm.pilot_vlm import GRAD_ACCUM, LR, PROMPT_PATH, _encode, _load_model

    cfg = _cfg()
    t0 = time.perf_counter()
    model, proc = _load_model(cfg.model_id, revision=cfg.model_revision, processor_kwargs=dict(cfg.processor_kwargs or {}))
    load_s = round(time.perf_counter() - t0, 3)
    loaded = loaded_identity(model, proc, cfg)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    conds = [_train_condition(model, proc, prompt, opt, params, make_rows(work, _mix_ks(12 * GRAD_ACCUM), tag="mix"),
                              warmup_steps=2, label="mix")]
    for k in POINTS:
        conds.append(_train_condition(model, proc, prompt, opt, params, make_rows(work, [k] * (5 * GRAD_ACCUM), tag=f"k{k}"),
                                      warmup_steps=1, label=f"k{k}"))
    worst = make_rows(work, [WORST_K], tag="worst")[0]
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    enc, labels, prompt_len = _encode(proc, worst, prompt)
    enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
    labels = labels.to("cuda")
    n_keep = int(labels.shape[1] - prompt_len + 1)
    out = model(**enc, logits_to_keep=n_keep)
    ce, n_tok = supervised_ce_sum(out.logits[:, :-1], labels[:, -(n_keep - 1):])
    ce.backward()
    torch.cuda.synchronize()
    worst_rec = {"n_defects": WORST_K, "supervised_tokens": int(n_tok), "wall_s": round(time.perf_counter() - t1, 3),
                 "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    opt.zero_grad()
    del model, opt
    torch.cuda.empty_cache()
    worst_rec["rows_sha256"] = rows_sha256([worst])
    return {"load_s_in_process": load_s, "loaded": loaded, "mix": {str(k): w for k, w in MIX.items()}, "conditions": conds,
            "worst_single": worst_rec}


def _gen_images(work: Path, n: int) -> list:
    from PIL import Image

    rows = make_rows(work, [2] * n, tag="gen", n_images_per_k=n)
    return [Image.open(r["image_path"]).convert("RGB") for r in rows]


def _fixed(model, proc, prompt, imgs, length: int) -> dict:
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig(max_new_tokens=length, min_new_tokens=length, do_sample=False, num_beams=1)
    lat, prompts = [], []
    for img in imgs:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(int(enc["input_ids"].shape[1]))
        enc = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        assert int(out.shape[1]) - prompts[-1] == length, "길이 고정이 듣지 않았다"
    print(f"    고정 {length}토큰 {len(imgs)}장 중앙 {statistics.median(lat):.3f}s", flush=True)
    return {"length": length, "n": len(imgs), "latency_s_median": round(statistics.median(lat), 4),
            "latency_s_minmax": [round(min(lat), 4), round(max(lat), 4)], "prompt_tokens": sorted(set(prompts))}


def _two_point(a: dict, b: dict) -> dict:
    tok_s = (b["length"] - a["length"]) / (b["latency_s_median"] - a["latency_s_median"])
    return {"tok_per_s": round(tok_s, 3), "prefill_s": round(a["latency_s_median"] - 1 / tok_s, 4)}


def _load_dtype(kind: str, cfg):
    """`infer_dtype.load` 와 같은 구성 — 판 · 프로세서 설정만 본실험 설정에서 준다."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    from fl.seeding import seeded, shared_init_seed
    from vlm.pilot_vlm import DEFAULT_INIT_SEED, TARGET_SUFFIXES

    proc = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.model_revision, **dict(cfg.processor_kwargs or {}))
    if kind == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                                            quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                                            dtype=torch.bfloat16, device_map={"": 0})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=TARGET_SUFFIXES)
    with seeded(shared_init_seed(DEFAULT_INIT_SEED)):
        model = get_peft_model(model, lora)
    model.eval()
    return model, proc


def measure_gen(work: Path) -> dict:
    from vlm.pilot_vlm import PROMPT_PATH, _load_model

    cfg = _cfg()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    imgs = _gen_images(work, max(PLAN_N, max(n for _, n in GEN_CURVE)))
    # (1) `cost_4b.measure_gen` 의 곡선 — 학습 적재기(4bit · 어댑터)로 연다
    model, proc = _load_model(cfg.model_id, revision=cfg.model_revision, processor_kwargs=dict(cfg.processor_kwargs or {}))
    loaded = {"curve": loaded_identity(model, proc, cfg)}
    model.eval()
    _fixed(model, proc, prompt, imgs[:1], 8)                  # 첫 커널 · 캐시 데우기 — 기록하지 않는다
    torch.cuda.reset_peak_memory_stats()
    curve = [_fixed(model, proc, prompt, imgs[:n], L) for L, n in GEN_CURVE]
    plan = [_fixed(model, proc, prompt, imgs[:PLAN_N], L) for L in PLAN_LENGTHS]
    curve_peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    del model
    torch.cuda.empty_cache()
    # (2) `infer_dtype.py` 의 통제 비교 — 같은 6장, 4bit 대 bf16
    dtype = []
    for kind in ("4bit", "bf16"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, proc = _load_dtype(kind, cfg)
        load_s = round(time.perf_counter() - t0, 3)
        loaded[kind] = loaded_identity(model, proc, cfg)
        weights_gb = round(torch.cuda.memory_allocated() / 1e9, 3)
        _fixed(model, proc, prompt, imgs[:1], 8)
        pts = [_fixed(model, proc, prompt, imgs[:DTYPE_N], L) for L in DTYPE_LENGTHS]
        dtype.append({"dtype": kind, "load_s": load_s, "weights_gb": weights_gb, "points": pts, **_two_point(*pts),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})
        del model
        torch.cuda.empty_cache()
    return {"loaded": loaded, "curve": curve, "curve_peak_gb": curve_peak, "plan_two_point": {"points": plan, **_two_point(*plan)},
            "dtype": dtype}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["tokens", "train", "gen"], required=True)
    ap.add_argument("--work", type=Path, required=True, help="합성 이미지를 그릴 폴더")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        print(f"[out_exists] {a.out.name} — 덮지 않는다", file=sys.stderr)
        return 2
    try:
        prepare_work(a.work)                                      # 모델을 올리기 전에 폴더부터 맞댄다
    except SynthWorkRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    cfg = _cfg()
    rep: dict = {"kind": f"synth_rate_{a.part}", "env": _env(), "gen_seed": GEN_SEED,
                 "inputs": "합성 이미지 1280x720 · 합성 타깃 — 실제 자료를 열지 않는다",
                 "model": {"model_id": cfg.model_id, "revision": cfg.model_revision,
                           "processor_kwargs": dict(cfg.processor_kwargs or {}),
                           "prompt_sha256": hashlib.sha256(_prompt_bytes()).hexdigest()}}
    t0 = time.perf_counter()
    if a.part == "tokens":
        from transformers import AutoProcessor

        from vlm.pilot_vlm import PROMPT_PATH, _encode

        proc = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.model_revision, **dict(cfg.processor_kwargs or {}))
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        per_k = {}
        for k in sorted(set(MIX) | set(POINTS) | {WORST_K}):
            rows = make_rows(a.work, [k] * 4, tag=f"tok{k}")
            n = []
            for r in rows:
                full, labels, pl = _encode(proc, r, prompt)
                n.append({"supervised": int((labels != -100).sum()), "prompt": int(pl), "total": int(full["input_ids"].shape[1])})
            per_k[str(k)] = n
        rep["per_k"] = per_k
        rep["loaded"] = loaded_identity(None, proc, cfg)
        rep["mix_mean_supervised"] = round(sum(w * statistics.fmean(x["supervised"] for x in per_k[str(k)]) for k, w in MIX.items()) / sum(MIX.values()), 2)
    elif a.part == "train":
        rep["train"] = measure_train(a.work)
    else:
        rep["gen"] = measure_gen(a.work)
    rep["synth_inputs"] = {"work_rule": WORK_RULE, "images": dict(sorted(USED_IMAGES.items())), "images_sha256": images_sha256()}
    rep["wall_s"] = round(time.perf_counter() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")
    print(f"총 {rep['wall_s']}s → {a.out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
