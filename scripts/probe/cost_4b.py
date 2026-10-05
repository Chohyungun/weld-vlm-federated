"""단가 실측 — 4B · 현행 프롬프트 · 템플릿 고정안 (02번 I-6 · m-3).

## 왜 다시 재는가

본실험 예산의 유일한 단가 근거가 **48행 · 2스텝 프로브**였다(`probe2_vlm_scale.json`).
02번 I-6 이 확인한 문제가 셋이다.

1. 그 48행은 페어 앞머리를 그대로 자른 것이어서 **결함 0개가 47행**이었다. 감독 토큰이
   행당 24 개다. 본실험 학습 분할의 평균 결함 수는 1.735 개이고 토큰은 80 개 안팎이다.
2. 워밍업이 0 스텝이라 모델 적재·첫 커널 컴파일이 2스텝에 그대로 실렸다. 같은 구성
   (0.8B · 판정 11 이전)의 프로브값 3.846초와 파일럿 전량 실측 2.696초가 1.43배 어긋난다.
3. 좌표 규약·프롬프트가 그 뒤 바뀌었고(정규화 → `ABS_ORIG`, v1 → v2), 템플릿 고정안도
   그때는 없었다.

그래서 점추정 하나로 안 A·D 를 가를 수 없다. 이 스크립트는 **범위**를 낸다.

## 무엇을 어떻게 재는가

**(가) 학습 정상상태 단가.** 학습 경로와 같은 부품을 조립한다 — `_load_model`(4bit +
LoRA + 체크포인팅), `_encode`, 토큰 총합 분모 손실, 창 토큰 총합 재정규화, AdamW.
`train_rounds` 를 그대로 부르지 않는 이유는 그쪽이 **스텝 단위 계측을 노출하지 않기**
때문이다(프로브 1의 한계가 그것이었다). 표본은 파일럿 페어에서 **결함 수 분포를 본실험
학습 분할에 맞춰** 층화 추출한다. 워밍업 스텝을 버리고 정상상태만 센다.

**(나) 생성 단가.** `val` 이미지로 잰다. 지연과 생성 토큰 수만 기록하고 **예측 내용은
채점하지 않는다**(평가셋은 열지 않는다). 한도 두 가지와 로그확률 저장 on/off 를 비교한다.
어댑터가 없으므로 **사전학습 모델**의 길이 분포다 — 미세조정 뒤에는 JSON 한 줄로 짧아질
것이 예상되지만 그것은 이 측정으로 알 수 없다. 토큰당 시간이 이 측정의 이전 가능한 값이다.

    uv run python scripts/probe/cost_4b.py --part both

산출: `outputs/probe_c/cost_4b.json`. 평가셋 이미지·정답은 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

#: 09-21 실행의 산출물. **지우지 않는다** — 11번 §3-1-2 의 수치가 이 파일에서 나왔다.
LEGACY_OUT = Path("outputs/probe_c/cost_4b.json")
#: 새 실행은 여기 아래 `{run_id}_{part}.json` 으로 쓴다. 고정 경로 하나에 쓰면 `--part gen`
#: 실행이 직전 `--part train` 산출물을 덮는다(실제로 그렇게 두 번 덮였다).
OUT_DIR = Path("outputs/probe_c/cost_4b")
MODEL = "Qwen/Qwen3.5-4B"
MANIFEST = Path("data/interim/manifest_v1/manifest.csv")

#: 학습 표본을 이 분포에 맞춘다. 본실험 학습 분할의 결함 수별 비율이다(매니페스트 집계).
#: 파일럿 페어에서 같은 비율로 뽑아 길이 분포를 본실험에 맞춘다.
SAMPLE_SEED = 20260921        # 계측 전용. 학습 시드(base_seed) 와 무관하다.

GEN_LIMITS = [512, 2368]      # 512 = 실무 후보 · 2368 = 83번 §11 잠정 권고
N_GEN = 20                    # 512 한도로 재는 장수
N_GEN_LONG = 10               # 2368 한도로 재는 장수(최악 시간이 커서 줄인다)

#: 생성 곡선 — (고정 길이, 장수). 길이를 고정해야 길이 분포와 무관한 값이 나온다.
#: 한 점만 재면 평가 시간 추정이 서지 않는다. 긴 길이는 장당 시간이 커서 장수를 줄인다.
GEN_CURVE: list[tuple[int, int]] = [(1, 6), (32, 6), (64, 6), (128, 6), (256, 4), (512, 3)]
CURVE_LOGPROB_LEN = 64        # 로그확률 저장 부담은 이 길이에서 켠·끈 것을 비교한다


# --------------------------------------------------------------------------
# (가) 학습 단가
# --------------------------------------------------------------------------

def _target_defect_mix() -> dict[int, float]:
    """본실험 학습 분할의 결함 수별 비율."""
    import csv
    from collections import Counter

    cnt: Counter[int] = Counter()
    with MANIFEST.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "train":
                cnt[int(row["n_defects"] or 0)] += 1
    n = sum(cnt.values())
    return {k: v / n for k, v in sorted(cnt.items())}


def _stratified_rows(n_want: int) -> tuple[list[dict], dict]:
    """파일럿 페어에서 결함 수 분포를 본실험에 맞춰 뽑는다.

    파일럿에 없는 결함 수 구간은 가장 가까운 구간으로 접고, 접은 사실을 기록한다.
    """
    from vlm.pilot_vlm import load_pairs

    rows = load_pairs("train")
    by_n: dict[int, list[dict]] = {}
    for r in rows:
        by_n.setdefault(len(r["skeleton"]["defects"]), []).append(r)
    mix = _target_defect_mix()
    rng = random.Random(SAMPLE_SEED)

    picked: list[dict] = []
    folded: dict[str, int] = {}
    for k, frac in mix.items():
        want = round(frac * n_want)
        if want <= 0:
            continue
        pool_key = k if k in by_n else min(by_n, key=lambda c: abs(c - k))
        if pool_key != k:
            folded[f"{k}->{pool_key}"] = folded.get(f"{k}->{pool_key}", 0) + want
        pool = by_n[pool_key]
        picked += [pool[rng.randrange(len(pool))] for _ in range(want)]
    while len(picked) < n_want:                      # 반올림 잔여
        picked.append(rows[rng.randrange(len(rows))])
    rng.shuffle(picked)
    picked = picked[:n_want]
    got = [len(r["skeleton"]["defects"]) for r in picked]
    return picked, {
        "목표_평균_결함수": round(sum(k * v for k, v in mix.items()), 4),
        "표본_평균_결함수": round(statistics.fmean(got), 4),
        "표본_결함수_분포": {str(k): got.count(k) for k in sorted(set(got))},
        "파일럿에_없어_접은_구간": folded,
        "추출_시드": SAMPLE_SEED,
    }


def measure_train(*, warmup_steps: int, measure_steps: int,
                  deadline_s: float | None = None) -> dict:
    """정상상태 학습 단가. 학습 경로와 같은 구성으로 조립한다."""
    from vlm.loss_norm import TokenAccumulator, rescale_grads_, supervised_ce_sum
    from vlm.pilot_vlm import GRAD_ACCUM, LR, PROMPT_PATH, _encode, _load_model

    total_steps = warmup_steps + measure_steps
    n_rows = total_steps * GRAD_ACCUM
    rows, mix = _stratified_rows(n_rows)
    print(f"  표본 {len(rows)}행 · 평균 결함 {mix['표본_평균_결함수']} "
          f"(목표 {mix['목표_평균_결함수']}) · {total_steps}스텝"
          f"(워밍업 {warmup_steps})", flush=True)

    model, proc = _load_model(MODEL)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    torch.cuda.reset_peak_memory_stats()

    acc = TokenAccumulator()
    t_begin = time.perf_counter()
    per_step: list[dict] = []
    per_sample_s: list[float] = []
    tok_in_step = 0
    t_step = time.perf_counter()
    for j, row in enumerate(rows):
        t_s = time.perf_counter()
        enc, labels, prompt_len = _encode(proc, row, prompt)
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
        labels = labels.to("cuda")
        n_keep = int(labels.shape[1] - prompt_len + 1)
        out = model(**enc, logits_to_keep=n_keep)
        logits = out.logits[:, :-1]
        tgt = labels[:, -(n_keep - 1):]
        ce, n_tok = supervised_ce_sum(logits, tgt)
        ce.backward()
        acc.add(n_tok)
        tok_in_step += n_tok
        torch.cuda.synchronize()
        per_sample_s.append(time.perf_counter() - t_s)
        if (j + 1) % GRAD_ACCUM == 0:
            rescale_grads_(params, acc.close())
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            step_i = (j + 1) // GRAD_ACCUM
            per_step.append({
                "step": step_i, "wall_s": round(time.perf_counter() - t_step, 3),
                "supervised_tokens": tok_in_step,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "warmup": step_i <= warmup_steps,
            })
            print(f"    step {step_i}/{total_steps} {per_step[-1]['wall_s']:.1f}s "
                  f"tok {tok_in_step} peak {per_step[-1]['peak_vram_gb']:.2f}GB"
                  f"{' (워밍업)' if step_i <= warmup_steps else ''}", flush=True)
            tok_in_step = 0
            t_step = time.perf_counter()
            # 벽시계 한도. 창을 넘기면 **잰 것만 들고 나간다** — 다 못 재는 것보다 낫다.
            # 워밍업 뒤 스텝이 둘은 있어야 정상상태라고 부를 수 있다.
            if (deadline_s is not None and time.perf_counter() - t_begin > deadline_s
                    and step_i >= warmup_steps + 2):
                print(f"    [한도] {deadline_s:.0f}s 를 넘겼다 — "
                      f"정상상태 {step_i - warmup_steps}스텝으로 멈춘다", flush=True)
                break

    steady = [s for s in per_step if not s["warmup"]]
    warm = [s for s in per_step if s["warmup"]]
    sec_per_sample = [s["wall_s"] / GRAD_ACCUM for s in steady]
    tail = per_sample_s[warmup_steps * GRAD_ACCUM:]

    # 최장 타깃의 피크 메모리 — 결함 수가 가장 많은 행 하나로 별도 측정한다.
    worst = max(rows, key=lambda r: len(r["skeleton"]["defects"]))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    enc, labels, prompt_len = _encode(proc, worst, prompt)
    enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
    labels = labels.to("cuda")
    n_keep = int(labels.shape[1] - prompt_len + 1)
    out = model(**enc, logits_to_keep=n_keep)
    ce, n_tok_worst = supervised_ce_sum(out.logits[:, :-1], labels[:, -(n_keep - 1):])
    ce.backward()
    torch.cuda.synchronize()
    worst_peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    worst_s = round(time.perf_counter() - t0, 3)
    opt.zero_grad()

    del model, opt
    torch.cuda.empty_cache()

    n_train = 44846
    return {
        "표본": mix,
        "스텝별": per_step,
        "워밍업_스텝_평균_s": round(statistics.fmean([s["wall_s"] for s in warm]), 2) if warm else None,
        "정상상태": {
            "n_steps": len(steady),
            "스텝_평균_s": round(statistics.fmean([s["wall_s"] for s in steady]), 2),
            "샘플당_s_평균": round(statistics.fmean(sec_per_sample), 4),
            "샘플당_s_중앙값": round(statistics.median(sec_per_sample), 4),
            "샘플당_s_최소최대": [round(min(sec_per_sample), 4), round(max(sec_per_sample), 4)],
            "샘플별_s_중앙값": round(statistics.median(tail), 4) if tail else None,
            "감독토큰_스텝평균": round(statistics.fmean([s["supervised_tokens"] for s in steady]), 1),
            "peak_vram_gb": max(s["peak_vram_gb"] for s in steady),
        },
        "최장타깃_단발": {"n_defects": len(worst["skeleton"]["defects"]),
                    "감독토큰": int(n_tok_worst), "wall_s": worst_s,
                    "peak_vram_gb": worst_peak},
        "환산": {
            "train_장수": n_train,
            "epoch당_시간_h_평균": round(statistics.fmean(sec_per_sample) * n_train / 3600, 2),
            "epoch당_시간_h_범위": [round(min(sec_per_sample) * n_train / 3600, 2),
                              round(max(sec_per_sample) * n_train / 3600, 2)],
        },
    }


# --------------------------------------------------------------------------
# (나) 생성 단가
# --------------------------------------------------------------------------

def _val_images(n: int) -> list[str]:
    """`val` 이미지 경로 표본. **eval 행은 읽지 않는다.**"""
    import csv

    paths = []
    with MANIFEST.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "val":
                paths.append(row["rel_path"])
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(paths)
    return paths[:n]


def measure_gen(*, n_gen: int, n_gen_long: int) -> dict:
    """생성 단가. 지연·생성 토큰 수만 남긴다 — 예측 내용은 저장·채점하지 않는다."""
    from PIL import Image
    from transformers import GenerationConfig

    from vlm.pilot_vlm import PROMPT_PATH, _load_model

    model, proc = _load_model(MODEL)
    model.eval()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    paths = _val_images(max(n_gen, n_gen_long))
    print(f"  val 이미지 {len(paths)}장", flush=True)

    def run(limit: int, k: int, *, logprobs: bool) -> dict:
        gen_cfg = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1)
        recs = []
        torch.cuda.reset_peak_memory_stats()
        for path in paths[:k]:
            img = Image.open(path).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": prompt}]}]
            enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True,
                                           enable_thinking=False)
            n_prompt = int(enc["input_ids"].shape[1])
            enc = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in enc.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, generation_config=gen_cfg,
                                     return_dict_in_generate=logprobs,
                                     output_logits=logprobs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ids = out.sequences if logprobs else out
            n_new = int(ids.shape[1] - n_prompt)
            lp_bytes = None
            if logprobs:
                # D 07번 계약 — **선택한 토큰의** 로그확률만 줄에 남긴다(생성 토큰 수만큼).
                sel = []
                for step_logits, tok in zip(out.logits, ids[0, n_prompt:]):
                    sel.append(float(torch.log_softmax(step_logits[0].float(), -1)[tok]))
                lp_bytes = len(json.dumps([round(v, 4) for v in sel]).encode())
            recs.append({"n_new_tokens": n_new, "latency_s": round(dt, 3),
                         "n_prompt_tokens": n_prompt,
                         "stop": "length" if n_new >= limit else "eos",
                         "logprob_json_bytes": lp_bytes})
            print(f"    한도 {limit}{' +logprob' if logprobs else ''}: "
                  f"{len(recs)}/{k} {dt:.1f}s {n_new}tok {recs[-1]['stop']}", flush=True)
        lat = [r["latency_s"] for r in recs]
        new = [r["n_new_tokens"] for r in recs]
        return {
            "한도": limit, "n": len(recs), "로그확률_저장": logprobs,
            "지연_s_중앙값": round(statistics.median(lat), 3),
            "지연_s_평균": round(statistics.fmean(lat), 3),
            "지연_s_최소최대": [round(min(lat), 3), round(max(lat), 3)],
            "생성토큰_중앙값": statistics.median(new),
            "생성토큰_최소최대": [min(new), max(new)],
            "초당_토큰_중앙값": round(statistics.median([n / l for n, l in zip(new, lat)]), 2),
            "length_종료_장수": sum(1 for r in recs if r["stop"] == "length"),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "로그확률_줄크기_bytes_중앙값": (
                round(statistics.median([r["logprob_json_bytes"] for r in recs]))
                if logprobs else None),
            "레코드": recs,
        }

    def run_fixed(length: int, k: int, *, logprobs: bool = False) -> dict:
        """**길이를 고정해서** 잰다 — `min_new_tokens = max_new_tokens`.

        길이를 놓아두고 재면 사전학습 모델이 장마다 20토큰에서 한도 소진까지 갈리므로
        (09-21 실측: 10장 중 5장이 512 를 소진, 나머지는 20~60토큰) 평균 지연이 **길이 분포의
        함수**가 된다. 미세조정 뒤 길이 분포는 달라지니 그 평균은 예산에 쓸 수 없다.
        길이를 고정하면 두 점으로 `선행 + 토큰당` 을 분리할 수 있고, 그 둘은 길이에 무관하다.
        """
        gen_cfg = GenerationConfig(max_new_tokens=length, min_new_tokens=length,
                                   do_sample=False, num_beams=1)
        lat, prompts, lp_bytes = [], [], []
        for path in paths[:k]:
            img = Image.open(path).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": prompt}]}]
            enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True,
                                           enable_thinking=False)
            prompts.append(int(enc["input_ids"].shape[1]))
            enc = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in enc.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, generation_config=gen_cfg,
                                     return_dict_in_generate=logprobs,
                                     output_logits=logprobs)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            ids = out.sequences if logprobs else out
            assert int(ids.shape[1]) - prompts[-1] == length, "길이 고정이 듣지 않았다"
            if logprobs:
                sel = [float(torch.log_softmax(sl[0].float(), -1)[tok])
                       for sl, tok in zip(out.logits, ids[0, prompts[-1]:])]
                lp_bytes.append(len(json.dumps([round(v, 4) for v in sel]).encode()))
        print(f"    고정 {length}토큰{' +logprob' if logprobs else ''}: "
              f"{k}장 중앙 {statistics.median(lat):.2f}s", flush=True)
        return {"고정_길이": length, "n": k, "로그확률_저장": logprobs,
                "지연_s_중앙값": round(statistics.median(lat), 3),
                "지연_s_평균": round(statistics.fmean(lat), 3),
                "지연_s_최소최대": [round(min(lat), 3), round(max(lat), 3)],
                "프롬프트_토큰_중앙값": statistics.median(prompts),
                "로그확률_줄크기_bytes_중앙값": (round(statistics.median(lp_bytes))
                                       if lp_bytes else None)}

    # ---- 곡선을 먼저 잰다. 예산 환산에 실제로 쓰이는 값이다. ----
    print("  [곡선] 길이를 고정해 장당 시간을 잰다", flush=True)
    curve = [run_fixed(L, k) for L, k in GEN_CURVE]
    curve_lp = run_fixed(CURVE_LOGPROB_LEN, 6, logprobs=True)

    # 최소제곱 직선 t = a + b·L. 기울기의 역수가 초당 토큰이다.
    xs = [c["고정_길이"] for c in curve]
    ys = [c["지연_s_중앙값"] for c in curve]
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else None

    fit = {
        "선행_s": round(a, 3),
        "토큰당_s": round(b, 5),
        "초당_토큰": round(1 / b, 2) if b > 0 else None,
        "R2": round(r2, 5) if r2 is not None else None,
        "잔차_s": [round(y - (a + b * x), 3) for x, y in zip(xs, ys)],
        "환산식": "장당 초 = 선행_s + 생성토큰수 x 토큰당_s",
        "왜_곡선인가": "한 점만 재면 선행(이미지 880토큰 프리필)과 토큰당 디코딩이 섞여 "
                  "다른 길이로 환산할 수 없다. R2 가 1 에 가까우면 이 두 항으로 충분하다는 뜻이고, "
                  "멀면 길이에 따라 비선형(캐시·메모리)이라 그 사실 자체가 결과다",
    }
    print(f"  [곡선] 선행 {a:.2f}s + 토큰당 {b*1000:.2f}ms "
          f"(초당 {1/b if b>0 else 0:.1f}토큰) R2={r2:.4f}", flush=True)

    # ---- 자유 길이(상한). 사전학습 길이 분포가 섞이므로 상한으로만 쓴다. ----
    runs = [run(GEN_LIMITS[0], n_gen, logprobs=False)]

    del model
    torch.cuda.empty_cache()

    base = runs[0]
    n_eval = 12461

    def per_image_s(n_tok: int) -> float:
        return fit["선행_s"] + fit["토큰당_s"] * n_tok

    lp_extra = None
    if curve_lp["로그확률_줄크기_bytes_중앙값"]:
        same = next(c for c in curve if c["고정_길이"] == CURVE_LOGPROB_LEN)
        lp_extra = {
            "길이": CURVE_LOGPROB_LEN,
            "지연_s_끈것": same["지연_s_중앙값"],
            "지연_s_켠것": curve_lp["지연_s_중앙값"],
            "증가_배": round(curve_lp["지연_s_중앙값"] / same["지연_s_중앙값"], 3),
            "줄당_추가_bytes": curve_lp["로그확률_줄크기_bytes_중앙값"],
            "토큰당_bytes": round(curve_lp["로그확률_줄크기_bytes_중앙값"] / CURVE_LOGPROB_LEN, 2),
            "12461장_5모델_추가_MB": round(
                curve_lp["로그확률_줄크기_bytes_중앙값"] * n_eval * 5 / 1e6, 1),
        }

    return {
        "생성곡선": {"측정": curve, "적합": fit, "로그확률_부담": lp_extra},
        "환산_곡선기준": {
            "n_eval": n_eval,
            "장당_초_예시": {str(L): round(per_image_s(L), 2) for L in (32, 64, 128, 256, 512)},
            "5모델_시간_h_예시": {
                str(L): round(per_image_s(L) * n_eval * 5 / 3600, 1)
                for L in (32, 64, 128, 256, 512)},
            "쓰는법": "SFT 뒤 n_new_tokens 중앙값이 정해지면 그 길이의 칸을 읽는다. "
                   "리허설(12번 8-2 의 9번)이 그 길이를 낸다",
        },
        "실행별_자유길이": runs,
        "환산_자유길이_상한": {
            "n_eval": n_eval,
            "한도512_5모델_h": round(base["지연_s_중앙값"] * n_eval * 5 / 3600, 1),
            "length_종료_장수": base["length_종료_장수"],
            "주의": "사전학습 모델의 길이 분포가 섞인 값이라 상한으로만 쓴다. "
                  "절반이 한도를 소진하므로 평균이 한도에 끌려간다. "
                  "예산 환산은 환산_곡선기준 을 쓴다.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["train", "gen", "both"], default="both")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--train-deadline-s", type=float, default=None,
                    help="학습 측정의 벽시계 한도. 넘기면 잰 스텝만으로 정리한다")
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--n-gen-long", type=int, default=N_GEN_LONG)
    ap.add_argument("--run-id", required=True,
                    help="실행 식별자. 산출 경로가 이것으로 갈린다 — 고정 경로를 덮지 않기 위해 필수다")
    ap.add_argument("--out", default=None, help="산출 JSON 경로를 직접 지정(진단용)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_DIR / f"{args.run_id}_{args.part}.json"
    # **있으면 거부한다.** 계측은 GPU 시간을 쓴 결과이고 덮어쓰면 되돌릴 수 없다.
    if out.exists():
        print(f"이미 있다: {out} — 기존 계측을 덮지 않는다. --run-id 를 바꾼다", flush=True)
        return 2

    rep: dict = {
        "무엇을_쟀나": "4B · 현행 프롬프트 · 템플릿 고정안(enable_thinking=False)의 "
                   "학습 정상상태 단가와 생성 단가",
        "model": MODEL, "커밋": os.popen("git rev-parse --short HEAD").read().strip(),
        "평가셋_무접촉": "학습은 파일럿 페어, 생성은 val 이미지. eval 행·정답 미열람",
    }
    t0 = time.perf_counter()
    if args.part in ("train", "both"):
        print("=== (가) 학습 정상상태 단가", flush=True)
        rep["학습"] = measure_train(warmup_steps=args.warmup, measure_steps=args.steps,
                                   deadline_s=args.train_deadline_s)
    if args.part in ("gen", "both"):
        print("=== (나) 생성 단가", flush=True)
        rep["생성"] = measure_gen(n_gen=args.n_gen, n_gen_long=args.n_gen_long)
    rep["wall_s"] = round(time.perf_counter() - t0, 1)

    rep["part"] = args.part
    rep["run_id"] = args.run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n총 {rep['wall_s']:.0f}s → {out}")
    if "학습" in rep:
        s = rep["학습"]["정상상태"]
        print(f"  학습: 샘플당 {s['샘플당_s_평균']}s (중앙 {s['샘플당_s_중앙값']}) · "
              f"peak {s['peak_vram_gb']}GB · epoch당 {rep['학습']['환산']['epoch당_시간_h_평균']}h")
    if "생성" in rep:
        # `환산_전량평가` 를 찾던 줄이 여기 있었다. `measure_gen` 이 그 키를 내지 않으므로
        # 저장은 끝난 뒤 이 요약에서 KeyError 가 났다 — 측정값을 잃는 오류가 아니라
        # 정상 종료를 막는 오류다. 실제 반환 키 둘로 바꾼다.
        print(f"  생성(곡선 기준 환산): {rep['생성']['환산_곡선기준']}")
        print(f"  생성(자유 길이, 상한): {rep['생성']['환산_자유길이_상한']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
