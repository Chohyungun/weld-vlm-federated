"""학습 한 행의 시간을 나눈다 — 입력 준비(CPU) · 순전파 · 역전파 · 최적화 단계.

`synth_rate.py --part train` 의 샘플당 시간이 GPU 연산으로 정해지는지, CPU 쪽 준비와 커널 띄우기로 정해지는지를 가린다.
같은 합성 행(혼합 조건)과 같은 부품을 쓰고, 구간마다 `torch.cuda.synchronize()` 로 끊는다 — 끊는 비용만큼 합이
`synth_rate.py` 의 샘플당보다 조금 크다. 실제 자료를 열지 않는다.

**이 값은 단계별 병목 진단값이다.** 행별 `total_s` 는 최적화 단계를 빼고(최적화는 모든 행 뒤 한 번만 잰다) 창 재정규화도 하지 않는다 —
누적 · 정규화를 따르는 학습의 샘플당 초와 같은 양이 아니므로 장비 배율의 분자 · 분모로 쓰지 않는다. 합성 폴더 규칙 · 입력 해시 ·
적재 판의 결속은 `synth_rate.py` 와 같다.

    uv run python scripts/probe/synth_rate_split.py --work <합성 폴더> --out <JSON> [--rows 64]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

import torch

from scripts.probe.synth_rate import (
    GEN_SEED,
    USED_IMAGES,
    WORK_RULE,
    SynthWorkRefused,
    _cfg,
    _env,
    _mix_ks,
    _prompt_bytes,
    images_sha256,
    loaded_identity,
    make_rows,
    prepare_work,
    rows_sha256,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--warmup-rows", type=int, default=8)
    a = ap.parse_args()
    if a.out.exists():
        print(f"[out_exists] {a.out.name} — 덮지 않는다", file=sys.stderr)
        return 2
    try:
        prepare_work(a.work)                                      # 모델을 올리기 전에 폴더부터 맞댄다
    except SynthWorkRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    import hashlib

    from vlm.loss_norm import supervised_ce_sum
    from vlm.pilot_vlm import LR, PROMPT_PATH, _encode, _load_model

    cfg = _cfg()
    model, proc = _load_model(cfg.model_id, revision=cfg.model_revision, processor_kwargs=dict(cfg.processor_kwargs or {}))
    loaded = loaded_identity(model, proc, cfg)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    model.train()
    rows = make_rows(a.work, _mix_ks(a.warmup_rows + a.rows), tag="split")
    rec = {"encode_s": [], "to_gpu_s": [], "forward_s": [], "backward_s": [], "total_s": []}
    for j, row in enumerate(rows):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc, labels, prompt_len = _encode(proc, row, prompt)
        t1 = time.perf_counter()
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
        labels = labels.to("cuda")
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        n_keep = int(labels.shape[1] - prompt_len + 1)
        out = model(**enc, logits_to_keep=n_keep)
        ce, _n = supervised_ce_sum(out.logits[:, :-1], labels[:, -(n_keep - 1):])
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        ce.backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if j >= a.warmup_rows:
            for k, v in (("encode_s", t1 - t0), ("to_gpu_s", t2 - t1), ("forward_s", t3 - t2), ("backward_s", t4 - t3),
                         ("total_s", t4 - t0)):
                rec[k].append(v)
    t5 = time.perf_counter()
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    step_s = time.perf_counter() - t5
    med = {k: round(statistics.median(v), 4) for k, v in rec.items()}
    share = {k: round(med[k] / med["total_s"], 3) for k in ("encode_s", "to_gpu_s", "forward_s", "backward_s")}
    rep = {"kind": "synth_rate_split", "env": _env(), "gen_seed": GEN_SEED, "rows": a.rows, "warmup_rows": a.warmup_rows,
           "median": med, "share_of_total": share, "optimizer_step_s": round(step_s, 4),
           "torch_threads": torch.get_num_threads(),
           "use": "단계별 병목 진단값 — 최적화 제외 · 창 재정규화 없음. 학습 샘플당 초나 장비 배율로 쓰지 않는다",
           "model": {"model_id": cfg.model_id, "revision": cfg.model_revision, "processor_kwargs": dict(cfg.processor_kwargs or {}),
                     "prompt_sha256": hashlib.sha256(_prompt_bytes()).hexdigest()},
           "loaded": loaded, "rows_sha256": rows_sha256(rows),
           "synth_inputs": {"work_rule": WORK_RULE, "images": dict(sorted(USED_IMAGES.items())), "images_sha256": images_sha256()},
           "inputs": "합성 이미지 1280x720 · 합성 타깃 — 실제 자료를 열지 않는다"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"median": med, "share": share}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
