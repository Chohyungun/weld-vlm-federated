"""⑤ 판정부 소요·VRAM 프로브 — 67번 §4-1 프로브 3. 71번 과제 4.

61번에 ⑤ 실측이 아예 없어 67번 §4-4 가 "1~7일" 자리값으로 채웠다. 그 폭을 닫는다.

**평가셋을 열지 않는다.** 67번 §5-4 가 "5칸 공통 디코딩 상수를 평가셋 통계로 맞추면
불변조건 1-4 를 실질에서 깬다"고 못박았고, 이 프로브의 산출(이미지당 시간·토큰·VRAM)은
그대로 `max_new_tokens` 와 예산 결정의 입력이 된다. 따라서 **train 분할에서만** 표본을
뽑는다. eval 12,461장으로의 외삽은 장수 곱셈만 하고 eval 라벨 통계는 쓰지 않는다.

프롬프트 입력은 **GT 결함 목록**이다 — 본실험 검출기가 GT 수준으로 검출하는 경우의
상한 프로파일이며, 미학습 파일럿 검출기(발화 10%)로 재면 본실험 비용을 과소평가한다.

검색은 실경로(`rag.retrieve.retrieve`)를 그대로 탄다. 두께는 이미지에 없으므로
**허용치 표 자신의 구간에서 유도한 대표 두께**를 쓴다(66번 §9-2 와 같은 근거) —
새 가정값을 도입하지 않으며, 이 프로브의 산출물은 소요·VRAM 뿐이고 지표가 아니다.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from evaluation.eval_set import read_manifest
from evaluation.gold import read_derived_csv
from rag.index import load_chunks, load_rag_config
from rag.judge import PROMPT_PATH, load_prompt_template, run_judge
from rag.retrieve import NO_CLAUSE, Query, RetrievalResult, retrieve

SNAP = Path("data/processed/aihub71761_rt_v1_pilot3000")
OUT = Path("outputs/pilot_d")
N_SAMPLE = 653          # 파일럿 평가셋과 같은 장수 — 단, 표본은 train 에서 뽑는다
SAMPLE_SEED = 20260825
EVAL_IMAGES_MAIN = 12461   # 동결 평가셋 장수(외삽 곱셈에만 쓴다. 라벨 통계 아님)


def representative_thickness(gold_rows) -> dict[tuple[str, str], Decimal]:
    """(검사방식, 결함코드) → 대표 두께. **허용치 표 자신의 구간 중점**이다.

    가정값이 아니다 — 표에 적힌 하한·상한에서 결정론적으로 유도한다. 상한이 비었으면
    하한 + 10 을 쓴다. 프로브가 검색 실경로를 타게 하는 것이 목적이고, 이 값으로 만든
    어떤 수치도 지표로 보고하지 않는다.
    """
    best: dict[tuple[str, str], Decimal] = {}
    for r in sorted(gold_rows, key=lambda x: str(x["rule_id"])):
        key = (str(r["inspection_method"]), str(r["defect_code"]))
        if key in best:
            continue
        lo = Decimal(str(r["thickness_min"] or 0))
        hi_raw = r["thickness_max"]
        hi = Decimal(str(hi_raw)) if hi_raw not in (None, "") else lo + Decimal(10)
        best[key] = (lo + hi) / 2
    return best


class Generator:
    """greedy 1회 생성기. 호출 수·지연·토큰을 센다."""

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

        torch.cuda.reset_peak_memory_stats()
        self.mem_before = torch.cuda.memory_allocated()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0}
        )
        self.model.eval()
        self.mem_weights = torch.cuda.memory_allocated()
        self.gen = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )
        assert self.gen.do_sample is False
        self.n_calls = 0
        self.latencies: list[float] = []
        self.in_tokens: list[int] = []
        self.out_tokens: list[int] = []

    def __call__(self, prompt: str) -> str:
        self.n_calls += 1
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        enc = self.proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**enc, generation_config=self.gen)
        torch.cuda.synchronize()
        self.latencies.append((time.perf_counter() - t0) * 1000)
        new = ids[:, n_in:]
        self.in_tokens.append(n_in)
        self.out_tokens.append(int(new.shape[1]))
        return self.proc.batch_decode(new, skip_special_tokens=True)[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_rag_config()
    chunks = load_chunks(cfg.chunk_meta)
    gold_rows = read_derived_csv("corpus/derived/gold_clauses.csv")
    thick = representative_thickness(gold_rows)

    rows = read_manifest(SNAP)
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["image_id"])
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(train), size=min(N_SAMPLE, len(train)), replace=False)
    sample = [train[i] for i in sorted(idx.tolist())]
    ids = {r["image_id"] for r in sample}

    # GT 결함 — train 분할 한정. eval 어노테이션은 읽지 않는다.
    defects: dict[str, list[dict]] = defaultdict(list)
    with (SNAP / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for a in csv.DictReader(fh):
            if a["image_id"] not in ids:
                continue
            defects[a["image_id"]].append({
                "iso_code": a["iso_code"],
                "size_px": float(a["major_axis_px"]) if a.get("major_axis_px") else None,
            })

    items = []
    n_no_defect = 0
    for r in sample:
        ds = defects.get(r["image_id"], [])
        if not ds:
            n_no_defect += 1
        hits: list[str] = []
        cand = []
        for code in sorted({d["iso_code"] for d in ds}):
            q = Query(
                inspection_method=r["modality"], defect_code=code,
                thickness_mm=thick.get((r["modality"], code)),
                quality_scheme="none", quality_level=None,
            )
            res = retrieve(chunks, q, {}, top_k=cfg.top_k)
            for cid in res.chunk_ids:
                if cid not in hits:
                    hits.append(cid)
                    cand.append(next(c for c in chunks if c.chunk_id == cid))
        union = RetrievalResult(tuple(hits), False, len(hits), "" if hits else NO_CLAUSE)
        items.append((r["image_id"], ds, union, cand))

    if cfg.judge_checkpoint is None:
        print("judge.checkpoint 미기재"); return 1
    _, prompt_sha = load_prompt_template(PROMPT_PATH)
    gen = Generator(cfg.judge_checkpoint, cfg.judge_max_new_tokens)
    print(f"표본 {len(items)}장(train) · 결함 0개 {n_no_defect}장 · "
          f"가중치 VRAM {gen.mem_weights / 2**30:.3f} GiB", flush=True)

    t0 = time.perf_counter()
    report = run_judge(items, gen, prompt_path=PROMPT_PATH)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()

    n_gen = gen.n_calls
    per_image_all = wall / len(items)
    per_gen = wall / n_gen if n_gen else 0.0

    payload = {
        "role": "소요·VRAM 프로브. 지표가 아니다",
        "checkpoint": cfg.judge_checkpoint,
        "precision": "bfloat16",
        "prompt_sha256": prompt_sha,
        "max_new_tokens": cfg.judge_max_new_tokens,
        "sample": {
            "split": "train", "n": len(items), "seed": SAMPLE_SEED,
            "n_images_without_defect": n_no_defect,
            "note": "eval 분할을 열지 않았다 (67번 §5-4)",
        },
        "generation": {
            "n_calls": n_gen,
            "n_no_hit": report.as_dict()["n_no_hit"],
            "greedy_once": n_gen == report.as_dict()["n_generated"],
            "failure_rate": report.failure_rate,
            "failure_counts": report.failure_counts,
            "wall_seconds": round(wall, 1),
            "seconds_per_image_all": round(per_image_all, 4),
            "seconds_per_generation": round(per_gen, 4),
            "latency_ms": {
                "p50": round(statistics.median(gen.latencies), 1) if gen.latencies else None,
                "p90": round(sorted(gen.latencies)[int(0.9 * len(gen.latencies))], 1)
                if gen.latencies else None,
                "max": round(max(gen.latencies), 1) if gen.latencies else None,
            },
            "prompt_tokens": {
                "p50": int(statistics.median(gen.in_tokens)) if gen.in_tokens else None,
                "max": max(gen.in_tokens) if gen.in_tokens else None,
            },
            "output_tokens": {
                "p50": int(statistics.median(gen.out_tokens)) if gen.out_tokens else None,
                "max": max(gen.out_tokens) if gen.out_tokens else None,
                "n_at_cap": sum(1 for t in gen.out_tokens if t >= cfg.judge_max_new_tokens),
            },
        },
        "vram": {
            "weights_gib": round(gen.mem_weights / 2**30, 4),
            "peak_allocated_gib": round(peak / 2**30, 4),
            "peak_reserved_gib": round(reserved / 2**30, 4),
        },
        "extrapolation": {
            "eval_images_main": EVAL_IMAGES_MAIN,
            "hours_per_pass_0p8b": round(EVAL_IMAGES_MAIN * per_image_all / 3600, 2),
            "note": (
                "장수 곱셈만 했다. 본실험 결함 밀도가 파일럿과 다르면 프롬프트 길이가 "
                "달라지는데, 그 보정은 train+val 결함 수 분포(A 산출)로 해야 한다"
            ),
        },
    }
    dest = OUT / "judge_cost_probe_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"생성 {n_gen}회 · 벽시계 {wall:.0f}s · 이미지당 {per_image_all:.3f}s "
          f"· 생성당 {per_gen:.3f}s", flush=True)
    print(f"VRAM 가중치 {gen.mem_weights / 2**30:.3f} / peak {peak / 2**30:.3f} GiB", flush=True)
    print(f"저장: {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
