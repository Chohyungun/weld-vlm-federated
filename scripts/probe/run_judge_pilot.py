"""⑤ 판정부 실행 — 조항 검색 + 기준 서술. 66번 과제 3.

두 실행을 낸다. **섞지 않는다.**

- **주 실행 (사전등록 그대로)**: 분리형 3칸의 계약 #4 레코드를 그대로 받아
  (이미지 × 결함코드)마다 조항을 검색하고 판정부를 태운다. 두께·품질수준은 매니페스트
  실측값이며 **가정값을 채우지 않는다**(`verdict_mode: clause_only`, 가정값 채택은 CTO
  승인 사항). 이것이 보고 대상 결과다.
- **생성부 시운전 (배관 검증, 지표 아님)**: 사전 등록된 임베딩 실측 질의 100건
  (`queries_from_gold_rows`, 시드 20260825)을 그대로 재사용해 검색이 실제로 걸리는
  입력으로 생성부만 돌린다. 질의의 두께는 **허용치 표 자신의 구간에서 나온 값**이라
  새 가정을 도입하지 않는다. 여기서 나오는 수치는 배관·인용 접지 진단이지 판정 근거
  신뢰도 지표가 아니다 — 보고에서 라벨을 붙여 분리한다.

**greedy 1회, 재시도·재프롬프트 금지.** 생성 호출 횟수를 직접 센다.
장기 GPU 작업이므로 pane 프로세스 트리에서 분리해 띄운다(C 의 고장 6 처방).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from evaluation.adapters import read_records
from evaluation.eval_set import eval_rows as select_eval
from evaluation.eval_set import parse_iso_codes, read_manifest
from evaluation.gold import build_gold_pairs, entries_from_derived, read_derived_csv
from evaluation.gold import ImageContext
from evaluation.metrics.clause import score_citation, score_retrieval
from rag.index import load_chunks, load_rag_config, queries_from_gold_rows
from rag.judge import PROMPT_PATH, load_prompt_template, run_judge
from rag.retrieve import NO_CLAUSE, Query, retrieve

SNAP = Path("data/processed/aihub71761_rt_v1_pilot3000")
OUT = Path("outputs/pilot_d")
SEED = 20260828
CELLS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")


def _dec(v: object) -> Decimal | None:
    return None if v in (None, "", "none") else Decimal(str(v))


class Generator:
    """greedy 1회 생성기. 호출 횟수를 센다 — 재시도 금지가 코드로 확인된다."""

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0}
        )
        self.model.eval()
        # 배포 generation_config 를 신뢰하지 않고 백지에서 구성한다 (레지스트리 #5)
        self.gen = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )
        assert self.gen.do_sample is False
        self.n_calls = 0
        self.latencies: list[float] = []
        self.out_tokens: list[int] = []

    def __call__(self, prompt: str) -> str:
        self.n_calls += 1
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        enc = self.proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**enc, generation_config=self.gen)
        self.latencies.append((time.perf_counter() - t0) * 1000)
        new = ids[:, enc["input_ids"].shape[1]:]
        self.out_tokens.append(int(new.shape[1]))
        return self.proc.batch_decode(new, skip_special_tokens=True)[0]


def raise_if_called(_prompt: str) -> str:
    raise AssertionError("검색 0건 경로에서 생성이 호출됐다 — judge_image 규약 위반")


def main() -> int:
    main_only = "--main-only" in sys.argv     # GPU 없이 주 실행만 검증할 때
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_rag_config()
    chunks = load_chunks(cfg.chunk_meta)
    index_ids = {c.chunk_id for c in chunks}
    grade_map: dict = {}
    _, prompt_sha = load_prompt_template(PROMPT_PATH)

    rows = read_manifest(SNAP)
    ev = select_eval(rows)
    by_id = {r["image_id"]: r for r in ev}

    gold_rows = read_derived_csv("corpus/derived/gold_clauses.csv")
    entries = entries_from_derived(gold_rows)

    # 정답 쌍 — 두께 결측이면 조회가 실패한다. 조용히 넘기지 않고 건수를 남긴다.
    contexts = [
        ImageContext(
            image_id=r["image_id"], inspection_method=r["modality"], material=r["material"],
            thickness_mm=_dec(r["thickness_mm"]), quality_scheme="none",
            quality_level=r["quality_level"] or "ALL",
        )
        for r in ev
    ]
    gt_codes: dict[str, list[str]] = {}
    for r in ev:
        gt_codes[r["image_id"]] = list(parse_iso_codes(r["iso_codes"]))
    gold_pairs, gold_skipped = build_gold_pairs(entries, contexts, gt_codes)
    print(f"정답 조항 쌍 {len(gold_pairs)}건 · 조회 실패 {gold_skipped}", flush=True)

    # ---------------- 주 실행: 사전등록 그대로 -------------------------------------
    main_run: dict[str, dict] = {}
    for cell in CELLS:
        path = OUT / f"{cell}_s{SEED}.jsonl"
        recs = read_records(path.read_text(encoding="utf-8").splitlines())
        items = []
        retrieved_pairs: dict[tuple[str, str], list[str]] = {}
        for rec in recs:
            ctx = by_id[rec.image_id]
            hits: list[str] = []
            cand_chunks = []
            for code in sorted(rec.iso_codes):
                q = Query(
                    inspection_method=ctx["modality"], defect_code=code,
                    thickness_mm=_dec(ctx["thickness_mm"]),
                    quality_scheme="none", quality_level=ctx["quality_level"] or None,
                )
                r = retrieve(chunks, q, grade_map, top_k=cfg.top_k)
                retrieved_pairs[(rec.image_id, code)] = list(r.chunk_ids)
                for cid in r.chunk_ids:
                    if cid not in hits:
                        hits.append(cid)
                        cand_chunks.append(next(c for c in chunks if c.chunk_id == cid))
            from rag.retrieve import RetrievalResult
            union = RetrievalResult(tuple(hits), False, len(hits),
                                    "" if hits else NO_CLAUSE)
            items.append((
                rec.image_id,
                [{"iso_code": d.iso_code, "size_px": d.size_px} for d in rec.defects],
                union, cand_chunks,
            ))
        report = run_judge(items, raise_if_called, prompt_path=PROMPT_PATH)
        retr = score_retrieval(retrieved_pairs, gold_pairs)
        cited = {o.image_id: o.cited_clauses for o in report.outputs}
        cite = score_citation(cited, gold_pairs)
        main_run[cell] = {
            **report.as_dict(),
            "n_queries": len(retrieved_pairs),
            "n_queries_with_hit": sum(1 for v in retrieved_pairs.values() if v),
            "retrieval": retr.as_dict(),
            "citation": cite.as_dict(),
        }
        print(f"[{cell}] 이미지 {report.as_dict()['n_images']} · 질의 {len(retrieved_pairs)} "
              f"· 검색 적중 {main_run[cell]['n_queries_with_hit']} "
              f"· 생성 {report.as_dict()['n_generated']} "
              f"· 무검색 {report.as_dict()['n_no_hit']}", flush=True)

    if main_only:
        dest = OUT / "judge_pilot_main_only.json"
        dest.write_text(json.dumps({"main_run": main_run}, ensure_ascii=False, indent=2)
                        + "\n", encoding="utf-8")
        print(f"주 실행만 저장: {dest}", flush=True)
        return 0

    # ---------------- 생성부 시운전: 검색이 걸리는 입력으로만 ------------------------
    queries = queries_from_gold_rows(gold_rows, n=cfg.trial_n_queries, seed=cfg.trial_seed)
    if cfg.judge_checkpoint is None:
        print("judge.checkpoint 미기재 — configs/rag.yaml 을 먼저 채운다")
        return 1
    gen = Generator(cfg.judge_checkpoint, cfg.judge_max_new_tokens)
    smoke_items = []
    for i, (q, gold) in enumerate(queries):
        r = retrieve(chunks, q, grade_map, top_k=cfg.top_k)
        cand = [c for c in chunks if c.chunk_id in r.chunk_ids]
        smoke_items.append((
            f"질의{i:03d}",
            [{"iso_code": q.defect_code, "size_px": None}],
            r, cand,
        ))
    t0 = time.perf_counter()
    smoke = run_judge(smoke_items, gen, prompt_path=PROMPT_PATH)
    dt = time.perf_counter() - t0

    presented = {item[0]: {c.chunk_id for c in item[3]} for item in smoke_items}
    gold_by_q = {f"질의{i:03d}": g for i, (_, g) in enumerate(queries)}
    n_cit = n_outside = n_not_in_index = n_gold_cited = 0
    for o in smoke.outputs:
        for c in o.cited_clauses:
            n_cit += 1
            if c not in index_ids:
                n_not_in_index += 1
            elif c not in presented[o.image_id]:
                n_outside += 1
        n_gold_cited += int(gold_by_q[o.image_id] in set(o.cited_clauses))

    smoke_report = {
        **smoke.as_dict(),
        "n_generation_calls": gen.n_calls,
        "greedy_once": gen.n_calls == sum(1 for o in smoke.outputs if o.generated),
        "wall_seconds": round(dt, 1),
        "latency_ms_median": float(sorted(gen.latencies)[len(gen.latencies) // 2])
        if gen.latencies else None,
        "out_tokens_max": max(gen.out_tokens) if gen.out_tokens else 0,
        "out_tokens_median": sorted(gen.out_tokens)[len(gen.out_tokens) // 2]
        if gen.out_tokens else 0,
        "max_new_tokens": gen.gen.max_new_tokens,
        "citations": {
            "n_citations": n_cit,
            "n_not_in_index": n_not_in_index,
            "n_in_index_but_not_presented": n_outside,
            "n_gold_clause_cited": n_gold_cited,
            "gold_citation_rate": n_gold_cited / len(smoke.outputs) if smoke.outputs else 0.0,
        },
        "verdicts": dict(Counter(o.verdict for o in smoke.outputs)),
        "label": "배관 시운전 — 판정 근거 신뢰도 지표가 아니다",
    }
    with (OUT / "judge_smoke_outputs.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for o in smoke.outputs:
            fh.write(json.dumps({**o.as_row(), "raw": o.raw_text}, ensure_ascii=False) + "\n")

    payload = {
        "prompt_sha256": prompt_sha,
        "checkpoint": cfg.judge_checkpoint,
        "precision": "bfloat16 (판정부는 어댑터 없이 기본 체크포인트로 돈다)",
        "config": {"top_k": cfg.top_k, "no_hit_output": cfg.no_hit_output},
        "gold_pairs": {"n": len(gold_pairs), "skipped": gold_skipped},
        "main_run": main_run,
        "smoke_run": smoke_report,
    }
    dest = OUT / "judge_pilot_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"시운전: 생성 {gen.n_calls}회 · 실패율 {smoke.failure_rate:.4f} "
          f"{smoke.failure_counts} · 인용 {n_cit}건 (색인 밖 {n_not_in_index}, "
          f"제시 밖 {n_outside}) · 정답조항 인용률 "
          f"{n_gold_cited}/{len(smoke.outputs)}", flush=True)
    print(f"저장: {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
